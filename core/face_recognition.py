"""
人脸识别模块
============
基于 OpenCV 内置的 YuNet（检测）+ SFace（识别），无需额外依赖。

存储布局（按人分目录）：
    data/faces/
    ├── 武凉烨/
    │   ├── 1.jpg          # 原图，可人工查看/删除
    │   ├── 2.jpg
    │   └── 3.jpg
    └── 李雷/
        └── 1.jpg
    data/face_database.json   # 特征索引，与目录一一对应

为什么要存原图 + 一人多照片：
  1) 认不出时能人工看一眼当初注册的是哪张脸（旧版拍完就丢，无从排查）
  2) 换模型/调分辨率时可重新提特征，不必让人重拍
  3) 单张特征对姿态/光照极敏感——实测同一个人在不同帧会「时认时不认」。
     多张照片时**只要有一张角度匹配上就能认出**，鲁棒性显著提升。

关键修复（旧版的致命 bug）：
  json.load 读出的特征是 Python float = float64，而 OpenCV feature() 返回
  float32。两者送进 recognizer.match() 会抛：
      (-5:Bad argument) ... different types ...
  导致「人脸库非空时身份识别必然失败」。现在统一强制 float32。
"""
from __future__ import annotations

import json
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import DATA_DIR, FaceConfig

# 模型存放目录
MODEL_DIR = Path(__file__).parent.parent / "models" / "face_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 模型下载地址（多个镜像，国内优先）
MODEL_URLS = {
    "yunet": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://gitee.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ],
    "sface": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://gitee.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ],
}


class FaceQualityError(Exception):
    """注册照片质量不达标"""


class FaceRecognizer:
    """人脸识别封装（支持一人多照片）"""

    def __init__(self, config: FaceConfig = None):
        self.config = config or FaceConfig()
        self.threshold = self.config.THRESHOLD
        self.detector = None
        self.recognizer = None

        # {姓名: {"photos": [文件名...], "features": [np.float32(128), ...]}}
        self.face_db: dict[str, dict] = {}

        self.face_dir: Path = Path(self.config.FACE_DIR)
        self.db_path: Path = Path(self.config.DB_FILE)
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self._load_db()

    # ==================================================================
    #  模型
    # ==================================================================
    def _download_model(self, model_name: str) -> Path:
        filename = MODEL_URLS[model_name][0].split("/")[-1]
        local_path = MODEL_DIR / filename
        if local_path.exists() and local_path.stat().st_size > 1000:
            return local_path

        print(f"[人脸模型] 正在下载 {filename} ...")
        last_error = None
        for url in MODEL_URLS[model_name]:
            try:
                urllib.request.urlretrieve(url, str(local_path))
                if local_path.exists() and local_path.stat().st_size > 1000:
                    print(f"[人脸模型] 下载完成: {filename} ({local_path.stat().st_size//1024}KB)")
                    return local_path
            except Exception as e:
                last_error = e
                print(f"[人脸模型] 从 {url} 下载失败: {e}")
                continue

        if local_path.exists() and local_path.stat().st_size > 0:
            print(f"[人脸模型] 使用本地已存在的 {filename}（下载未成功，可能不完整）")
            return local_path
        raise RuntimeError(
            f"模型 {filename} 下载失败，请手动下载放到 {MODEL_DIR} 目录\n"
            f"下载地址: {MODEL_URLS[model_name][0]}\n最后错误: {last_error}"
        )

    def load(self, silent: bool = False, allow_download: bool = True) -> bool:
        """加载检测/识别模型

        Args:
            silent: 不打印日志（后台监控高频调用时用）
            allow_download: False 时不允许联网下载（避免后台线程阻塞在网络上）
        """
        if self.detector and self.recognizer:
            return True

        if not allow_download:
            yunet = MODEL_DIR / MODEL_URLS["yunet"][0].split("/")[-1]
            sface = MODEL_DIR / MODEL_URLS["sface"][0].split("/")[-1]
            if not (yunet.exists() and sface.exists()):
                if not silent:
                    print("[人脸识别] 模型文件不存在，且当前不允许下载，跳过加载")
                return False

        try:
            yunet_path = self._download_model("yunet")
            sface_path = self._download_model("sface")
        except Exception as e:
            if not silent:
                print(f"[人脸识别] 模型准备失败: {e}")
            return False

        size = int(getattr(self.config, "DETECT_SIZE", 640))
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=str(yunet_path),
                config="",
                input_size=(size, size),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
            )
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=str(sface_path),
                config="",
            )
        except Exception as e:
            if not silent:
                print(f"[人脸识别] 模型初始化失败: {e}")
            self.detector = None
            self.recognizer = None
            return False

        if not silent:
            print(f"[人脸识别] 模型加载完成（检测分辨率 {size}x{size}，"
                  f"阈值 {self.threshold}）")
        return True

    # ==================================================================
    #  人脸库读写
    # ==================================================================
    def _load_db(self):
        """加载人脸库

        兼容两种格式：
          旧: {姓名: [128个float]}                 -> 自动升级为新格式
          新: {姓名: {"photos": [...], "features": [[...], ...]}}
        """
        if not self.db_path.exists():
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[人脸库] 加载失败: {e}")
            return

        upgraded = 0
        for name, value in raw.items():
            if isinstance(value, dict):
                feats = [np.asarray(v, dtype=np.float32).reshape(-1)
                         for v in value.get("features", [])]
                photos = list(value.get("photos", []))
                self.face_db[name] = {"photos": photos, "features": feats}
            else:
                # 旧格式：只有一组特征，没有原图
                feats = [np.asarray(value, dtype=np.float32).reshape(-1)]
                self.face_db[name] = {"photos": [], "features": feats}
                upgraded += 1

        total_photos = sum(len(v["features"]) for v in self.face_db.values())
        print(f"[人脸库] 已加载 {len(self.face_db)} 人 / {total_photos} 组特征"
              + (f"（其中 {upgraded} 条从旧格式升级，原图缺失）" if upgraded else ""))

    def _save_db(self):
        """保存人脸库（统一 float32，避免精度漂移）"""
        data = {}
        for name, entry in self.face_db.items():
            data[name] = {
                "photos": list(entry.get("photos", [])),
                "features": [np.asarray(f, dtype=np.float32).reshape(-1).tolist()
                             for f in entry.get("features", [])],
            }
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ==================================================================
    #  检测与特征
    # ==================================================================
    def detect_faces(self, image: np.ndarray) -> list:
        """检测人脸，返回 [[x, y, w, h, ...], ...]"""
        if not self.load(silent=True, allow_download=False):
            return []
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []
        return faces.tolist()

    def extract_feature(self, image: np.ndarray,
                        face_box: list = None) -> Optional[np.ndarray]:
        """提取人脸特征（强制 float32，与库内保持一致）"""
        if not self.load(silent=True, allow_download=False):
            return None

        if face_box is None:
            faces = self.detect_faces(image)
            if not faces:
                return None
            face_box = max(faces, key=lambda f: f[2] * f[3])

        face_box_np = np.array(face_box, dtype=np.float32)
        aligned = self.recognizer.alignCrop(image, face_box_np)
        feature = self.recognizer.feature(aligned)
        return np.asarray(feature, dtype=np.float32).reshape(-1)

    # ==================================================================
    #  注册质量检查
    # ==================================================================
    @staticmethod
    def _sharpness(image: np.ndarray) -> float:
        """拉普拉斯方差，越小越模糊"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def check_quality(self, image: np.ndarray) -> tuple[bool, str, Optional[list]]:
        """检查这张照片是否适合注册

        Returns: (是否合格, 说明, 选中的脸框)
        """
        cfg = self.config
        if not self.load(silent=True, allow_download=False):
            return False, "人脸识别模型未就绪", None

        h, w = image.shape[:2]
        frame_area = float(h * w)
        faces = self.detect_faces(image)

        if not faces:
            return False, "没有检测到人脸，请正对摄像头", None
        if cfg.REQUIRE_SINGLE_FACE and len(faces) > 1:
            return False, f"画面里有 {len(faces)} 张脸，请确保只有你一个人", None

        face = max(faces, key=lambda f: f[2] * f[3])
        ratio = (face[2] * float(face[3])) / max(frame_area, 1.0)
        if ratio < cfg.MIN_FACE_RATIO:
            return False, f"脸太小（占画面 {ratio*100:.1f}%），请靠近摄像头", None

        # YuNet 输出的最后一位是检测置信度
        score = float(face[-1]) if len(face) > 14 else 1.0
        if score < cfg.MIN_DET_SCORE:
            return False, f"人脸不够清晰（置信度 {score:.2f}），请正对摄像头", None

        sharp = self._sharpness(image)
        if sharp < cfg.MIN_SHARPNESS:
            return False, f"画面偏模糊（清晰度 {sharp:.0f}），请拿稳或调亮光线", None

        return True, f"质量合格（占比 {ratio*100:.1f}%，清晰度 {sharp:.0f}）", face

    # ==================================================================
    #  注册
    # ==================================================================
    def _person_dir(self, name: str) -> Path:
        d = self.face_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _next_photo_name(self, name: str) -> str:
        d = self._person_dir(name)
        used = set()
        for p in d.glob("*.jpg"):
            try:
                used.add(int(p.stem))
            except ValueError:
                continue
        i = 1
        while i in used:
            i += 1
        return f"{i}.jpg"

    def register_face(self, name: str, image: np.ndarray,
                      quality_checked: bool = False,
                      face_box: list = None) -> tuple[bool, str]:
        """注册一张人脸照片（追加，不覆盖）

        Args:
            name: 姓名
            image: BGR 图像
            quality_checked: 是否已通过 check_quality（避免重复检测）
            face_box: 已知的人脸框
        Returns: (是否成功, 说明)
        """
        name = (name or "").strip()
        if not name:
            return False, "姓名不能为空"

        if not quality_checked:
            ok, msg, face = self.check_quality(image)
            if not ok:
                return False, msg
            face_box = face

        feature = self.extract_feature(image, face_box)
        if feature is None:
            return False, "无法提取人脸特征，请重试"

        entry = self.face_db.setdefault(name, {"photos": [], "features": []})
        if len(entry["features"]) >= self.config.MAX_PHOTOS_PER_PERSON:
            return False, (f"{name} 已有 {len(entry['features'])} 张照片，"
                           f"达到上限（{self.config.MAX_PHOTOS_PER_PERSON}）。"
                           f"可用 /forget {name} N 删除某张后再加")

        # 保存原图（保留证据，便于日后排查）
        fname = self._next_photo_name(name)
        path = self._person_dir(name) / fname
        try:
            cv2.imwrite(str(path), image)
        except Exception as e:
            return False, f"照片保存失败: {e}"

        entry["photos"].append(fname)
        entry["features"].append(np.asarray(feature, dtype=np.float32).reshape(-1))
        self._save_db()

        n = len(entry["features"])
        return True, f"已记住 {name} 的第 {n} 张（照片: {path}）"

    # ==================================================================
    #  识别
    # ==================================================================
    def recognize(self, image: np.ndarray) -> list:
        """识别画面中所有人脸

        Returns: [(姓名, 置信度, 脸框), ...]，未匹配到返回 ("unknown", 0.0, box)
        """
        if not self.load(silent=True, allow_download=False):
            return []
        faces = self.detect_faces(image)
        if not faces:
            return []

        results = []
        for face_box in faces:
            feature = self.extract_feature(image, face_box)
            if feature is None:
                results.append(("unknown", 0.0, face_box))
                continue

            feature = np.asarray(feature, dtype=np.float32).reshape(-1)
            best_name, best_score = "unknown", 0.0

            for name, entry in self.face_db.items():
                # 一人多特征：取该人的**最高分**。
                # 比"取平均"更好——只要有一张照片角度匹配上就能认出，
                # 不会被那张拍歪的照片拉低平均分。
                person_best = 0.0
                for db_feature in entry.get("features", []):
                    db_feature = np.asarray(db_feature, dtype=np.float32).reshape(-1)
                    try:
                        score = float(self.recognizer.match(
                            feature, db_feature, cv2.FaceRecognizerSF_FR_COSINE))
                    except cv2.error as e:
                        print(f"[人脸] 与 {name} 比对失败（已跳过）: {e}")
                        continue
                    if score > person_best:
                        person_best = score
                if person_best > best_score:
                    best_score, best_name = person_best, name

            if best_score < self.threshold:
                best_name = "unknown"
            results.append((best_name, round(best_score, 3), face_box))

        return results

    # ==================================================================
    #  查询与删除
    # ==================================================================
    def get_registered_names(self) -> list:
        return list(self.face_db.keys())

    def get_photo_count(self, name: str) -> int:
        entry = self.face_db.get(name)
        return len(entry.get("features", [])) if entry else 0

    def delete_face(self, name: str) -> bool:
        """删除整个人的记录（含照片目录）"""
        if name not in self.face_db:
            return False
        del self.face_db[name]
        self._save_db()
        d = self.face_dir / name
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        print(f"[删除] 已移除 {name}（含照片目录）")
        return True

    def delete_photo(self, name: str, index: int) -> tuple[bool, str]:
        """删除某人的第 index 张（从 1 开始）"""
        entry = self.face_db.get(name)
        if not entry:
            return False, f"没有找到 {name}"
        n = len(entry["features"])
        if not (1 <= index <= n):
            return False, f"{name} 只有 {n} 张照片，序号 {index} 无效"

        i = index - 1
        photos = entry.get("photos", [])
        if i < len(photos) and photos[i]:
            p = self.face_dir / name / photos[i]
            if p.exists():
                p.unlink()
            photos.pop(i)
        entry["features"].pop(i)
        self._save_db()
        remain = len(entry["features"])
        if remain == 0:
            return self.delete_face(name), f"{name} 已无照片，已整体移除"
        return True, f"已删除 {name} 的第 {index} 张，剩余 {remain} 张"

    def clear_db(self):
        """清空整个人脸库（含所有照片目录）"""
        names = list(self.face_db.keys())
        self.face_db = {}
        self._save_db()
        for n in names:
            d = self.face_dir / n
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
        print(f"[人脸库] 已清空 {len(names)} 人")


# ========== 便捷函数 ==========
def draw_faces(image: np.ndarray, results: list) -> np.ndarray:
    """在图片上绘制人脸框和姓名"""
    for name, score, face_box in results:
        x, y, w, h = int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"{name} ({score:.2f})"
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image
