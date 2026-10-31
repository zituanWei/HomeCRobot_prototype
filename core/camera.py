"""
摄像头采集模块
负责：打开摄像头、捕获帧、保存图片、实时预览

并发说明（重要）：
后台上视觉监控线程（在场检测，默认每 1.5 秒）与主对话线程（用户提问时）
会同时使用同一个 VideoCapture。OpenCV 的 VideoCapture 不是线程安全的，
两个线程同时 read() 可能导致某一方拿到损坏的帧甚至异常。
因此这里用一把锁把硬件访问串行化：同一时刻只允许一个线程读摄像头，
不同的连续 read()（监控一次、对话一次）是安全的。
"""
import os
import cv2
import time
import threading
from pathlib import Path
from datetime import datetime
from PIL import Image

from config import CameraConfig, DATA_DIR


def _suppress_libpng_warnings():
    """
    抑制 OpenCV 在 Windows 上的 libpng iCCP 警告。
    原理：临时重定向 stderr 到 devnull，触发一次窗口创建让警告静默输出，
    之后恢复 stderr，后续正常使用就不会再弹警告了。
    """
    try:
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        # 触发一次窗口创建/销毁，让所有 iCCP 警告在静默期输出
        try:
            cv2.namedWindow("_warmup_suppress", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("_warmup_suppress")
        except Exception:
            pass
        # 恢复 stderr
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
    except Exception:
        pass  # 抑制失败也不影响主流程


class CameraCapture:
    """摄像头采集封装类"""

    def __init__(self, config: CameraConfig = None):
        self.config = config or CameraConfig()
        self.cap = None
        self._last_capture_time = 0
        # 保护 VideoCapture 的硬件访问（监控线程与主线程共享）
        self._lock = threading.RLock()

    def open(self) -> bool:
        """打开摄像头，返回是否成功"""
        with self._lock:
            self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # 尝试默认后端
                self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)

            if not self.cap.isOpened():
                print(f"[错误] 无法打开摄像头 (index={self.config.CAMERA_INDEX})")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)

            # 预热：丢弃前几帧，让摄像头自动曝光/白平衡稳定
            for _ in range(3):
                self.cap.read()
                time.sleep(0.05)

            print(f"[摄像头] 已打开: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            return True

    def read_frame(self):
        """读取一帧，返回 (success, frame_bgr)（线程安全）"""
        with self._lock:
            if self.cap is None or not self.cap.isOpened():
                return False, None
            ret, frame = self.cap.read()
            return ret, frame

    def capture_pil_image(self, max_retries: int = 3) -> Image.Image | None:
        """捕获一帧并转为 PIL Image（RGB格式，供VL模型使用）
        带重试机制，解决摄像头刚打开时前几帧为空的问题
        线程安全：整个重试过程持有锁，避免与监控线程交错读同一设备
        """
        frame = None
        with self._lock:
            for attempt in range(max_retries):
                if self.cap is None or not self.cap.isOpened():
                    break
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    break
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # 等待摄像头预热

        if frame is None:
            print("[警告] 读取摄像头帧失败（已重试3次）")
            print("  可能原因：摄像头被其他程序占用，或设备异常")
            return None

        # OpenCV 是 BGR，转为 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        self._last_capture_time = time.time()
        return pil_img

    # ------------------------------------------------------------------
    #  拍照存储（带容量上限，超出后删最早的）
    # ------------------------------------------------------------------
    def _photo_dir(self) -> Path:
        d = Path(self.config.PHOTO_DIR)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def enforce_photo_quota(self) -> int:
        """把拍照目录压到容量/张数上限内，返回删除的文件数

        策略：**先删最早的**（按修改时间）。
        注意：只清理普通拍照目录，不动 data/faces/ 下的人脸注册照片
        ——那些删了就没法重新提特征，属于必须有原图的资料。
        """
        d = self._photo_dir()
        max_bytes = int(self.config.PHOTO_MAX_MB) * 1024 * 1024
        max_count = int(self.config.PHOTO_MAX_COUNT)

        photos = sorted(
            (p for p in d.glob("*.jpg") if p.is_file()),
            key=lambda p: p.stat().st_mtime,
        )
        removed = 0

        # 先按数量删
        while len(photos) > max_count:
            oldest = photos.pop(0)
            try:
                oldest.unlink()
                removed += 1
            except Exception:
                break

        # 再按容量删
        total = sum(p.stat().st_size for p in photos if p.exists())
        idx = 0
        while total > max_bytes and idx < len(photos):
            p = photos[idx]
            idx += 1
            if not p.exists():
                continue
            try:
                size = p.stat().st_size
                p.unlink()
                total -= size
                removed += 1
            except Exception:
                break

        if removed:
            print(f"[照片] 超出上限，已删除最早的 {removed} 张")
        return removed

    def save_frame(self, frame=None, filename: str = None) -> Path | None:
        """保存当前帧，返回保存路径

        使用 Pillow 保存（明确不嵌入 ICC profile，避免 libpng iCCP 警告），
        并在写入后执行容量清理。
        """
        if frame is None:
            ret, frame = self.read_frame()
            if not ret:
                return None

        d = self._photo_dir()
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"capture_{timestamp}.jpg"
        save_path = d / filename

        # BGR -> RGB，用 Pillow 保存（明确不嵌入 ICC profile）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img.save(str(save_path), "JPEG",
                     quality=int(getattr(self.config, "PHOTO_JPEG_QUALITY", 95)),
                     optimize=True)

        # 写入后检查容量，防止无限增长占满磁盘
        try:
            self.enforce_photo_quota()
        except Exception as e:
            print(f"[照片] 容量清理失败: {e}")
        return save_path

    def get_photo_stats(self) -> dict:
        """拍照目录占用情况（供 /photos 或状态展示）"""
        d = self._photo_dir()
        files = [p for p in d.glob("*.jpg") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        return {
            "dir": str(d),
            "count": len(files),
            "mb": round(total / 1024 / 1024, 1),
            "max_count": int(self.config.PHOTO_MAX_COUNT),
            "max_mb": int(self.config.PHOTO_MAX_MB),
        }

    def preview(self, window_name: str = "Camera Preview"):
        """
        实时预览摄像头画面，按 q 退出，按 s 保存当前帧
        这是一个阻塞函数，用于单独测试摄像头
        """
        _suppress_libpng_warnings()  # 抑制 OpenCV libpng iCCP 警告
        if not self.open():
            return
        print("[预览] 按 q 退出，按 s 保存图片")
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = self.save_frame(frame)
                if path:
                    print(f"[保存] {path}")
        self.close()

    def should_auto_capture(self) -> bool:
        """判断是否到了自动拍照时间"""
        if self.config.AUTO_CAPTURE_INTERVAL <= 0:
            return False
        return (time.time() - self._last_capture_time) >= self.config.AUTO_CAPTURE_INTERVAL

    def close(self):
        """释放摄像头（线程安全）"""
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        cv2.destroyAllWindows()
        print("[摄像头] 已关闭")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
