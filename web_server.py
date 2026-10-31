# -*- coding: utf-8 -*-
"""
HomeCRobot Web 服务
===================
在现有 Python 能力之上套一层 Web 界面，**不改动** core/ 与 modules/ 里的任何模块。

架构：
    浏览器 (web/index.html)
      ├── WebSocket /ws     双向：文字、语音、状态、播报、主动消息
      ├── HTTP  /video      摄像头 MJPEG 流
      └── HTTP  /audio/*    语音播报音频

设计要点：
  1. **推理不阻塞事件循环**：模型推理是同步且耗时的（数秒），
     全部丢到线程池执行，否则 WebSocket 会被卡死。
  2. **复用现有模块**：VisionTrigger / IntentRecognizer / FaceRecognizer /
     VLModel / TTS / ASR / VisionMonitor 全部直接调用，逻辑与命令行版一致。
  3. **优雅降级**：摄像头/模型/麦克风任一不可用时不崩溃，前端能看到原因。

运行:
    python web_server.py
    然后浏览器打开 http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import signal
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

import hf_env  # noqa: F401  配置镜像，必须最先导入
from config import (VLConfig, CameraConfig, VoiceConfig, MonitorConfig,
                    ConversationConfig, FaceConfig, WebConfig)
from core.camera import CameraCapture
from core.conversation import ConversationManager
from core.face_recognition import FaceRecognizer
from core.intent_recognizer import IntentRecognizer, Intent
from core.monitor import VisionMonitor
from core.vision_trigger import VisionTrigger
from core.vl_model import VLModel
from modules.asr import ASR
from modules.tts import TTS
from utils.utterance_filter import UtteranceFilter

HERE = Path(__file__).parent
INDEX = HERE / "web" / "index.html"

# 推理用线程池：模型推理是同步阻塞的，必须移出事件循环
POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="infer")


# ======================================================================
#  机器人后端（复用现有模块）
# ======================================================================
class WebRobot:
    """把现有能力组装成 Web 可用的服务"""

    def __init__(self):
        self.camera: Optional[CameraCapture] = None
        self.vl: Optional[VLModel] = None
        self.face: Optional[FaceRecognizer] = None
        self.tts: Optional[TTS] = None
        self.asr: Optional[ASR] = None
        self.monitor: Optional[VisionMonitor] = None
        self.conversation = ConversationManager(ConversationConfig())
        self.filter = UtteranceFilter(VoiceConfig())

        self.ready = False
        self.loading = False
        self.load_error = ""
        # 最近一次摄像头帧（供 MJPEG 与快照复用，避免多线程抢摄像头）
        self._last_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # 待发送的消息队列：监控线程产生的主动消息从这里取
        self.pending: list[dict] = []
        self._pending_lock = threading.Lock()

        # 前端连接（可能多个标签页）
        self.clients: set[WebSocket] = set()

        self.stats = {
            "对话轮数": 0,
            "语音播报": 0,
            "主动消息": 0,
            "启动时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # ------------------------------------------------------------------
    #  加载
    # ------------------------------------------------------------------
    def load_all(self, with_camera=True, with_tts=True):
        """加载全部资源（耗时，在线程池里调用）"""
        if self.loading or self.ready:
            return
        self.loading = True
        try:
            # 摄像头
            if with_camera:
                try:
                    self.camera = CameraCapture(CameraConfig())
                    if not self.camera.open():
                        self.camera = None
                except Exception as e:
                    print(f"[Web] 摄像头初始化失败: {e}")
                    self.camera = None

            # 人脸识别
            try:
                self.face = FaceRecognizer(FaceConfig())
                self.face.load()
            except Exception as e:
                print(f"[Web] 人脸识别初始化失败: {e}")
                self.face = None

            # 视觉模型（最慢）
            self.vl = VLModel(VLConfig())
            self.vl.load()

            # 语音合成
            if with_tts:
                try:
                    self.tts = TTS(VoiceConfig(),
                                   on_start=None, on_end=None)
                except Exception as e:
                    print(f"[Web] TTS 初始化失败: {e}")
                    self.tts = None

            # 语音识别（后台预热，不阻塞）
            try:
                self.asr = ASR(VoiceConfig())
                if VoiceConfig.ENABLE_ASR:
                    self.asr.load(background=True)
            except Exception as e:
                print(f"[Web] ASR 初始化失败: {e}")
                self.asr = None

            # 后台监控
            if self.camera and self.vl and MonitorConfig.ENABLE_MONITOR:
                try:
                    self.monitor = VisionMonitor(
                        self.camera, self.vl, MonitorConfig(),
                        face_recognizer=self.face,
                        on_proactive=self._on_proactive,
                    )
                    self.monitor.start()
                except Exception as e:
                    print(f"[Web] 监控启动失败: {e}")
                    self.monitor = None

            self.ready = True
            self.load_error = ""
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self.loading = False

    # ------------------------------------------------------------------
    #  监控回调（在监控线程里执行，不能直接 await）
    # ------------------------------------------------------------------
    def _on_proactive(self, message: str, meta: dict = None):
        with self._pending_lock:
            self.pending.append({
                "type": "proactive",
                "text": message,
                "meta": meta or {},
                "ts": time.time(),
            })
        self.stats["主动消息"] += 1

    def take_pending(self) -> list[dict]:
        with self._pending_lock:
            items, self.pending = self.pending, []
        return items

    # ------------------------------------------------------------------
    #  摄像头
    # ------------------------------------------------------------------
    def snapshot_jpeg(self) -> Optional[bytes]:
        """取一帧并编码为 JPEG"""
        if self.camera is None:
            return None
        try:
            img = self.camera.capture_pil_image(max_retries=1)
            if img is None:
                return None
            arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            with self._frame_lock:
                self._last_frame = arr
            ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            return buf.tobytes() if ok else None
        except Exception:
            return None

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._last_frame is None else self._last_frame.copy()

    # ------------------------------------------------------------------
    #  对话（在线程池中执行）
    # ------------------------------------------------------------------
    def _identity(self) -> Optional[str]:
        """识别镜头前是谁（复用监控快照，避免重复拍照）"""
        if self.monitor is not None:
            snap = self.monitor.get_snapshot(max_age=6.0)
            name = snap.get("name")
            if name and name != "unknown":
                return name
        if self.face is None or self.camera is None:
            return None
        try:
            img = self.camera.capture_pil_image(max_retries=1)
            if img is None:
                return None
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            res = self.face.recognize(bgr)
            known = [r[0] for r in res if r[0] != "unknown"]
            return known[0] if known else None
        except Exception:
            return None

    def chat_sync(self, text: str, want_vision: bool = False) -> dict:
        """同步对话（在池里跑）。返回 {reply, used_vision, identity}"""
        self.stats["对话轮数"] += 1
        who = self._identity()

        # 判断是否需要看画面
        with_vision = want_vision
        if not with_vision and self.camera is not None:
            try:
                hist = self.conversation.get_history_for_model()
                with_vision, _reason = VisionTrigger.should_use_vision(text, hist)
            except Exception:
                with_vision = False

        image = None
        prompt = text
        if who:
            prompt = f"[当前和你说话的用户叫「{who}」]\n{text}"
        if with_vision and self.camera is not None:
            pil = self.camera.capture_pil_image()
            if pil is not None:
                image = pil

        history = self.conversation.get_history_for_model()
        reply = self.vl.generate(text=prompt, image=image, history=history)

        # 视觉推理是最吃显存的操作（一张 1280x720 图会产生大量视觉 token）。
        # 做完立刻归还缓存，降低下一次申请时因碎片而 OOM 的概率。
        # 背景：7B 在 8GB 卡上占用已接近上限，实测曾出现进程被系统杀掉。
        if image is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        self.conversation.add_user_message(text)
        self.conversation.add_assistant_message(reply)
        return {"reply": reply, "used_vision": image is not None, "identity": who}

    def recognize_only(self) -> dict:
        """只做一次人脸识别（前端"我是谁"按钮）"""
        if self.face is None or self.camera is None:
            return {"ok": False, "reason": "人脸识别或摄像头不可用"}
        try:
            img = self.camera.capture_pil_image()
            if img is None:
                return {"ok": False, "reason": "拍照失败"}
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            res = self.face.recognize(bgr)
            named = [{"name": n, "score": s} for n, s, _ in res]
            return {"ok": True, "faces": named}
        except Exception as e:
            return {"ok": False, "reason": str(e)}

    # ------------------------------------------------------------------
    #  人脸注册 / 管理
    # ------------------------------------------------------------------
    def face_list(self) -> dict:
        if self.face is None:
            return {"ok": False, "reason": "人脸识别不可用", "people": []}
        people = []
        for n in self.face.get_registered_names():
            entry = self.face.face_db.get(n, {})
            people.append({
                "name": n,
                "count": len(entry.get("features", [])),
                "photos": list(entry.get("photos", [])),
            })
        return {"ok": True, "people": people}

    def register_from_camera(self, name: str) -> dict:
        """用当前摄像头画面注册一张人脸

        流程与命令行版一致：先质量检查，不合格明确告知原因。
        """
        name = (name or "").strip()
        if not name:
            return {"ok": False, "reason": "请先填写名字"}
        if self.face is None or self.camera is None:
            return {"ok": False, "reason": "人脸识别或摄像头不可用"}
        try:
            img = self.camera.capture_pil_image()
            if img is None:
                return {"ok": False, "reason": "拍照失败"}
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            ok, msg, face_box = self.face.check_quality(bgr)
            if not ok:
                return {"ok": False, "reason": msg}
            success, detail = self.face.register_face(
                name, bgr, quality_checked=True, face_box=face_box)
            if not success:
                return {"ok": False, "reason": detail}
            return {"ok": True, "message": detail,
                    "count": self.face.get_photo_count(name)}
        except Exception as e:
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    def delete_person(self, name: str) -> dict:
        if self.face is None:
            return {"ok": False, "reason": "人脸识别不可用"}
        if self.face.delete_face(name):
            return {"ok": True, "message": f"已删除 {name}"}
        return {"ok": False, "reason": f"没有找到 {name}"}

    def delete_one_photo(self, name: str, index: int) -> dict:
        if self.face is None:
            return {"ok": False, "reason": "人脸识别不可用"}
        ok, msg = self.face.delete_photo(name, index)
        if ok:
            return {"ok": True, "message": msg}
        return {"ok": False, "reason": msg}

    def clear_faces(self) -> dict:
        if self.face is None:
            return {"ok": False, "reason": "人脸识别不可用"}
        n = len(self.face.get_registered_names())
        self.face.clear_db()
        return {"ok": True, "message": f"已清空 {n} 人的记录"}

    # ------------------------------------------------------------------
    #  头像
    # ------------------------------------------------------------------
    def avatar_info(self) -> dict:
        p = Path(WebConfig.AVATAR_FILE)
        if p.exists():
            return {"ok": True, "custom": True, "url": f"/api/avatar?t={int(p.stat().st_mtime)}"}
        return {"ok": True, "custom": False, "emoji": WebConfig.AVATAR_DEFAULT}

    def set_avatar(self, data: bytes) -> dict:
        """保存上传的头像图片（统一转成 PNG 并限制尺寸）"""
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            img = img.convert("RGBA")
            # 限制到 512x512 以内，等比缩放
            img.thumbnail((512, 512))
            p = Path(WebConfig.AVATAR_FILE)
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p, "PNG")
            return {"ok": True, "message": f"头像已更新（{img.width}x{img.height}）",
                    "url": f"/api/avatar?t={int(p.stat().st_mtime)}"}
        except Exception as e:
            return {"ok": False, "reason": f"图片处理失败: {e}"}

    def reset_avatar(self) -> dict:
        p = Path(WebConfig.AVATAR_FILE)
        if p.exists():
            p.unlink()
        return {"ok": True, "message": "已恢复默认头像"}

    # ------------------------------------------------------------------
    #  语音播报
    # ------------------------------------------------------------------
    def speak_sync(self, text: str) -> Optional[bytes]:
        """合成语音，返回 wav 字节（供前端播放）

        注意：这里**不能**直接调 tts.synthesize()。
        那样会绕过引擎降级链——在线引擎（edge-tts）失败时直接返回空，
        前端就听不到声音。必须按「主引擎 → 降级引擎」依次尝试。
        """
        if self.tts is None or not text.strip():
            return None
        # 语音开关：edge-tts 不可用（如额度到期）时可整体关闭，
        # 避免每次对话都白等一次合成失败
        if not getattr(VoiceConfig, "ENABLE_TTS", False):
            return None
        try:
            # 注意：TTS.enabled 是只读 property（由 config.ENABLE_TTS 决定），
            # 不能赋值，否则抛 "property has no setter"。
            result = self.tts.synthesize(text)

            # 主引擎失败 -> 尝试降级引擎
            if result is None:
                fb = getattr(VoiceConfig, "TTS_FALLBACK_ENGINE", None)
                if fb and fb != VoiceConfig.TTS_ENGINE:
                    print(f"[Web] {VoiceConfig.TTS_ENGINE} 合成失败，降级到 {fb}")
                    result = self.tts.synthesize(text, engine=fb)

            if result is None:
                return None

            samples, sr = result
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, samples, sr, format="WAV")
            self.stats["语音播报"] += 1
            return buf.getvalue()
        except Exception as e:
            print(f"[Web] 语音合成失败: {e}")
            return None

    # ------------------------------------------------------------------
    #  状态
    # ------------------------------------------------------------------
    def status(self) -> dict:
        names = self.face.get_registered_names() if self.face else []
        people = []
        if self.face:
            for n in names:
                entry = self.face.face_db.get(n, {})
                people.append({"name": n,
                               "count": len(entry.get("features", []))})
        out = {
            "ready": self.ready,
            "loading": self.loading,
            "error": self.load_error,
            "camera": self.camera is not None,
            "face_count": len(names),
            "faces": names,
            "people": people,
            "asr_ready": bool(self.asr and self.asr.ready),
            "tts_engine": VoiceConfig.TTS_ENGINE,
            "voice_on": bool(getattr(VoiceConfig, "ENABLE_TTS", False)),
            "typewriter_interval": int(getattr(WebConfig, "TYPEWRITER_INTERVAL", 0.018) * 1000),
            "avatar": self.avatar_info(),
            "stats": dict(self.stats),
        }
        if self.monitor is not None:
            out["monitor"] = self.monitor.get_status()
        return out

    def shutdown(self):
        """释放所有资源：监控线程 → 音频 → 摄像头 → 模型显存

        顺序很重要：监控线程还在用摄像头/模型时不能先关它们，
        否则会抛异常或留下占用。
        """
        print("[Web] 正在释放资源...")
        # 1) 先停监控（它在后台调用模型和摄像头）
        try:
            if self.monitor:
                self.monitor.stop()
                self.monitor = None
        except Exception as e:
            print(f"[Web] 停止监控失败: {e}")

        # 2) 音频链路
        try:
            if self.tts:
                self.tts.close()
        except Exception:
            pass
        try:
            if self.asr:
                self.asr.unload()
        except Exception:
            pass

        # 3) 摄像头（释放设备，否则其他程序用不了）
        try:
            if self.camera:
                self.camera.close()
                self.camera = None
                print("[Web] 摄像头已释放")
        except Exception as e:
            print(f"[Web] 关闭摄像头失败: {e}")

        # 4) 模型显存
        try:
            if self.vl:
                self.vl.unload()
                self.vl = None
        except Exception as e:
            print(f"[Web] 卸载模型失败: {e}")

        self.ready = False
        print("[Web] 资源释放完成")

    def unload_model(self) -> dict:
        """只卸载模型与摄像头，保留 Web 服务在跑

        用于「暂时不用了，但要释放显存/摄像头」的场景。
        """
        self.shutdown()
        return {"ok": True, "message": "已释放摄像头与显存，可随时重新启动"}


robot = WebRobot()


# ======================================================================
#  FastAPI 应用
# ======================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时不立刻加载模型（会让服务卡住无法响应），
    # 由前端点"启动系统"或直接调 /api/load 触发。
    loop = asyncio.get_event_loop()
    loop.run_in_executor(POOL, robot.load_all)
    try:
        yield
    finally:
        # 无论是正常退出、Ctrl+C 还是被 kill，都尝试释放资源。
        # 这一步很关键：不释放的话摄像头会被一直占用。
        try:
            await loop.run_in_executor(POOL, robot.shutdown)
        except Exception as e:
            print(f"[Web] 收尾失败: {e}")
        POOL.shutdown(wait=False)


app = FastAPI(title="HomeCRobot Web", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    if not INDEX.exists():
        return HTMLResponse(
            "<h1>缺少前端文件</h1><p>请确认 web/index.html 存在</p>",
            status_code=500)
    return HTMLResponse(INDEX.read_text(encoding="utf-8"))


@app.get("/api/status")
async def api_status():
    return JSONResponse(robot.status())


# ---------------- 头像 ----------------
@app.get("/api/avatar")
async def api_avatar():
    """返回头像图片；没有自定义图片时返回 404，前端用 emoji"""
    p = Path(WebConfig.AVATAR_FILE)
    if not p.exists():
        return JSONResponse({"ok": False, "reason": "使用默认头像"}, status_code=404)
    from fastapi.responses import FileResponse
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/avatar/info")
async def api_avatar_info():
    return JSONResponse(robot.avatar_info())


@app.post("/api/avatar")
async def api_avatar_upload(file: UploadFile = File(...)):
    data = await file.read()
    limit = WebConfig.AVATAR_MAX_MB * 1024 * 1024
    if len(data) > limit:
        return JSONResponse({"ok": False,
                             "reason": f"图片过大（上限 {WebConfig.AVATAR_MAX_MB}MB）"},
                            status_code=400)
    loop = asyncio.get_event_loop()
    return JSONResponse(await loop.run_in_executor(POOL, robot.set_avatar, data))


@app.post("/api/avatar/reset")
async def api_avatar_reset():
    return JSONResponse(robot.reset_avatar())


# ---------------- 人脸管理 ----------------
@app.get("/api/faces")
async def api_faces():
    loop = asyncio.get_event_loop()
    return JSONResponse(await loop.run_in_executor(POOL, robot.face_list))


@app.post("/api/faces/register")
async def api_face_register(payload: dict):
    """用当前摄像头画面注册（name 为姓名）"""
    loop = asyncio.get_event_loop()
    return JSONResponse(await loop.run_in_executor(
        POOL, robot.register_from_camera, payload.get("name", "")))


@app.post("/api/faces/delete")
async def api_face_delete(payload: dict):
    name = payload.get("name", "")
    idx = payload.get("index")
    loop = asyncio.get_event_loop()
    if idx:
        return JSONResponse(await loop.run_in_executor(
            POOL, robot.delete_one_photo, name, int(idx)))
    return JSONResponse(await loop.run_in_executor(
        POOL, robot.delete_person, name))


@app.post("/api/faces/clear")
async def api_faces_clear():
    loop = asyncio.get_event_loop()
    return JSONResponse(await loop.run_in_executor(POOL, robot.clear_faces))


@app.post("/api/load")
async def api_load():
    loop = asyncio.get_event_loop()
    loop.run_in_executor(POOL, robot.load_all)
    return {"ok": True, "message": "开始加载"}


@app.post("/api/unload")
async def api_unload():
    """释放摄像头与显存，但保留 Web 服务在跑"""
    loop = asyncio.get_event_loop()
    return JSONResponse(await loop.run_in_executor(POOL, robot.unload_model))


@app.post("/api/shutdown")
async def api_shutdown():
    """完全关闭服务并释放所有资源

    实现说明（踩过的坑）：
    Windows 上 os.kill(SIGINT) 只对主线程有效，在线程池里发信号
    进程不会退出。所以这里直接「显式释放资源 + 强制退出」，
    不依赖信号 —— 这样关闭一定生效，不会留下占用摄像头的孤儿进程。
    """

    async def _later():
        await asyncio.sleep(0.5)          # 先让 HTTP 响应发出去
        loop = asyncio.get_event_loop()
        # 必须释放完再退出，否则摄像头会被操作系统一直占着
        try:
            await loop.run_in_executor(POOL, robot.shutdown)
        except Exception as e:
            print(f"[Web] 关闭时释放资源失败: {e}")
        print("[Web] 服务关闭")
        os._exit(0)

    asyncio.create_task(_later())
    return {"ok": True, "message": "服务正在关闭，摄像头与显存已释放"}


@app.post("/api/recognize")
async def api_recognize():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(POOL, robot.recognize_only)
    return JSONResponse(result)


@app.get("/api/snapshot")
async def api_snapshot():
    """单张快照（调试用）"""
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(POOL, robot.snapshot_jpeg)
    if data is None:
        return JSONResponse({"ok": False, "reason": "摄像头不可用"}, status_code=503)
    return HTMLResponse(
        content=base64.b64encode(data).decode(),
        media_type="text/plain")


@app.get("/video")
async def video():
    """MJPEG 流"""
    async def gen():
        while True:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(POOL, robot.snapshot_jpeg)
            if data:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                       + data + b"\r\n")
            await asyncio.sleep(0.4)      # 约 2.5 fps，够看画面又不吃满 CPU
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------- WebSocket ----------------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    robot.clients.add(ws)
    loop = asyncio.get_event_loop()

    async def send(obj: dict):
        try:
            await ws.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass

    await send({"type": "hello", "status": robot.status()})

    # 推送循环：主动消息 + 状态
    async def pusher():
        while True:
            try:
                for item in robot.take_pending():
                    await send(item)
                    # 主动消息顺带合成语音
                    audio = await loop.run_in_executor(
                        POOL, robot.speak_sync, item.get("text", ""))
                    if audio:
                        await ws.send_bytes(audio)
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.5)

    push_task = asyncio.create_task(pusher())

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")

            if mtype == "ping":
                await send({"type": "pong"})

            elif mtype == "status":
                await send({"type": "status", "status": robot.status()})

            elif mtype == "load":
                loop.run_in_executor(POOL, robot.load_all)
                await send({"type": "info", "text": "开始加载模型..."})

            elif mtype == "chat":
                text = (msg.get("text") or "").strip()
                if not text:
                    continue
                if not robot.ready:
                    await send({"type": "error",
                                "text": "系统尚未就绪，请先点「启动系统」"})
                    continue
                await send({"type": "user", "text": text})
                await send({"type": "thinking"})
                try:
                    result = await loop.run_in_executor(
                        POOL, robot.chat_sync, text, bool(msg.get("vision")))
                except Exception as e:
                    await send({"type": "error", "text": f"处理失败: {e}"})
                    continue
                await send({"type": "reply", **result})
                # 合成并推送语音
                audio = await loop.run_in_executor(
                    POOL, robot.speak_sync, result.get("reply", ""))
                if audio:
                    await ws.send_bytes(audio)

            elif mtype == "recognize":
                result = await loop.run_in_executor(POOL, robot.recognize_only)
                await send({"type": "recognize", **result})

            elif mtype == "register":
                # 用摄像头注册一张人脸（前端「拍照注册」按钮）
                name = (msg.get("name") or "").strip()
                result = await loop.run_in_executor(
                    POOL, robot.register_from_camera, name)
                listing = await loop.run_in_executor(POOL, robot.face_list)
                await send({"type": "faces",
                            "people": listing.get("people", []),
                            "message": result.get("message") or result.get("reason"),
                            "ok": result.get("ok")})

            elif mtype == "face_delete":
                name = (msg.get("name") or "").strip()
                idx = msg.get("index")
                if idx:
                    result = await loop.run_in_executor(
                        POOL, robot.delete_one_photo, name, int(idx))
                else:
                    result = await loop.run_in_executor(
                        POOL, robot.delete_person, name)
                listing = await loop.run_in_executor(POOL, robot.face_list)
                await send({"type": "faces",
                            "people": listing.get("people", []),
                            "message": result.get("message") or result.get("reason"),
                            "ok": result.get("ok")})

            elif mtype == "faces_clear":
                result = await loop.run_in_executor(POOL, robot.clear_faces)
                await send({"type": "faces", "people": [],
                            "message": result.get("message"),
                            "ok": result.get("ok")})

            elif mtype == "faces_list":
                listing = await loop.run_in_executor(POOL, robot.face_list)
                await send({"type": "faces", "people": listing.get("people", [])})

            elif mtype == "speak":
                audio = await loop.run_in_executor(
                    POOL, robot.speak_sync, msg.get("text", ""))
                if audio:
                    await ws.send_bytes(audio)
                else:
                    await send({"type": "info", "text": "语音合成不可用"})

            elif mtype == "asr_bytes":
                # 前端录音（PCM16 单声道 16k），直接送现有 ASR
                if robot.asr is None:
                    await send({"type": "error", "text": "语音识别未启用"})
                    continue
                try:
                    b64 = msg.get("data") or ""
                    pcm = np.frombuffer(base64.b64decode(b64), dtype=np.int16)
                    audio = pcm.astype(np.float32) / 32768.0
                    text = await loop.run_in_executor(
                        POOL, robot.asr.transcribe_array, audio, 16000)
                    await send({"type": "asr", "text": text or ""})
                except Exception as e:
                    await send({"type": "error", "text": f"识别失败: {e}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] 异常: {e}")
    finally:
        push_task.cancel()
        robot.clients.discard(ws)


# ======================================================================
if __name__ == "__main__":
    import uvicorn
    print("=" * 64)
    print(" HomeCRobot Web 服务")
    print("=" * 64)
    print(" 浏览器打开: http://127.0.0.1:8000")
    print(" 接口文档  : http://127.0.0.1:8000/docs")
    print()
    print(" 提示: 模型在后台加载，页面打开后会自动就绪")
    print("=" * 64)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
