"""
家庭陪护机器人 - 主程序入口
============================
功能：摄像头视觉 + 语音/文字对话 + 后台主动关怀

运行: python main.py

架构（事件驱动）：
    键盘输入线程 ──┐
    语音识别线程 ──┼──> 事件队列 ──> 主循环统一处理 ──> VL 推理 ──> 文字 + TTS
    视觉监控线程 ──┘

相比旧版的改动：
1. 主循环不再卡在 console.input()，改为从队列取事件，
   这样语音、键盘、监控三个来源可以自然共存。
2. 对话处理放到后台线程，机器人说话时依然能响应"别说了"。
3. 语音输入链路：麦克风常开 -> VAD 断句 -> FunASR 识别 -> 事件入队。
4. TTS 播报期间自动静音麦克风，避免机器人听到自己的声音（自激）。
"""
import hf_env  # noqa: F401  必须在最前面，配置 HuggingFace 国内镜像
import argparse
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config import (VLConfig, CameraConfig, ConversationConfig, VoiceConfig,
                    MonitorConfig, FaceConfig)
from core.audio import AudioRecorder, EnergyVAD, list_audio_devices
from core.camera import CameraCapture
from core.vl_model import VLModel
from core.conversation import ConversationManager
from core.vision_trigger import VisionTrigger
from core.monitor import VisionMonitor
from core.face_recognition import FaceRecognizer
from core.intent_recognizer import IntentRecognizer, Intent
from core.guided_enroll import GuidedEnroll
from modules.tts import TTS
from modules.asr import ASR
from utils.helpers import print_banner, print_help, graceful_exit, check_gpu_memory
from utils.utterance_filter import UtteranceFilter

console = Console()


# ======================================================================
#  事件定义
# ======================================================================
class EventType:
    TEXT_INPUT = "text"        # 键盘输入
    VOICE_INPUT = "voice"      # 语音识别结果
    PROACTIVE = "proactive"    # 监控主动消息
    COMMAND = "command"        # 内部命令（模式切换等）
    EXIT = "exit"


@dataclass
class Event:
    type: str
    payload: object = None
    meta: dict = field(default_factory=dict)


# ======================================================================
#  主类
# ======================================================================
class HomeCRobot:
    """家庭陪护机器人主类"""

    def __init__(self, args):
        self.args = args
        self.vl_config = VLConfig()
        self.camera_config = CameraConfig()
        self.conv_config = ConversationConfig()
        self.voice_config = VoiceConfig()
        self.monitor_config = MonitorConfig()

        # --- 组件 ---
        self.camera = None
        self.vl_model = None
        self.conversation = ConversationManager(self.conv_config)
        self.face_rec = None

        # TTS：播报开始/结束时要静音麦克风，避免自激
        self.tts = TTS(self.voice_config, on_start=self._on_tts_start,
                       on_end=self._on_tts_end)
        self.asr = ASR(self.voice_config)
        self.recorder: Optional[AudioRecorder] = None
        self.segmenter = None
        self.vad = None
        self.monitor = None

        # --- 状态 ---
        self.voice_reply_enabled = self.voice_config.ENABLE_TTS
        self.voice_input_enabled = self.voice_config.ENABLE_ASR
        self.auto_vision_enabled = True
        self.running = True
        self.pending_intent = None
        self.pending_params = {}

        # 认不出人时的引导注册状态机（多轮对话：问名字 -> 复述确认 -> 拍照）
        self.guided = GuidedEnroll(
            FaceConfig(),
            on_ask=self._guided_ask,
            on_confirm=self._guided_confirm,
            on_enroll=self._guided_do_enroll,
            on_giveup=self._guided_giveup,
        )

        # --- 事件与并发 ---
        self.events: queue.Queue[Event] = queue.Queue()
        self._conversation_lock = threading.Lock()   # 保护对话历史
        self._processing = threading.Event()         # 是否正在处理一轮对话
        self._barge_in_stop = threading.Event()      # 通知打断监听线程退出
        # 输入过滤：挡掉重复/纠正句、语气词、空内容
        self.utterance_filter = UtteranceFilter(self.voice_config)
        # 是否启用键盘输入线程。非交互环境（stdin 被重定向/关闭）下
        # 键盘线程会立刻读到 EOF 并投递退出事件，导致主循环瞬间结束，
        # 这种场景可以关掉它，改用语音或程序化事件驱动。
        self.keyboard_enabled = True

    # ==================================================================
    #  初始化
    # ==================================================================
    def initialize(self):
        """初始化摄像头、模型、人脸识别、语音链路"""
        self._init_audio()

        # 摄像头
        console.print("[cyan]📷 正在初始化摄像头...[/cyan]")
        self.camera = CameraCapture(self.camera_config)
        if not self.camera.open():
            console.print("[yellow]⚠️  摄像头打开失败，将仅使用文字对话模式[/yellow]")
            self.camera = None
        else:
            console.print("[green]✓ 摄像头就绪[/green]")

        # 人脸识别（在监控之前，让主动打招呼能用上身份）
        console.print("[cyan]👤 正在加载人脸识别模型...[/cyan]")
        try:
            self.face_rec = FaceRecognizer()
            if self.face_rec.load():
                names = self.face_rec.get_registered_names()
                console.print(f"[green]✓ 人脸识别就绪"
                              f"{f'（已注册 {len(names)} 人）' if names else '（人脸库为空）'}[/green]")
            else:
                self.face_rec = None
                console.print("[yellow]⚠️  人脸识别不可用[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠️  人脸识别加载失败: {e}[/yellow]")
            self.face_rec = None

        # VL 模型
        console.print("[cyan]🧠 正在加载视觉语言模型（首次会自动下载，约3-6GB）...[/cyan]")
        self.vl_model = VLModel(self.vl_config)
        self.vl_model.load()

        mem = check_gpu_memory()
        if "error" not in mem:
            console.print(f"[dim]显存: 已用 {mem['allocated_gb']}GB / 总计 {mem['total_gb']}GB[/dim]")

        # 后台监控（传入人脸识别 + 主动消息回调）
        if self.monitor_config.ENABLE_MONITOR and self.camera and self.vl_model:
            self.monitor = VisionMonitor(
                self.camera, self.vl_model, self.monitor_config,
                face_recognizer=self.face_rec,
                on_proactive=self._on_proactive,
            )
            self.monitor.start()

        # 键盘输入线程（语音模式下也保留，方便调试）
        if self.keyboard_enabled:
            threading.Thread(target=self._keyboard_loop, daemon=True,
                             name="KeyboardInput").start()
        else:
            console.print("[dim]⌨️  键盘输入已关闭（仅语音/事件驱动）[/dim]")

        # 预合成常用语：让问候/固定追问这类高频回复"零延迟"出声。
        # edge-tts 在线合成慢（实测 1.6~13.9s），预合成是唯一不牺牲音质的加速手段。
        if self.voice_config.TTS_PRELOAD_PHRASES and self.voice_reply_enabled:
            try:
                from utils.phrases import all_preload_phrases
                phrases = all_preload_phrases()
                self.tts.preload_phrases(
                    phrases,
                    background=True,
                    delay=float(getattr(self.voice_config, "TTS_PRELOAD_DELAY", 8.0)),
                )
            except Exception as e:
                console.print(f"[dim]预合成跳过: {e}[/dim]")

    def _init_audio(self):
        """初始化音频链路（麦克风采集 + VAD + ASR）"""
        if not self.voice_config.ENABLE_ASR:
            console.print("[dim]🔇 语音输入未启用（config.py 里 VoiceConfig.ENABLE_ASR）[/dim]")
            return

        console.print("[cyan]🎤 正在启动麦克风...[/cyan]")
        self.recorder = AudioRecorder(
            sample_rate=self.voice_config.ASR_SAMPLE_RATE,
            device=self.voice_config.MIC_DEVICE,
            block_ms=self.voice_config.BLOCK_MS,
        )
        if not self.recorder.start():
            console.print(f"[yellow]⚠️  麦克风启动失败: {self.recorder.last_error}[/yellow]")
            console.print("[dim]可用音频设备列表：[/dim]")
            console.print(f"[dim]{list_audio_devices()}[/dim]")
            self.recorder = None
            return
        console.print("[green]✓ 麦克风就绪[/green]")

        # 组装：录音机 + VAD + ASR -> 识别结果进事件队列
        self.recorder, self.segmenter, self.vad = self.asr.build_live_pipeline(
            on_text=self._on_voice_text,
            recorder=self.recorder,
        )
        # VAD 需要知道机器人是否在说话（说话时跳过，避免自激）
        self.segmenter.is_speaking = lambda: self.tts.speaking
        self.segmenter.poll_interval = self.voice_config.VAD_POLL_INTERVAL
        self.segmenter.idle_flush_seconds = (
            getattr(self.voice_config, "VAD_IDLE_FLUSH_MS", 1800) / 1000.0)
        self.segmenter.start()
        console.print("[green]✓ 语音输入已启动（说话即可，停顿一下我就明白你说完了）[/green]")

        # 麦克风健康自检：很多"说话没反应"其实是麦克风没在拾音
        # （实测见过默认设备 peak 只有 0.00002，比正常低几千倍）。
        # 这里在启动时给一次明确警告，省得靠猜。
        self._check_mic_health()

        # 后台预热 ASR 模型（首次要从 ModelScope 下载）
        if self.voice_config.ASR_PRELOAD:
            self.asr.load(background=True)
            console.print("[dim]⏳ ASR 模型正在后台加载，就绪前说的语音会被跳过[/dim]")

    def _check_mic_health(self, seconds: float = 1.2):
        """启动时检查麦克风是否真的在拾音

        为什么要做：音频链路的故障最难排查——程序不报错、VAD 不触发、
        只是"说了没反应"。实测踩过：默认设备 peak 仅 0.00002（几乎无声），
        而换成正确的设备后有 0.08。启动时给一次提示能省很多时间。
        """
        if self.recorder is None:
            return
        try:
            peak = 0.0
            for _ in range(int(seconds * 10)):
                time.sleep(0.1)
                peak = max(peak, self.recorder.level)
        except Exception:
            return

        threshold = float(getattr(self.voice_config, "MIC_HEALTH_MIN", 0.0008))
        if peak >= threshold:
            console.print(f"[dim]🎤 麦克风信号正常（peak={peak:.5f}）[/dim]")
            return

        # 信号过弱 -> 明确警告并给出可执行的下一步
        console.print(Panel(
            Text(
                f"麦克风几乎收不到声音（peak={peak:.5f}）。\n"
                f"这会导致：说话没有反应、语音识别乱码、VAD 不触发。\n"
                f"\n"
                f"常见原因与处理：\n"
                f"  1) MIC_DEVICE 选错了（当前={self.voice_config.MIC_DEVICE}）\n"
                f"     >>> 跑 python scripts/calibrate_mic.py --scan 找可用设备\n"
                f"  2) 蓝牙耳机处于 A2DP（音乐）模式 —— 该模式没有麦克风输入，\n"
                f"     要在系统「声音设置」里切到 Hands-Free / 通讯模式\n"
                f"  3) 麦克风被静音（系统音量面板 / 耳机物理静音键）\n"
                f"  4) 系统隐私设置未允许应用访问麦克风\n"
                f"\n"
                f"你仍可以用键盘文字继续对话。",
                style="yellow",
            ),
            title="⚠️ 麦克风信号异常", border_style="yellow",
        ))

    # ==================================================================
    #  音频回调
    # ==================================================================
    def _on_tts_start(self):
        """TTS 开始播报：静音麦克风，避免把自己的声音识别成用户说话"""
        if self.recorder:
            self.recorder.mute()

    def _on_tts_end(self):
        """TTS 结束：恢复麦克风"""
        if self.recorder:
            self.recorder.unmute()

    def _on_voice_text(self, text: str):
        """VAD 检测到一整句话并识别完成的回调（在 VAD 线程中执行）"""
        if not text:
            return
        # 正在处理上一轮对话时，忽略新语音（避免排队堆积）
        if self._processing.is_set():
            return
        self.events.put(Event(EventType.VOICE_INPUT, text))

    def _on_proactive(self, message: str, meta: dict = None):
        """监控线程的主动消息回调"""
        self.events.put(Event(EventType.PROACTIVE, message, meta or {}))

    def _keyboard_loop(self):
        """后台读取键盘输入，转成事件（这样语音模式下键盘依然可用）"""
        while self.running:
            try:
                text = console.input("[bold blue]你:[/bold blue] ")
            except (KeyboardInterrupt, EOFError):
                self.events.put(Event(EventType.EXIT))
                return
            except Exception:
                return
            text = text.strip()
            if text:
                self.events.put(Event(EventType.TEXT_INPUT, text))

    # ==================================================================
    #  打断监听（用户开口时让机器人闭嘴）
    # ==================================================================
    def _start_barge_in_watcher(self):
        """TTS 播报期间监听麦克风音量，用户一开口就打断

        为什么要单独做：ASR 识别要等一句话说完（VAD 静音 700ms + 识别耗时），
        等识别出来再打断就太晚了。这里直接看音量，几十毫秒内就能反应。
        """
        if not (self.voice_config.BARGE_IN and self.recorder):
            return
        if self._barge_in_stop.is_set():
            return
        self._barge_in_stop.clear()

        threshold = self.voice_config.BARGE_IN_RMS

        def _watch():
            # 播放刚开始的一小段不判定，避免把播放起音当成用户插话
            time.sleep(0.25)
            while not self._barge_in_stop.is_set():
                if not self.tts.player.playing:
                    break
                level = self.recorder.level
                if level > threshold:
                    console.print("[dim]👂 检测到你在说话，停止播报[/dim]")
                    self.tts.interrupt()
                    break
                time.sleep(0.05)

        threading.Thread(target=_watch, daemon=True, name="BargeIn").start()

    def _stop_barge_in_watcher(self):
        self._barge_in_stop.set()

    # ==================================================================
    #  语音播报（统一入口）
    # ==================================================================
    def _speak(self, text: str, wait: bool = True):
        """语音播报：可被打断，播报时自动静音麦克风"""
        if not self.voice_reply_enabled:
            return
        if not text or not text.strip():
            return

        # 太长的话截断，避免念一大段
        spoken = text.strip()
        limit = self.voice_config.TTS_MAX_CHARS
        if len(spoken) > limit:
            spoken = spoken[:limit] + "……"

        try:
            if wait:
                # 先启动打断监听，再阻塞播放
                self._start_barge_in_watcher()
                try:
                    self.tts.speak(spoken, wait=True)
                finally:
                    self._stop_barge_in_watcher()
            else:
                self.tts.speak(spoken, wait=False)
        except Exception as e:
            console.print(f"[yellow]⚠️  语音播放失败: {e}[/yellow]")

    # ==================================================================
    #  命令处理
    # ==================================================================
    def handle_command(self, user_input: str) -> bool:
        cmd = user_input.strip().lower()

        if cmd in ("/exit", "/quit", "/q"):
            self.running = False
            return True

        if cmd == "/help":
            print_help()
            self._print_voice_help()
            return True

        if cmd == "/clear":
            with self._conversation_lock:
                self.conversation.clear()
            console.print("[green]✓ 对话历史已清空[/green]")
            return True

        if cmd == "/history":
            with self._conversation_lock:
                summary = self.conversation.get_recent_summary(n=10)
            console.print(Panel(summary, title="📜 最近对话", border_style="blue"))
            return True

        if cmd == "/camera":
            if self.camera is None:
                console.print("[red]✗ 摄像头不可用[/red]")
            else:
                console.print("[cyan]📷 打开摄像头预览，按 q 退出，按 s 保存[/cyan]")
                self.camera.preview()
            return True

        if cmd == "/save":
            with self._conversation_lock:
                self.conversation.save_history()
            console.print("[green]✓ 对话历史已保存[/green]")
            return True

        # ---------- 语音开关 ----------
        if cmd.startswith("/voice"):
            self._handle_voice_cmd(cmd)
            return True

        if cmd == "/listen":
            console.print(f"语音输入: {'开启' if self.segmenter else '未启动'}"
                          f" | 语音回复: {'开启' if self.voice_reply_enabled else '关闭'}"
                          f" | ASR就绪: {'是' if self.asr.ready else '否'}")
            return True

        if cmd == "/gpu":
            mem = check_gpu_memory()
            console.print(f"显存信息: {mem}")
            return True

        if cmd == "/audio":
            console.print(Panel(list_audio_devices(), title="🎧 音频设备",
                                border_style="cyan"))
            return True

        if cmd.startswith("/autovision"):
            parts = cmd.split()
            if len(parts) > 1:
                if parts[1] == "on":
                    self.auto_vision_enabled = True
                    console.print("[green]👁️  自动视觉识别已开启[/green]")
                elif parts[1] == "off":
                    self.auto_vision_enabled = False
                    console.print("[yellow]👁️  自动视觉识别已关闭（需手动 /look）[/yellow]")
            else:
                status = "开启" if self.auto_vision_enabled else "关闭"
                console.print(f"自动视觉识别当前状态: {status}")
            return True

        if cmd == "/monitor":
            if self.monitor:
                status = self.monitor.get_status()
                lines = [f"  {k}: {v}" for k, v in status.items()]
                console.print(Panel("\n".join(lines), title="📊 监控状态", border_style="cyan"))
            else:
                console.print("[yellow]监控未启动[/yellow]")
            return True

        if cmd.startswith("/register"):
            if self.face_rec is None:
                console.print("[red]✗ 人脸识别未启用[/red]")
                return True
            if self.camera is None:
                console.print("[red]✗ 摄像头不可用[/red]")
                return True
            parts = user_input.strip().split(maxsplit=1)
            name = parts[1].strip() if len(parts) > 1 else None
            if not name:
                console.print("[yellow]用法: /register 姓名  （例如: /register 小明）[/yellow]")
                return True
            console.print(f"[cyan]📸 正在拍照注册 {name}...[/cyan]")
            image = self.camera.capture_pil_image()
            if image is None:
                console.print("[red]✗ 拍照失败[/red]")
                return True
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            ok, msg, _face = self.face_rec.check_quality(frame_bgr)
            if not ok:
                console.print(f"[red]✗ {msg}[/red]")
                self._speak(msg)
                return True
            success, detail = self.face_rec.register_face(
                name, frame_bgr, quality_checked=True, face_box=_face)
            if success:
                console.print(f"[green]✓ {detail}[/green]")
                self._speak(f"记住了，你是{name}")
            else:
                console.print(f"[red]✗ {detail}[/red]")
                self._speak(detail)
            return True

        if cmd == "/faces":
            if self.face_rec is None:
                console.print("[yellow]人脸识别未启用[/yellow]")
            else:
                names = self.face_rec.get_registered_names()
                if names:
                    lines = []
                    for n in names:
                        cnt = self.face_rec.get_photo_count(n)
                        lines.append(f"  {n}：{cnt} 张照片")
                    console.print(Panel(
                        f"已注册 {len(names)} 人：\n" + "\n".join(lines),
                        title="👥 人脸库", border_style="green"))
                else:
                    console.print("[yellow]人脸库为空，用 /register 姓名 来注册[/yellow]")
            return True

        if cmd.startswith("/photos"):
            # 查看某人的照片清单
            if self.face_rec is None:
                console.print("[yellow]人脸识别未启用[/yellow]")
                return True
            parts = user_input.strip().split(maxsplit=1)
            name = parts[1].strip() if len(parts) > 1 else None
            if not name:
                console.print("[yellow]用法: /photos 姓名[/yellow]")
                return True
            entry = self.face_rec.face_db.get(name)
            if not entry:
                console.print(f"[yellow]没有找到 {name}[/yellow]")
                return True
            d = self.face_rec.face_dir / name
            lines = [f"  {name} 共 {len(entry['features'])} 张特征"]
            for i, f in enumerate(entry.get("photos", []), 1):
                p = d / f
                exists = "✓" if p.exists() else "✗缺失"
                lines.append(f"    {i}. {f}  {exists}")
            if not entry.get("photos"):
                lines.append("    （旧格式记录，原图未保存）")
            lines.append(f"\n  目录: {d}")
            console.print(Panel("\n".join(lines), title="🖼️ 照片", border_style="cyan"))
            return True

        if cmd.startswith("/forget"):
            if self.face_rec is None:
                console.print("[yellow]人脸识别未启用[/yellow]")
                return True
            parts = user_input.strip().split(maxsplit=2)
            name = parts[1].strip() if len(parts) > 1 else None
            index = None
            if len(parts) > 2:
                try:
                    index = int(parts[2])
                except ValueError:
                    console.print("[yellow]序号必须是数字，例如 /forget 小明 2[/yellow]")
                    return True
            if not name:
                console.print("[yellow]用法: /forget 姓名        （删掉这个人全部）[/yellow]")
                console.print("[yellow]      /forget 姓名 2      （只删第 2 张）[/yellow]")
                return True
            if index is not None:
                ok, msg = self.face_rec.delete_photo(name, index)
                console.print(f"[green]✓ {msg}[/green]" if ok else f"[yellow]{msg}[/yellow]")
            elif self.face_rec.delete_face(name):
                console.print(f"[green]✓ 已忘记 {name}[/green]")
            else:
                console.print(f"[yellow]没有找到 {name}[/yellow]")
            return True

        if cmd in ("/clearfaces", "/forgetall"):
            if self.face_rec is None:
                console.print("[yellow]人脸识别未启用[/yellow]")
                return True
            names = self.face_rec.get_registered_names()
            if not names:
                console.print("[yellow]人脸库已经是空的[/yellow]")
                return True
            console.print(f"[yellow]将删除全部 {len(names)} 人的记录和照片[/yellow]")
            self.face_rec.clear_db()
            console.print("[green]✓ 人脸库已清空[/green]")
            self._speak("好的，我把认识的人都忘掉了")
            return True

        if cmd.startswith("/look"):
            parts = user_input.strip().split(maxsplit=1)
            question = parts[1].strip() if len(parts) > 1 else ""
            if not question:
                question = "请描述你看到的画面，并关心一下我的状态"
            self.chat_with_vision(question)
            return True

        return False

    def _handle_voice_cmd(self, cmd: str):
        """/voice 系列命令"""
        parts = cmd.split()
        sub = parts[1] if len(parts) > 1 else ""

        if sub in ("on", ""):
            if sub == "on":
                self.voice_reply_enabled = True
            console.print(f"[green]🔊 语音回复: 开启[/green]"
                          if self.voice_reply_enabled else "语音回复: 关闭")
            if self.voice_reply_enabled and not self.voice_config.ENABLE_TTS:
                console.print("[yellow]⚠️  config.py 里 VoiceConfig.ENABLE_TTS 是 False，"
                              "需要改成 True 才能真正出声[/yellow]")
            return

        if sub == "off":
            self.voice_reply_enabled = False
            self.tts.interrupt()
            console.print("[yellow]🔇 语音回复已关闭[/yellow]")
            return

        if sub == "stop":
            self.tts.interrupt()
            console.print("[green]⏹️  已停止播报[/green]")
            return

        if sub == "input":
            self.voice_input_enabled = not self.voice_input_enabled
            state = "开启" if self.voice_input_enabled else "关闭"
            console.print(f"[green]🎤 语音输入: {state}[/green]")
            return

        console.print("用法: /voice on | off | stop | input")

    def _print_voice_help(self):
        lines = [
            "语音相关命令：",
            "  /voice on      开启语音回复",
            "  /voice off     关闭语音回复",
            "  /voice stop    立即停止正在播报的话",
            "  /voice input   切换语音输入开关",
            "  /listen        查看语音链路状态",
            "  /audio         列出可用音频设备",
            "",
        ]
        if not self.voice_config.ENABLE_TTS and not self.voice_config.ENABLE_ASR:
            lines.append("提示：语音默认关闭，在 config.py 的 VoiceConfig 里把")
            lines.append("      ENABLE_TTS / ENABLE_ASR 改成 True 即可启用。")
        console.print(Panel("\n".join(lines), title="🎤 语音帮助", border_style="green"))

    # ==================================================================
    #  意图执行
    # ==================================================================
    def execute_intent(self, intent: str, params: dict) -> bool:
        if intent == Intent.REGISTER_FACE:
            if self.face_rec is None or self.camera is None:
                console.print("[yellow]人脸识别或摄像头不可用[/yellow]")
                return True
            name = params.get("name")
            if not name:
                self.pending_intent = Intent.REGISTER_FACE
                self.pending_params = {}
                console.print("[cyan]好的，我会记住你的脸。你叫什么名字？[/cyan]")
                self._speak("好的，我会记住你的脸。你叫什么名字？")
                return True
            self._do_register_face(name)
            return True

        if intent == Intent.DELETE_FACE:
            if self.face_rec is None:
                console.print("[yellow]人脸识别未启用[/yellow]")
                return True
            name = params.get("name")
            if not name and self.camera:
                image = self.camera.capture_pil_image()
                if image:
                    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    results = self.face_rec.recognize(frame_bgr)
                    known = [r[0] for r in results if r[0] != "unknown"]
                    if known:
                        name = known[0]
            if name:
                if self.face_rec.delete_face(name):
                    console.print(f"[green]好的，我已经忘记 {name} 了[/green]")
                    self._speak(f"好的，我已经忘记{name}了")
                else:
                    console.print(f"[yellow]我没有找到 {name} 的记录[/yellow]")
            else:
                console.print("[yellow]你想让我忘记谁？可以直接说名字[/yellow]")
            return True

        if intent == Intent.LIST_FACES:
            if self.face_rec is None:
                console.print("[yellow]人脸识别未启用[/yellow]")
            else:
                names = self.face_rec.get_registered_names()
                if names:
                    console.print(f"[green]我认识 {len(names)} 个人: {', '.join(names)}[/green]")
                    self._speak(f"我认识{len(names)}个人")
                else:
                    console.print("[yellow]我还不认识任何人[/yellow]")
                    self._speak("我还不认识任何人，你可以说记住我的脸，让我认识你")
            return True

        # --- 清空人脸库（旧版缺失这个意图，「把所有人删除」会掉进普通聊天）---
        if intent == Intent.CLEAR_FACES:
            if self.face_rec is None:
                console.print("[yellow]人脸识别未启用[/yellow]")
                return True
            names = self.face_rec.get_registered_names()
            if not names:
                console.print("[yellow]人脸库已经是空的[/yellow]")
                self._speak("我本来就不认识任何人")
                return True
            self.face_rec.clear_db()
            console.print(f"[green]✓ 已清空 {len(names)} 人的记录和照片[/green]")
            self._speak("好的，我把认识的人都忘掉了")
            return True

        # --- 「我是谁」：先看摄像头认出身份，再把结果交给模型组织语言 ---
        # 旧版这条会掉进普通聊天，模型不知道镜头前是谁，
        # 于是把自己（小护）当成了用户，回答"我是小护"。
        if intent == Intent.WHO_AM_I:
            if self.face_rec is None or self.camera is None:
                console.print("[yellow]人脸识别或摄像头不可用，我认不出你[/yellow]")
                return True
            image = self.camera.capture_pil_image()
            if image is None:
                console.print("[yellow]拍照失败，我暂时看不清[/yellow]")
                return True
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = self.face_rec.recognize(frame_bgr)
            known = [r[0] for r in results if r[0] != "unknown"]
            if known:
                who = known[0]
                console.print(f"[dim]👤 认出: {who}[/dim]")
                # 给出明确的句式指令：
                # 只说"识别结果是XX"不够——实测模型会用「我就是XX」回答，
                # 把自己当成了用户。这里直接规定人称和句式。
                self.chat_with_vision(
                    f"用户问「我是谁」。摄像头识别结果：用户是「{who}」。\n"
                    f"请用第二人称回答，正确句式是「你是{who}」。\n"
                    f"注意：「{who}」是用户的名字，不是你的名字；"
                    f"你是机器人小护，不要用「我是{who}」这种说法。"
                )
            else:
                # 没认出 -> 主动询问名字，走两步确认的引导注册。
                # 这里不直接把识别结果丢给模型，因为"你是谁"这个问题
                # 需要真实身份，靠模型猜只会得到编造的答案。
                if self._maybe_offer_enroll():
                    console.print("[dim]👤 没认出，进入引导注册[/dim]")
                else:
                    self._speak("我暂时没认出你，你可以说「记住我的脸」让我先认识你")
                    console.print("[yellow]没认出画面中的人（未注册或角度不好）[/yellow]")
            return True

        if intent == Intent.CAMERA_PREVIEW:
            if self.camera is None:
                console.print("[red]摄像头不可用[/red]")
            else:
                console.print("[cyan]打开摄像头预览，按 q 退出，按 s 保存[/cyan]")
                self.camera.preview()
            return True

        if intent == Intent.CLEAR_HISTORY:
            with self._conversation_lock:
                self.conversation.clear()
            console.print("[green]好的，对话已清空，我们重新开始吧！[/green]")
            self._speak("好的，我们重新开始吧")
            return True

        if intent == Intent.SHOW_HISTORY:
            with self._conversation_lock:
                summary = self.conversation.get_recent_summary(n=10)
            console.print(Panel(summary, title="📜 最近对话", border_style="blue"))
            return True

        if intent == Intent.MONITOR_STATUS:
            if self.monitor:
                status = self.monitor.get_status()
                lines = [f"  {k}: {v}" for k, v in status.items()]
                console.print(Panel("\n".join(lines), title="📊 监控状态", border_style="cyan"))
            else:
                console.print("[yellow]监控未启动[/yellow]")
            return True

        if intent == Intent.HELP:
            self._print_natural_help()
            return True

        if intent == Intent.EXIT:
            # 告别要真的说出来再走。
            # 旧实现直接 running=False，主循环立刻 break，
            # 用户看到的就是"程序无声无息地关了"。
            import random
            farewell = random.choice([
                "再见，我在这儿等你回来",
                "好的，那你先忙，我随时都在",
                "再见，记得好好照顾自己",
            ])
            console.print(Panel(Text(farewell, style="green"),
                                title="🤖 小护（告别）", border_style="green"))
            with self._conversation_lock:
                self.conversation.add_assistant_message(farewell)
            self._speak(farewell, wait=True)   # 阻塞播完再退
            console.print("[dim]再见~ 正在退出...[/dim]\n")
            self.running = False
            return True

        return False

    def _do_register_face(self, name: str):
        console.print(f"[cyan]📸 好的 {name}，正对摄像头，我来记住你的脸...[/cyan]")
        image = self.camera.capture_pil_image()
        if image is None:
            console.print("[red]拍照失败，请重试[/red]")
            return
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 先做质量检查，不合格就明确告诉用户哪里不行
        # （旧版只要检测到脸就存，导致注册进垃圾特征、之后永远认不出）
        ok, msg, face = self.face_rec.check_quality(frame_bgr)
        if not ok:
            console.print(f"[yellow]✗ {msg}[/yellow]")
            self._speak(msg)
            return

        success, detail = self.face_rec.register_face(
            name, frame_bgr, quality_checked=True, face_box=face)
        if success:
            console.print(f"[green]✓ {detail}[/green]")
            n = self.face_rec.get_photo_count(name)
            tip = "以后我就能认出你了"
            if n > 1:
                tip = f"这是第 {n} 张了，认你会更稳"
            self._speak(f"记住了，你是{name}，{tip}")
        else:
            console.print(f"[yellow]✗ {detail}[/yellow]")
            self._speak(detail)

    def _print_natural_help(self):
        examples = IntentRecognizer.get_examples()
        lines = ["你可以直接对我说以下的话，不需要记命令：\n"]
        for say, do in examples:
            lines.append(f"  「{say}」→ {do}")
        lines.append("\n其他问题直接问我就行，我会自动判断要不要看摄像头~")
        console.print(Panel("\n".join(lines), title="💡 我能做什么", border_style="green"))

    # ==================================================================
    #  对话
    # ==================================================================
    def chat_with_vision(self, text: str):
        if self.camera is None:
            self.chat_text(text)
            return

        console.print("[cyan]📸 正在拍照...[/cyan]")
        image = self.camera.capture_pil_image()
        if image is None:
            console.print("[red]✗ 拍照失败，使用纯文字模式[/red]")
            self.chat_text(text)
            return

        # 身份信息注入 prompt
        enhanced_text = text
        if self.face_rec:
            try:
                frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                if self.face_rec.load(silent=True, allow_download=False):
                    face_results = self.face_rec.recognize(frame_bgr)
                    names = [r[0] for r in face_results if r[0] != "unknown"]
                    if names:
                        who = "、".join(names)
                        console.print(f"[dim]👤 识别到: {who}[/dim]")
                        # 措辞要明确「这是用户的名字」，否则模型可能用
                        # 第一人称复述，把自己当成用户
                        enhanced_text = (
                            f"[画面中的用户是「{who}」]\n{text}"
                        )
            except Exception as e:
                console.print(f"[dim]人脸识别跳过: {e}[/dim]")

        console.print("[cyan]🤔 正在思考（结合画面）...[/cyan]")
        with self._conversation_lock:
            history = self.conversation.get_history_for_model()
        response = self.vl_model.generate(text=enhanced_text, image=image, history=history)

        with self._conversation_lock:
            self.conversation.add_user_message(f"[看图] {text}")
            self.conversation.add_assistant_message(response)

        self._print_response(response)
        self._speak(response)

    # ==================================================================
    #  引导注册（认不出人时的两步确认）
    # ==================================================================
    def _guided_ask(self, prompt: str):
        """提问：你叫什么名字？"""
        console.print(Panel(Text(prompt, style="cyan"),
                            title="🤖 小护（不认识你）", border_style="cyan"))
        self._speak(prompt)
        self._guided_pending_speak = prompt

    def _guided_confirm(self, name: str):
        """复述确认：你叫 XX，对吗？

        为什么要复述：语音识别名字常出错（「武凉烨」->「五粮液」），
        让用户确认一遍能挡掉误存。
        """
        if not name:
            return
        text = f"好的，你叫「{name}」，是这个吗？"
        console.print(Panel(Text(text, style="cyan"),
                            title="🤖 小护（确认名字）", border_style="cyan"))
        self._speak(text)

    def _guided_do_enroll(self, name: str):
        """确认通过 -> 拍照并注册"""
        if self.face_rec is None or self.camera is None:
            console.print("[yellow]人脸识别或摄像头不可用，无法登记[/yellow]")
            return
        console.print(f"[cyan]📸 好的 {name}，我记住你的脸...[/cyan]")
        image = self.camera.capture_pil_image()
        if image is None:
            console.print("[red]拍照失败[/red]")
            self._speak("我没能拍到你的脸，稍后再试吧")
            return
        frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        ok, msg, face = self.face_rec.check_quality(frame_bgr)
        if not ok:
            console.print(f"[yellow]✗ {msg}[/yellow]")
            self._speak(msg)
            return
        success, detail = self.face_rec.register_face(
            name, frame_bgr, quality_checked=True, face_box=face)
        if success:
            console.print(f"[green]✓ {detail}[/green]")
            self._speak(f"记住了，你是{name}")
            with self._conversation_lock:
                self.conversation.add_assistant_message(f"（已登记人脸：{name}）")
        else:
            console.print(f"[yellow]✗ {detail}[/yellow]")
            self._speak(detail)

    def _guided_giveup(self, reason: str):
        """放弃引导"""
        console.print(f"[dim]引导注册结束：{reason}[/dim]")
        self._speak("没关系，等你想告诉我的时候再说")

    def _maybe_offer_enroll(self):
        """认不出人时，判断要不要主动问名字

        触发条件（三个都满足）：
          · 用户这句话涉及身份（"我是谁""你认识我吗"）
          · 人脸识别没认出他
          · 人脸库里已经有别人（全新环境不追问，避免对客人突兀）
        """
        if not self.guided.should_offer(len(self.face_rec.face_db)
                                       if self.face_rec else 0):
            return False
        return self.guided.start()

    @staticmethod
    def _is_identity_related(text: str) -> bool:
        """判断这句话是否在问"我是谁"这类身份问题

        注意排除"记住我的脸"——那走的是显式注册流程（会问名字），
        不需要再叠加一层引导。
        """
        t = (text or "").strip().lower()
        if any(k in t for k in ["记住我", "注册", "认识我", "我叫"]):
            return False
        return any(k in t for k in [
            "我是谁", "认得我", "认识我吗", "你认识我", "记得我吗",
            "知道我是谁", "看出来我是", "认识一下我",
        ])

    def _detect_identity(self, max_age: float = 6.0) -> str | None:
        """快速识别镜头前是谁（用于身份注入）

        优先复用监控线程最近一次在场检测的结果——它每 1.5 秒就跑一次，
        通常比主对话重新拍一张更新，也避免两处同时抢摄像头。
        缓存过期或没认出时才自己拍一张识别。
        """
        # 1) 复用监控快照
        if self.monitor is not None:
            try:
                snap = self.monitor.get_snapshot(max_age=max_age)
                name = snap.get("name")
                if name and name != "unknown":
                    return name
            except Exception:
                pass

        # 2) 自己识别
        if self.face_rec is None or self.camera is None:
            return None
        try:
            if not self.face_rec.load(silent=True, allow_download=False):
                return None
            image = self.camera.capture_pil_image(max_retries=1)
            if image is None:
                return None
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = self.face_rec.recognize(frame_bgr)
            known = [r[0] for r in results if r[0] != "unknown"]
            return known[0] if known else None
        except Exception as e:
            console.print(f"[dim]身份识别跳过: {e}[/dim]")
            return None

    def chat_text(self, text: str):
        """纯文字对话

        ⚠️ 旧版这里**没有做身份识别**，导致：
          - 用户问「我是谁」时，模型不知道镜头前是谁，把自己当成了用户
          - 任何需要知道"在和谁说话"的对话都缺上下文
        现在每轮都做一次轻量身份识别并注入提示。
        """
        console.print("[cyan]🤔 正在思考...[/cyan]")

        # 身份注入：让模型知道镜头前是谁
        # 注意要**简洁**——之前写了一大段规则（"他是你的家人…注意你叫小护…"），
        # 实测会把模型带偏：用户问"我有没有戴眼镜"，模型却回"好的我们继续聊吧"。
        # 这里只给一行事实，行为约束交给 SYSTEM_PROMPT。
        who = self._detect_identity()
        if who:
            # 措辞要与 SYSTEM_PROMPT 里的说明一致：
            # 明确这是「用户」的名字，避免模型用「我」去复述它
            enhanced = f"[当前和你说话的用户叫「{who}」]\n{text}"
            console.print(f"[dim]👤 身份: {who}[/dim]")
        else:
            enhanced = text
            # 没认出人、而这句话又在聊身份 -> 主动询问名字（两步确认）
            # 覆盖「你认识我吗」「认得我是谁吗」这类没命中 WHO_AM_I 意图的说法
            if self._is_identity_related(text):
                if self._maybe_offer_enroll():
                    console.print("[dim]👤 没认出 + 身份话题 -> 引导注册[/dim]")
                    return

        with self._conversation_lock:
            history = self.conversation.get_history_for_model()
        response = self.vl_model.generate(text=enhanced, image=None, history=history)

        with self._conversation_lock:
            self.conversation.add_user_message(text)
            self.conversation.add_assistant_message(response)

        self._print_response(response)
        self._speak(response)

    def _print_response(self, response: str):
        console.print(Panel(Text(response, style="white"),
                            title="🤖 小护", border_style="green"))

    def _print_proactive(self, message: str):
        console.print(Panel(Text(message, style="yellow"),
                            title="🤖 小护（主动关心）", border_style="yellow"))

    def _process_auto_vision(self, text: str):
        use_vision = False
        reason = ""
        if self.auto_vision_enabled and self.camera:
            with self._conversation_lock:
                history = self.conversation.get_history_for_model()
            use_vision, reason = VisionTrigger.should_use_vision(text, history)

        if use_vision:
            console.print(f"[dim]👁️  自动视觉触发: {reason}[/dim]")
            self.chat_with_vision(VisionTrigger.get_vision_prompt(text))
        else:
            self.chat_text(text)

    # ==================================================================
    #  事件处理（在后台线程中执行一轮对话）
    # ==================================================================
    def _handle_user_text(self, text: str, source: str = "text"):
        """处理用户一句话（键盘或语音）—— 可能耗时，所以跑在线程里"""
        self._processing.set()
        try:
            # 用户开口了，先让机器人闭嘴
            self.tts.interrupt()

            # 输入过滤（只对语音生效——键盘输入的"嗯"是用户故意的）
            if source == "voice":
                ok, reason = self.utterance_filter.check(text)
                if not ok:
                    console.print(f"[dim]🔇 已忽略：{reason}[/dim]")
                    return
                self.utterance_filter.accept(text)

            tag = "🎤" if source == "voice" else "你"
            if source == "voice":
                console.print(f"[bold blue]{tag}:[/bold blue] {text}")

            # 1.5 引导注册流程优先：正在等用户报名字/确认时，
            #     这句话属于该流程，不能被当成普通对话或独立意图。
            if self.guided.active:
                consumed = self.guided.handle(text)
                if consumed:
                    return
                # 未被消费（例如超时放弃）→ 继续走正常处理

            # 2. 待处理意图（上一轮在追问参数）
            if self.pending_intent:
                pending = self.pending_intent
                self.pending_intent = None
                if pending == Intent.REGISTER_FACE:
                    name = text.strip().rstrip("。！？.?!")
                    self._do_register_face(name)
                return

            # 2.5 命令（斜杠）
            if self.handle_command(text):
                return

            # 3. 自然语言意图
            intent, params = IntentRecognizer.recognize(text)
            if self.execute_intent(intent, params):
                return

            # 4. 普通对话
            self._process_auto_vision(text)
        except Exception as e:
            console.print(f"[red]✗ 处理失败: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            self._processing.clear()

    def _handle_proactive(self, message: str, meta: dict):
        """处理监控主动消息"""
        self._print_proactive(message)
        with self._conversation_lock:
            self.conversation.add_assistant_message(message)

        # 打招呼/异常类消息：语音播报（打招呼默认要出声）
        kind = meta.get("kind", "")
        should_speak = self.voice_reply_enabled
        if kind == "greeting" and self.monitor_config.GREETING_WITH_TTS:
            should_speak = self.voice_reply_enabled
        if should_speak:
            self._speak(message)

    # ==================================================================
    #  主循环（事件驱动）
    # ==================================================================
    def run(self):
        print_banner()
        self.initialize()
        print_help()
        self._print_voice_help()

        voice_state = []
        if self.recorder:
            voice_state.append("语音输入✓")
        if self.voice_reply_enabled:
            voice_state.append("语音回复✓")
        state = "、".join(voice_state) if voice_state else "纯文字模式"
        console.print(f"[green]✓ 系统就绪（{state}），开始对话吧！[/green]\n")

        while self.running:
            try:
                event = self.events.get(timeout=0.3)
            except queue.Empty:
                # 兜底：兼容没有回调时监控的内部队列
                if self.monitor and self.monitor.has_message():
                    msg = self.monitor.get_message()
                    if msg:
                        self._handle_proactive(msg, {"kind": "legacy"})
                continue

            if event.type == EventType.EXIT:
                self.running = False
                break

            if event.type == EventType.PROACTIVE:
                self._handle_proactive(event.payload, event.meta)
                continue

            if event.type in (EventType.TEXT_INPUT, EventType.VOICE_INPUT):
                source = "voice" if event.type == EventType.VOICE_INPUT else "text"
                # 对话处理放后台线程：这样播报期间还能响应打断/新命令
                threading.Thread(
                    target=self._handle_user_text,
                    args=(event.payload, source),
                    daemon=True,
                    name="DialogWorker",
                ).start()

        self.shutdown()

    def shutdown(self):
        """优雅退出：按顺序收尾，避免留下音频设备句柄"""
        console.print("\n[cyan]正在收尾...[/cyan]")
        self._stop_barge_in_watcher()
        try:
            self.tts.interrupt()
            self.tts.close()
        except Exception:
            pass
        if self.segmenter:
            self.segmenter.stop()
        if self.recorder:
            self.recorder.stop()
        if self.monitor:
            self.monitor.stop()
        if self.asr:
            self.asr.unload()
        graceful_exit(self.camera, self.vl_model)


# ======================================================================
#  入口
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="家庭陪护机器人 HomeCRobot")
    parser.add_argument("--no-camera", action="store_true",
                        help="禁用摄像头，仅文字对话")
    parser.add_argument("--no-voice", action="store_true",
                        help="强制关闭语音输入输出")
    parser.add_argument("--voice", action="store_true",
                        help="强制开启语音输入输出（覆盖 config.py）")
    parser.add_argument("--model", type=str, default=None,
                        help="指定模型名称（覆盖config.py中的设置）")
    parser.add_argument("--cpu", action="store_true",
                        help="强制使用CPU（不推荐，速度极慢）")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.model:
        VLConfig.MODEL_NAME = args.model

    # 命令行覆盖语音开关
    if args.voice:
        VoiceConfig.ENABLE_ASR = True
        VoiceConfig.ENABLE_TTS = True
    if args.no_voice:
        VoiceConfig.ENABLE_ASR = False
        VoiceConfig.ENABLE_TTS = False

    if args.cpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    robot = HomeCRobot(args)

    if args.no_camera:
        robot.camera_config.CAMERA_INDEX = -1

    try:
        robot.run()
    except KeyboardInterrupt:
        robot.shutdown()
    except Exception as e:
        console.print(f"[red]✗ 发生错误: {e}[/red]")
        import traceback
        traceback.print_exc()
        try:
            robot.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
