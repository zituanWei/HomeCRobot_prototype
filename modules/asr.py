"""
语音识别模块 (ASR - Automatic Speech Recognition)
=================================================
引擎：FunASR（阿里达摩院，中文识别效果好，支持本地离线）

关键设计（相比旧版的变化）：
1. **补上了完全缺失的麦克风采集**
   旧版只有 transcribe_from_array 的 NotImplementedError，机器人根本听不到声音。
   现在由 core.audio.AudioRecorder + VADSegmenter 负责采集与断句，
   本模块专注"把一句话的音频转成文字"。
2. **支持 numpy 数组直接识别**（无需先落盘成 wav）
3. **模型可后台预热**，避免第一次说话时卡住好几秒
4. **加锁保护**，FunASR 模型不是线程安全的

启用方式：config.py 中 VoiceConfig.ENABLE_ASR = True
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from config import VoiceConfig
from core.audio import AudioRecorder, EnergyVAD, VADSegmenter, load_audio


def is_periodic_voice(audio: np.ndarray, sample_rate: int,
                      min_periodicity: float = 0.30) -> tuple[bool, float]:
    """判断音频是否像「人声」而不是「噪声」

    原理：
      人说话时有**基频**（声带振动），波形呈现明显周期性；
      咳嗽/清嗓/敲桌/拍手是宽带噪声爆发，没有稳定周期。

    做法：对每帧算归一化自相关，取基频范围（70~350Hz）内的峰值。
      峰值高 => 有稳定周期 => 像人声
      峰值低 => 无周期 => 像噪声

    ⚠️ 局限：唱歌同样有基频，所以这个方法**挡不住唱歌**。
       那需要唤醒词或声纹识别。

    Returns: (是否像人声, 平均周期性得分)
    """
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size < sample_rate * 0.1:
        return False, 0.0

    # 分帧（50ms，25ms 跳步），只取能量足够高的帧
    win = int(sample_rate * 0.05)
    hop = max(1, int(sample_rate * 0.025))
    lo = max(2, int(sample_rate / 350))     # 350Hz 对应最小周期
    hi = min(win - 2, int(sample_rate / 70))  # 70Hz 对应最大周期
    if hi <= lo:
        return True, 0.0

    scores = []
    for start in range(0, a.size - win, hop):
        frame = a[start:start + win]
        energy = float(np.sqrt(np.mean(frame ** 2)))
        if energy < 0.01:          # 太安静，跳过
            continue
        frame = frame - float(np.mean(frame))
        denom = float(np.dot(frame, frame))
        if denom <= 1e-9:
            continue
        # 归一化自相关（只算需要的滞后区间）
        ac = np.correlate(frame, frame, mode="full")[win - 1:]
        ac = ac[:hi + 1] / denom
        if lo >= len(ac):
            continue
        peak = float(np.max(ac[lo:hi + 1])) if hi + 1 <= len(ac) else float(np.max(ac[lo:]))
        scores.append(peak)
        if len(scores) >= 40:      # 够用就停，控制耗时
            break

    if not scores:
        return False, 0.0

    mean_score = float(np.mean(scores))
    return mean_score >= min_periodicity, mean_score


class ASR:
    """语音识别封装（FunASR）"""

    def __init__(self, config: VoiceConfig = None):
        self.config = config or VoiceConfig()
        self.model = None
        self._lock = threading.Lock()        # 串行化识别调用
        self._loading = threading.Event()    # 是否正在加载
        self._ready = threading.Event()      # 是否加载完成
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    #  模型加载
    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self, background: bool = False) -> bool:
        """加载 FunASR 模型

        Args:
            background: True 则在后台线程加载（首次要从 ModelScope 下载，
                        约几百 MB，会卡住主流程，所以推荐后台预热）
        Returns:
            同步模式下返回是否加载成功；后台模式一律返回 True（结果看 ready）
        """
        if not self.config.ENABLE_ASR:
            print("[ASR] 未启用语音识别")
            return False
        if self.model is not None:
            return True

        if background:
            if not self._loading.is_set():
                threading.Thread(target=self._load_sync, daemon=True,
                                 name="ASRLoader").start()
            return True
        return self._load_sync()

    def _load_sync(self) -> bool:
        self._loading.set()
        try:
            from funasr import AutoModel
            print("[ASR] 正在加载语音识别模型（首次会自动下载）...")
            t0 = time.time()
            self.model = AutoModel(
                model=self.config.ASR_MODEL,
                vad_model=self.config.ASR_VAD_MODEL,
                disable_update=True,
            )
            print(f"[ASR] 模型加载完成 ✓ 耗时 {time.time() - t0:.1f}s")
            self._ready.set()
            return True
        except ImportError:
            self.last_error = "未安装 funasr，请运行: pip install funasr modelscope"
            print(f"[ASR] {self.last_error}")
        except Exception as e:
            self.last_error = f"加载失败: {e}"
            print(f"[ASR] {self.last_error}")
        finally:
            self._loading.clear()
        return False

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """等待后台加载完成"""
        if self.model is not None:
            return True
        return self._ready.wait(timeout=timeout)

    # ------------------------------------------------------------------
    #  识别
    # ------------------------------------------------------------------
    def transcribe(self, audio_path: str) -> str:
        """转写音频文件"""
        if self.model is None:
            return "[ASR未启用]"
        try:
            samples, sr = load_audio(audio_path, self.config.ASR_SAMPLE_RATE)
            return self.transcribe_array(samples, sr)
        except Exception as e:
            print(f"[ASR] 文件识别失败: {e}")
            return ""

    def transcribe_array(self, audio: np.ndarray,
                         sample_rate: Optional[int] = None) -> str:
        """识别 numpy 音频数组（实时麦克风主路径）

        Args:
            audio: float32 [-1,1] 音频
            sample_rate: 采样率，默认取配置值
        Returns:
            识别出的文字（失败或太短返回空串）

        注意：FunASR 的 frontend 会对波形做 mean() 之类的浮点运算，
        传入 int16 会直接抛
        "mean(): could not infer output dtype ... Got: Short"。
        所以这里必须保持 **float32**，不要做 ×32768 转 int16。
        """
        if self.model is None:
            if not self._ready.is_set():
                # 模型还在后台加载，本次先放弃，避免阻塞对话
                print("[ASR] 模型尚未加载完成，本次语音已跳过")
            return ""

        sample_rate = sample_rate or self.config.ASR_SAMPLE_RATE
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        # --- 过滤1：太短。咳嗽/清嗓/敲击大多在 0.4 秒内 ---
        min_ms = getattr(self.config, "MIN_SPEECH_MS", 400)
        if audio.size < sample_rate * (min_ms / 1000.0):
            return ""

        # 幅度过小视为静音（VAD 漏网的情况）
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < 0.002:
            return ""

        # 有效语音占比检查：整段里"有明显声音"的帧太少，
        # 说明是以噪声/静音为主的片段（VAD 被噪声误触发是常见情况）。
        frame = max(1, int(sample_rate * 0.03))
        n_frames = audio.size // frame
        if n_frames >= 3:
            frames = audio[:n_frames * frame].reshape(n_frames, frame)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            voiced = float(np.mean(frame_rms > max(peak * 0.15, 0.002)))
            if voiced < 0.10:
                return ""

        # 统一到 float32 [-1, 1]
        if peak > 1.5:
            audio = audio / 32768.0
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        # --- 过滤2：周期性。咳嗽/清嗓是宽带噪声（无基频），人声有基频 ---
        if getattr(self.config, "ENABLE_PERIODICITY_CHECK", True):
            ok, score = is_periodic_voice(
                audio, sample_rate,
                getattr(self.config, "MIN_PERIODICITY", 0.30))
            if not ok:
                print(f"[ASR] 判定为非人声（周期得分 {score:.2f}），跳过"
                      f"（多半是咳嗽/清嗓/敲击）")
                return ""

        try:
            with self._lock:
                result = self.model.generate(
                    input=audio,
                    fs=sample_rate,
                    cache={},
                    language="zh",
                    use_itn=True,
                )
            if not result:
                return ""
            text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            text = text.replace(" ", "").strip()

            # FunASR 没检测到有效语音时返回 "empty speech" 之类的提示
            low = text.lower()
            if low in ("emptyspeech", "empty", "[emptyspeech]"):
                return ""

            # --- 过滤3：单个语气词。清嗓常被识别成"嗯"、"呃" ---
            if getattr(self.config, "DROP_FILLER_WORDS", True):
                fillers = set(getattr(self.config, "FILLER_WORDS", []))
                if text in fillers:
                    print(f"[ASR] 丢弃语气词 {text!r}")
                    return ""

            return text
        except Exception as e:
            print(f"[ASR] 识别失败: {e}")
            return ""

    # ------------------------------------------------------------------
    #  麦克风实时识别（采集 + 断句 + 识别 的完整链路）
    # ------------------------------------------------------------------
    def build_live_pipeline(self, on_text, recorder: AudioRecorder = None):
        """构造实时语音输入链路

        组装 AudioRecorder（常开采集） + EnergyVAD（断句） + 本模块（识别），
        在检测到一整句话后自动识别并回调文字。

        Args:
            on_text: 回调 (text: str) -> None
            recorder: 可复用外部已创建的录音机（便于主程序统一管理）
        Returns:
            (recorder, segmenter, vad) —— 需要由调用方 start()/stop()
        """
        recorder = recorder or AudioRecorder(
            sample_rate=self.config.ASR_SAMPLE_RATE,
            device=self.config.MIC_DEVICE,
        )
        vad = EnergyVAD(
            sample_rate=self.config.ASR_SAMPLE_RATE,
            start_threshold=self.config.VAD_START_THRESHOLD,
            end_threshold=self.config.VAD_END_THRESHOLD,
            min_silence_ms=self.config.VAD_MIN_SILENCE_MS,
            max_speech_ms=self.config.VAD_MAX_SPEECH_MS,
            prebuffer_ms=getattr(self.config, "VAD_PREBUFFER_MS", 600),
            soft_ratio=getattr(self.config, "VAD_SOFT_THRESHOLD_RATIO", 0.5),
        )

        def _handle(audio: np.ndarray):
            text = self.transcribe_array(audio)
            if text:
                on_text(text)

        segmenter = VADSegmenter(recorder, vad, _handle)
        return recorder, segmenter, vad

    def unload(self):
        """卸载模型"""
        with self._lock:
            if self.model is not None:
                del self.model
                self.model = None
        self._ready.clear()
        print("[ASR] 已卸载")


class ASRUnavailable(RuntimeError):
    """ASR 不可用（未启用或缺依赖）"""
