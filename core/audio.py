"""
音频 I/O 基础层
===============
为语音输入输出提供统一的地基能力：

- AudioPlayer  : 后台线程播放，支持"等待播完"与"立即打断"
- AudioRecorder: 麦克风 16kHz 单声道常开采集，写入环形缓冲
- RingBuffer   : 线程安全的环形缓冲，供 VAD 读取最近音频
- EnergyVAD    : 基于短时能量的语音端点检测（零额外依赖，中文/嘈杂环境可调）
- VADSegmenter : 把连续音频切成"一句一句"的话音片段

设计原则：
1. 采集与播放解耦，各自独立声卡流，避免争用
2. 播放走独立线程，绝不阻塞主循环
3. 所有硬件异常都被捕获并降级，不因为没麦克风就整个程序起不来
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:  # pragma: no cover - 环境缺库时降级
    sd = None
    HAS_SOUNDDEVICE = False


# ============================================================
#  环形缓冲
# ============================================================
class RingBuffer:
    """固定容量的 float32 环形缓冲（线程安全）

    写入端是声卡回调线程，读取端是 VAD / 识别线程，
    用长度与最后写入时间做粗粒度同步，足够本项目使用。
    """

    def __init__(self, capacity_samples: int):
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._capacity = capacity_samples
        self._write = 0          # 下一个写入位置
        self._filled = 0         # 已填充样本数
        self._total = 0          # 累计写入样本总数（只增不减，用于增量读取）
        self._lock = threading.Lock()
        self.last_write_time = 0.0

    def write(self, data: np.ndarray):
        data = np.asarray(data, dtype=np.float32).reshape(-1)
        n = len(data)
        if n == 0:
            return
        with self._lock:
            if n >= self._capacity:
                # 一次写入超过容量，只保留最后 capacity 个
                self._buf[:] = data[-self._capacity:]
                self._write = 0
                self._filled = self._capacity
            else:
                end = self._write + n
                if end <= self._capacity:
                    self._buf[self._write:end] = data
                else:
                    first = self._capacity - self._write
                    self._buf[self._write:] = data[:first]
                    self._buf[:end - self._capacity] = data[first:]
                self._write = end % self._capacity
                self._filled = min(self._capacity, self._filled + n)
            self._total += n
            self.last_write_time = time.time()

    @property
    def total_written(self) -> int:
        """累计写入的样本总数，用于增量读取（只读新增音频）"""
        with self._lock:
            return self._total

    def read_since(self, consumed_total: int) -> tuple[np.ndarray, int]:
        """读取自 consumed_total 之后新增的样本

        Returns: (新增音频, 新的 consumed 位置)
        环形缓冲溢出时，超出容量的旧数据已被覆盖，此时从缓冲最早可用处返回。
        """
        with self._lock:
            new_count = self._total - consumed_total
            if new_count <= 0:
                return np.zeros(0, dtype=np.float32), self._total
            # 最多只能拿到缓冲容量内的数据
            available = min(new_count, self._filled)
            start = (self._write - available) % self._capacity
            if start + available <= self._capacity:
                data = self._buf[start:start + available].copy()
            else:
                first = self._capacity - start
                data = np.concatenate([self._buf[start:], self._buf[:available - first]])
            return data, self._total

    def read_last(self, n_samples: int) -> np.ndarray:
        """读取最近 n_samples 个样本（不足则返回已有的全部）"""
        with self._lock:
            n = min(n_samples, self._filled)
            if n <= 0:
                return np.zeros(0, dtype=np.float32)
            start = (self._write - n) % self._capacity
            if start + n <= self._capacity:
                return self._buf[start:start + n].copy()
            first = self._capacity - start
            return np.concatenate([self._buf[start:], self._buf[:n - first]])

    def clear(self):
        """清空缓冲内容（例如 TTS 播放前，避免把自己的声音当成用户说话）

        只丢弃已缓存的数据，不动 _total，这样增量读取方会自动跳过这段。
        """
        with self._lock:
            self._filled = 0
            self._write = 0

    @property
    def filled(self) -> int:
        with self._lock:
            return self._filled


# ============================================================
#  播放器
# ============================================================
class AudioPlayer:
    """后台播放音频，支持等待与打断

    用法：
        player.play(samples, 24000)
        player.wait(timeout=10)      # 等它播完
        player.stop()                # 立即打断（用户插话时用）
    """

    def __init__(self, sample_rate: int = 24000, channels: int = 1,
                 device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._playing = threading.Event()
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return HAS_SOUNDDEVICE

    @property
    def playing(self) -> bool:
        return self._playing.is_set()

    def play(self, samples: np.ndarray, sample_rate: Optional[int] = None):
        """异步开始播放；会先打断上一次未播完的内容"""
        if not HAS_SOUNDDEVICE:
            self.last_error = "sounddevice 未安装"
            return
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.size == 0:
            return

        self.stop()

        with self._lock:
            self._stop_event.clear()
            self._playing.set()
            sr = sample_rate or self.sample_rate
            self._thread = threading.Thread(
                target=self._play_worker,
                args=(samples, sr),
                daemon=True,
                name="AudioPlayer",
            )
            self._thread.start()

    def _play_worker(self, samples: np.ndarray, sample_rate: int):
        """在后台线程里分块写入声卡，每块之间检查是否需要打断"""
        block = max(1, int(sample_rate * 0.05))  # 50ms 一块，打断响应足够快
        try:
            with sd.OutputStream(samplerate=sample_rate,
                                 channels=self.channels,
                                 dtype="float32",
                                 device=self.device) as stream:
                for i in range(0, len(samples), block):
                    if self._stop_event.is_set():
                        return
                    chunk = samples[i:i + block]
                    stream.write(chunk.reshape(-1, self.channels))
        except Exception as e:  # 声卡被占用 / 设备不支持
            self.last_error = f"播放失败: {e}"
        finally:
            self._playing.clear()

    def stop(self):
        """立即打断当前播放"""
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        self._playing.clear()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待播放结束；返回 True 表示已播完，False 表示超时/被打断"""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()


# ============================================================
#  录音机
# ============================================================
class AudioRecorder:
    """麦克风常开采集，音频写入环形缓冲

    - 单声道 int16 -> float32 归一化到 [-1, 1]
    - mute() 用于 TTS 播放期间屏蔽自己的声音（防自激）
    """

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 device: Optional[int] = None,
                 buffer_seconds: float = 12.0,
                 block_ms: int = 50):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.block_size = int(sample_rate * block_ms / 1000)
        self.buffer = RingBuffer(int(sample_rate * buffer_seconds))

        self._stream = None
        self._running = False
        self._muted = threading.Event()
        self.level = 0.0                  # 最近一块的 RMS 音量，供 UI 显示
        self.last_error: Optional[str] = None

    @property
    def available(self) -> bool:
        return HAS_SOUNDDEVICE

    def _callback(self, indata, frames, time_info, status):  # noqa: ARG002
        if self._muted.is_set():
            return
        # indata: (frames, channels) int16
        mono = indata[:, 0].astype(np.float32) / 32768.0
        self.buffer.write(mono)
        # 计算 RMS 供音量指示
        if mono.size:
            self.level = float(np.sqrt(np.mean(mono ** 2)) + 1e-9)

    def start(self) -> bool:
        """打开麦克风；失败返回 False 并记录原因（不抛异常）

        若显式指定了 device 但打不开（设备被拔掉 / 编号变了），
        自动回退到系统默认设备，避免因为一个过期编号就完全没声音。
        """
        if not HAS_SOUNDDEVICE:
            self.last_error = "sounddevice 未安装，无法录音"
            return False
        if self._running:
            return True

        candidates = [self.device]
        if self.device is not None:
            candidates.append(None)      # 回退：系统默认

        for dev in candidates:
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    blocksize=self.block_size,
                    device=dev,
                    callback=self._callback,
                )
                self._stream.start()
                self._running = True
                if dev != self.device:
                    print(f"[音频] 指定设备 {self.device} 打开失败，已回退到系统默认设备")
                    self.device = None
                return True
            except Exception as e:
                self.last_error = f"麦克风打开失败(device={dev}): {e}"
                self._stream = None
                self._running = False
        return False

    def stop(self):
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def mute(self):
        """屏蔽输入（TTS 播放时调用）"""
        self._muted.set()

    def unmute(self):
        self._muted.clear()
        # 清掉静音期间的残留，避免把旧音频当新说话
        self.buffer.clear()

    @property
    def muted(self) -> bool:
        return self._muted.is_set()

    def read_recent(self, seconds: float) -> np.ndarray:
        return self.buffer.read_last(int(self.sample_rate * seconds))

    def clear(self):
        self.buffer.clear()


class PlaybackBuffer:
    """流式播放的字节缓冲：生产者不断写入编码音频，消费者逐块解码

    用于"边合成边播放"——edge-tts 是分块返回 MP3 的，
    不必等整句合成完才开始出声，首字延迟能降低数倍。
    """

    def __init__(self):
        self._buf = bytearray()
        self._eof = False
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

    def write(self, data: bytes):
        with self._cond:
            self._buf.extend(data)
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self._eof = True
            self._cond.notify_all()

    def read(self, size: int = -1) -> bytes:
        """阻塞读取；EOF 且无数据时返回 b''"""
        with self._cond:
            while not self._buf and not self._eof:
                self._cond.wait(timeout=0.2)
            if not self._buf:
                return b""
            if size is None or size < 0:
                out = bytes(self._buf)
                self._buf.clear()
                return out
            out = bytes(self._buf[:size])
            del self._buf[:size]
            return out

    @property
    def eof(self) -> bool:
        with self._lock:
            return self._eof and not self._buf


class StreamingPlayer:
    """流式播放器：一边收编码音频一边解码出声

    与 AudioPlayer 的区别：
    - AudioPlayer  需要拿到完整样本再播（简单，但要等合成完）
    - StreamingPlayer 拿到第一块就开始播（首字延迟低）

    实现方式：生产者线程写入 MP3 字节，本类的工作线程用 PyAV 增量解码，
    每解出一帧就写声卡。打断通过 stop_event，块间检查，响应很快。
    """

    def __init__(self, device: Optional[int] = None):
        self.device = device
        self._buffer = PlaybackBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._playing = threading.Event()
        self._lock = threading.Lock()
        self.last_error: Optional[str] = None
        self.frames_played = 0

    @property
    def available(self) -> bool:
        return HAS_SOUNDDEVICE

    @property
    def playing(self) -> bool:
        return self._playing.is_set()

    def start(self):
        """开始一次流式播放会话；之后用 feed() 投喂音频数据"""
        self.stop()
        with self._lock:
            self._buffer = PlaybackBuffer()
            self._stop.clear()
            self._playing.set()
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="StreamingPlayer")
            self._thread.start()

    def feed(self, data: bytes):
        """投喂一块编码音频（MP3 等）"""
        self._buffer.write(data)

    def end(self):
        """声明音频已全部投喂完毕"""
        self._buffer.close()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """等待播放结束"""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def stop(self):
        """立即打断"""
        self._stop.set()
        try:
            self._buffer.close()
        except Exception:
            pass
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)
        self._thread = None
        self._playing.clear()

    def _run(self):
        """工作线程：增量解码 + 写声卡"""
        try:
            import av
        except ImportError:
            self.last_error = "缺少 PyAV，无法流式解码"
            self._playing.clear()
            return

        try:
            with av.open(self._buffer, mode="r", format="mp3") as container:
                stream = container.streams.audio[0]
                sample_rate = stream.sample_rate
                channels = stream.channels or 1

                with sd.OutputStream(samplerate=sample_rate,
                                     channels=min(channels, 2),
                                     dtype="float32",
                                     device=self.device) as out:
                    for frame in container.decode(stream):
                        if self._stop.is_set():
                            return
                        arr = frame.to_ndarray()
                        # 统一成 (frames, channels) float32
                        if arr.ndim == 1:
                            arr = arr.reshape(-1, 1)
                        else:
                            arr = arr.T if arr.shape[0] <= 2 else arr
                        data = arr.astype(np.float32)
                        if data.shape[1] > 1:
                            data = data[:, :1]     # 单声道输出
                        peak = float(np.max(np.abs(data))) if data.size else 0.0
                        if peak > 1.5:
                            data = data / 32768.0
                        np.clip(data, -1.0, 1.0, out=data)
                        out.write(data)
                        self.frames_played += 1
        except Exception as e:
            # 打断时 PyAV 会因为流被关闭而报错，这里不当成失败
            if not self._stop.is_set():
                self.last_error = f"流式播放失败: {e}"
        finally:
            self._playing.clear()


# ============================================================
#  语音端点检测（短时能量 + 双阈值）
# ============================================================
class EnergyVAD:
    """基于短时能量的 VAD

    机器人跑在本地电脑上，环境噪声相对稳定，短时能量法足够可靠，
    且不需要额外模型、不占 GPU、零下载。用双阈值 + 最短静音时长
    来实现"说完一句才收口"的效果。
    """

    def __init__(self, sample_rate: int = 16000,
                 frame_ms: int = 30,
                 start_threshold: float = 0.018,
                 end_threshold: float = 0.010,
                 min_speech_ms: int = 300,
                 min_silence_ms: int = 700,
                 max_speech_ms: int = 15000,
                 prebuffer_ms: int = 240,
                 soft_ratio: float = 0.8):
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.min_silence_frames = max(1, min_silence_ms // frame_ms)
        self.max_speech_frames = max(1, max_speech_ms // frame_ms)
        # 「软」起始阈值：语音起头（第一个字）音量往往低于稳定说话时，
        # 若只用 start_threshold 判定，第一个字会被整段丢掉。
        self.soft_threshold = max(start_threshold * soft_ratio, end_threshold)

        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._utterance: list[np.ndarray] = []
        # 说话起点前的音频缓冲。
        # 旧值固定 8 帧 = 仅 240ms，第一个字的前半截会被吃掉——
        # 实测就表现为「用户说的头几个字在识别结果里消失了」。
        prebuffer_frames = max(8, int(prebuffer_ms // frame_ms))
        self._pending: deque[np.ndarray] = deque(maxlen=prebuffer_frames)

    def reset(self):
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._utterance = []
        self._pending.clear()

    def flush(self, force: bool = True) -> Optional[np.ndarray]:
        """强制收口当前正在累积的语音段

        为什么需要它：VAD 靠"持续 N 毫秒静音"来判定一句话结束。
        但如果用户说完后**一直没再出声**，这个"静音"虽然实际存在，
        却只有在有音频帧进来时才会被计数——最后一段就会一直悬着不吐出来。

        VADSegmenter 在检测到"麦克风长时间没有新数据"时会调用本方法，
        把悬着的那段收口送出去。

        Args:
            force: True 时即使没达到 min_speech_frames 也返回
                  （因为此时已经是"用户说完并且安静了很久"）
        """
        if not self._in_speech or not self._utterance:
            self.reset()
            return None
        utterance = (np.concatenate(self._utterance)
                     if self._utterance else np.zeros(0, np.float32))
        speech_frames = self._speech_frames
        self.reset()
        if not force and speech_frames < self.min_speech_frames:
            return None
        return utterance

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame ** 2)))

    def push(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """喂入一段音频，返回检测到的完整一句话（float32），否则 None"""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        pos = 0
        results = None
        while pos + self.frame_size <= len(audio):
            frame = audio[pos:pos + self.frame_size]
            pos += self.frame_size
            finished = self._push_frame(frame)
            if finished is not None:
                results = finished  # 只取最后一句，防止积压
        return results

    def _push_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        rms = self._rms(frame)

        if not self._in_speech:
            # 保留起点前的音频，避免第一个字被切掉
            self._pending.append(frame)
            # 用「软阈值」触发，而不是 start_threshold：
            # 语音起头（第一个字）音量通常低于稳定说话，用硬阈值会让
            # 第一个字整段丢失。软阈值更早进入说话态，从而把 pending
            # 里积累的前置音频一起纳入这一句。
            if rms >= self.soft_threshold:
                self._in_speech = True
                self._silence_frames = 0
                # 从缓冲里找出"真正开始有声"的那一帧，只保留它前面
                # 一小段上下文（约 300ms）。
                # 这样既不丢第一个字的开头，也不会把大段静音塞进音频里
                # （静音太多会稀释语音、影响识别）。
                pending = list(self._pending)
                onset = 0
                for i in range(len(pending) - 1, -1, -1):
                    if self._rms(pending[i]) >= self.soft_threshold:
                        onset = i
                    else:
                        break
                keep_ctx = max(1, int(300 // self.frame_ms))
                start = max(0, onset - keep_ctx)
                self._utterance = pending[start:]
                self._speech_frames = len(self._utterance)
                self._pending.clear()
            return None

        # 说话中
        self._utterance.append(frame)
        if rms < self.end_threshold:
            self._silence_frames += 1
        else:
            self._silence_frames = 0
            self._speech_frames += 1

        too_short = self._speech_frames < self.min_speech_frames
        too_long = len(self._utterance) >= self.max_speech_frames
        end_of_speech = self._silence_frames >= self.min_silence_frames

        if end_of_speech or too_long:
            utterance = np.concatenate(self._utterance) if self._utterance else np.zeros(0, np.float32)
            self.reset()
            # 太短的多半是咳嗽/关门声，丢弃
            if too_short and not too_long:
                return None
            return utterance
        return None


class VADSegmenter:
    """把常开麦克风的音频流，切成一句一句的话音

    在独立线程里运行：定期从 RingBuffer 取新增音频 -> 喂给 VAD ->
    检测到整句话就回调出去。
    """

    def __init__(self, recorder: AudioRecorder, vad: EnergyVAD,
                 on_speech, poll_interval: float = 0.1,
                 skip_while_playing: bool = True,
                 idle_flush_seconds: float = 1.8):
        self.recorder = recorder
        self.vad = vad
        self.on_speech = on_speech
        self.poll_interval = poll_interval
        self.skip_while_playing = skip_while_playing
        # 麦克风安静这么久（无新数据）时，强制收口悬着的语音段。
        # 必须大于 VAD 的 min_silence，否则会和正常端点检测抢着收口。
        self.idle_flush_seconds = idle_flush_seconds

        self._thread: Optional[threading.Thread] = None
        self._running = False
        # 播放状态查询函数，由外部注入（避免本模块依赖 TTS）
        self.is_speaking = lambda: False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="VADSegmenter")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        consumed = self.recorder.buffer.total_written
        while self._running:
            try:
                # 机器人在说话时跳过，避免把 TTS 的声音识别成用户说话
                if self.skip_while_playing and self.is_speaking():
                    self.vad.reset()
                    self.recorder.buffer.clear()
                    consumed = self.recorder.buffer.total_written
                    time.sleep(self.poll_interval)
                    continue

                chunk, consumed = self.recorder.buffer.read_since(consumed)
                if chunk.size:
                    utterance = self.vad.push(chunk)
                    if utterance is not None and utterance.size > 0:
                        self.on_speech(utterance)
                else:
                    # 没有新音频。如果安静得够久，把悬着的语音段收口送出去。
                    # 否则用户说完后一直不出声时，最后一段会永远卡在 VAD 里。
                    last_write = self.recorder.buffer.last_write_time
                    if (last_write
                            and time.time() - last_write > self.idle_flush_seconds):
                        pending = self.vad.flush()
                        if pending is not None and pending.size > 0:
                            self.on_speech(pending)
            except Exception as e:
                print(f"[VAD] 异常: {e}")
            time.sleep(self.poll_interval)


# ============================================================
#  音频文件读写
# ============================================================
def load_audio(path, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    """读取音频文件为 float32 单声道；优先用 soundfile，失败再用 torchaudio"""
    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    except Exception:
        pass
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).numpy().astype(np.float32), sr
    except Exception as e:
        raise RuntimeError(f"无法读取音频文件 {path}: {e}")


def decode_audio_bytes(data: bytes, hint: str = "mp3") -> tuple[np.ndarray, int]:
    """把编码音频字节（mp3/wav/ogg 等）解码成 float32 单声道样本

    为什么需要这个：
    edge-tts 的 stream() 返回的是 **MP3** 字节流（帧头 fff3...），
    不是裸 PCM。早期版本直接 np.frombuffer(int16) 会把它当成
    采样点解释，导致时长只有实际的几分之一（听起来像被"掐头去尾"）。
    这里统一走真正的解码器。

    Returns: (float32 样本, 采样率)
    """
    # 1) soundfile（能直接吃 mp3 时最轻量）
    try:
        import io
        import soundfile as sf
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.size:
            return samples.astype(np.float32), int(sr)
    except Exception:
        pass

    # 2) torchaudio 兜底
    try:
        import io
        import torchaudio
        waveform, sr = torchaudio.load(io.BytesIO(data))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0).numpy().astype(np.float32), int(sr)
    except Exception:
        pass

    # 3) PyAV 最后兜底
    try:
        import io
        import av
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.audio[0]
            chunks = [frame.to_ndarray() for frame in container.decode(audio=0)]
            sr = stream.sample_rate
        if not chunks:
            raise RuntimeError("没有解出音频帧")
        mono = np.concatenate([c if c.ndim == 1 else c.mean(axis=0) for c in chunks])
        return mono.astype(np.float32), int(sr)
    except Exception as e:
        raise RuntimeError(f"音频解码失败（{hint}）: {e}")


def save_wav(path, samples: np.ndarray, sample_rate: int = 16000):
    """保存 float32 音频为 wav"""
    import soundfile as sf
    samples = np.asarray(samples, dtype=np.float32)
    sf.write(str(path), samples, sample_rate)


def list_audio_devices() -> str:
    """列出音频设备（供 check_env / 调试使用）"""
    if not HAS_SOUNDDEVICE:
        return "sounddevice 未安装"
    try:
        lines = []
        for i, d in enumerate(sd.query_devices()):
            lines.append(f"  [{i}] {d['name']}  in={d['max_input_channels']} out={d['max_output_channels']}")
        return "\n".join(lines)
    except Exception as e:
        return f"查询失败: {e}"
