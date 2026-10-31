"""
语音合成模块 (TTS - Text To Speech)
====================================
引擎（按延迟从低到高）：

| 引擎 | 延迟 | 音质 | 依赖 |
|---|---|---|---|
| `sapi`     | **~0.2s**（几乎瞬时） | 机械 | 无（Windows 自带） |
| `edge-tts` | 首字 0.5~2s（流式）/ 2~13s（整句） | 最自然 | 网络 |
| `cosyvoice`| 未实现 | — | 需部署 |

关键设计：
1. **流式播放**：edge-tts 是分块返回 MP3 的，本模块边收边解码边播，
   不必等整句合成完才出声——首字延迟从"十几秒"降到 1 秒级。
2. **本地引擎兜底**：`sapi` 引擎走 Windows 自带语音，完全离线、几乎零延迟，
   代价是音质机械。适合"响应速度优先"的场景。
3. **合成缓存**：同一句话（问候语、固定提示）不重复合成。
4. 不再写临时 mp3、不再用 os.startfile（那会弹窗且无法打断）。

启用方式：config.py 中 VoiceConfig.ENABLE_TTS = True
"""
from __future__ import annotations

import asyncio
import base64
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import VoiceConfig, DATA_DIR
from core.audio import (AudioPlayer, StreamingPlayer, decode_audio_bytes, save_wav)

# edge-tts 输出 MP3，解码后为 24kHz 单声道
EDGE_TTS_SAMPLE_RATE = 24000


class TTS:
    """语音合成 + 播放封装（支持流式与本地引擎）"""

    def __init__(self, config: VoiceConfig = None,
                 on_start=None, on_end=None):
        """
        Args:
            config: 语音配置
            on_start: 开始播报时的回调（主程序用来静音麦克风）
            on_end:   播报结束时的回调（主程序用来恢复麦克风）
        """
        self.config = config or VoiceConfig()
        self.on_start = on_start
        self.on_end = on_end

        self.player = AudioPlayer(sample_rate=EDGE_TTS_SAMPLE_RATE)
        self.stream_player = StreamingPlayer()
        self._speaking = threading.Event()
        self._lock = threading.Lock()       # 串行化合成调用
        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        self.cache_hits = 0
        self.last_error: Optional[str] = None

        # Piper 本地引擎（懒加载，模型只读一次）
        self._piper_voice = None
        self._piper_lock = threading.Lock()

    # ------------------------------------------------------------------
    #  状态
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.config.ENABLE_TTS)

    @property
    def speaking(self) -> bool:
        """当前是否正在播报（供 VAD 判断要不要屏蔽输入）"""
        return (self._speaking.is_set()
                or self.player.playing
                or self.stream_player.playing)

    @property
    def engine(self) -> str:
        return (self.config.TTS_ENGINE or "edge-tts").lower()

    # ------------------------------------------------------------------
    #  合成（返回完整样本，用于缓存与非流式路径）
    # ------------------------------------------------------------------
    def synthesize(self, text: str,
                   engine: Optional[str] = None) -> Optional[tuple[np.ndarray, int]]:
        """把文字合成为音频样本（带缓存）

        Args:
            text: 待合成文字
            engine: 指定引擎；None 表示用配置里的引擎
                    （降级时显式传入，避免缓存键串味）
        Returns: (float32 单声道样本, 采样率) 或 None（失败/未启用）
        """
        if not text or not text.strip():
            return None

        text = text.strip()
        eng = (engine or self.engine).lower()
        cache_key = f"{eng}|{self.config.TTS_VOICE}|{text}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        if eng == "sapi":
            result = self._synthesize_sapi(text)
        elif eng == "piper":
            result = self._synthesize_piper(text)
        elif eng == "cosyvoice":
            result = self._synthesize_cosyvoice(text)
        else:
            result = self._synthesize_edge(text)

        if result is not None:
            if len(self._cache) >= self.config.TTS_CACHE_SIZE:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = result
        return result

    def clear_cache(self):
        self._cache.clear()

    # ---------------- edge-tts（在线，音质好）----------------
    async def _edge_collect(self, text: str) -> bytes:
        """收集 edge-tts 的 MP3 字节

        注意：edge_tts.Communicate 没有 close()/__aexit__，
        它靠 async generator 的关闭来做 aiohttp 清理。
        因此必须用 aclosing 显式关闭，否则会抛出
        "Unclosed client session" / "Event loop is closed"。
        """
        from contextlib import aclosing
        import edge_tts
        communicate = edge_tts.Communicate(
            text,
            voice=self.config.TTS_VOICE,
            rate=self.config.TTS_RATE,
            volume=self.config.TTS_VOLUME,
        )
        payload = bytearray()
        async with aclosing(communicate.stream()) as stream:
            async for chunk in stream:
                if chunk.get("type") == "audio" and chunk.get("data"):
                    payload.extend(chunk["data"])
        return bytes(payload)

    def _synthesize_edge(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """edge-tts 整句合成（注意：返回的是 MP3，必须解码）"""
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            self.last_error = "未安装 edge-tts，请运行: pip install edge-tts"
            print(f"[TTS] {self.last_error}")
            return None

        try:
            with self._lock:
                encoded = self._run_async(self._edge_collect(text))
        except Exception as e:
            self.last_error = f"合成失败: {e}"
            print(f"[TTS] {self.last_error}")
            return None

        if not encoded:
            self.last_error = "合成结果为空（网络问题或音色名错误）"
            print(f"[TTS] {self.last_error}")
            return None

        try:
            samples, sr = decode_audio_bytes(encoded, hint="edge-tts mp3")
        except Exception as e:
            self.last_error = str(e)
            print(f"[TTS] {self.last_error}")
            return None

        if samples.size == 0 or samples.size < sr * 0.1:
            self.last_error = "解码后音频异常短"
            print(f"[TTS] {self.last_error}")
            return None
        return samples, sr

    @staticmethod
    def _run_async(coro):
        """在无事件循环的线程里安全地跑协程"""
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False
        if not running:
            return asyncio.run(coro)
        # 已在事件循环中（理论上不会发生），用新线程规避
        box: dict = {}

        def _run():
            box["r"] = asyncio.run(coro)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join()
        return box.get("r")

    # ---------------- Windows SAPI（本地，低延迟）----------------
    def _synthesize_sapi(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """用 Windows 自带 SAPI 合成（完全离线，延迟极低）

        原理：调 PowerShell 的 System.Speech 合成到内存流，
        导出 WAV 字节后用同一套解码器转成 float32 样本。
        """
        safe = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {int(self.config.SAPI_RATE)}; "
            f"$s.Volume = {int(self.config.SAPI_VOLUME)}; "
            "$ms = New-Object System.IO.MemoryStream; "
            "$s.SetOutputToWaveStream($ms); "
            f"$s.Speak('{safe}'); "
            "$s.SetOutputToNull(); $s.Dispose(); "
            "[Convert]::ToBase64String($ms.ToArray())"
        )
        try:
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, timeout=60,
            )
        except Exception as e:
            self.last_error = f"SAPI 调用失败: {e}"
            print(f"[TTS] {self.last_error}")
            return None

        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", "ignore").strip()[:200]
            self.last_error = f"SAPI 返回错误: {err}"
            print(f"[TTS] {self.last_error}")
            return None

        out = proc.stdout.decode("ascii", "ignore").strip()
        if not out:
            self.last_error = "SAPI 未返回音频数据"
            print(f"[TTS] {self.last_error}")
            return None

        try:
            wav = base64.b64decode(out)
            samples, sr = decode_audio_bytes(wav, hint="sapi wav")
        except Exception as e:
            self.last_error = f"SAPI 音频解码失败: {e}"
            print(f"[TTS] {self.last_error}")
            return None

        if samples.size == 0:
            self.last_error = "SAPI 音频为空"
            return None
        return samples, sr

    # ---------------- Piper（本地神经 TTS，推荐）----------------
    def _ensure_piper_voice(self):
        """懒加载 Piper 语音模型（只加载一次）

        实测（zh_CN-huayan-medium，CPU）：
            模型加载 2.25s（一次性）
            冷启动合成 0.27s / 热启动 0.068s
            RTF 0.057 —— 比实时快约 17 倍
        对比 edge-tts 的 2.5~13s，延迟低两个数量级，且完全离线。
        """
        if self._piper_voice is not None:
            return self._piper_voice
        with self._piper_lock:
            if self._piper_voice is not None:
                return self._piper_voice
            try:
                from piper import PiperVoice
            except ImportError:
                self.last_error = "未安装 piper-tts，请运行: pip install piper-tts"
                print(f"[TTS] {self.last_error}")
                return None

            from config import PROJECT_ROOT
            model_path = Path(self.config.PIPER_MODEL)
            if not model_path.is_absolute():
                model_path = PROJECT_ROOT / model_path
            if not model_path.exists():
                self.last_error = (
                    f"Piper 模型不存在: {model_path}\n"
                    f"      请运行: python scripts/download_piper.py"
                )
                print(f"[TTS] {self.last_error}")
                return None

            print(f"[TTS] 正在加载 Piper 模型 {model_path.name} ...")
            t0 = time.time()
            try:
                self._piper_voice = PiperVoice.load(
                    str(model_path),
                    use_cuda=bool(self.config.PIPER_USE_CUDA),
                )
            except Exception as e:
                self.last_error = f"Piper 模型加载失败: {e}"
                print(f"[TTS] {self.last_error}")
                return None
            print(f"[TTS] Piper 就绪（{time.time() - t0:.2f}s）")
            return self._piper_voice

    def _synthesize_piper(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """用 Piper 本地合成（无需联网）"""
        voice = self._ensure_piper_voice()
        if voice is None:
            return None

        try:
            from piper import SynthesisConfig
        except ImportError:
            SynthesisConfig = None

        syn_config = None
        if SynthesisConfig is not None:
            try:
                syn_config = SynthesisConfig(
                    length_scale=self.config.PIPER_LENGTH_SCALE,
                    noise_scale=self.config.PIPER_NOISE_SCALE,
                    noise_w_scale=self.config.PIPER_NOISE_W,
                )
            except Exception:
                syn_config = None

        try:
            chunks = []
            sample_rate = 22050
            for chunk in voice.synthesize(text, syn_config=syn_config):
                sample_rate = getattr(chunk, "sample_rate", sample_rate)
                audio = getattr(chunk, "audio_float_array", None)
                if audio is None:
                    raw = getattr(chunk, "audio_int16_bytes", None)
                    if raw:
                        audio = (np.frombuffer(raw, dtype=np.int16)
                                 .astype(np.float32) / 32768.0)
                if audio is not None and len(audio):
                    chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        except Exception as e:
            self.last_error = f"Piper 合成失败: {e}"
            print(f"[TTS] {self.last_error}")
            return None

        if not chunks:
            self.last_error = "Piper 没有产出音频"
            print(f"[TTS] {self.last_error}")
            return None

        samples = np.concatenate(chunks)
        return samples, int(sample_rate)

    def _synthesize_cosyvoice(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """CosyVoice 本地合成（未实现，回退 edge-tts）"""
        print("[TTS] CosyVoice 尚未实现，回退 edge-tts")
        return self._synthesize_edge(text)

    # ------------------------------------------------------------------
    #  播放
    # ------------------------------------------------------------------
    def speak(self, text: str, wait: bool = True) -> bool:
        """合成并播报（带流式优化 + 慢网络自动降级）

        Args:
            text: 要说的内容
            wait: True 阻塞到播完；False 立即返回（后台播）
        Returns:
            是否成功开始播报
        """
        if not self.enabled or not text or not text.strip():
            return False
        if not (self.player.available and self.stream_player.available):
            self.last_error = "sounddevice 未安装，无法播放"
            print(f"[TTS] {self.last_error}")
            return False

        text = text.strip()

        # 已缓存（例如重复的问候语）：直接播，零合成延迟
        cache_key = f"{self.engine}|{self.config.TTS_VOICE}|{text}"
        if cache_key in self._cache:
            self.cache_hits += 1
            samples, sr = self._cache[cache_key]
            return self._play_samples(samples, sr, wait)

        # 本地引擎（Piper / SAPI）本身就很快，直接走常规路径，
        # 不需要流式播放和在线降级那一套。
        if self.engine in ("sapi", "piper"):
            result = self.synthesize(text)
            if result is None:
                # Piper 模型缺失等情况下，退到 SAPI 保底出声
                return self._try_local_fallback(
                    text, wait, reason=f"{self.engine} 不可用")
            return self._play_samples(result[0], result[1], wait)

        # ---- edge-tts：先流式、再整句、最后本地，逐级降级 ----
        stream_timeout = self.config.TTS_FALLBACK_AFTER or 99.0
        synth_wait = self.config.TTS_SYNTH_WAIT or 0.0

        if self.config.TTS_STREAMING:
            try:
                started = self._speak_streaming_edge(
                    text, wait, first_timeout=stream_timeout,
                )
                if started:
                    return True
            except Exception as e:
                print(f"[TTS] 流式播报异常: {e}")

        # 阶段2：整句合成，但最多等 synth_wait 秒
        # 说明：合成是在工作线程里跑的。超时后那个线程可能仍在跑完，
        # 因此降级前必须先 interrupt() 掐掉它，否则会双重出声。
        result: dict = {}

        def _worker():
            try:
                result["r"] = self.synthesize(text)
            except Exception as e:
                result["err"] = e

        th = threading.Thread(target=_worker, daemon=True, name="TTSSynth")
        th.start()
        th.join(timeout=synth_wait)

        if th.is_alive():
            # 在线合成太慢 -> 降级
            print(f"[TTS] 在线合成超过 {synth_wait:.1f}s 未完成")
            self.interrupt()                      # 掐掉可能已开始的播放
            return self._try_local_fallback(text, wait, reason="在线合成过慢")

        if result.get("err"):
            print(f"[TTS] 合成失败: {result['err']}")
            return self._try_local_fallback(text, wait, reason="合成报错")

        s = result.get("r")
        if s is None:
            return self._try_local_fallback(text, wait, reason="合成无结果")
        return self._play_samples(s[0], s[1], wait)

    def _try_local_fallback(self, text: str, wait: bool,
                            reason: str = "") -> bool:
        """在线引擎不可用时，改用本地 SAPI 引擎先出声

        为什么要这一步：edge-tts 是云服务，网络差时首块能拖到十几秒，
        而陪伴机器人的问候需要及时——先出声比音质更重要。
        """
        fb = self.config.TTS_FALLBACK_ENGINE
        if not fb or fb == self.engine:
            return False

        print(f"[TTS] {reason} -> 自动降级到本地引擎 {fb}")
        try:
            # 注意：缓存键里会带上 fb，下次同样的句子能直接命中本地缓存，
            # 不会因为引擎切换而互相污染。
            result = self.synthesize(text, engine=fb)
            if result is None:
                print(f"[TTS] 本地引擎 {fb} 也不可用")
                return False
            return self._play_samples(result[0], result[1], wait)
        except Exception as e:
            print(f"[TTS] 本地降级失败: {e}")
            return False

    def _play_samples(self, samples: np.ndarray, sr: int, wait: bool) -> bool:
        """播放已合成的样本"""
        self._fire_start()
        self.player.play(samples, sr)

        est = len(samples) / float(sr) + 5.0

        def _finish():
            self.player.wait(timeout=est)
            self._fire_end()

        if wait:
            _finish()
        else:
            threading.Thread(target=_finish, daemon=True, name="TTSWaiter").start()
        return True

    def _speak_streaming_edge(self, text: str, wait: bool,
                              first_timeout: Optional[float] = None,
                              on_timeout=None) -> bool:
        """流式播报：边收 edge-tts 的块边解码播放

        这是降低"文字已显示、声音迟迟不来"的手段之一。
        但要注意：edge-tts 的首块延迟取决于服务端合成速度，
        网络差时首块可能十几秒——所以支持 first_timeout 超时后
        交由调用方降级到本地引擎。
        """
        import edge_tts

        sp = self.stream_player
        sp.start()
        self._fire_start()

        timeout = first_timeout or self.config.TTS_STREAM_FIRST_TIMEOUT
        t0 = time.time()
        state = {"first": None, "error": None, "done": False, "timed_out": False}

        async def producer():
            """在协程内部自己控超时

            关键：不能从外部强行取消协程——那样 edge-tts 的 aiohttp 会话
            来不及清理，会抛 "Event loop is closed" / "Unclosed client session"。
            让协程自己到时退出，清理就能在事件循环还活着时完成。
            """
            from contextlib import aclosing
            deadline = time.time() + timeout
            started = False
            try:
                communicate = edge_tts.Communicate(
                    text,
                    voice=self.config.TTS_VOICE,
                    rate=self.config.TTS_RATE,
                    volume=self.config.TTS_VOLUME,
                )
                stream = communicate.stream()
                async with aclosing(stream) as s:
                    async for chunk in s:
                        if chunk.get("type") == "audio" and chunk.get("data"):
                            if not started:
                                started = True
                                state["first"] = time.time()
                            sp.feed(chunk["data"])
                        elif not started and time.time() > deadline:
                            # 超时前一块音频都没来 -> 放弃
                            state["timed_out"] = True
                            break
            except Exception as e:
                state["error"] = e
            finally:
                sp.end()
                state["done"] = True

        def _run_producer():
            try:
                asyncio.run(producer())
            except Exception as e:
                state["error"] = e
                try:
                    sp.end()
                except Exception:
                    pass

        threading.Thread(target=_run_producer, daemon=True,
                         name="TTSProducer").start()

        # 等第一块音频到达（生产者会自行在 timeout 后退出）
        while state["first"] is None and not state["done"]:
            if time.time() - t0 > timeout + 1.0:
                break
            time.sleep(0.05)

        if state["first"] is None:
            sp.stop()
            self._fire_end()
            if state["error"]:
                print(f"[TTS] 流式合成失败: {state['error']}")
            elif state.get("timed_out") or time.time() - t0 > timeout:
                print(f"[TTS] 流式合成超过 {timeout:.1f}s 未出第一块")
            if on_timeout:
                try:
                    on_timeout()
                except Exception as e:
                    print(f"[TTS] 降级回调异常: {e}")
            return False

        first_latency = state["first"] - t0
        print(f"[TTS] 流式播放已启动（首块 {first_latency:.2f}s）")

        def _finish():
            sp.wait(timeout=len(text) * 1.0 + 30.0)
            self._fire_end()

        if wait:
            _finish()
        else:
            threading.Thread(target=_finish, daemon=True,
                             name="TTSStreamWait").start()
        return True

    def _fire_start(self):
        self._speaking.set()
        if self.on_start:
            try:
                self.on_start()
            except Exception as e:
                print(f"[TTS] on_start 回调异常: {e}")

    def _fire_end(self):
        self._speaking.clear()
        if self.on_end:
            try:
                self.on_end()
            except Exception as e:
                print(f"[TTS] on_end 回调异常: {e}")

    def interrupt(self):
        """立即打断正在进行的播报（用户插话时调用）"""
        if self.speaking:
            try:
                self.stream_player.stop()
            except Exception:
                pass
            try:
                self.player.stop()
            except Exception:
                pass
            self._fire_end()
            print("[TTS] 已打断播报")

    def preload_phrases(self, phrases: list[str], background: bool = True,
                        delay: float = 0.0) -> threading.Thread | None:
        """预合成常用语，让高频回复"零延迟"出声

        为什么需要：edge-tts 是微软在线服务，实测合成要 1.6~13.9 秒。
        问候语、固定追问这类**会反复出现**的句子，提前合成好放进缓存，
        之后命中缓存就是 0 延迟——这是在不牺牲音质前提下唯一能压缩等待的手段。

        实测收益：同一句话 12.73s -> 0.000s。

        Args:
            phrases: 要预合成的句子列表
            background: True 则起后台线程（不阻塞启动）
            delay: 等主流程稳定多久再开始，避免和模型加载抢资源
        """
        # 去重并跳过已缓存的
        todo: list[str] = []
        seen = set()
        for p in phrases:
            p = (p or "").strip()
            if not p or p in seen:
                continue
            seen.add(p)
            key = f"{self.engine}|{self.config.TTS_VOICE}|{p}"
            if key not in self._cache:
                todo.append(p)

        if not todo:
            return None

        def _worker():
            if delay > 0:
                time.sleep(delay)
            if not self.enabled:
                return
            ok, fail = 0, 0
            print(f"[TTS] 开始后台预合成 {len(todo)} 句常用语...")
            for p in todo:
                try:
                    if self.synthesize(p) is not None:
                        ok += 1
                    else:
                        fail += 1
                except Exception:
                    fail += 1
            print(f"[TTS] 预合成完成：成功 {ok} 句"
                  + (f"，失败 {fail} 句" if fail else "")
                  + "（这些句子之后播报零延迟）")

        if background:
            th = threading.Thread(target=_worker, daemon=True, name="TTSPreload")
            th.start()
            return th
        _worker()
        return None

    def save_to_file(self, text: str, path=None) -> Optional[Path]:
        """合成并保存为 wav（调试/素材制作）"""
        result = self.synthesize(text)
        if result is None:
            return None
        samples, sr = result
        if path is None:
            from datetime import datetime
            path = DATA_DIR / f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        save_wav(path, samples, sr)
        return Path(path)

    def close(self):
        """释放播放资源"""
        try:
            self.stream_player.stop()
        except Exception:
            pass
        self.player.stop()
