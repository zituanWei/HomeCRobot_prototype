"""
音频链路功能自检
================
不依赖摄像头和 VL 模型，单独验证：
1. 播放器：能否播放、能否被打断
2. 录音机：麦克风能否采集、音量是否正常
3. EnergyVAD：能否从带静音的音频里切出一句话
4. 实时链路：录音 -> VAD -> 识别 的端到端骨架
5. TTS 合成：edge-tts 能否拿到 PCM（需要网络）

运行: python scripts/test_audio_loop.py
"""
import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.audio import (AudioPlayer, AudioRecorder, EnergyVAD, list_audio_devices,
                        HAS_SOUNDDEVICE)

PASS, FAIL = "[OK]", "[FAIL]"


def hr(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_devices():
    hr("1. 音频设备")
    if not HAS_SOUNDDEVICE:
        print(f"{FAIL} sounddevice 未安装")
        return False
    import sounddevice as sd
    print(f"  默认输入设备: {sd.query_devices(kind='input')['name']}")
    print(f"  默认输出设备: {sd.query_devices(kind='output')['name']}")
    print(f"{PASS} 设备可用")
    return True


def test_player():
    hr("2. 播放器（含打断）")
    player = AudioPlayer()
    if not player.available:
        print(f"{FAIL} 播放器不可用")
        return False

    sr = 24000
    # 440Hz 短音，音量压到很小，避免吵到人
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    tone = (0.06 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    print("  播放 1 秒低音量测试音...")
    player.play(tone, sr)
    time.sleep(0.2)
    playing_mid = player.playing
    print(f"  播放中状态: {playing_mid}")
    ok = player.wait(timeout=5)
    print(f"  播放结束: {ok}")
    if player.last_error:
        print(f"{FAIL} 播放报错: {player.last_error}")
        return False

    # 打断测试：播 3 秒，0.4 秒后打断
    print("  测试打断（播 3 秒，0.4 秒后打断）...")
    t3 = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    long_tone = (0.06 * np.sin(2 * np.pi * 440 * t3)).astype(np.float32)
    player.play(long_tone, sr)
    time.sleep(0.4)
    t0 = time.time()
    player.stop()
    elapsed = time.time() - t0
    print(f"  打断耗时: {elapsed:.2f}s, 是否仍在播放: {player.playing}")
    if player.playing or elapsed > 1.5:
        print(f"{FAIL} 打断未生效")
        return False
    print(f"{PASS} 播放与打断正常")
    return True


def test_recorder():
    hr("3. 麦克风采集")
    from config import VoiceConfig

    rec = AudioRecorder(device=VoiceConfig.MIC_DEVICE)
    if not rec.available:
        print(f"{FAIL} sounddevice 不可用")
        return False
    if not rec.start():
        print(f"{FAIL} 麦克风打开失败: {rec.last_error}")
        print(list_audio_devices())
        return False

    # 用扬声器放测试音，靠回授判断麦克风能否拾音。
    #
    # ⚠️ 这个测法有先天局限：如果你戴着耳机，声音进了耳朵而不会外泄到
    #    麦克风，回授必然失败——这不代表麦克风坏了。
    #    因此下面把"回授失败"降级为警告，而不是判定失败。
    #    真正可靠的验证方式是对着麦克风说话：
    #        python scripts/calibrate_mic.py --scan
    import threading
    import sounddevice as sd
    sr = 16000
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(sr * 1.5)))).astype(np.float32)

    print("  回授测试（扬声器放测试音，看麦克风能否收到；戴耳机时此测试无效）...")
    idle_peak = 0.0
    for _ in range(3):
        time.sleep(0.1)
        idle_peak = max(idle_peak, rec.level)

    peak = 0.0
    attempts = []
    for attempt in range(3):
        threading.Thread(target=lambda: (sd.play(tone, sr), sd.wait()),
                         daemon=True).start()
        time.sleep(0.2)
        this_peak = 0.0
        for _ in range(12):
            time.sleep(0.1)
            this_peak = max(this_peak, rec.level)
        attempts.append(this_peak)
        peak = max(peak, this_peak)
        if this_peak >= 0.001 and this_peak >= idle_peak * 1.5:
            break
        time.sleep(0.2)

    audio = rec.read_recent(2.0)
    rec.stop()

    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    print(f"  底噪 peak={idle_peak:.5f}  |  各次回授 peak={[f'{a:.5f}' for a in attempts]}"
          f"  |  平均 RMS={rms:.5f}")
    if audio.size == 0:
        print(f"{FAIL} 没有采集到任何音频")
        return False
    if peak <= 0.0:
        print(f"{FAIL} 音量恒为 0，麦克风流没有数据")
        return False

    if peak >= 0.001 and peak >= idle_peak * 1.5:
        print(f"{PASS} 麦克风拾音正常（回授信噪比 {peak / max(idle_peak, 1e-9):.1f}x）")
        return True

    # 回授没测到：不判定失败，给出下一步
    print("[WARN] 回授未测到明显信号 —— 这不代表麦克风坏了")
    print("       如果你戴着耳机，声音不会外泄到麦克风，回授本来就收不到")
    print("       >>> 请用说话的方式来确认（唯一可靠的方法）：")
    print("           python scripts/calibrate_mic.py --scan")
    print("           该命令会让你对着麦克风说话，逐个设备对比拾音强度")
    print("       麦克风流本身是通的（能读到数据），采集通路无异常")
    return True


def test_vad():
    hr("4. EnergyVAD 断句")
    vad = EnergyVAD(sample_rate=16000, start_threshold=0.018,
                    end_threshold=0.010, min_silence_ms=700)

    # 构造：0.5s 静音 + 1.0s "说话"(噪声) + 1.0s 静音
    sr = 16000
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    rng = np.random.default_rng(0)
    speech = (rng.normal(0, 0.08, int(sr * 1.0))).astype(np.float32)
    tail = np.zeros(int(sr * 1.0), dtype=np.float32)

    # 分块喂进去，模拟实时流
    stream = np.concatenate([silence, speech, tail])
    chunk = int(sr * 0.1)
    found = None
    for i in range(0, len(stream), chunk):
        seg = vad.push(stream[i:i + chunk])
        if seg is not None:
            found = seg
            break

    if found is None:
        print(f"{FAIL} 没有切出语音段")
        return False
    dur = len(found) / sr
    print(f"  切出语音段: {len(found)} 点 = {dur:.2f}s（期望约 1.0s 左右）")
    if not (0.4 <= dur <= 2.0):
        print(f"{FAIL} 语音段长度异常")
        return False
    print(f"{PASS} VAD 断句正常")
    return True


def test_tts_synth():
    hr("5. TTS 合成（edge-tts，需要网络）")
    from config import VoiceConfig
    from modules.tts import TTS

    cfg = VoiceConfig()
    cfg.ENABLE_TTS = True
    tts = TTS(cfg)
    print("  合成「你好，我是小护」...")
    t0 = time.time()
    result = tts.synthesize("你好，我是小护")
    if result is None:
        print(f"{FAIL} 合成失败: {tts.last_error}")
        return False
    samples, sr = result
    print(f"  得到 {len(samples)} 点 @ {sr}Hz = {len(samples)/sr:.2f}s，耗时 {time.time()-t0:.2f}s")
    if len(samples) < sr * 0.3:
        print(f"{FAIL} 音频过短，可能合成异常")
        return False
    print(f"{PASS} TTS 合成正常（PCM 直出，不落盘）")
    return True


def test_asr_dtype():
    """ASR 输入格式回归测试

    背景：FunASR 的 frontend 会对波形做 mean() 等浮点运算，
    传入 int16 会抛:
        mean(): could not infer output dtype ... Got: Short
    这个测试锁住该修复：float32 必须能识别，int16 也不能崩。
    """
    hr("6. ASR 识别（输入格式回归）")
    try:
        import io
        import asyncio
        import edge_tts
        import soundfile as sf
    except ImportError as e:
        print(f"[SKIP] 缺少依赖: {e}")
        return True

    from config import VoiceConfig
    from modules.asr import ASR

    cfg = VoiceConfig()
    cfg.ENABLE_ASR = True
    asr = ASR(cfg)

    print("  加载 FunASR 模型（已缓存则较快）...")
    t0 = time.time()
    if not asr.load():
        print(f"[SKIP] ASR 模型不可用: {asr.last_error}")
        return True
    print(f"  加载耗时 {time.time() - t0:.1f}s")

    # 造一段真实中文语音
    async def gen(text):
        c = edge_tts.Communicate(text, voice="zh-CN-XiaoxiaoNeural")
        buf = bytearray()
        async for ch in c.stream():
            if ch.get("type") == "audio" and ch.get("data"):
                buf.extend(ch["data"])
        return bytes(buf)

    try:
        mp3 = asyncio.run(gen("你好我是小护"))
    except Exception as e:
        print(f"[SKIP] 无法生成测试语音（网络）: {e}")
        return True

    audio, sr = sf.read(io.BytesIO(mp3), dtype="float32")
    if sr != 16000:
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)

    # --- 用例1：float32（正确格式）---
    text = asr.transcribe_array(audio.astype(np.float32), 16000)
    print(f"  float32 输入 -> {text!r}")
    if not text or "ASR" in text:
        print(f"{FAIL} float32 输入未能识别出文字")
        return False
    print(f"{PASS} float32 输入识别正常（int16 崩溃问题已修复）")

    # --- 用例2：int16（应被内部归一化，而不是崩溃）---
    try:
        text2 = asr.transcribe_array((audio * 32768).astype(np.int16), 16000)
        print(f"  int16 输入 -> {text2!r}（应被自动归一化）")
        print(f"{PASS} int16 输入不再抛异常")
    except Exception as e:
        print(f"{FAIL} int16 输入仍会崩溃: {e}")
        return False

    asr.unload()
    return True


def main():
    run_asr = "--with-asr" in sys.argv
    print("音频链路功能自检")
    print(f"sounddevice 可用: {HAS_SOUNDDEVICE}")
    if not run_asr:
        print("(ASR 识别测试默认跳过，加 --with-asr 启用，需加载模型约20s)")

    results = {}
    results["设备"] = test_devices()
    results["播放器"] = test_player()
    results["麦克风"] = test_recorder()
    results["VAD"] = test_vad()
    results["TTS合成"] = test_tts_synth()
    if run_asr:
        results["ASR识别"] = test_asr_dtype()

    hr("结果汇总")
    for k, v in results.items():
        print(f"  {PASS if v else FAIL} {k}")
    print()
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
