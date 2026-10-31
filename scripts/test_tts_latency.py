"""
TTS 延迟对比测试
================
对比三种合成方案的"首字延迟"（用户感知的：文字已出现 -> 声音开始响）。

测法：记录 speak() 调用到声卡真正开始输出之间的时间。
      流式路径用 TTS 内部打印的"首块延迟"；整句路径用"合成耗时"近似。

运行: python scripts/test_tts_latency.py
     python scripts/test_tts_latency.py --no-play   # 不实际出声，只测合成
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SENTENCE = "你好，我是小护，今天天气不错，要不要一起聊聊天"


def hr(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


def test_engine(engine: str, streaming: bool, no_play: bool):
    from config import VoiceConfig
    from modules.tts import TTS

    cfg = VoiceConfig()
    cfg.ENABLE_TTS = True
    cfg.TTS_ENGINE = engine
    cfg.TTS_STREAMING = streaming

    print(f"\n--- 引擎={engine}  流式={streaming} ---")
    tts = TTS(cfg)

    # 模型加载是**一次性**开销，不能算进每句话的延迟里，
    # 否则 Piper 这种本地模型会被严重冤枉（要加载 60MB 的 onnx）。
    t0 = time.time()
    tts.synthesize("预热")
    first_call = time.time() - t0
    if engine == "piper":
        print(f"  首次调用(含模型加载): {first_call:.2f}s  <- 只发生一次")

    # 1) 合成耗时（模型已就绪）
    t0 = time.time()
    result = tts.synthesize(SENTENCE)
    synth_time = time.time() - t0
    if result is None:
        print(f"  [FAIL] 合成失败: {tts.last_error}")
        return None
    samples, sr = result
    dur = len(samples) / sr
    print(f"  合成: {synth_time:.3f}s  ->  音频 {dur:.2f}s @ {sr}Hz"
          f"  RTF={synth_time / dur:.3f}")

    # 2) 缓存命中
    t0 = time.time()
    tts.synthesize(SENTENCE)
    cache_time = time.time() - t0
    print(f"  缓存命中: {cache_time:.3f}s")

    # 3) 感知延迟（首字延迟）
    if no_play:
        perceived = synth_time
        print(f"  [no-play] 以合成耗时近似首字延迟: {perceived:.2f}s")
    elif streaming and engine == "edge-tts":
        # 流式：测首块到达时间
        t0 = time.time()
        ok = tts.speak(SENTENCE, wait=True)
        perceived = time.time() - t0      # 这里包含完整播放，另看打印的首块延迟
        print(f"  流式 speak 全程: {perceived:.2f}s（首块延迟见上方 [TTS] 打印）")
        # 单独测首块延迟
        tts2 = TTS(cfg)
        t0 = time.time()
        got = {"t": None}

        import asyncio
        async def first_chunk():
            import edge_tts
            c = edge_tts.Communicate(SENTENCE, voice=cfg.TTS_VOICE)
            async for ch in c.stream():
                if ch.get("type") == "audio" and ch.get("data"):
                    return time.time() - t0
            return None
        got["t"] = asyncio.run(first_chunk())
        perceived = got["t"] if got["t"] else synth_time
        print(f"  >>> 真实首块延迟: {perceived:.2f}s")
    else:
        t0 = time.time()
        tts.speak(SENTENCE, wait=True)
        perceived = time.time() - t0
        print(f"  整句 speak 全程: {perceived:.2f}s")

    return {
        "engine": f"{engine}{'(流式)' if streaming and engine=='edge-tts' else ''}",
        "合成": round(synth_time, 2),
        "缓存": round(cache_time, 3),
        "首字延迟": round(perceived, 2),
    }


def main():
    no_play = "--no-play" in sys.argv
    print("TTS 延迟对比")
    print(f"测试句: 「{SENTENCE}」")
    if no_play:
        print("模式: 不实际出声")

    rows = []
    # Piper 本地神经 TTS（新增，推荐）
    r = test_engine("piper", False, no_play)
    if r:
        r["engine"] = "piper(本地神经)"
        rows.append(r)

    # edge-tts 整句（旧行为）
    r = test_engine("edge-tts", False, no_play)
    if r:
        r["engine"] = "edge-tts(整句,旧)"
        rows.append(r)

    # edge-tts 流式（新）
    r = test_engine("edge-tts", True, no_play)
    if r:
        rows.append(r)

    # Windows SAPI 本地
    r = test_engine("sapi", False, no_play)
    if r:
        rows.append(r)

    hr("汇总")
    print(f"  {'方案':<20} {'合成耗时':>10} {'缓存命中':>10} {'首字延迟':>10}")
    print("  " + "-" * 54)
    for r in rows:
        print(f"  {r['engine']:<20} {r['合成']:>9.2f}s {r['缓存']:>9.3f}s {r['首字延迟']:>9.2f}s")
    print()
    if len(rows) >= 2:
        base = rows[0]["首字延迟"]
        best = min(r["首字延迟"] for r in rows)
        print(f"  最快方案首字延迟 {best:.2f}s，相对旧方案提速 {base/max(best,0.01):.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
