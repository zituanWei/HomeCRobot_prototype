"""
VAD 切分诊断工具
================
让你看清"系统到底把你的话切成了几段、每段多长、每段识别成了什么"。

为什么需要它：VAD 把一句话切成几段、分别识别，是"识别结果变成乱码"的
头号原因——每个碎片都会被模型硬猜成一段文字，而**开头几个字最容易丢**。

用法:
    python scripts/diagnose_vad.py                  # 录 8 秒，显示切分过程
    python scripts/diagnose_vad.py --seconds 12
    python scripts/diagnose_vad.py --file xxx.wav   # 分析已有音频
    python scripts/diagnose_vad.py --no-asr         # 只看切分，不做识别
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def hr(t):
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


def analyze_segments(audio: np.ndarray, sr: int, cfg, label: str):
    """用给定 VAD 参数切分音频，返回分段列表"""
    from core.audio import EnergyVAD
    vad = EnergyVAD(
        sample_rate=sr,
        start_threshold=cfg.VAD_START_THRESHOLD,
        end_threshold=cfg.VAD_END_THRESHOLD,
        min_silence_ms=cfg.VAD_MIN_SILENCE_MS,
        max_speech_ms=cfg.VAD_MAX_SPEECH_MS,
        prebuffer_ms=getattr(cfg, "VAD_PREBUFFER_MS", 600),
        soft_ratio=getattr(cfg, "VAD_SOFT_THRESHOLD_RATIO", 0.5),
    )
    segs = []
    chunk = int(sr * 0.1)
    for i in range(0, len(audio), chunk):
        r = vad.push(audio[i:i + chunk])
        if r is not None and r.size:
            segs.append(r)
    # 收口：音频放完后 VAD 里可能还悬着最后一段没吐出来
    # （因为它还在等"足够长的静音"来判定结束）
    tail = vad.flush()
    if tail is not None and tail.size:
        segs.append(tail)
    print(f"  {label}: min_silence={cfg.VAD_MIN_SILENCE_MS}ms, "
          f"prebuffer={getattr(cfg,'VAD_PREBUFFER_MS',600)}ms "
          f"-> 切出 {len(segs)} 段")
    return segs


def main():
    ap = argparse.ArgumentParser(description="VAD 切分诊断")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--file", default=None)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--no-asr", action="store_true", help="不做识别，只看切分")
    args = ap.parse_args()

    from config import VoiceConfig, DATA_DIR
    from core.audio import AudioRecorder, save_wav, load_audio
    cfg = VoiceConfig()
    sr = cfg.ASR_SAMPLE_RATE

    # ---------- 获取音频 ----------
    if args.file:
        hr("分析已有音频")
        audio, sr = load_audio(args.file, sr)
        wav = args.file
    else:
        hr("录制音频")
        device = args.device if args.device is not None else cfg.MIC_DEVICE
        rec = AudioRecorder(device=device)
        if not rec.start():
            print(f"[FAIL] 麦克风打开失败: {rec.last_error}")
            print("  >>> python scripts/calibrate_mic.py --scan")
            return 1
        print(f"  设备: {device if device is not None else '系统默认'}")
        print(f"  >>> 请用平时说话的语速，说一句完整的话（含名字更好）")
        print(f"      例如：「记住我的脸，我叫张伟强」 <<<")
        print(f"  采集中 {args.seconds:.0f} 秒...")
        time.sleep(args.seconds)
        audio = rec.read_recent(args.seconds)
        rec.stop()
        if audio.size == 0:
            print("[FAIL] 没录到音频")
            return 1
        name = time.strftime("vad_diag_%Y%m%d_%H%M%S.wav")
        wav = DATA_DIR / name
        save_wav(wav, audio, sr)
        print(f"  已保存: {wav}")

    dur = len(audio) / sr
    peak = float(np.max(np.abs(audio))) if audio.size else 0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0
    hr("音频概况")
    print(f"  时长 {dur:.2f}s | 峰值 {peak:.4f} | RMS {rms:.5f}")

    # ---------- 对比新旧参数 ----------
    hr("VAD 切分对比")
    # 旧参数（问题版本）
    class Old:
        VAD_START_THRESHOLD = cfg.VAD_START_THRESHOLD
        VAD_END_THRESHOLD = 0.001
        VAD_MIN_SILENCE_MS = 700
        VAD_MAX_SPEECH_MS = 15000
        VAD_PREBUFFER_MS = 240
        VAD_SOFT_THRESHOLD_RATIO = 0.8
    old_segs = analyze_segments(audio, sr, Old, "旧参数")
    new_segs = analyze_segments(audio, sr, cfg, "新参数(当前)")

    def show(segs, tag):
        if not segs:
            print(f"      [{tag}] 没有切出任何语音段 —— VAD 阈值可能太高")
            return
        for i, s in enumerate(segs, 1):
            print(f"      [{tag}] 段{i}: {len(s)/sr:.2f}s")

    show(old_segs, "旧")
    show(new_segs, "新")

    if len(old_segs) > len(new_segs):
        print(f"\n  ✓ 新参数把 {len(old_segs)} 段合并成了 {len(new_segs)} 段"
              f"（少切 {len(old_segs)-len(new_segs)} 次）")
    elif len(new_segs) > 1:
        print(f"\n  ⚠️ 仍然切成了 {len(new_segs)} 段 —— 说明你的停顿比较长，"
              f"可继续调大 VAD_MIN_SILENCE_MS")
    else:
        print(f"\n  ✓ 切分正常（{len(new_segs)} 段）")

    # ---------- 分段识别 ----------
    if not args.no_asr:
        hr("分段识别结果（看碎片是怎么变成乱码的）")
        from modules.asr import ASR
        asr = ASR(cfg)
        if not asr.load():
            print(f"[FAIL] ASR 加载失败: {asr.last_error}")
            return 1

        if new_segs and len(new_segs) > 1:
            print("  各碎片分别识别（这就是乱码的来源）：")
            for i, s in enumerate(new_segs, 1):
                t = asr.transcribe_array(s, sr)
                print(f"    段{i} ({len(s)/sr:.2f}s) -> {t!r}")
            print()

        full_text = asr.transcribe_array(audio, sr)
        print(f"  整段一次性识别 -> {full_text!r}")
        if len(new_segs) > 1:
            joined = "".join(asr.transcribe_array(s, sr) for s in new_segs)
            print(f"  分片识别拼接   -> {joined!r}")
            print()
            print("  对比上面两条：如果「整段一次性」明显更通顺，")
            print("  说明问题就在 VAD 切分，把 VAD_MIN_SILENCE_MS 继续调大即可。")
        asr.unload()

    print()
    print(f"  录音保留在: {wav}")
    print("  （可反复用于对比不同参数/模型，无需重新录音）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
