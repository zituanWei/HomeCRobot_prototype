"""
语音识别诊断工具
================
用途：判断 ASR 识别错误到底是「音频信号问题」还是「模型能力问题」。

原理：录一段音 -> 存成 wav 让你亲耳听 -> 同一条音频喂给 ASR -> 对比。

  · 如果你听 wav 也听不清  -> 是麦克风/环境问题，换更大的模型也没用
  · 如果 wav 清晰但识别错  -> 是模型问题，可以换模型或加热词

同时输出音频质量指标（RMS、削波比例、静音比例），辅助判断。

用法:
    python scripts/diagnose_asr.py                 # 录 5 秒并诊断
    python scripts/diagnose_asr.py --seconds 8
    python scripts/diagnose_asr.py --file xxx.wav  # 诊断已有文件
    python scripts/diagnose_asr.py --device 15     # 指定麦克风
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def hr(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64)


def audio_quality_report(audio: np.ndarray, sr: int) -> dict:
    """输出音频质量指标，用来判断信号是否适合识别"""
    a = np.asarray(audio, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return {"ok": False, "reason": "空音频"}

    peak = float(np.max(np.abs(a)))
    rms = float(np.sqrt(np.mean(a ** 2)))
    # 削波：接近满幅的样本比例（麦克风增益过高会削波，识别会崩）
    clipped = float(np.mean(np.abs(a) > 0.98))
    # 静音比例：能量低于阈值的帧占比
    frame = max(1, int(sr * 0.03))
    n = a.size // frame
    if n > 1:
        frames = a[:n * frame].reshape(n, frame)
        fr = np.sqrt(np.mean(frames ** 2, axis=1))
        silence_ratio = float(np.mean(fr < max(rms * 0.15, 1e-4)))
    else:
        silence_ratio = 0.0

    return {
        "ok": True,
        "duration": a.size / sr,
        "peak": peak,
        "rms": rms,
        "clipped_ratio": clipped,
        "silence_ratio": silence_ratio,
    }


def print_quality(q: dict):
    if not q.get("ok"):
        print(f"  [FAIL] {q.get('reason')}")
        return
    print(f"  音频时长 : {q['duration']:.2f}s")
    print(f"  峰值     : {q['peak']:.4f}")
    print(f"  平均 RMS : {q['rms']:.5f}")
    print(f"  削波比例 : {q['clipped_ratio']*100:.2f}%"
          f"{'   ← 过高！麦克风增益太大，会严重破坏识别' if q['clipped_ratio'] > 0.01 else ''}")
    print(f"  静音比例 : {q['silence_ratio']*100:.1f}%"
          f"{'   ← 偏高，可能没录到有效说话' if q['silence_ratio'] > 0.7 else ''}")

    print()
    print("  诊断：")
    if q["rms"] < 0.002:
        print("    ⚠️  音量过低 —— 大概率是麦克风设备选错（比如选到了没有麦克风的条目）")
        print("        请跑: python scripts/calibrate_mic.py --scan")
    elif q["clipped_ratio"] > 0.01:
        print("    ⚠️  削波严重 —— 在 Windows 声音设置里调低麦克风输入音量")
    elif q["silence_ratio"] > 0.7:
        print("    ⚠️  大部分是静音 —— 可能 VAD 切错，或说话时离麦太远")
    elif q["rms"] < 0.01:
        print("    · 音量偏低但可能可用 —— 建议靠近麦克风、或换灵敏度更高的设备")
    else:
        print("    ✓ 音量正常，音频信号本身没问题")
        print("      如果识别仍然错，那问题在模型侧（可考虑换模型/加热词）")


def main():
    parser = argparse.ArgumentParser(description="语音识别诊断")
    parser.add_argument("--seconds", type=float, default=5.0, help="录音时长")
    parser.add_argument("--file", default=None, help="改为诊断已有音频文件")
    parser.add_argument("--device", type=int, default=None, help="麦克风设备编号")
    parser.add_argument("--save", default=None, help="保存路径（默认 data/asr_diag_*.wav）")
    args = parser.parse_args()

    from config import VoiceConfig, DATA_DIR
    from core.audio import AudioRecorder, save_wav, load_audio

    if args.file:
        hr("诊断已有音频文件")
        audio, sr = load_audio(args.file, VoiceConfig.ASR_SAMPLE_RATE)
        wav_path = args.file
        print(f"  文件: {args.file}  ({len(audio)/sr:.2f}s @ {sr}Hz)")
    else:
        hr("录制一段音频")
        device = args.device if args.device is not None else VoiceConfig.MIC_DEVICE
        rec = AudioRecorder(device=device)
        if not rec.start():
            print(f"[FAIL] 麦克风打开失败: {rec.last_error}")
            print("  提示: python scripts/calibrate_mic.py --scan")
            return 1
        print(f"  使用设备: {device if device is not None else '系统默认'}")
        print(f"  >>> 请现在开始说话，持续 {args.seconds:.0f} 秒 <<<")
        time.sleep(args.seconds)
        audio = rec.read_recent(args.seconds)
        rec.stop()
        if audio.size == 0:
            print("[FAIL] 没有录到音频")
            return 1
        sr = VoiceConfig.ASR_SAMPLE_RATE
        name = time.strftime("asr_diag_%Y%m%d_%H%M%S.wav")
        wav_path = args.save or (DATA_DIR / name)
        save_wav(wav_path, audio, sr)
        print(f"  已保存: {wav_path}")

    # ---------------- 音频质量 ----------------
    hr("音频质量指标")
    q = audio_quality_report(audio, sr)
    print_quality(q)
    print()
    print(f"  📢 请亲耳听一下这段录音：{wav_path}")
    print("     （用系统播放器打开即可）")
    print("     你听得清楚 → 信号没问题，识别错是模型侧的问题")
    print("     你也听不清 → 信号有问题，换更大模型也没用")

    # ---------------- ASR 识别 ----------------
    hr("ASR 识别结果")
    from modules.asr import ASR
    cfg = VoiceConfig()
    cfg.ENABLE_ASR = True
    print(f"  模型: {cfg.ASR_MODEL}")
    asr = ASR(cfg)
    t0 = time.time()
    if not asr.load():
        print(f"[FAIL] 模型加载失败: {asr.last_error}")
        return 1
    print(f"  加载耗时: {time.time()-t0:.1f}s")

    t0 = time.time()
    text = asr.transcribe_array(audio, sr)
    dt = time.time() - t0
    print(f"  识别耗时: {dt:.2f}s")
    print()
    print(f"  ┌─ 识别结果 ─────────────────────────────────")
    print(f"  │ {text if text else '（空）'}")
    print(f"  └────────────────────────────────────────────")
    print()
    print("  对照判断：")
    print(f"    你说的是      : （你自己知道）")
    print(f"    模型识别成    : {text if text else '（空）'}")
    print()
    print("  如果差别很大，且上面录音你听着清楚：")
    print("    · 可能是口音/语速超出 paraformer-zh 的适应范围")
    print("    · 可以在 config.py 换更大的模型试试，例如：")
    print('        ASR_MODEL = "paraformer-zh"          # 当前（约220MB）')
    print('        ASR_MODEL = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"')
    print("    · 或为专有名词加「热词」（FunASR 支持 hotword 参数）")
    print()
    print(f"  该 wav 已保留，可用于对比不同模型：{wav_path}")
    asr.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
