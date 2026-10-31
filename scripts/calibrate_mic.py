"""
麦克风音量校准与设备选择工具
============================
不同麦克风（耳机麦 / 内置麦 / USB麦）灵敏度差别很大，而且蓝牙耳机在
A2DP（音乐）模式下属正常现象——它根本没有麦克风输入。默认设备选错时，
语音识别会完全失效。

功能：
  1. --scan          : 逐个测试每个输入设备，找出真正能拾音的那个
  2. --device N      : 指定设备做完整校准
  3. 无参数          : 用默认设备做完整校准，并给出建议阈值

用法:
    python scripts/calibrate_mic.py --scan
    python scripts/calibrate_mic.py --device 15
    python scripts/calibrate_mic.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.audio import AudioRecorder, list_audio_devices


BAR = "█"
WIDTH = 40


def draw_level(rms: float, peak_scale: float = 0.2):
    ratio = min(1.0, rms / peak_scale)
    filled = int(ratio * WIDTH)
    return f"[{BAR * filled}{' ' * (WIDTH - filled)}] {rms:.4f}"


def measure(rec: AudioRecorder, seconds: float, label: str) -> dict:
    print(f"\n  >>> {label}（{seconds:.0f} 秒）")
    samples = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        time.sleep(0.1)
        samples.append(rec.level)
        sys.stdout.write("\r      " + draw_level(rec.level))
        sys.stdout.flush()
    print()
    samples = np.array(samples) if samples else np.array([0.0])
    return {
        "peak": float(samples.max()),
        "mean": float(samples.mean()),
        "p95": float(np.percentile(samples, 95)),
    }


def scan_devices():
    """逐个输入设备测试，找出真正能拾音的设备

    关键点：必须让用户**说话**来测。用扬声器回授的测法对耳机无效
    （声音进了耳朵，不会外泄到麦克风）。
    """
    print("=" * 62)
    print(" 输入设备扫描")
    print("=" * 62)
    try:
        import sounddevice as sd
    except ImportError:
        print("[FAIL] sounddevice 未安装")
        return 1

    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    if not inputs:
        print("[FAIL] 没有找到任何输入设备")
        return 1

    print(f"共 {len(inputs)} 个输入设备。")
    print()
    print("⚠️  测试方法说明：")
    print("   每个设备会采集约 2 秒，**请在采集期间持续说话**")
    print("   （说「一二三四五」这样即可）。")
    print("   不要用放音乐的方式测——如果你戴耳机，声音进了耳朵，")
    print("   麦克风根本收不到回授，会误判成「没信号」。")
    print()
    print("   如果设备很多很慢，可以中途按 Ctrl+C 退出，改用：")
    print("     python scripts/calibrate_mic.py --device <编号>")
    print()

    results = []
    for idx, d in inputs:
        name = d["name"][:36]
        try:
            input(f"  按回车开始测 [{idx}] {name} ...")
        except EOFError:
            pass
        rec = AudioRecorder(device=idx)
        if not rec.start():
            print(f"       [{idx:>2}] {name:<36} 打开失败")
            results.append((idx, name, 0.0, False))
            continue

        peak = 0.0
        print("       >>> 现在说话！（2 秒）")
        for _ in range(20):
            time.sleep(0.1)
            peak = max(peak, rec.level)
            sys.stdout.write("\r       " + draw_level(rec.level))
            sys.stdout.flush()
        print()
        rec.stop()
        ok = peak > 0.002
        print(f"       peak={peak:.5f}  {'✓ 能拾音' if ok else '✗ 信号很弱'}")
        results.append((idx, name, peak, ok))

    print("\n" + "=" * 62)
    working = [r for r in results if r[3]]
    if working:
        working.sort(key=lambda r: -r[2])
        best = working[0]
        print(" 可拾音的设备（按信号强度排序）：")
        for pk, i, n in [(r[2], r[0], r[1]) for r in working[:5]]:
            print(f"    [{i:>2}] {n:<36} peak={pk:.5f}")
        print(f"\n 建议使用信号最强的：")
        print(f"    MIC_DEVICE = {best[0]}    # {best[1]}")
        print(f"\n 然后用它做完整校准（测阈值）：")
        print(f"    python scripts/calibrate_mic.py --device {best[0]}")
    else:
        print(" ⚠️  没有任何设备检测到明显信号。请检查：")
        print("      1. 麦克风是否被静音（系统音量面板 / 耳机物理静音键）")
        print("      2. Windows 设置 → 隐私和安全性 → 麦克风，允许桌面应用访问")
        print("      3. 蓝牙耳机：需要一个「Hands-Free」条目才有麦克风，")
        print("         若列表里只有 A2DP 条目，先在系统声音设置里切换模式")
        print("      4. 设备是否被其他程序独占（会议软件等）")
    return 0 if working else 1


def full_calibration(device=None):
    print("=" * 62)
    print(" 麦克风音量校准" + (f"（设备 {device}）" if device is not None else "（默认设备）"))
    print("=" * 62)
    print("\n可用音频设备（若默认设备不对，用 --scan 找，再填 config.py 的 MIC_DEVICE）：")
    print(list_audio_devices())
    print()

    rec = AudioRecorder(device=device)
    if not rec.available:
        print("[FAIL] sounddevice 未安装")
        return 1
    if not rec.start():
        print(f"[FAIL] 麦克风打开失败: {rec.last_error}")
        print("\n提示：用 --scan 找出可用的输入设备")
        return 1

    print("提示：请确保麦克风没有被静音，并且系统隐私设置允许应用访问麦克风。")

    try:
        input("\n准备好后按回车开始（先测环境噪声）...")
    except EOFError:
        pass

    quiet = measure(rec, 3.0, "保持安静，不要说话")

    try:
        input("\n接下来测说话音量，按回车后请正常音量说话...")
    except EOFError:
        pass

    speech = measure(rec, 5.0, "请正常说话（随便说什么都行）")

    rec.stop()

    print("\n" + "=" * 62)
    print(" 测量结果")
    print("=" * 62)
    print(f"  环境噪声:  峰值 {quiet['peak']:.4f}  平均 {quiet['mean']:.4f}")
    print(f"  说话音量:  峰值 {speech['peak']:.4f}  平均 {speech['mean']:.4f}  "
          f"P95 {speech['p95']:.4f}")

    ratio = speech['peak'] / max(quiet['peak'], 1e-6)
    print(f"  信噪比:    {ratio:.1f}x")

    noise_floor = max(quiet['p95'], quiet['peak'], 1e-4)
    speech_level = max(speech['p95'], speech['peak'])

    start_th = float(np.clip(noise_floor * 2.5, 1e-4, speech_level * 0.8))
    end_th = float(np.clip(noise_floor * 1.4, 5e-5, start_th * 0.9))
    start_th = round(start_th, 4)
    end_th = round(end_th, 4)

    print("\n" + "=" * 62)
    print(" 建议配置（复制到 config.py 的 VoiceConfig）")
    print("=" * 62)
    if device is not None:
        print(f"    MIC_DEVICE = {device}")
    print(f"    VAD_START_THRESHOLD = {start_th}")
    print(f"    VAD_END_THRESHOLD = {end_th}")
    print(f"    BARGE_IN_RMS = {round(float(np.clip(speech_level * 0.5, 0.01, 0.3)), 4)}")
    print()

    if speech['peak'] < 0.005:
        print("[WARN] 说话时音量依然极低，可能原因：")
        print("       1. 麦克风被静音（系统音量面板 / 耳机上的物理静音键）")
        print("       2. Windows 隐私设置未允许应用访问麦克风")
        print("       3. MIC_DEVICE 指向了错误的设备（或蓝牙耳机处于 A2DP 模式无麦克风）")
        print("       -> 先跑 python scripts/calibrate_mic.py --scan 找到能拾音的设备")
        return 1
    if ratio < 2.0:
        print("[WARN] 环境噪声偏大，语音识别可能不稳，建议在安静一点的环境重试")
    else:
        print("[OK] 信噪比正常，把上面的阈值填进 config.py 即可")
    return 0


def main():
    args = sys.argv[1:]
    if "--scan" in args:
        return scan_devices()

    device = None
    if "--device" in args:
        i = args.index("--device")
        if i + 1 < len(args):
            try:
                device = int(args[i + 1])
            except ValueError:
                print(f"[FAIL] --device 需要一个整数编号，收到: {args[i+1]}")
                return 1
    return full_calibration(device)


if __name__ == "__main__":
    sys.exit(main())
