"""
环境检查脚本
运行: python check_env.py
检查：Python版本、GPU、CUDA、关键依赖包、摄像头
"""
import sys
import importlib


def check_python():
    print(f"🐍 Python 版本: {sys.version}")
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 10):
        print("  ✓ Python 版本符合要求 (>=3.10)")
    else:
        print("  ⚠️  建议 Python >= 3.10")


def check_gpu():
    print("\n🎮 GPU / CUDA 检查:")
    try:
        import torch
        print(f"  PyTorch 版本: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✓ GPU: {gpu_name}")
            print(f"  ✓ 显存: {vram:.1f} GB")
            print(f"  ✓ CUDA 版本: {torch.version.cuda}")
            if vram < 4:
                print("  ⚠️  显存较小，建议使用 2B 模型或 4bit 量化")
            elif vram < 6:
                print("  ℹ️  建议使用 3B-4bit 或 2B 模型")
            else:
                print("  ✓ 显存充足，可运行 3B-4bit 甚至 7B-4bit")
        else:
            print("  ✗ CUDA 不可用，将使用 CPU（速度极慢，不推荐）")
    except ImportError:
        print("  ✗ PyTorch 未安装，请运行: pip install torch torchvision")


def check_packages():
    print("\n📦 关键依赖包检查:")
    packages = {
        # 核心
        "transformers": "transformers",
        "accelerate": "accelerate",
        "bitsandbytes": "bitsandbytes",
        "qwen_vl_utils": "qwen-vl-utils",
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "numpy": "numpy",
        "rich": "rich",
        # 音频（语音输入输出的地基）
        "sounddevice": "sounddevice (音频I/O)",
        "soundfile": "soundfile (音频解码)",
        "edge_tts": "edge-tts (语音合成)",
        "funasr": "funasr (语音识别)",
    }
    missing = []
    for module, name in packages.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "已安装")
            print(f"  ✓ {name}: {version}")
        except ImportError:
            print(f"  ✗ {name}: 未安装")
            missing.append(name)
    if missing:
        print(f"\n  ⚠️  缺失 {len(missing)} 个包，请运行: pip install -r requirements.txt")


def check_audio():
    print("\n🎧 音频设备检查（语音功能依赖）:")
    try:
        import sounddevice as sd
    except ImportError:
        print("  ✗ sounddevice 未安装，语音输入输出不可用")
        print("    安装: pip install sounddevice")
        return
    try:
        din = sd.query_devices(kind="input")
        dout = sd.query_devices(kind="output")
        print(f"  ✓ 默认麦克风: {din['name']}")
        print(f"  ✓ 默认扬声器: {dout['name']}")
    except Exception as e:
        print(f"  ⚠️  未找到默认输入或输出设备: {e}")
        print("     语音功能需要麦克风（输入）和扬声器（输出）")

    try:
        devices = sd.query_devices()
        ins = [d for d in devices if d["max_input_channels"] > 0]
        outs = [d for d in devices if d["max_output_channels"] > 0]
        print(f"  ℹ️  共发现 {len(ins)} 个输入设备、{len(outs)} 个输出设备")
        print("     如果默认设备不对，改 config.py 的 MIC_DEVICE / SPEAKER_DEVICE")
        print("     查看编号: python -c \"import sounddevice as sd; print(sd.query_devices())\"")
    except Exception as e:
        print(f"  ⚠️  枚举设备失败: {e}")

    print("\n  💡 麦克风音量校准（强烈建议语音使用前跑一次）:")
    print("     python scripts/calibrate_mic.py")


def check_camera():
    print("\n📷 摄像头检查:")
    try:
        import cv2
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  ✓ 摄像头可用，分辨率: {w}x{h}")
            else:
                print("  ⚠️  摄像头已打开但无法读取帧")
            cap.release()
        else:
            print("  ✗ 无法打开摄像头 (index=0)")
            print("  提示: 检查摄像头是否被其他程序占用，或在 config.py 中修改 CAMERA_INDEX")
        cv2.destroyAllWindows()
    except ImportError:
        print("  ✗ opencv-python 未安装")


def main():
    print("=" * 55)
    print("🔍 家庭陪护机器人 - 环境检查")
    print("=" * 55)
    check_python()
    check_gpu()
    check_packages()
    check_camera()
    check_audio()
    print("\n" + "=" * 55)
    print("检查完成！如有缺失的包，请运行: pip install -r requirements.txt")
    print("=" * 55)


if __name__ == "__main__":
    main()
