"""
使用 ModelScope 下载 Qwen2.5-VL-7B-Instruct 到本地
=================================================
专为切换到 7B 模型准备（3B 版本见 download_model.py）。

为什么用 ModelScope 而不是 HuggingFace：
  国内网络下载 HF 经常超时，ModelScope 是阿里的镜像，速度快且支持断点续传。

显存提醒（重要）：
  RTX 4060 Laptop 8GB 跑 7B **必须**用 4bit 量化：
      7B 4bit  ≈ 4.5~5GB（权重）+ 视觉编码器 ≈ 总 5.5GB
      7B 8bit  ≈ 8GB+  ->  必然 OOM
  下载完成后记得确认 config.py 里：
      LOAD_IN_4BIT = True
      LOAD_IN_8BIT = False

运行: python scripts/download_model_7b.py
"""
import sys
from pathlib import Path

# ============ 要下载的模型 ============
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
DIR_NAME = "Qwen2.5-VL-7B-Instruct"
LOCAL_DIR = Path("D:/PyCharm_projects/HomeCRobot/models") / DIR_NAME
# 预计占用空间（GB），用于提前检查磁盘
EXPECT_GB = 17.0


def check_disk():
    import shutil
    root = LOCAL_DIR.parent
    root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(str(root)).free / (1024 ** 3)
    print(f"  磁盘可用空间: {free_gb:.1f} GB   预计需要: {EXPECT_GB:.1f} GB")
    if free_gb < EXPECT_GB:
        print(f"  ✗ 空间不足，请先清理磁盘（至少留 {EXPECT_GB:.0f} GB）")
        return False
    print("  ✓ 空间充足")
    return True


def check_modelscope():
    try:
        import modelscope
        print(f"  ✓ modelscope 已安装: {modelscope.__version__}")
        return True
    except ImportError:
        print("  ✗ modelscope 未安装")
        return False


def download_model():
    from modelscope import snapshot_download

    print()
    print("=" * 62)
    print(f" 开始下载: {MODEL_NAME}")
    print(f" 保存到  : {LOCAL_DIR}")
    print("=" * 62)
    print(" 支持断点续传；中断后重新运行本脚本即可继续。")
    print()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        local_path = snapshot_download(
            model_id=MODEL_NAME,
            cache_dir=str(LOCAL_DIR.parent),
            local_dir=str(LOCAL_DIR),
        )
        print(f"\n✓ 下载完成: {local_path}")
        return local_path
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能原因：")
        print("  1. 网络不稳定 —— 直接重新运行本脚本（断点续传）")
        print("  2. 磁盘空间不足")
        print("  3. ModelScope 需要登录 —— 可先运行: modelscope login")
        return None


def verify_download(path: str) -> bool:
    p = Path(path)
    required = ["config.json", "preprocessor_config.json",
                "tokenizer_config.json", "vocab.json", "merges.txt"]
    weights = list(p.glob("*.safetensors")) + list(p.glob("model*.bin"))

    print()
    print("=" * 62)
    print(" 验证下载文件")
    print("=" * 62)
    ok = True
    for f in required:
        fp = p / f
        if fp.exists():
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} 缺失")
            ok = False

    if weights:
        total = sum(f.stat().st_size for f in weights) / (1024 ** 3)
        print(f"  ✓ 权重: {len(weights)} 个文件，共 {total:.2f} GB")
    else:
        print("  ✗ 未找到权重文件（*.safetensors）")
        ok = False
    return ok


def print_next_steps(path: str):
    print()
    print("=" * 62)
    print(" 下一步：改 config.py（两处，都要改）")
    print("=" * 62)
    print(f"""
打开 D:\\PyCharm_projects\\HomeCRobot\\config.py，找到 VLConfig：

    # ① 模型路径改成 7B
    MODEL_NAME = r"{path}"

    # ② 必须是 4bit（8bit 会爆显存！）
    LOAD_IN_4BIT = True
    LOAD_IN_8BIT = False

保存后验证：

    python scripts/check_env.py          # 看显存与依赖
    python test_model.py                 # 只测模型能否加载

再启动：

    python main.py --voice

⚠️ 如果启动时报 "CUDA out of memory"，按顺序试：
    1) 确认 LOAD_IN_4BIT=True 且 LOAD_IN_8BIT=False
    2) 把 MonitorConfig.INTERVAL_SECONDS 调大到 120（VL 深度分析少跑）
    3) 把 MonitorConfig.GREETING_USE_VL 设为 False（打招呼不调大模型）
    4) 关闭其他占显存的程序

回滚：把 MODEL_NAME 改回 3B 路径即可（3B 模型仍在本地，不会被删）
""")


def main():
    print("=" * 62)
    print(" Qwen2.5-VL-7B 下载工具 (ModelScope)")
    print("=" * 62)

    if not check_disk():
        return 1
    if not check_modelscope():
        print("\n  请先安装: pip install modelscope modelscope[framework]")
        print(f"  或直接运行: {sys.executable} -m pip install modelscope")
        return 1

    if LOCAL_DIR.exists() and list(LOCAL_DIR.glob("*.safetensors")):
        print(f"\n  检测到已存在: {LOCAL_DIR}")
        try:
            c = input("  是否重新下载？(y/N): ").strip().lower()
        except EOFError:
            c = "n"
        if c != "y":
            if verify_download(str(LOCAL_DIR)):
                print_next_steps(str(LOCAL_DIR))
            return 0

    path = download_model()
    if not path:
        return 1
    if verify_download(path):
        print_next_steps(path)
        return 0
    print("\n⚠️ 文件不完整，请重新运行本脚本（支持断点续传）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
