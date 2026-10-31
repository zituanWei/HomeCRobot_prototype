"""
使用 ModelScope 下载 Qwen2.5-VL 模型到本地
国内最稳定的下载方式，不会卡住

运行: python scripts/download_model.py
下载完成后会自动提示修改 config.py
"""
import os
import sys
from pathlib import Path

# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
# 本地保存路径（项目目录下的 models 文件夹）
LOCAL_DIR = Path(__file__).parent.parent / "models" / "Qwen2.5-VL-3B-Instruct"


def check_modelscope():
    """检查 modelscope 是否安装"""
    try:
        import modelscope
        print(f"✓ modelscope 已安装: {modelscope.__version__}")
        return True
    except ImportError:
        print("✗ modelscope 未安装")
        return False


def install_modelscope():
    """安装 modelscope"""
    print("\n正在安装 modelscope...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "modelscope"],
        capture_output=False
    )
    if result.returncode != 0:
        print("✗ modelscope 安装失败，请手动运行: pip install modelscope")
        sys.exit(1)
    print("✓ modelscope 安装成功")


def download_model():
    """使用 ModelScope 下载模型"""
    from modelscope import snapshot_download

    print(f"\n{'='*60}")
    print(f" 开始下载模型: {MODEL_NAME}")
    print(f" 保存到: {LOCAL_DIR}")
    print(f"{'='*60}\n")

    # 创建目录
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # snapshot_download 会下载整个模型仓库
        local_path = snapshot_download(
            model_id=MODEL_NAME,
            cache_dir=str(LOCAL_DIR.parent),  # 缓存目录
            local_dir=str(LOCAL_DIR),          # 最终保存目录
        )
        print(f"\n✓ 模型下载完成!")
        print(f"  路径: {local_path}")
        return local_path
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能的原因:")
        print("  1. 网络不稳定 - 重试即可，支持断点续传")
        print("  2. 磁盘空间不足 - 确保有 10GB 以上空闲空间")
        sys.exit(1)


def verify_download(local_path: str):
    """验证下载的文件是否完整"""
    path = Path(local_path)
    required_files = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    # 模型权重文件（可能是单个或分片）
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("model*.bin"))

    print(f"\n{'='*60}")
    print(" 验证下载文件")
    print(f"{'='*60}")

    all_ok = True
    for f in required_files:
        fpath = path / f
        if fpath.exists():
            size_mb = fpath.stat().st_size / 1024 / 1024
            print(f"  ✓ {f} ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {f} 缺失")
            all_ok = False

    if weight_files:
        total_size = sum(f.stat().st_size for f in weight_files) / 1024 / 1024 / 1024
        print(f"  ✓ 模型权重: {len(weight_files)} 个文件, 共 {total_size:.2f} GB")
    else:
        print("  ✗ 未找到模型权重文件 (*.safetensors / *.bin)")
        all_ok = False

    return all_ok


def print_next_steps(local_path: str):
    """打印后续操作说明"""
    print(f"\n{'='*60}")
    print(" 下一步：修改 config.py 使用本地模型")
    print(f"{'='*60}")
    print(f"""
打开 config.py，找到 VLConfig 类，修改 MODEL_NAME 为本地路径:

    class VLConfig:
        MODEL_NAME = r"{local_path}"   # ← 改成本地路径
        # MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"  # 注释掉在线下载

保存后运行:
    python test_model.py

以后加载模型就完全离线，不会再卡下载了。
""")


def main():
    print("=" * 60)
    print(" Qwen2.5-VL 模型下载工具 (ModelScope)")
    print("=" * 60)

    # 检查并安装 modelscope
    if not check_modelscope():
        install_modelscope()

    # 检查是否已经下载过
    if LOCAL_DIR.exists() and list(LOCAL_DIR.glob("*.safetensors")):
        print(f"\n检测到模型已存在: {LOCAL_DIR}")
        choice = input("是否重新下载？(y/N): ").strip().lower()
        if choice != "y":
            print("跳过下载，直接验证...")
            if verify_download(str(LOCAL_DIR)):
                print_next_steps(str(LOCAL_DIR))
            return

    # 下载模型
    local_path = download_model()

    # 验证
    if verify_download(local_path):
        print_next_steps(local_path)
    else:
        print("\n⚠️  文件不完整，请重新运行本脚本（支持断点续传）")


if __name__ == "__main__":
    main()
