"""
模型快速测试脚本（不启动完整对话系统）
运行: python test_model.py
功能：加载模型，用一张测试图片或纯文字验证模型是否正常工作
"""
import hf_env  # noqa: F401  必须在最前面，配置 HuggingFace 国内镜像
import torch
from PIL import Image
from core.vl_model import VLModel
from config import VLConfig


def test_text_only():
    """纯文字测试"""
    print("\n" + "=" * 50)
    print("📝 纯文字推理测试")
    print("=" * 50)

    vl = VLModel(VLConfig)
    vl.load()

    test_questions = [
        "你好，请用一句话介绍你自己",
        "1+1等于几？",
    ]

    for q in test_questions:
        print(f"\n问: {q}")
        response = vl.generate(text=q, image=None, history=None)
        print(f"答: {response}")

    vl.unload()


def test_with_image():
    """带图片测试（需要先运行 test_camera.py 保存一张图片到 data 目录）"""
    print("\n" + "=" * 50)
    print("🖼️  图文推理测试")
    print("=" * 50)

    from pathlib import Path
    data_dir = Path("data")
    images = list(data_dir.glob("capture_*.jpg")) + list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))

    if not images:
        print("未找到测试图片，请先运行 test_camera.py 按 s 保存一张图片")
        print("或者将任意图片放入 data/ 目录")
        return

    img_path = images[0]
    print(f"使用图片: {img_path}")
    image = Image.open(img_path).convert("RGB")

    vl = VLModel(VLConfig)
    vl.load()

    questions = [
        "请详细描述这张图片中的内容",
        "图片里有人吗？如果有，描述一下他的状态",
    ]

    for q in questions:
        print(f"\n问: {q}")
        response = vl.generate(text=q, image=image, history=None)
        print(f"答: {response}")

    vl.unload()


def main():
    print("🧠 Qwen-VL 模型测试")
    print("1. 纯文字测试")
    print("2. 图文测试（需要 data/ 目录下有图片）")
    print("3. 全部测试")

    choice = input("请选择 (1/2/3，默认1): ").strip() or "1"

    if choice == "1":
        test_text_only()
    elif choice == "2":
        test_with_image()
    elif choice == "3":
        test_text_only()
        test_with_image()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
