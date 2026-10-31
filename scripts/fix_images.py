"""
批量修复图片的 iCCP 颜色配置警告
用法: python scripts/fix_images.py [目录路径]
默认修复 data/ 目录下所有 jpg/jpeg/png 文件
"""
import sys
from pathlib import Path
from PIL import Image
import warnings

# 抑制 Pillow 本身的警告，只看我们的输出
warnings.filterwarnings("ignore")


def fix_image(image_path: Path) -> bool:
    """
    修复单张图片：重新保存，去除异常 ICC profile 和多余元数据
    返回是否修复成功
    """
    try:
        # 打开图片
        with Image.open(image_path) as img:
            # 转换为 RGB（PNG 可能是 RGBA，保存为 jpg 需要转）
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # 获取原始大小用于对比
            original_size = image_path.stat().st_size

            # 保存：不保留 ICC profile，优化编码
            # 先保存到临时文件，成功后再替换
            temp_path = image_path.with_suffix(".tmp" + image_path.suffix)
            img.save(str(temp_path), quality=95, optimize=True)

            # 替换原文件
            temp_path.replace(image_path)

            new_size = image_path.stat().st_size
            print(f"  ✓ {image_path.name} ({original_size//1024}KB -> {new_size//1024}KB)")
            return True
    except Exception as e:
        print(f"  ✗ {image_path.name}: {e}")
        return False


def main():
    # 确定要修复的目录
    if len(sys.argv) > 1:
        target_dir = Path(sys.argv[1])
    else:
        target_dir = Path(__file__).parent.parent / "data"

    if not target_dir.exists():
        print(f"目录不存在: {target_dir}")
        return

    print("=" * 60)
    print(f" 批量修复图片 iCCP 警告")
    print(f" 目标目录: {target_dir}")
    print("=" * 60)

    # 查找所有图片
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_files = []
    for ext in extensions:
        image_files.extend(target_dir.glob(ext))
        # 也包括子目录
        image_files.extend(target_dir.rglob(ext))

    # 去重
    image_files = list(set(image_files))

    if not image_files:
        print("未找到图片文件")
        return

    print(f"\n找到 {len(image_files)} 张图片，开始修复...\n")

    success = 0
    for img_path in sorted(image_files):
        if fix_image(img_path):
            success += 1

    print(f"\n{'=' * 60}")
    print(f" 修复完成: {success}/{len(image_files)} 张")
    print(f" 现在用 Pillow/模型加载这些图片不会再报 iCCP 警告了")
    print("=" * 60)


if __name__ == "__main__":
    main()
