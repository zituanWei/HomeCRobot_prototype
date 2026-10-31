"""
摄像头单独测试脚本
运行: python test_camera.py
功能：打开摄像头预览，按 s 保存图片，按 q 退出
"""
from core.camera import CameraCapture


def main():
    print("📷 摄像头测试程序")
    print("操作: q=退出, s=保存图片")
    print("-" * 40)

    cam = CameraCapture()
    cam.preview(window_name="Camera Test - 按q退出 按s保存")


if __name__ == "__main__":
    main()
