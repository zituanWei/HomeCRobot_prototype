"""
通用工具函数
"""
import sys
import time


def print_banner():
    """打印项目启动横幅"""
    banner = r"""
╔══════════════════════════════════════════════╗
║       🏠 家庭陪护机器人 - HomeCRobot         ║
║       视觉语言模型: Qwen2.5-VL               ║
╚══════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """打印帮助信息（自然语言为主）"""
    help_text = """
💡 直接对我说就行，不需要记命令：

  「记住我的脸，我叫小明」  → 注册人脸，以后我能认出你
  「你都认识谁？」            → 查看已注册人员
  「忘记我吧」                → 删除人脸记录
  「打开摄像头看看」          → 摄像头实时预览
  「我现在状态怎么样？」      → 自动拍照，分析你的状态
  「这是什么东西？」          → 自动拍照，识别物品
  「清空对话，重新开始」      → 清空对话历史
  「我们之前聊了什么？」      → 查看最近对话
  「你会什么？」              → 查看完整功能列表
  「再见」                    → 退出程序

🏠 后台监控会自动：
  - 每60秒分析一次画面
  - 检测到异常主动关心
  - 久坐30分钟提醒活动
  - 长时间看不到人主动询问

📌 斜杠命令仍然可用（/look /register /faces 等），但推荐用自然语言~
"""
    print(help_text)


def timed(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  ⏱️  耗时: {elapsed:.2f}s")
        return result
    return wrapper


def check_gpu_memory() -> dict:
    """检查GPU显存使用情况"""
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            return {
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "cached_gb": round(cached, 2),
                "free_gb": round(total - allocated, 2),
            }
    except Exception:
        pass
    return {"error": "GPU不可用"}


def graceful_exit(camera=None, vl_model=None):
    """优雅退出，释放资源"""
    print("\n👋 正在退出...")
    if camera:
        camera.close()
    if vl_model:
        vl_model.unload()
    print("再见！祝你有美好的一天 🌟")
    sys.exit(0)
