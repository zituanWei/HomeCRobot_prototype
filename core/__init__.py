"""核心模块包

使用延迟导入，避免 import core.camera 时触发 vl_model 的依赖加载。
"""

def __getattr__(name):
    """延迟导入：只有访问具体类时才加载对应模块"""
    if name == "CameraCapture":
        from .camera import CameraCapture
        return CameraCapture
    if name == "VLModel":
        from .vl_model import VLModel
        return VLModel
    if name == "ConversationManager":
        from .conversation import ConversationManager
        return ConversationManager
    raise AttributeError(f"module 'core' has no attribute {name!r}")
