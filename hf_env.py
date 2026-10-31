"""
HuggingFace 国内镜像配置
⚠️ 必须在导入 transformers / huggingface_hub / torch 之前导入本模块才生效

国内网络访问 huggingface.co 超时，使用 hf-mirror.com 镜像加速下载。
如需关闭镜像，设置环境变量 HF_ENDPOINT="" 或删除本文件的导入。
"""
import os

# 国内镜像站（优先）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 禁用 Xet 下载协议（镜像站不支持，会导致 401 Unauthorized）
# 回退到传统 HTTP 分块下载
os.environ["HF_HUB_DISABLE_XET"] = "1"

# ============ CUDA 显存分配策略（必须在 torch 之前设置）============
# 背景：7B 模型在 8GB 显卡上显存占用接近上限（实测 7.8/8.2 GB）。
# 视觉推理瞬间申请更多显存时若不足，Windows 会触发 WDDM 换页
# （显存↔内存，速度下降几十倍），严重时进程被系统直接杀掉，
# 表现为「发了消息之后一直没有回应」。
#
# 注意：经实测，expandable_segments 在 Windows 上**不支持**，
# 设置它只会得到 UserWarning: expandable_segments not supported on
# this platform，没有任何效果。所以这里改用 Windows 可用的参数：
#
#   garbage_collection_threshold:0.8
#     当显存占用超过该比例时，主动回收未使用的缓存块，
#     缓解「总量够但碎片太多」导致的 OOM。这是 Windows 上可用的方案。
#
# 更根本的显存控制手段见 config.VLConfig.IMAGE_MAX_PIXELS（限制视觉 token 数）。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "garbage_collection_threshold:0.8")

# 模型下载缓存到项目目录（避免C盘空间不足，可选）
# from pathlib import Path
# os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))
# os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(__file__).parent / "models" / "hub"))
