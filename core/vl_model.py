"""
Qwen-VL 视觉语言模型模块
负责：模型加载、图文推理、对话生成
支持 4bit/8bit 量化，适配 8GB 显存

并发说明（重要）：
后台上视觉监控线程与主对话线程会共用同一个模型实例。
HuggingFace 模型 + 共享 CUDA context 都不是线程安全的，
两个线程同时进 generate() 会导致输出错乱甚至显存崩溃。
因此这里用一把可重入锁把推理串行化，并暴露 busy 状态
供监控线程判断"主线程正在说话，这次先别抢 GPU"。
"""
import threading
import time

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

# 兼容新旧版本 transformers：
# 旧版(4.x): AutoModelForVision2Seq
# 新版(5.x): AutoModelForImageTextToText
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq

from qwen_vl_utils import process_vision_info

from config import VLConfig


class VLModel:
    """Qwen2.5-VL 模型封装（推理串行化）"""

    def __init__(self, config: VLConfig = None):
        self.config = config or VLConfig()
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 推理锁：保证同一时刻只有一次 generate 在跑
        self._infer_lock = threading.RLock()
        self._busy = threading.Event()
        self.last_infer_seconds = 0.0
        print(f"[设备] 使用: {self.device}")

    @property
    def busy(self) -> bool:
        """当前是否有推理正在进行（监控线程据此避让）"""
        return self._busy.is_set()

    def load(self):
        """加载模型和处理器（耗时操作，只需执行一次）"""
        print(f"[模型] 正在加载 {self.config.MODEL_NAME} ...")

        # 量化配置
        quantization_config = None
        if self.config.LOAD_IN_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("[量化] 启用 4bit 量化 (NF4)")
        elif self.config.LOAD_IN_8BIT:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            print("[量化] 启用 8bit 量化")

        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            self.config.MODEL_NAME,
            trust_remote_code=True,
        )

        # 加载模型
        model_kwargs = dict(
            trust_remote_code=True,
            device_map="auto",
        )
        # 兼容新旧版本 transformers：5.x 用 dtype，4.x 用 torch_dtype
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        import transformers
        if int(transformers.__version__.split(".")[0]) >= 5:
            model_kwargs["dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = dtype
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.config.MODEL_NAME,
            **model_kwargs,
        )
        self.model.eval()
        print("[模型] 加载完成 ✓")

    def _build_messages(self, text: str, image: Image.Image = None,
                        history: list = None) -> list:
        """
        构建 Qwen-VL 格式的消息列表
        history: 历史对话列表，每个元素是 {"role": "user"/"assistant", "content": "..."}
        """
        messages = []

        # 系统提示
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": self.config.SYSTEM_PROMPT}]
        })

        # 历史对话
        if history:
            for turn in history:
                role = turn["role"]
                content = turn["content"]
                if isinstance(content, str):
                    messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    })
                else:
                    messages.append({"role": role, "content": content})

        # 当前用户输入（图文）
        user_content = []
        if image is not None:
            item = {"type": "image", "image": image}
            # 限制视觉 token 数（patch 数），避免显存峰值过高导致进程被杀。
            # 详见 config.VLConfig 中 IMAGE_MAX_PIXELS 的注释与换算表。
            max_px = getattr(self.config, "IMAGE_MAX_PIXELS", 0)
            min_px = getattr(self.config, "IMAGE_MIN_PIXELS", 0)
            if max_px:
                item["max_pixels"] = max_px
            if min_px:
                item["min_pixels"] = min_px
            user_content.append(item)
        user_content.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": user_content})

        return messages

    def generate(self, text: str, image: Image.Image = None,
                 history: list = None) -> str:
        """
        核心推理方法（线程安全：内部串行化）

        Args:
            text: 用户文字输入
            image: PIL Image 或 None
            history: 历史对话列表
        Returns:
            模型回复文本
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load()")

        with self._infer_lock:
            self._busy.set()
            t0 = time.time()
            try:
                return self._generate_locked(text, image, history)
            finally:
                self.last_infer_seconds = time.time() - t0
                self._busy.clear()

    def _generate_locked(self, text: str, image: Image.Image = None,
                         history: list = None) -> str:
        """真正的推理实现，必须在持有 _infer_lock 时调用"""
        messages = self._build_messages(text, image, history)

        # 应用聊天模板
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)

        # 构建输入
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 推理生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                do_sample=self.config.DO_SAMPLE,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P,
            )

        # 截取生成部分（去掉输入prompt）
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()

    def unload(self):
        """卸载模型，释放显存（等待正在进行的推理结束，避免半途销毁模型）"""
        with self._infer_lock:
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("[模型] 已卸载，显存已释放")
