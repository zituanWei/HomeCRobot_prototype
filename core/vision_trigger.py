"""
自动视觉触发模块
================
判断用户这句话要不要调用摄像头拍照。

核心原则：宁可少拍，不可乱拍。
纯问候、纠错、闲聊知识一律不拍；但只要涉及**用户自身的外观/状态/行为**，
就必须拍——那类问题不看画面根本无法回答。

历史踩过的坑（都已修）：
  1. "我头上戴着什么" 不触发 —— 词表缺少"戴着"
  2. "我有没有在戴眼镜" 被误判成「否定/纠正: 没有」—— "有没有"是疑问，
     不是否定。旧的 startswith 否定判断过于粗暴。
  3. "今天天气不错" 被误判成「否定/纠正: 不」—— 单个"不"字不该算否定。
"""
import re


class VisionTrigger:
    """自动判断是否需要视觉输入"""

    # ========== 强触发：明确要求看，或必然需要画面 ==========
    STRONG_TRIGGERS = [
        # --- 直接要求看 ---
        "你看", "看看", "看一下", "看我", "看这", "看图", "看照片",
        "看到了什么", "能看见吗", "看得见吗", "瞧一瞧", "瞅瞅",
        # --- 询问当前状态/画面 ---
        "我现在", "我这是在", "我在干", "我在做什么", "我在忙什么",
        "这是什么", "这是啥", "这地方", "这个东西",
        "状态怎么样", "状态如何", "我怎么样", "我还好吗",
        "我看起来", "我看上去",
        # --- 头部 / 面部 ---
        "我头上", "头顶", "戴着", "戴的", "戴了", "我戴",
        "眼镜", "帽子", "口罩", "耳机", "发型", "头发",
        "我脸上", "表情", "脸色", "气色", "妆容",
        # --- 衣着 ---
        "我穿的", "我穿", "我今天穿", "我衣服", "我的衣服",
        "我好看吗", "我帅吗", "我漂亮吗",
        # --- 手 / 持物 ---
        "我手里", "手上", "拿着", "拿的", "我端的",
        # --- 姿态 / 动作 ---
        "我坐姿", "我姿势", "我站姿", "我的姿势", "我坐得",
        "我在不在", "我有没有",
        # --- 明确要求识别/描述 ---
        "识别一下", "描述一下画面", "分析一下画面", "看看我",
        # --- 明确要求拍照 ---
        "摄像头", "拍一张", "拍个照", "截图看看",
    ]

    # ========== 弱触发：需要同时是疑问句才拍照 ==========
    WEAK_TRIGGERS = [
        "这是什么", "那是什么", "这啥", "那啥",
        "怎么回事", "什么情况",
        "谁在", "有人吗",
        "我穿", "我戴", "我拿",
    ]

    # ========== 绝不触发：问候寒暄 ==========
    GREETINGS = [
        "你好", "您好", "泥嚎", "嗨", "哈喽", "hello", "hi", "hey",
        "在吗", "在不在", "在么",
        "早上好", "晚上好", "中午好", "下午好",
        "我来了", "我回来",
    ]

    # ========== 绝不触发：用户在纠正模型 ==========
    # 注意：只收"完整的纠正语"，不要收单个"不""没"——
    # 那会把"今天天气不错""我有没有戴眼镜"这类正常句子误判成否定。
    DENIALS = [
        "不是", "不对", "错了", "搞错了", "理解错", "你说错",
        "我没有问你", "我没问你", "别说了", "停一下",
        "我不在", "我没有说",
    ]

    # ========== 绝不触发：纯知识/闲聊 ==========
    NO_VISION_KEYWORDS = [
        "几点", "天气", "日期", "星期", "时间",
        "讲个故事", "唱首歌", "背首诗", "猜个谜语", "笑话",
        "谢谢", "感谢", "再见", "拜拜",
        "1+1", "等于几", "计算", "换算", "数学",
        "翻译", "什么意思", "解释一下",
        "设置", "配置", "命令", "帮助", "功能",
        "吃饭了吗", "吃了吗", "吃没",
        "无聊", "陪我聊", "聊聊天", "说说话",
    ]

    QUESTION_MARKS = ["?", "？", "吗", "呢", "吧", "啊", "嘛", "有没有", "是不是"]

    @classmethod
    def should_use_vision(cls, text: str, history: list = None) -> tuple[bool, str]:
        """
        判断是否需要调用摄像头
        Returns: (是否需要视觉, 触发原因说明)
        """
        text_clean = (text or "").strip()
        text_lower = text_clean.lower()

        # 1. 斜杠命令不处理
        if text_clean.startswith("/"):
            return False, "命令行指令"
        if not text_clean:
            return False, "空输入"

        is_question = any(m in text_clean for m in cls.QUESTION_MARKS)

        # 2. 纯问候寒暄 -> 不触发
        for kw in cls.GREETINGS:
            if text_lower == kw or text_lower.startswith(kw):
                return False, f"问候寒暄: {kw}"

        # 3. 明确的纠正语 -> 不触发
        #    关键：先看是不是疑问句。"我有没有在戴眼镜"含"没有"却是疑问，
        #    绝不能因为它含"没"就判成纠正。
        if not is_question:
            for kw in cls.DENIALS:
                if text_lower.startswith(kw) or (len(text_clean) <= 8 and kw in text_lower):
                    return False, f"否定/纠正: {kw}"

        # 4. 纯知识/闲聊 -> 不触发
        #    同样要在"强触发"之前判断，避免"这是什么意思"被当视觉请求
        for kw in cls.NO_VISION_KEYWORDS:
            if kw in text_lower:
                return False, f"纯知识/闲聊: {kw}"

        # 5. 强触发词 -> 触发
        #    放在闲聊过滤之后，但放在弱触发之前
        for kw in cls.STRONG_TRIGGERS:
            if kw in text_lower:
                return True, f"明确要求视觉: {kw}"

        # 6. 疑问句 + 弱触发词 -> 可能要看
        if is_question:
            for kw in cls.WEAK_TRIGGERS:
                if kw in text_lower:
                    return True, f"疑问句+视觉相关: {kw}"

        # 7. 上下文延续（严格条件）
        if history:
            last_user_input = ""
            for msg in reversed(history):
                if msg.get("role") == "user":
                    last_user_input = msg.get("content", "")
                    break
            last_was_vision = any(kw in last_user_input for kw in
                                  ["你看", "看看", "看一下", "看我", "这是什么",
                                   "我现在", "状态", "拍照", "摄像头", "/look",
                                   "戴着", "穿着", "表情"])
            if last_was_vision and is_question and 3 <= len(text_clean) <= 15:
                if any(f in text_lower for f in ["然后", "还有", "呢", "接着", "继续", "再"]):
                    return True, "视觉话题追问"

        return False, "默认纯文字模式"

    @classmethod
    def get_vision_prompt(cls, text: str) -> str:
        """为视觉对话生成更合适的 prompt"""
        if len(text.strip()) <= 4 and any(kw in text for kw in ["你看", "看看", "看一下"]):
            return "请描述你看到的画面，并关心一下我的状态"
        return text
