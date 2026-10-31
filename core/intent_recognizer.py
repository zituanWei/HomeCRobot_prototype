"""
自然语言意图识别模块
功能：将用户的日常对话转化为系统命令意图
用户不需要记斜杠命令，直接说人话即可

支持的意图：
- register_face: 注册人脸（记住我的脸）
- delete_face: 删除人脸（忘记我吧）
- list_faces: 查看已注册人员（你认识谁）
- camera_preview: 摄像头预览（打开摄像头）
- clear_history: 清空对话（重新开始）
- show_history: 查看历史（我们聊了什么）
- monitor_status: 查看监控状态（监控怎么样）
- help: 帮助（你会什么）
- exit: 退出（再见）
- normal_chat: 普通对话（默认）
"""
import re
from typing import Optional


class Intent:
    """意图类型常量"""
    REGISTER_FACE = "register_face"
    DELETE_FACE = "delete_face"
    LIST_FACES = "list_faces"
    LIST_PHOTOS = "list_photos"
    CLEAR_FACES = "clear_faces"
    WHO_AM_I = "who_am_i"
    CAMERA_PREVIEW = "camera_preview"
    CLEAR_HISTORY = "clear_history"
    SHOW_HISTORY = "show_history"
    MONITOR_STATUS = "monitor_status"
    HELP = "help"
    EXIT = "exit"
    NORMAL_CHAT = "normal_chat"


class IntentRecognizer:
    """自然语言意图识别器"""

    # ========== 意图定义：(意图类型, 触发关键词列表, 优先级) ==========
    # 优先级数字越小越先匹配
    INTENT_PATTERNS = [
        # --- 人脸相关 ---
        (Intent.REGISTER_FACE, [
            "记住我的脸", "记住我", "认识我", "认识一下我",
            "注册我的脸", "注册一下我", "存一下我的脸", "存我的脸",
            "记下我", "记下我的脸", "记一下我长什么样", "记住我长什么样",
            "把我的脸存下来", "录入我的脸", "添加我的脸",
            "我叫.*记住", "我叫.*认识", "记住.*我叫", "认识.*我叫",
        ], 1),
        (Intent.DELETE_FACE, [
            "忘记我", "忘了我", "忘掉我", "不要记住我",
            "删掉我的脸", "删除我的脸", "清除我的脸", "移除我的脸",
            "把我从人脸库删掉", "别记住我了",
        ], 1),
        (Intent.LIST_FACES, [
            "你认识谁", "你都认识谁", "你记得谁", "你都记得谁",
            "人脸库", "注册了哪些人", "有哪些人", "都有谁",
            "你认识哪些人", "名单", "认不认识我",
        ], 2),

        # --- 清空人脸库（旧版缺失，导致「把所有人删除」被当成普通聊天）---
        (Intent.CLEAR_FACES, [
            "清空人脸库", "清空人脸", "删除所有人脸", "把所有人删除",
            "把所有人删掉", "忘记所有人", "忘掉所有人", "清空所有人",
            "把所有脸删掉", "删掉全部人脸", "清除人脸库", "重置人脸库",
        ], 1),

        # --- 询问自己的身份（问"我是谁"时不该被当成自我介绍）---
        (Intent.WHO_AM_I, [
            "我是谁", "你知道我是谁", "认得我吗", "你认得我吗",
            "认识我吗", "你认识我吗", "认得出我吗", "你还记得我吗",
            "看出来我是谁",
        ], 1),

        # --- 查看某人存了几张照片 ---
        (Intent.LIST_PHOTOS, [
            "存了几张", "几张照片", "照片列表", "有几张",
        ], 3),

        # --- 摄像头相关 ---
        (Intent.CAMERA_PREVIEW, [
            "打开摄像头", "看看摄像头", "摄像头画面", "实时画面",
            "看看实时", "打开预览", "摄像头预览",
        ], 2),

        # --- 对话管理 ---
        (Intent.CLEAR_HISTORY, [
            "清空对话", "清空历史", "重新开始", "从头开始",
            "忘掉之前的对话", "忘记我们聊过什么", "新的对话",
            "重置对话", "清除记录",
        ], 1),
        (Intent.SHOW_HISTORY, [
            "我们聊了什么", "历史记录", "聊天记录", "之前聊了什么",
            "看看历史", "对话历史",
        ], 2),

        # --- 系统相关 ---
        (Intent.MONITOR_STATUS, [
            "监控怎么样", "监控状态", "后台监控", "监控在运行吗",
            "监控正常吗",
        ], 2),
        (Intent.HELP, [
            "你会什么", "你能做什么", "你有什么功能", "帮助",
            "怎么用", "使用说明", "你能帮我做什么", "功能列表",
        ], 2),
        (Intent.EXIT, [
            "再见", "拜拜", "退出", "关闭", "下次聊",
            "我走了", "先这样",
        ], 1),
    ]

    # 姓名提取正则（匹配"我叫XX""我的名字是XX""我是XX"）
    NAME_PATTERNS = [
        r"我叫([\u4e00-\u9fa5a-zA-Z0-9]{1,10})",
        r"我的名字是([\u4e00-\u9fa5a-zA-Z0-9]{1,10})",
        r"我是([\u4e00-\u9fa5a-zA-Z0-9]{1,10})",
        r"名字叫([\u4e00-\u9fa5a-zA-Z0-9]{1,10})",
        r"叫我([\u4e00-\u9fa5a-zA-Z0-9]{1,10})",
    ]

    @classmethod
    def recognize(cls, text: str) -> tuple[str, dict]:
        """
        识别用户输入的意图
        Args:
            text: 用户输入文本
        Returns:
            (意图类型, 提取的参数字典)
        """
        text_clean = text.strip()
        text_lower = text_clean.lower()

        # 斜杠命令直接透传（兼容老用户）
        if text_clean.startswith("/"):
            return cls._parse_slash_command(text_clean)

        # 按优先级匹配意图
        for intent_type, keywords, priority in sorted(cls.INTENT_PATTERNS, key=lambda x: x[2]):
            for kw in keywords:
                if "*" in kw:
                    # 正则模式
                    if re.search(kw, text_lower):
                        params = cls._extract_params(intent_type, text_clean)
                        return intent_type, params
                else:
                    # 普通关键词
                    if kw in text_lower:
                        params = cls._extract_params(intent_type, text_clean)
                        return intent_type, params

        # 默认：普通对话
        return Intent.NORMAL_CHAT, {}

    @classmethod
    def _parse_slash_command(cls, text: str) -> tuple[str, dict]:
        """解析斜杠命令（兼容旧用法）"""
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        mapping = {
            "/register": Intent.REGISTER_FACE,
            "/forget": Intent.DELETE_FACE,
            "/faces": Intent.LIST_FACES,
            "/photos": Intent.LIST_PHOTOS,
            "/clearfaces": Intent.CLEAR_FACES,
            "/forgetall": Intent.CLEAR_FACES,
            "/whoami": Intent.WHO_AM_I,
            "/camera": Intent.CAMERA_PREVIEW,
            "/clear": Intent.CLEAR_HISTORY,
            "/history": Intent.SHOW_HISTORY,
            "/monitor": Intent.MONITOR_STATUS,
            "/help": Intent.HELP,
            "/exit": Intent.EXIT,
            "/quit": Intent.EXIT,
            "/q": Intent.EXIT,
        }

        intent = mapping.get(cmd, Intent.NORMAL_CHAT)
        params = {}
        if intent == Intent.REGISTER_FACE and arg:
            params["name"] = arg
        if intent == Intent.DELETE_FACE and arg:
            params["name"] = arg
        return intent, params

    @classmethod
    def _extract_params(cls, intent: str, text: str) -> dict:
        """从文本中提取参数"""
        params = {}

        if intent == Intent.REGISTER_FACE:
            # 尝试提取姓名
            name = cls._extract_name(text)
            if name:
                params["name"] = name

        if intent == Intent.DELETE_FACE:
            name = cls._extract_name(text)
            if name:
                params["name"] = name

        return params

    @classmethod
    def _extract_name(cls, text: str) -> Optional[str]:
        """从文本中提取人名"""
        for pattern in cls.NAME_PATTERNS:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                # 过滤掉常见的非名字词
                if name not in ("谁", "什么", "哪个", "哪里", "怎么", "为什么"):
                    return name
        return None

    @classmethod
    def get_examples(cls) -> list:
        """获取自然语言示例（用于帮助展示）"""
        return [
            ("记住我的脸，我叫小明", "注册人脸（可多次注册提高准确率）"),
            ("你都认识谁？", "查看已注册人员和照片数"),
            ("我是谁？", "拍照认出你是谁"),
            ("忘记我吧", "删除人脸记录"),
            ("把所有人删除", "清空整个人脸库"),
            ("打开摄像头看看", "摄像头预览"),
            ("我现在状态怎么样？", "自动拍照+视觉分析"),
            ("这是什么东西？", "自动拍照+物品识别"),
            ("清空对话，重新开始", "清空历史"),
            ("我们之前聊了什么？", "查看历史"),
            ("你会什么？", "查看帮助"),
            ("再见，下次聊", "退出"),
        ]
