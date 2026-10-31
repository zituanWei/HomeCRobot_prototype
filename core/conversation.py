"""
对话管理模块
负责：维护对话历史、上下文截断、历史持久化
"""
import json
from pathlib import Path
from config import ConversationConfig


class ConversationManager:
    """对话历史管理器"""

    def __init__(self, config: ConversationConfig = None):
        self.config = config or ConversationConfig()
        self.history: list[dict] = []
        self._load_history()

    def _load_history(self):
        """从文件加载历史对话"""
        if self.config.HISTORY_FILE.exists():
            try:
                with open(self.config.HISTORY_FILE, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                print(f"[对话] 已加载 {len(self.history)} 条历史记录")
            except Exception as e:
                print(f"[警告] 历史记录加载失败: {e}")
                self.history = []

    def save_history(self):
        """保存历史对话到文件"""
        try:
            with open(self.config.HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[警告] 历史记录保存失败: {e}")

    def add_user_message(self, text: str):
        """添加用户消息"""
        self.history.append({"role": "user", "content": text})
        self._truncate()

    def add_assistant_message(self, text: str):
        """添加助手消息"""
        self.history.append({"role": "assistant", "content": text})
        self._truncate()
        self.save_history()

    def _truncate(self):
        """截断历史，保留最近 N 轮（一轮=用户+助手各一条）"""
        max_messages = self.config.MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_history_for_model(self) -> list:
        """
        获取传给模型的历史格式
        注意：系统提示词由 VLModel 单独添加，这里只返回对话历史
        """
        return self.history.copy()

    def clear(self):
        """清空对话历史"""
        self.history = []
        self.save_history()
        print("[对话] 历史已清空")

    def get_recent_summary(self, n: int = 5) -> str:
        """获取最近 n 条对话的摘要（用于调试显示）"""
        recent = self.history[-n:]
        lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "小护"
            content = msg["content"] if isinstance(msg["content"], str) else "[图文]"
            lines.append(f"{role}: {content[:80]}")
        return "\n".join(lines)
