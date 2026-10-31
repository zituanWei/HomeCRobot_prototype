"""
用户输入过滤
============
语音识别会把三类"不是用户想说的话"混进来，这里统一挡掉：

1. **重复/纠正**：用户说错了重说一遍，两遍都进了对话历史，
   模型看到两条几乎一样的话会困惑（实测表现为回答驴唇不对马嘴）。
2. **语气词**：清嗓常被识别成「嗯」「呃」，被当成一次提问。
3. **无意义短句**：单个标点、空白、纯符号。

只做"明显该丢的"过滤，不猜语义——避免误伤正常对话。
"""
from __future__ import annotations

import re
import time


def _edit_distance(a: str, b: str) -> int:
    """标准编辑距离（内部实现，避免引入额外依赖）"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,        # 删除
                cur[j - 1] + 1,     # 插入
                prev[j - 1] + cost,  # 替换
            ))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    """归一化相似度，0~1（1 表示完全相同）"""
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    dist = _edit_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


# 只由标点/符号组成的内容
_PUNCT_ONLY = re.compile(r"^[\s\W_]+$", re.UNICODE)


class UtteranceFilter:
    """判断一段识别文本该不该交给模型"""

    def __init__(self, config=None, enabled: bool = True):
        self.config = config
        self.enabled = enabled
        self.last_text: str = ""
        self.last_time: float = 0.0
        # 去重时间窗：超过这么久就算用户真的想重复说一遍
        self.dedup_window = float(getattr(config, "DEDUP_WINDOW_SECONDS", 10.0)) if config else 10.0
        self.rejected_count = 0

    def _fillers(self) -> set:
        if self.config is None:
            return set()
        if not getattr(self.config, "DROP_FILLER_WORDS", True):
            return set()
        return set(getattr(self.config, "FILLER_WORDS", []))

    def check(self, text: str) -> tuple[bool, str]:
        """检查是否接受这句话

        Returns: (是否接受, 拒绝原因)；接受时原因为空
        """
        if not self.enabled:
            return True, ""

        t = (text or "").strip()

        # --- 1. 空 / 纯标点 ---
        if not t or _PUNCT_ONLY.match(t):
            self.rejected_count += 1
            return False, "空内容或纯标点"

        # --- 2. 单个语气词（清嗓常被识别成"嗯"）---
        if t in self._fillers():
            self.rejected_count += 1
            return False, f"语气词「{t}」"

        # --- 3. 与上一句高度重复（用户重说/纠正）---
        if getattr(self.config, "DEDUP_REPEAT", True) and self.last_text:
            threshold = float(getattr(self.config, "DEDUP_SIMILARITY", 0.8))
            within_window = (time.time() - self.last_time) <= self.dedup_window
            if within_window:
                sim = similarity(t, self.last_text)
                if sim >= threshold:
                    self.rejected_count += 1
                    return False, f"与上一句重复（相似度 {sim:.2f}）"

        return True, ""

    def accept(self, text: str):
        """记录已接受的这句（供下一轮去重比较）"""
        self.last_text = (text or "").strip()
        self.last_time = time.time()

    def reset(self):
        self.last_text = ""
        self.last_time = 0.0
