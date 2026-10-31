"""快速测试视觉触发逻辑"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.vision_trigger import VisionTrigger

tests = [
    ("泥嚎", None, "问候-应不触发"),
    ("我不在自拍", None, "纠正-应不触发"),
    ("你看到了什么？", None, "强触发-应触发"),
    ("我现在状态怎么样", None, "强触发-应触发"),
    ("1+1等于几", None, "知识-应不触发"),
    ("这是什么东西", None, "强触发-应触发"),
    ("今天天气怎么样", None, "闲聊-应不触发"),
    ("然后呢", [{"role": "user", "content": "/look 你看到了什么"}], "追问-应触发"),
    ("不是这样的", [{"role": "user", "content": "你看我在做什么"}], "纠正-应不触发"),
    ("嗨", None, "问候-应不触发"),
    ("我帅吗", None, "强触发-应触发"),
    ("讲个笑话", None, "闲聊-应不触发"),
]

print("测试结果:")
print("-" * 60)
all_pass = True
for text, hist, desc in tests:
    result, reason = VisionTrigger.should_use_vision(text, hist)
    should_trigger = "应触发" in desc
    passed = (result == should_trigger)
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    label = "视觉" if result else "文字"
    print(f"[{status}] [{label}] {text:<14} | {desc} | {reason}")

print("-" * 60)
print(f"全部通过: {all_pass}")
