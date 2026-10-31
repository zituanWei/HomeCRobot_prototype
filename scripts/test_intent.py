"""快速测试意图识别"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.intent_recognizer import IntentRecognizer, Intent

tests = [
    ("记住我的脸，我叫小明", Intent.REGISTER_FACE, {"name": "小明"}),
    ("记住我的脸", Intent.REGISTER_FACE, {}),
    ("你都认识谁", Intent.LIST_FACES, {}),
    ("忘记我吧", Intent.DELETE_FACE, {}),
    ("打开摄像头看看", Intent.CAMERA_PREVIEW, {}),
    ("清空对话，重新开始", Intent.CLEAR_HISTORY, {}),
    ("我们之前聊了什么", Intent.SHOW_HISTORY, {}),
    ("你会什么", Intent.HELP, {}),
    ("再见，下次聊", Intent.EXIT, {}),
    ("我现在状态怎么样", Intent.NORMAL_CHAT, {}),
    ("这是什么东西", Intent.NORMAL_CHAT, {}),
    ("今天天气怎么样", Intent.NORMAL_CHAT, {}),
    ("/register 小红", Intent.REGISTER_FACE, {"name": "小红"}),
    ("/faces", Intent.LIST_FACES, {}),
    ("监控怎么样", Intent.MONITOR_STATUS, {}),
]

print("意图识别测试:")
print("-" * 60)
all_pass = True
for text, expected_intent, expected_params in tests:
    intent, params = IntentRecognizer.recognize(text)
    intent_match = intent == expected_intent
    params_match = all(params.get(k) == v for k, v in expected_params.items())
    passed = intent_match and params_match
    if not passed:
        all_pass = False
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {text:<20} → {intent:<16} params={params}")

print("-" * 60)
print(f"全部通过: {all_pass}")
