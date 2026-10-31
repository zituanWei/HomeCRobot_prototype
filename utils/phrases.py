"""
固定语音文案集中管理
====================
把这些"会反复说"的句子集中在一处，好处：

1. **预合成**：启动时能一次性拿到全部高频文案，提前合成好放缓存，
   命中时零延迟（edge-tts 在线合成慢，这是唯一能压缩等待的手段）。
2. **避免漏改**：以前同一句问候散落在 monitor.py / main.py 里，
   改一处容易漏另一处。
3. **便于统一语气**：人设调整时只改这一个文件。

注意：只有**完全固定**的句子放在这里。带变量的（如"记住了，你是XX"）
没法预合成，仍在各自位置动态生成。
"""

# ---------------- 主动打招呼 ----------------
# {name} 占位由调用方填充
GREETING_KNOWN = [
    "{name}，你回来啦！我一直在这儿呢，今天感觉怎么样？",
    "{name}，看到你了，今天过得还顺利吗？",
    "{name}，你回来啦，要不要先歇一会儿？",
]

GREETING_UNKNOWN = [
    "你好呀，我看到你了，需要我陪你聊聊天吗？",
    "你好呀，最近过得怎么样？",
    "你好，我看到你了，有什么想聊的吗？",
]

# ---------------- 注册人脸 ----------------
ASK_NAME = "好的，我会记住你的脸。你叫什么名字？"

# ---------------- 认不出时的引导注册 ----------------
# 注意：复述确认那句含具体名字（"你叫XX，对吗"），名字无限多，
# 没法预合成，只能即时合成；这里只收固定句。
GUIDED_ASK = "我好像还不太认识你，能告诉我你叫什么名字吗？"
GUIDED_RETRY_NAME = "抱歉，我没听清，你能再说一次名字吗？"
GUIDED_ASK_AGAIN = "好的，那请再说一次你的名字？"
GUIDED_GIVEUP = "没关系，等你想告诉我的时候再说"

# ---------------- 人脸库状态 ----------------
NO_FACES_KNOWN = "我还不认识任何人，你可以说记住我的脸，让我认识你"
FACES_CLEARED = "好的，我把认识的人都忘掉了"
FACES_ALREADY_EMPTY = "我本来就不认识任何人"
NOT_RECOGNIZED = "我暂时没认出你，你可以说「记住我的脸」让我先认识你"

# ---------------- 对话管理 ----------------
HISTORY_CLEARED = "好的，我们重新开始吧"

# ---------------- 告别 ----------------
FAREWELL = [
    "再见，我在这儿等你回来",
    "好的，那你先忙，我随时都在",
    "再见，记得好好照顾自己",
]

# ---------------- 主动关怀 ----------------
CARE_INACTIVE = "你已经坐了一会儿了，要不要起来活动活动？喝杯水或者走动一下对身体好哦~"

CARE_ABNORMAL = [
    "我注意到你可能不太舒服，你还好吗？需要我帮你联系家人吗？",
    "你看起来状态不太好，需要我帮忙吗？",
]


def all_preload_phrases() -> list[str]:
    """返回可预合成的全部固定文案

    只收**不含占位符**的句子——带 {name} 的问候依赖具体人名，
    没法逐个预合成（可能的人名是无限的）。
    """
    phrases: list[str] = []
    for group in (GREETING_UNKNOWN, FAREWELL, CARE_ABNORMAL):
        phrases.extend(group)

    phrases.extend([
        ASK_NAME,
        GUIDED_ASK,
        GUIDED_RETRY_NAME,
        GUIDED_ASK_AGAIN,
        GUIDED_GIVEUP,
        NO_FACES_KNOWN,
        FACES_CLEARED,
        FACES_ALREADY_EMPTY,
        NOT_RECOGNIZED,
        HISTORY_CLEARED,
        CARE_INACTIVE,
    ])
    # 去重保序
    seen = set()
    out = []
    for p in phrases:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out
