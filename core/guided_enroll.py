"""
引导注册：认不出人时主动询问并两步确认
======================================
场景：用户说了身份相关的话（"我是谁""你认识我吗"），但人脸识别没认出他。

流程（两步确认）：
    系统：我好像还不太认识你，能告诉我你叫什么吗？
    用户：我叫武凉烨
    系统：好的，你叫「武凉烨」，是这个吗？
    用户：对
    系统：（拍照）记住了，你是武凉烨！

为什么要两步确认，而不是"说个名字就存"：
  1) **语音识别名字经常出错**。实测「武凉烨」被识别成「五粮液」、
     「小护」被识别成「小户」。一步到位会把错名字写进人脸库，
     而且越积越多。
  2) **防冒充**。如果"说个名字就注册"，任何人都能对着摄像头说
     "我是你爸爸"，系统就把他存成家人了。之后打招呼会叫他爸爸，
     真正的家人反而被当成陌生人。
  3) 注册必须**拍照**——不能只凭一句话就把一张陌生的脸存进去。

状态机设计：
    IDLE ──(身份相关对话 + 没认出 + 库非空)──> AWAIT_NAME
    AWAIT_NAME ──(用户说了名字)──> CONFIRM
    AWAIT_NAME ──(超时/用户说别的)──> IDLE
    CONFIRM ──(用户确认)──> 回调执行注册 ──> IDLE
    CONFIRM ──(用户否认)──> AWAIT_NAME（重说，超过次数则放弃）
"""
from __future__ import annotations

import re
import time


class EnrollState:
    IDLE = "idle"
    AWAIT_NAME = "await_name"
    CONFIRM = "confirm"


# 从"我叫XXX""我是XXX"里抽名字
_NAME_PATTERNS = [
    r"我叫([\u4e00-\u9fa5a-zA-Z]{1,12})",
    r"我的名字是([\u4e00-\u9fa5a-zA-Z]{1,12})",
    r"名字叫([\u4e00-\u9fa5a-zA-Z]{1,12})",
    r"我是([\u4e00-\u9fa5a-zA-Z]{1,12})",
    r"叫我([\u4e00-\u9fa5a-zA-Z]{1,12})",
    r"^([\u4e00-\u9fa5a-zA-Z]{1,12})$",      # 直接回一个名字
]

# 确认词
_YES = ["对", "是", "是的", "对的", "没错", "正确", "是我", "好", "yes", "yeah"]
# 否认词
# ⚠️ 注意顺序陷阱："没错"里含"错"，若先判否认就会被误判成否认。
# 因此 _handle_confirm 必须先判"明确确认"，再判否认（见 _CONFIRM_FIRST）。
_NO = ["不是", "不对", "弄错", "搞错", "重说", "重新", "不", "no", "wrong"]
# 一定表示"确认"的词，优先于否定词判断
_CONFIRM_FIRST = ["没错", "对的", "是的", "正确", "是我", "yes", "yeah"]

# 需要忽略的、不可能是名字的词
_NOT_NAMES = {"什么", "谁", "哪个", "哪里", "怎么", "为什么", "你", "我", "他",
              "不知道", "忘了", "没有", "再见", "谢谢",
              # 单字语气词：语音识别常把清嗓/思考声识别成这些，不能当名字
              "呃", "嗯", "啊", "哦", "额", "唉", "哎", "呀", "嘛", "唔",
              "对", "是", "好", "不", "错"}
# 单字名字容易被语气词误撞，要求至少 2 个字
# （现实里单字名很少，且用文字 /register 可以强制注册任意名字）
_NAME_MIN_LEN = 2


class GuidedEnroll:
    """认不出人时的引导注册状态机"""

    def __init__(self, config=None, on_ask=None, on_confirm=None,
                 on_enroll=None, on_giveup=None):
        """
        Args:
            config: FaceConfig
            on_ask:      提问回调 (prompt_text) -> None
            on_confirm:  复述确认回调 (name) -> None
            on_enroll:   确认通过后真正执行注册 (name) -> (ok, detail)
            on_giveup:   放弃时回调 (reason) -> None
        """
        self.config = config
        self.on_ask = on_ask
        self.on_confirm = on_confirm
        self.on_enroll = on_enroll
        self.on_giveup = on_giveup

        self.state = EnrollState.IDLE
        self.pending_name: str | None = None
        self.retries = 0
        self._entered_at = 0.0
        # 状态超时：用户半天不回就放弃，避免卡在等待态
        self.timeout = 30.0

    # ------------------------------------------------------------------
    #  对外接口
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """是否正在引导注册流程中"""
        return self.state != EnrollState.IDLE

    def should_offer(self, face_db_count: int) -> bool:
        """判断当前该不该主动询问

        条件（三个都要满足）：
          · 功能已开启
          · 人脸库里已经有人（全新环境不追问，避免对客人突兀）
          · 当前不在引导流程中
        """
        cfg = self.config
        if cfg is None or not getattr(cfg, "ENABLE_GUIDED_ENROLL", False):
            return False
        if getattr(cfg, "GUIDED_ENROLL_REQUIRE_EXISTING", True) and face_db_count <= 0:
            return False
        return self.state == EnrollState.IDLE

    def start(self) -> bool:
        """开始引导：询问名字"""
        self.state = EnrollState.AWAIT_NAME
        self.pending_name = None
        self.retries = 0
        self._entered_at = time.time()
        prompt = "我好像还不太认识你，能告诉我你叫什么名字吗？"
        if self.on_ask:
            self.on_ask(prompt)
        return True

    def handle(self, text: str) -> bool:
        """处理用户的一句话

        Returns: True 表示这句话已被引导流程消费掉（不要当普通对话处理）
        """
        if self.state == EnrollState.IDLE:
            return False

        # 超时保护
        if time.time() - self._entered_at > self.timeout:
            self._giveup("等待超时")
            return False

        text = (text or "").strip()
        if not text:
            return True

        if self.state == EnrollState.AWAIT_NAME:
            return self._handle_name(text)
        if self.state == EnrollState.CONFIRM:
            return self._handle_confirm(text)
        return False

    def cancel(self):
        """外部主动取消（例如用户切换话题、退出）"""
        if self.active:
            self.state = EnrollState.IDLE
            self.pending_name = None
            self.retries = 0

    # ------------------------------------------------------------------
    #  内部
    # ------------------------------------------------------------------
    def _handle_name(self, text: str) -> bool:
        """用户在回答"你叫什么" """
        name = self._extract_name(text)

        if not name:
            # 没听出名字，再问一次
            self.retries += 1
            max_retries = getattr(self.config, "GUIDED_ENROLL_MAX_RETRIES", 2)
            if self.retries > max_retries:
                self._giveup("没能听清名字")
                return False
            if self.on_ask:
                self.on_ask("抱歉，我没听清，你能再说一次名字吗？")
            self._entered_at = time.time()
            return True

        # 拿到名字 -> 进入确认环节
        self.pending_name = name
        self.state = EnrollState.CONFIRM
        self._entered_at = time.time()
        if self.on_confirm:
            self.on_confirm(name)
        return True

    def _handle_confirm(self, text: str) -> bool:
        """用户在回答"你叫 XX，对吗？" """
        low = text.lower().strip()

        # 1) 先判"明确确认"。
        #    必须放在否认之前：因为「没错」里含「错」，
        #    若先跑否认匹配就会被误判成否认，用户点头反而被当成否认。
        if any(k in low for k in _CONFIRM_FIRST):
            return self._do_enroll()

        # 2) 再判否认
        if any(k in low for k in _NO):
            self.retries += 1
            max_retries = getattr(self.config, "GUIDED_ENROLL_MAX_RETRIES", 2)
            if self.retries > max_retries:
                self._giveup("名字多次未确认")
                return False
            self.state = EnrollState.AWAIT_NAME
            self.pending_name = None
            self._entered_at = time.time()
            if self.on_ask:
                self.on_ask("好的，那请再说一次你的名字？")
            return True

        # 3) 一般确认词（对/是/好）
        if any(k in low for k in _YES):
            return self._do_enroll()

        # 4) 既不是确认也不是否认：可能直接纠正成新名字
        new_name = self._extract_name(text)
        if new_name and new_name != self.pending_name:
            self.pending_name = new_name
            self._entered_at = time.time()
            if self.on_confirm:
                self.on_confirm(new_name)
            return True

        # 5) 听不懂 -> 再复述一次
        if self.on_confirm:
            self.on_confirm(self.pending_name or "")
        self._entered_at = time.time()
        return True

    def _do_enroll(self) -> bool:
        """执行注册并复位"""
        name = self.pending_name
        self.state = EnrollState.IDLE
        self.pending_name = None
        self.retries = 0
        if name and self.on_enroll:
            self.on_enroll(name)
        return True

    @staticmethod
    def _extract_name(text: str) -> str | None:
        """从用户回答里抽出名字

        过滤规则（都是实测踩出来的）：
          · 单字不认：语音识别常把清嗓/思考声识别成「呃」「嗯」，
            单字正则会把它们当成名字。（单字名很少见，需要时用
            文字命令 /register 注册即可。）
          · 常见疑问词/语气词不认：什么、谁、不知道、呃、嗯…
        """
        t = text.strip().rstrip("。！？.!?，,")
        if not t:
            return None
        for pat in _NAME_PATTERNS:
            m = re.search(pat, t)
            if not m:
                continue
            name = m.group(1).strip()
            if not name:
                continue
            if name in _NOT_NAMES:
                continue
            # 单字（含单个字母）不作为名字
            if len(name) < _NAME_MIN_LEN:
                continue
            if len(name) > 12:
                continue
            return name
        return None

    def _giveup(self, reason: str):
        self.state = EnrollState.IDLE
        self.pending_name = None
        self.retries = 0
        if self.on_giveup:
            try:
                self.on_giveup(reason)
            except Exception:
                pass
