"""
后台视觉监控模块（实时在场感知 + 主动打招呼）
================================================
相比旧版的重构要点：

1. **从"每分钟问一次大模型"改为"高频轻量检测 + 按需大模型"**
   旧版每 INTERVAL_SECONDS 秒调一次 Qwen2.5-VL，一次推理要好几秒，
   根本做不到"有人路过就打招呼"。
   现在主循环跑的是 YuNet 人脸检测（毫秒级、不占 GPU），
   只有状态发生变化（没人→有人）时才做身份识别与打招呼。

2. **接入人脸识别，能认出是谁**
   旧版 monitor 完全没拿 face_rec，所以永远只会说"有人"。
   现在"有人出现"时做一次人脸识别，认识的人名字会进入问候语。

3. **与主对话线程共享 GPU，必须避让**
   每次要调 VL 之前检查 vl_model.busy，主线程正在说话就让路，
   避免两个线程同时进模型（那是旧版的并发 bug）。

4. **打招呼有冷却与去抖**，避免对着同一个人反复问候。

使用方式：
    monitor = VisionMonitor(camera, vl_model, config=config,
                            face_recognizer=face_rec,
                            on_proactive=my_callback)
    monitor.start()
"""
import threading
import time
import queue
from typing import Callable, Optional

from config import MonitorConfig

# 在场状态
STATE_ABSENT = "absent"        # 画面里没人
STATE_PRESENT = "present"      # 有人（本轮刚出现，待打招呼）


class VisionMonitor:
    """后台视觉监控线程：在场感知 + 身份识别 + 主动关怀"""

    def __init__(self, camera, vl_model, config: MonitorConfig = None,
                 face_recognizer=None,
                 on_proactive: Optional[Callable[[str], None]] = None):
        """
        Args:
            camera: CameraCapture
            vl_model: VLModel（用于深度分析，可为 None 则只做在场检测）
            config: MonitorConfig
            face_recognizer: FaceRecognizer（用于识别身份）
            on_proactive: 主动消息回调 (text, meta) -> None
                          meta 形如 {"kind": "greeting", "name": "小明"}
        """
        self.camera = camera
        self.vl_model = vl_model
        self.config = config or MonitorConfig()
        self.face_rec = face_recognizer
        self.on_proactive = on_proactive

        # 兼容旧接口：没有回调时退回内部队列
        self.message_queue = queue.Queue(maxsize=10)

        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # --- 在场状态 ---
        self._present = False                 # 当前是否检测到人
        self._last_seen_ts = 0.0              # 最近一次检测到人的时间
        self._absent_since = time.time()      # 从什么时候开始没人
        self._state = STATE_ABSENT
        self._last_known_name = "unknown"     # 最近一次认出的人名（供主对话复用）

        # --- 打招呼冷却（按人区分）---
        self._greet_last: dict[str, float] = {}

        # --- 久坐 / 消息冷却 ---
        self._last_position = None
        self._last_active_time = time.time()
        self._last_proactive_time = 0.0
        self._consecutive_no_person = 0
        self._last_deep_analysis = 0.0
        self._abnormal_streak = 0        # 连续判定异常的次数（用于去抖）

        # 统计（供 /monitor 展示）
        self.stats = {
            "在场检测次数": 0,
            "打招呼次数": 0,
            "深度分析次数": 0,
            "识别成功次数": 0,
        }

        # 最近一次在场检测的快照，供主对话复用（避免重复拍照/识别）。
        # 主对话问"我戴着什么"时，通常刚发生过一次在场检测，
        # 直接用它比再拍一张更省事，也避免两处同时抢摄像头。
        self.last_snapshot: dict = {}

        self._analysis_prompt = (
            "用一句话回答：画面中有人吗？如果有人，他在做什么？"
            "是否有异常情况（如摔倒、躺着不动、看起来不舒服）？"
            "只回答事实，不要寒暄。"
        )

    # ==================================================================
    #  生命周期
    # ==================================================================
    def start(self):
        if self._running:
            return
        if self.camera is None:
            print("[监控] 摄像头未就绪，监控未启动")
            return
        if not self.config.ENABLE_MONITOR:
            print("[监控] 配置中已关闭监控")
            return
        self._running = True
        self._absent_since = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="VisionMonitor")
        self._thread.start()
        mode = "实时在场检测" if self.config.ENABLE_PRESENCE_WATCH else "定时深度分析"
        print(f"[监控] 已启动（{mode}，在场检测每 {self.config.PRESENCE_INTERVAL_SECONDS}s，"
              f"深度分析每 {self.config.INTERVAL_SECONDS}s）")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[监控] 已停止")

    def has_message(self) -> bool:
        return not self.message_queue.empty()

    def get_message(self, timeout: float = 0) -> Optional[str]:
        try:
            return self.message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ==================================================================
    #  主循环
    # ==================================================================
    def _loop(self):
        time.sleep(3)  # 等主程序初始化完成（模型加载等）
        while self._running:
            try:
                if self.config.ENABLE_PRESENCE_WATCH:
                    self._presence_tick()
                self._maybe_deep_analysis()
            except Exception as e:
                print(f"[监控] 异常: {e}")
            time.sleep(self.config.PRESENCE_INTERVAL_SECONDS
                       if self.config.ENABLE_PRESENCE_WATCH else 1.0)

    # ==================================================================
    #  实时在场检测（轻量，高频）
    # ==================================================================
    def _presence_tick(self):
        """检测画面里有没有人，并在"没人→有人"时触发打招呼"""
        if self.face_rec is None:
            return

        image = self.camera.capture_pil_image(max_retries=1)
        if image is None:
            return

        self.stats["在场检测次数"] += 1
        now = time.time()

        # 快速检测：只做检测，不取特征
        faces, frame_area = self._detect_faces(image)

        # 过滤掉太小的脸（远处误检 / 画面里的路人照片）
        # 注意：过滤必须发生在"是否有人"判定之前。
        # 否则小脸会把 _present 置为 True，等真正的脸出现时
        # 因为没有"没人->有人"的状态跳变，就永远不会打招呼。
        if frame_area > 0:
            faces = [f for f in faces
                     if (f[2] * f[3]) / frame_area >= self.config.PRESENCE_MIN_FACE_RATIO]
        else:
            faces = []

        has_person = len(faces) > 0

        # 更新快照，供主对话复用
        with self._lock:
            self.last_snapshot = {
                "ts": now,
                "has_person": has_person,
                "face_count": len(faces),
                "name": self._last_known_name,
                "present": self._present,
            }

        if has_person:
            self._last_seen_ts = now
            self._consecutive_no_person = 0
            if not self._present:
                # ===== 状态变化：没人 → 有人 =====
                self._present = True
                self._state = STATE_PRESENT
                away = now - self._absent_since
                if away > self.config.PRESENCE_ABSENT_SECONDS:
                    print(f"[监控] 检测到有人出现（此前空了 {int(away)}s）")
                else:
                    print("[监控] 检测到有人出现")
                self._handle_person_appeared(image, faces, frame_area, now)
        else:
            if self._present:
                # 可能只是检测抖动，等连续没人超过阈值才判定离开
                if now - self._last_seen_ts >= self.config.PRESENCE_ABSENT_SECONDS:
                    self._present = False
                    self._state = STATE_ABSENT
                    self._absent_since = now
                    self._last_position = None
                    print("[监控] 人已离开")


    def _detect_faces(self, image):
        """返回 (人脸列表, 画面面积)；不触发模型下载"""
        try:
            import numpy as np
            import cv2
            if not self.face_rec.load(silent=True, allow_download=False):
                return [], 0
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            faces = self.face_rec.detect_faces(frame)
            return faces, frame.shape[0] * frame.shape[1]
        except Exception as e:
            print(f"[监控] 人脸检测失败: {e}")
            return [], 0

    # ==================================================================
    #  有人出现：识身份 -> 打招呼
    # ==================================================================
    def _handle_person_appeared(self, image, faces, frame_area, now):
        """人刚出现：识别身份并按冷却策略打招呼

        注意：faces 已在 _presence_tick 里按面积过滤过。
        """
        if not self.config.ENABLE_GREETING:
            return
        if not faces:
            return

        name = "unknown"
        try:
            import numpy as np
            import cv2
            if self.face_rec.load(silent=True, allow_download=False):
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                results = self.face_rec.recognize(frame)
                known = [r[0] for r in results if r[0] != "unknown"]
                if known:
                    name = known[0]
                    self.stats["识别成功次数"] += 1
                    with self._lock:
                        self._last_known_name = name
                    print(f"[监控] 认出这是: {name}")
                elif results:
                    print("[监控] 检测到人脸，但不在已注册名单中")
        except Exception as e:
            print(f"[监控] 身份识别失败: {e}")

        # --- 冷却检查（按人区分）---
        key = name if name != "unknown" else "__unknown__"
        cooldown = (self.config.GREETING_COOLDOWN_SECONDS if name != "unknown"
                    else self.config.GREETING_UNKNOWN_COOLDOWN)
        last = self._greet_last.get(key, 0.0)
        if now - last < cooldown:
            remain = int(cooldown - (now - last))
            print(f"[监控] {name} 在冷却期内（还需 {remain}s），本次不重复打招呼")
            return

        message = self._compose_greeting(name, image)
        if not message:
            return

        self._greet_last[key] = now
        self._last_proactive_time = now
        self.stats["打招呼次数"] += 1
        self._emit(message, {"kind": "greeting", "name": name})

    def _compose_greeting(self, name: str, image) -> str:
        """生成问候语；可选让 VL 看一眼场景，让问候更自然"""
        from utils.phrases import GREETING_KNOWN, GREETING_UNKNOWN
        import random

        # 先给出稳定的兜底问候（保证 VL 不可用时也能打招呼）
        if name != "unknown":
            fallback = random.choice(GREETING_KNOWN).format(name=name)
        else:
            fallback = random.choice(GREETING_UNKNOWN)

        if not self.config.GREETING_USE_VL or self.vl_model is None:
            return fallback

        # 主对话线程正在用模型，这次就用兜底问候，不抢 GPU
        if getattr(self.vl_model, "busy", False):
            print("[监控] 模型正忙，使用简洁问候")
            return fallback

        who = (f"画面中的用户是「{name}」。" if name != "unknown"
               else "画面中是陌生人。")
        prompt = (
            f"{who}请用一句话主动打个招呼，语气亲切自然，像家人一样，"
            f"20字以内，不要描述画面细节。"
            f"注意：你是机器人小护，「我」指你自己，「你」指这位用户，"
            f"不要用「我是{name}」这种说法。"
        )
        try:
            text = self.vl_model.generate(text=prompt, image=image, history=None)
            text = (text or "").strip().replace("\n", " ")
            if 0 < len(text) <= 60:
                self.stats["深度分析次数"] += 1
                return text
        except Exception as e:
            print(f"[监控] 生成问候语失败，改用简洁问候: {e}")
        return fallback

    # ==================================================================
    #  VL 深度分析（低频）：异常检测 / 久坐提醒 / 长时间没人
    # ==================================================================
    def _maybe_deep_analysis(self):
        if self.vl_model is None:
            return
        now = time.time()
        if now - self._last_deep_analysis < self.config.INTERVAL_SECONDS:
            return
        # 主线程正在推理就让路
        if getattr(self.vl_model, "busy", False):
            return
        self._last_deep_analysis = now
        try:
            self._deep_analysis_once()
        except Exception as e:
            print(f"[监控] 深度分析异常: {e}")

    def _deep_analysis_once(self):
        image = self.camera.capture_pil_image(max_retries=2)
        if image is None:
            return

        self.stats["深度分析次数"] += 1
        response = self.vl_model.generate(
            text=self._analysis_prompt,
            image=image,
            history=None,
        )
        response_lower = (response or "").lower()
        now = time.time()

        has_person = not any(kw in response_lower for kw in
                             ["没有人", "没看到人", "无人", "空无一人", "画面中没有人"])

        # --- 情况A：异常（摔倒/不适）---
        # 判定必须排除否定表述，否则「没有看到他摔倒」也会被当成摔倒。
        # 这是实测踩过的坑：模型常写"没有看到摔倒/躺着不动的情况"，
        # 旧的关键词匹配会命中"摔倒"，导致误报"你是不是不舒服"。
        is_abnormal, hit = self._detect_abnormal(response or "")
        if (self.config.ENABLE_ABNORMAL_DETECT and is_abnormal
                and self._can_proactive(now)):
            self._last_proactive_time = now
            self._abnormal_streak += 1
            self._emit(
                f"⚠️ 我注意到画面里可能有不舒服的情况（{hit}）。{response}\n"
                f"你还好吗？需要我帮你联系家人吗？",
                {"kind": "abnormal"},
            )
            return
        else:
            self._abnormal_streak = 0

        # --- 情况B：有人，判断久坐 ---
        if has_person:
            self._last_seen_ts = now
            self._consecutive_no_person = 0
            if self._last_position and self._last_position in (response or ""):
                inactive = now - self._last_active_time
                if inactive > self.config.INACTIVE_THRESHOLD_SECONDS and self._can_proactive(now):
                    self._last_proactive_time = now
                    self._last_active_time = now
                    self._emit(
                        f"你已经安静地待了 {int(inactive/60)} 分钟了，"
                        f"要不要起来活动活动？喝杯水或者走动一下对身体好哦~",
                        {"kind": "inactive"},
                    )
                    return
            else:
                self._last_active_time = now
            self._last_position = (response or "")[:50]

        # --- 情况C：长时间没人 ---
        else:
            self._consecutive_no_person += 1
            if (self._consecutive_no_person >= self.config.NO_PERSON_THRESHOLD_COUNT
                    and self._last_seen_ts and self._can_proactive(now)):
                away_minutes = int((now - self._last_seen_ts) / 60)
                self._last_proactive_time = now
                self._consecutive_no_person = 0
                self._emit(
                    f"我有 {away_minutes} 分钟没看到你了，一切都好吗？"
                    f"如果你在附近，跟我说说话呀~",
                    {"kind": "no_person"},
                )

    # ==================================================================
    #  工具
    # ==================================================================
    # 明确表示"整句无异常"的说法
    _NO_ABNORMAL_PHRASES = [
        "没有异常", "没有发现异常", "未见异常", "未发现异常", "无明显异常",
        "一切正常", "状态正常", "看起来正常", "没有不舒服",
    ]
    # 异常关键词
    _ABNORMAL_KEYS = [
        "摔倒", "跌倒", "倒地", "躺在地上", "躺着不动", "昏迷",
        "痛苦", "明显不适", "看起来不舒服", "表情痛苦", "需要急救",
        "身体不适", "面色不好",
        # 受伤类（之前遗漏，"他头部有伤"会漏报）
        "有伤", "受伤", "流血", "出血", "伤口", "淤青", "肿了",
    ]
    # 句内否定词：同一个分句里出现这些词，就认为该句是在说"没有/不是"异常
    _NEG_WORDS = ["没有", "没", "未", "无", "不是", "并不", "不太", "未见"]

    def _detect_abnormal(self, response: str) -> tuple[bool, str]:
        """判断深度分析结果是否真的表示异常

        为什么要单独抽出来：单纯的关键词匹配会把否定句也算进来。
        实测模型输出「没有看到他摔倒或躺着不动的情况」时，
        旧逻辑命中「摔倒」-> 误报 -> 机器人莫名问"你还好吗"。

        策略（分句判定，比穷举否定短语稳健）：
          1) 先看整段有没有"未见异常"这类明确的整体否定 -> 直接放行
          2) 把回答按标点切成分句，逐个分句判断：
             如果某分句里既有异常关键词、又有否定词 -> 该句是"没有异常"，跳过
             只有分句里出现异常关键词、且**没有**否定词时，才判为异常
        """
        import re
        text = (response or "").strip()
        if not text:
            return False, ""

        # 1) 整段明确否定
        for p in self._NO_ABNORMAL_PHRASES:
            if p in text:
                return False, "（明确否定了异常）"

        # 2) 按标点分句判定
        clauses = re.split(r"[，。；！？,;!?\n]", text)
        for clause in clauses:
            if not clause.strip():
                continue
            hit = next((k for k in self._ABNORMAL_KEYS if k in clause), None)
            if not hit:
                continue
            # 同一分句里有否定词 -> 这句是在说"没有异常"
            if any(n in clause for n in self._NEG_WORDS):
                continue
            return True, hit
        return False, ""

    def _can_proactive(self, now: float) -> bool:
        return (now - self._last_proactive_time) >= self.config.PROACTIVE_COOLDOWN_SECONDS

    def get_snapshot(self, max_age: float = 5.0) -> dict:
        """获取最近一次在场检测的快照（供主对话复用）

        Args:
            max_age: 超过这个秒数就认为快照过期，返回空
        Returns: {ts, has_person, face_count, name, present} 或 {}
        """
        with self._lock:
            snap = dict(self.last_snapshot)
        if not snap:
            return {}
        if time.time() - snap.get("ts", 0) > max_age:
            return {}
        return snap

    def _emit(self, message: str, meta: dict = None):
        """发出主动消息：优先走回调，否则进内部队列"""
        meta = meta or {}
        if self.on_proactive:
            try:
                self.on_proactive(message, meta)
                return
            except Exception as e:
                print(f"[监控] 主动回调异常: {e}")
        try:
            self.message_queue.put_nowait(message)
            print(f"[监控] 主动消息: {message[:60]}")
        except queue.Full:
            pass

    def get_status(self) -> dict:
        now = time.time()
        return {
            "运行中": self._running,
            "当前状态": "有人在场" if self._present else "无人",
            "在场检测间隔": f"{self.config.PRESENCE_INTERVAL_SECONDS}秒",
            "深度分析间隔": f"{self.config.INTERVAL_SECONDS}秒",
            "上次看到人": (
                f"{int((now - self._last_seen_ts)/60)}分钟前"
                if self._last_seen_ts else "从未"
            ),
            "上次活动": f"{int((now - self._last_active_time)/60)}分钟前",
            "打招呼次数": self.stats["打招呼次数"],
            "识别成功次数": self.stats["识别成功次数"],
            "深度分析次数": self.stats["深度分析次数"],
            "待处理消息": self.message_queue.qsize(),
        }
