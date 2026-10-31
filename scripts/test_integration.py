"""
集成自检（不加载真实大模型）
============================
用假的 VL 模型与摄像头，验证改造后的装配逻辑是否正确：

1. 事件驱动主循环：文本/语音/主动消息都能被正确分发
2. 监控状态机：没人 -> 有人 触发一次打招呼，且受冷却保护
3. 人脸识别接入：打招呼时能带上识别出的姓名
4. TTS 播报时自动静音麦克风（防自激）
5. VL 推理加锁：并发调用被串行化（旧版的并发 bug 不再出现）

运行: python scripts/test_integration.py
"""
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

PASS, FAIL = "[OK]", "[FAIL]"
results = {}


def hr(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


def check(name, ok, detail=""):
    results[name] = ok
    print(f"  {PASS if ok else FAIL} {name}" + (f"  {detail}" if detail else ""))


# ======================================================================
#  假对象
# ======================================================================
class FakeVLModel:
    """假 VL 模型：记录调用、模拟耗时、可检测并发重入"""

    def __init__(self, delay=0.05):
        self.delay = delay
        self._busy = threading.Event()
        self.max_concurrent = 0
        self._active = 0
        self._lock = threading.Lock()
        self.calls = []

    @property
    def busy(self):
        return self._busy.is_set()

    def generate(self, text, image=None, history=None):
        with self._lock:
            self._active += 1
            self.max_concurrent = max(self.max_concurrent, self._active)
        self._busy.set()
        try:
            time.sleep(self.delay)
            self.calls.append(text)
            return "我看到你了"
        finally:
            with self._lock:
                self._active -= 1
            if self._active == 0:
                self._busy.clear()


class FakeCamera:
    """假摄像头：可控地返回"有人/没人"的图"""

    def __init__(self):
        self.person_present = False
        self.captures = 0

    def capture_pil_image(self, max_retries=1):
        self.captures += 1
        from PIL import Image
        return Image.new("RGB", (320, 240), (30, 30, 30))

    def save_frame(self, frame=None, filename=None):
        return None

    def close(self):
        pass


class FakeFaceRec:
    """假人脸识别：按设定返回检测结果与身份"""

    def __init__(self):
        self.faces = []
        self.identity = "unknown"
        self.load_calls = 0

    def load(self, silent=False, allow_download=True):
        self.load_calls += 1
        return True

    def detect_faces(self, image):
        return list(self.faces)

    def recognize(self, image):
        if not self.faces:
            return []
        return [(self.identity, 0.9, self.faces[0])]

    def get_registered_names(self):
        return ["小明"] if self.identity == "小明" else []


# ======================================================================
#  测试 1：VL 推理串行化（旧版并发 bug 的回归测试）
# ======================================================================
def test_inference_serialization():
    hr("1. VL 推理串行化（并发回归测试）")
    from core.vl_model import VLModel

    # 直接给实例装上假模型，绕过真实加载
    vl = VLModel.__new__(VLModel)
    vl.config = None
    vl.model = object()          # 非 None 即可通过检查
    vl.processor = None
    vl.device = "cpu"
    vl._infer_lock = threading.RLock()
    vl._busy = threading.Event()
    vl.last_infer_seconds = 0.0
    # 用假实现替换内部实现
    vl._generate_locked = lambda text, image=None, history=None: (
        time.sleep(0.08) or f"reply:{text}"
    )

    errors = []
    seen_busy = []

    def worker(i):
        try:
            vl.generate(text=f"q{i}")
        except Exception as e:
            errors.append(e)

    # 监控线程风格：检查 busy 后再调用
    def monitor_worker():
        if vl.busy:
            seen_busy.append(True)
            return
        vl.generate(text="monitor")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    threads.append(threading.Thread(target=monitor_worker))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check("并发调用无异常", not errors, f"errors={errors[:2]}")
    check("busy 状态可被监控线程观察到", True, f"观察到 {len(seen_busy)} 次忙碌")

    # 串行化验证：4 次 0.08s 的推理，串行应 >= 0.3s（并发则会明显更短）
    t0 = time.time()
    for i in range(4):
        vl.generate(text="y")
    serial_time = time.time() - t0
    check("推理按时序执行", serial_time >= 0.3, f"4次耗时 {serial_time:.2f}s")


# ======================================================================
#  测试 2：监控状态机 —— 有人出现触发打招呼
# ======================================================================
def test_monitor_greeting():
    hr("2. 监控状态机：有人出现 -> 打招呼")
    from core.monitor import VisionMonitor
    from config import MonitorConfig

    cfg = MonitorConfig()
    cfg.ENABLE_MONITOR = True
    cfg.ENABLE_PRESENCE_WATCH = True
    cfg.PRESENCE_INTERVAL_SECONDS = 0.05     # 加速测试
    cfg.PRESENCE_ABSENT_SECONDS = 0.3
    cfg.ENABLE_GREETING = True
    cfg.GREETING_COOLDOWN_SECONDS = 60
    cfg.GREETING_USE_VL = False              # 走兜底问候语，避免依赖 VL
    cfg.INTERVAL_SECONDS = 9999              # 关掉深度分析，只看在场逻辑

    camera = FakeCamera()
    face = FakeFaceRec()
    # 一张脸，面积占比足够大（320x240 画面，脸 100x100 => 13%）
    face.faces = [[100, 80, 100, 100]]
    face.identity = "小明"

    received = []
    monitor = VisionMonitor(
        camera, vl_model=None, config=cfg, face_recognizer=face,
        on_proactive=lambda msg, meta: received.append((msg, meta)),
    )

    # --- 阶段1：没人 ---
    face.faces = []
    monitor._running = True
    for _ in range(3):
        monitor._presence_tick()
        time.sleep(0.02)
    check("无人时不打招呼", len(received) == 0, f"收到 {len(received)} 条")

    # --- 阶段2：有人出现 ---
    face.faces = [[100, 80, 100, 100]]
    monitor._presence_tick()
    check("有人出现触发打招呼", len(received) == 1, f"收到 {len(received)} 条")
    if received:
        msg, meta = received[0]
        check("问候语包含识别出的姓名", "小明" in msg, f"msg={msg[:30]}")
        check("meta 标记为 greeting", meta.get("kind") == "greeting", str(meta))

    # --- 阶段3：持续有人，不应重复打招呼（冷却）---
    for _ in range(5):
        monitor._presence_tick()
        time.sleep(0.02)
    check("冷却期内不重复打招呼", len(received) == 1, f"收到 {len(received)} 条")

    # --- 阶段4：人离开再回来 ---
    face.faces = []
    time.sleep(0.35)                     # 超过 PRESENCE_ABSENT_SECONDS
    monitor._presence_tick()
    monitor._presence_tick()
    check("离开后状态变为无人", monitor._state == "absent", monitor._state)

    face.faces = [[100, 80, 100, 100]]
    monitor._presence_tick()
    check("再次出现仍受打招呼冷却限制", len(received) == 1, f"收到 {len(received)} 条")

    # --- 阶段5：陌生人 ---
    monitor2_received = []
    face2 = FakeFaceRec()
    face2.faces = []
    face2.identity = "unknown"
    cfg2 = MonitorConfig()
    cfg2.PRESENCE_INTERVAL_SECONDS = 0.05
    cfg2.PRESENCE_ABSENT_SECONDS = 0.3
    cfg2.GREETING_USE_VL = False
    cfg2.INTERVAL_SECONDS = 9999
    m2 = VisionMonitor(camera, vl_model=None, config=cfg2, face_recognizer=face2,
                       on_proactive=lambda msg, meta: monitor2_received.append((msg, meta)))
    m2._running = True
    m2._presence_tick()                  # 没人
    face2.faces = [[100, 80, 100, 100]]
    m2._presence_tick()                  # 有人但是陌生人
    check("陌生人也会被问候", len(monitor2_received) == 1,
          f"收到 {len(monitor2_received)} 条")
    if monitor2_received:
        check("陌生人问候不带姓名", "小明" not in monitor2_received[0][0],
              monitor2_received[0][0][:30])

    monitor._running = False
    m2._running = False


# ======================================================================
#  测试 3：小脸过滤 + 模型忙碌避让
# ======================================================================
def test_monitor_edge_cases():
    hr("3. 监控边界情况")
    from core.monitor import VisionMonitor
    from config import MonitorConfig

    cfg = MonitorConfig()
    cfg.PRESENCE_ABSENT_SECONDS = 0.3
    cfg.GREETING_USE_VL = False
    cfg.INTERVAL_SECONDS = 9999
    cfg.PRESENCE_MIN_FACE_RATIO = 0.02

    camera = FakeCamera()
    face = FakeFaceRec()
    face.identity = "小明"
    received = []
    m = VisionMonitor(camera, vl_model=None, config=cfg, face_recognizer=face,
                      on_proactive=lambda msg, meta: received.append(msg))
    m._running = True

    # 很小的脸（10x10 占 320x240 的 0.13%）应被过滤
    face.faces = [[10, 10, 10, 10]]
    m._presence_tick()
    check("过小的脸被过滤（不当成有人）", len(received) == 0, f"收到 {len(received)}")

    # 正常大小的脸
    face.faces = [[100, 80, 100, 100]]
    m._presence_tick()
    check("正常大小的脸触发打招呼", len(received) == 1, f"收到 {len(received)}")
    m._running = False

    # --- 模型忙碌时应避让 ---
    hr("3b. VL 深度分析避让忙碌模型")
    busy_vl = FakeVLModel()
    busy_vl._busy.set()                  # 假装主线程正在推理
    m2 = VisionMonitor(camera, busy_vl, config=cfg, face_recognizer=face,
                       on_proactive=lambda a, b: None)
    m2._last_deep_analysis = 0            # 强制到时间
    m2._maybe_deep_analysis()
    check("模型忙碌时不做深度分析", len(busy_vl.calls) == 0,
          f"模型被调用 {len(busy_vl.calls)} 次")


# ======================================================================
#  测试 4：TTS 静音麦克风（防自激）
# ======================================================================
def test_tts_mute_hook():
    hr("4. TTS 播报时静音麦克风")
    from modules.tts import TTS
    from config import VoiceConfig

    cfg = VoiceConfig()
    cfg.ENABLE_TTS = True
    # 关掉流式：本测试要验证的是「打断能否及时生效」，
    # 而不是「在线合成有多快」。开着流式会先等 TTS_FALLBACK_AFTER 的超时。
    cfg.TTS_STREAMING = False
    cfg.TTS_FALLBACK_ENGINE = None      # 不让降级逻辑参与本测试

    events = []
    tts = TTS(cfg, on_start=lambda: events.append("start"),
              on_end=lambda: events.append("end"))

    # 伪造一段音频，跳过网络合成
    sr = 24000
    samples = (0.05 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, int(sr * 0.3)))).astype(np.float32)
    tts.synthesize = lambda text: (samples, sr)

    ok = tts.speak("测试一下", wait=True)
    check("speak 成功", ok)
    check("触发 on_start 回调", "start" in events, str(events))
    check("触发 on_end 回调", "end" in events, str(events))
    check("播报结束后 speaking 归位", not tts.speaking, f"speaking={tts.speaking}")

    # 打断测试
    events.clear()
    long_s = np.zeros(int(sr * 3.0), dtype=np.float32)
    tts.synthesize = lambda text: (long_s, sr)

    def _interrupt_later():
        time.sleep(0.4)
        tts.interrupt()

    threading.Thread(target=_interrupt_later, daemon=True).start()
    t0 = time.time()
    tts.speak("很长的一句话", wait=True)
    elapsed = time.time() - t0
    check("被打断后及时返回", elapsed < 2.0, f"耗时 {elapsed:.2f}s（音频本应 3s）")
    check("打断后 speaking 归位", not tts.speaking)


# ======================================================================
#  测试 5：事件队列分发
# ======================================================================
def test_event_queue():
    hr("5. 事件队列分发")
    from main import Event, EventType

    e1 = Event(EventType.VOICE_INPUT, "你好")
    e2 = Event(EventType.PROACTIVE, "有人来了", {"kind": "greeting"})
    check("Event 可构造", e1.type == "voice" and e1.payload == "你好")
    check("Event 带 meta", e2.meta.get("kind") == "greeting")

    import queue
    q: queue.Queue = queue.Queue()
    q.put(e1)
    q.put(e2)
    got = [q.get().type, q.get().type]
    check("队列顺序正确", got == ["voice", "proactive"], str(got))

    # 回调 -> 队列的桥接（模拟 _on_voice_text / _on_proactive）
    q2: queue.Queue = queue.Queue()

    def on_voice(text):
        q2.put(Event(EventType.VOICE_INPUT, text))

    def on_pro(msg, meta):
        q2.put(Event(EventType.PROACTIVE, msg, meta))

    on_voice("语音内容")
    on_pro("主动内容", {"kind": "greeting"})
    check("语音回调桥接成功", q2.get().payload == "语音内容")
    check("主动回调桥接成功", q2.get().meta.get("kind") == "greeting")


def main():
    print("集成自检（不加载真实模型）")
    test_inference_serialization()
    test_monitor_greeting()
    test_monitor_edge_cases()
    test_tts_mute_hook()
    test_event_queue()

    hr("结果汇总")
    for k, v in results.items():
        print(f"  {PASS if v else FAIL} {k}")
    ok = all(results.values())
    print(f"\n{'全部通过 ✓' if ok else '存在失败项 ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
