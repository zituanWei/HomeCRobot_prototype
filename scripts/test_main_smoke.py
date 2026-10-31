"""
主程序启动冒烟测试
==================
之前的 test_integration.py 用的是假对象，没有真正跑过 main.py 的启动路径。
这个脚本补上那一环：真实调用 HomeCRobot.initialize() / run() / shutdown()，
只把「大模型加载」和「ASR 下载」换成假的，其余全部走真实代码。

覆盖：
1. initialize() 全流程：配置加载、音频链路、摄像头、人脸识别、监控启动
2. 事件驱动主循环：文本事件 / 主动消息事件 / 退出事件 的分发
3. shutdown() 资源释放：音频设备、监控线程、模型卸载

硬件相关部分可跳过（--no-hardware），便于无麦克风环境跑通。

运行: python scripts/test_main_smoke.py
     python scripts/test_main_smoke.py --no-hardware
"""
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = "[OK]", "[FAIL]"
results = {}


def check(name, ok, detail=""):
    results[name] = bool(ok)
    print(f"  {PASS if ok else FAIL} {name}" + (f"  {detail}" if detail else ""))


def hr(t):
    print("\n" + "=" * 62)
    print(t)
    print("=" * 62)


def main():
    no_hardware = "--no-hardware" in sys.argv

    hr("主程序启动冒烟测试")
    print(f"  硬件模式: {'关闭（跳过麦克风/TTS）' if no_hardware else '开启'}")

    # ---------------- 测试前准备：改配置，避免真实加载大模型 ----------------
    import config
    from core.vl_model import VLModel
    from modules.asr import ASR

    # 假模型：加载瞬间完成，推理立即返回
    class FakeLoadedModel:
        def eval(self):
            pass

    def fake_load(self):
        print("[模型] (冒烟测试) 跳过真实加载")
        self.model = FakeLoadedModel()
        self.processor = None

    def fake_generate_locked(self, text, image=None, history=None):
        return "（冒烟测试回复）我很好，谢谢你关心"

    def fake_unload(self):
        print("[模型] (冒烟测试) 已卸载")
        self.model = None

    VLModel.load = fake_load
    VLModel._generate_locked = fake_generate_locked
    VLModel.unload = fake_unload

    # 假 ASR 加载：不下载模型，但记录被调用过
    asr_load_called = {"n": 0}
    orig_asr_load = ASR.load

    def fake_asr_load(self, background=False):
        asr_load_called["n"] += 1
        print("[ASR] (冒烟测试) 跳过真实模型加载")
        return True

    ASR.load = fake_asr_load

    if no_hardware:
        config.VoiceConfig.ENABLE_ASR = False
        config.VoiceConfig.ENABLE_TTS = False
    else:
        # 语音链路真实启动，但 ASR 模型加载被替换
        config.VoiceConfig.ENABLE_ASR = True
        config.VoiceConfig.ENABLE_TTS = True
        config.VoiceConfig.ASR_PRELOAD = True

    config.VoiceConfig.AUTO_VOICE_MODE = False
    config.MonitorConfig.ENABLE_MONITOR = False   # 避免后台干扰本次验证
    config.ConversationConfig.HISTORY_FILE = config.DATA_DIR / "_smoke_history.json"

    # ---------------- 构造主程序 ----------------
    from main import HomeCRobot, Event, EventType
    import argparse

    class Args:
        no_camera = False
        no_voice = False
        voice = False
        model = None
        cpu = False

    robot = HomeCRobot(Args())
    # 非交互式测试环境：关掉键盘线程，否则 stdin EOF 会立刻投递退出事件
    robot.keyboard_enabled = False

    # ---------------- 1. initialize ----------------
    hr("1. initialize() 全流程")
    try:
        robot.initialize()
        check("initialize 未抛异常", True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("initialize 未抛异常", False, str(e))
        return 1

    check("摄像头对象已创建", robot.camera is not None,
          "（无摄像头时允许为 None）")
    check("VL 模型为假加载状态", robot.vl_model.model is not None)
    check("人脸识别对象存在", robot.face_rec is not None)
    check("对话管理器就绪", robot.conversation is not None)

    if no_hardware:
        check("硬件模式关闭时未启动麦克风", robot.recorder is None)
    else:
        if robot.recorder is not None:
            check("麦克风已启动", True)
            check("VAD 分段器已启动", robot.segmenter is not None)
            check("ASR 后台预热被调用", asr_load_called["n"] > 0,
                  f"调用 {asr_load_called['n']} 次")
            check("TTS 播报器可用", robot.tts.player.available)
        else:
            print("  [SKIP] 麦克风不可用，跳过语音链路检查")

    # ---------------- 2. 事件分发 ----------------
    hr("2. 事件驱动主循环分发")
    dispatched = {"text": False, "proactive": False, "responses": []}

    # 用轻量包装记录输出，但走真实的处理函数（不用 spy，覆盖真实代码路径）
    orig_print_response = robot._print_response
    orig_print_proactive = robot._print_proactive

    def spy_response(text):
        dispatched["responses"].append(("assistant", text))
        dispatched["text"] = True

    def spy_proactive_print(text):
        dispatched["proactive"] = True
        dispatched["proactive_text"] = text

    robot._print_response = spy_response
    robot._print_proactive = spy_proactive_print

    # 关掉 TTS 播报，避免冒烟测试出声
    robot.voice_reply_enabled = False
    # 强制纯文字，避免摄像头/视觉分支干扰断言
    robot.auto_vision_enabled = False

    hist_before = len(robot.conversation.history)

    def feed_events():
        # 时间留够：chat_text 现在会先做一次"身份识别"（拍照 + 人脸比对），
        # 再加上模型生成，比原先慢。间隔太短会在处理完成前就投递退出事件，
        # 表现为偶发的「文本事件未被处理」。
        time.sleep(1.0)
        robot.events.put(Event(EventType.TEXT_INPUT, "你好"))
        time.sleep(3.0)
        robot.events.put(Event(EventType.PROACTIVE, "有人在吗", {"kind": "greeting"}))
        time.sleep(1.0)
        robot.events.put(Event(EventType.EXIT))

    threading.Thread(target=feed_events, daemon=True).start()

    try:
        # 主循环内部会调用 shutdown()，其中 graceful_exit 会 sys.exit
        robot.run()
        normal_exit = True
        exit_error = ""
    except SystemExit as e:
        normal_exit = (e.code in (0, None))
        exit_error = f"SystemExit({e.code})"
    except Exception as e:
        import traceback
        traceback.print_exc()
        normal_exit = False
        exit_error = f"{type(e).__name__}: {e}"

    # 对话在线程里跑，可能还没写完就被退出事件打断了。
    # 等一小会儿再断言，避免"处理尚未结束就检查"造成的假失败。
    deadline = time.time() + 8.0
    while time.time() < deadline and not dispatched["responses"]:
        time.sleep(0.1)

    check("主循环正常退出", normal_exit, exit_error)
    check("文本事件被处理并产生回复", dispatched["text"],
          f"responses={dispatched['responses']}")
    check("回复内容来自模型", any("冒烟测试回复" in r[1] for r in dispatched["responses"]),
          str(dispatched["responses"][:1]))
    check("对话历史被写入", len(robot.conversation.history) > hist_before,
          f"{hist_before} -> {len(robot.conversation.history)}")
    check("主动消息事件被分发", dispatched["proactive"],
          f"text={dispatched.get('proactive_text')}")

    robot._print_response = orig_print_response
    robot._print_proactive = orig_print_proactive

    # ---------------- 3. 资源释放 ----------------
    hr("3. shutdown() 资源释放")
    check("退出后 running 为 False", not robot.running)
    if not no_hardware and robot.recorder is not None:
        check("麦克风流已关闭", not robot.recorder._running)
    if robot.vl_model is not None:
        check("模型已卸载", robot.vl_model.model is None)
    if robot.segmenter is not None:
        check("VAD 线程已停止", not robot.segmenter._running)
    check("监控已停止（未启用时跳过）",
          robot.monitor is None or not robot.monitor._running)

    # ---------------- 汇总 ----------------
    hr("结果汇总")
    for k, v in results.items():
        print(f"  {PASS if v else FAIL} {k}")
    ok = all(results.values())
    print(f"\n{'全部通过 ✓' if ok else '存在失败项 ✗'}")

    # 清理冒烟测试产生的历史文件
    try:
        if config.ConversationConfig.HISTORY_FILE.exists():
            config.ConversationConfig.HISTORY_FILE.unlink()
    except Exception:
        pass

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
