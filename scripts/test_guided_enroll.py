"""
引导注册自检
============
验证「认不出人 -> 主动问名字 -> 两步确认 -> 拍照注册」的状态机。

覆盖路径：
  1. 正常流程：问名字 -> 报名字 -> 确认 -> 注册
  2. 名字没听清：重问
  3. 用户否认：回到重说名字
  4. 反复否认：达到上限后放弃
  5. 超时：卡住太久自动放弃
  6. 触发条件：全新环境（库为空）不追问
  7. 超时/边界：空输入、纯标点

运行: python scripts/test_guided_enroll.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.guided_enroll import GuidedEnroll, EnrollState

PASS, FAIL = "[OK]", "[FAIL]"
results = {}


def check(name, ok, detail=""):
    results[name] = bool(ok)
    print(f"  {PASS if ok else FAIL} {name}" + (f"  {detail}" if detail else ""))


def hr(t):
    print("\n" + "=" * 64)
    print(t)
    print("=" * 64)


class FakeCfg:
    ENABLE_GUIDED_ENROLL = True
    GUIDED_ENROLL_REQUIRE_EXISTING = True
    GUIDED_ENROLL_MAX_RETRIES = 2


def make(log):
    """建一个带回调记录的 GuidedEnroll"""
    return GuidedEnroll(
        FakeCfg(),
        on_ask=lambda p: log.append(("ask", p)),
        on_confirm=lambda n: log.append(("confirm", n)),
        on_enroll=lambda n: (log.append(("enroll", n)) or (True, f"记住了 {n}")),
        on_giveup=lambda r: log.append(("giveup", r)),
    )


# ----------------------------------------------------------------------
def test_happy_path():
    hr("1. 正常流程：问名字 -> 报名 -> 确认 -> 注册")
    log = []
    g = make(log)

    g.start()
    check("开始后进入等待名字态", g.state == EnrollState.AWAIT_NAME, g.state)
    check("发出了提问", log and log[0][0] == "ask", str(log[:1]))

    consumed = g.handle("我叫武凉烨")
    check("名字被消费", consumed)
    check("进入确认态", g.state == EnrollState.CONFIRM, g.state)
    check("复述了名字", any(k == "confirm" and "武凉烨" in str(v) for k, v in log),
          str([x for x in log if x[0] == "confirm"]))

    g.handle("对")
    check("确认后触发注册", any(k == "enroll" and v == "武凉烨" for k, v in log),
          str([x for x in log if x[0] == "enroll"]))
    check("回到空闲态", g.state == EnrollState.IDLE, g.state)


def test_name_variants():
    hr("2. 各种说法都能抽出名字")
    cases = [
        ("我叫张伟", "张伟"),
        ("我的名字是李雷", "李雷"),
        ("我是韩梅梅", "韩梅梅"),
        ("叫我小王", "小王"),
        ("王小明", "王小明"),          # 直接回名字
        ("我叫武凉烨。", "武凉烨"),
    ]
    for text, expect in cases:
        got = GuidedEnroll._extract_name(text)
        ok = got == expect
        check(f"{text!r} -> {got!r}", ok, "" if ok else f"期望 {expect!r}")


def test_confirm_parsing():
    hr("3. 确认/否认判断（'不是'含'是'，必须先判否认）")
    # 否认
    for deny in ["不是", "不对", "你弄错了", "重新说一个"]:
        log = []
        g = make(log)
        g.start()
        g.handle("我叫张三")
        g.handle(deny)
        ok = (g.state == EnrollState.AWAIT_NAME
              and not any(k == "enroll" for k, _ in log))
        check(f"否认 {deny!r} -> 回到等待名字", ok, g.state)
    # 确认（注意「没错」含「错」，必须先判确认再判否认）
    for yes in ["对", "是的", "没错", "正确", "对的"]:
        log = []
        g = make(log)
        g.start()
        g.handle("我叫张三")
        g.handle(yes)
        ok = any(k == "enroll" for k, _ in log)
        check(f"确认 {yes!r} -> 触发注册", ok, "" if ok else f"state={g.state}")


def test_retry_limit():
    hr("4. 反复否认 -> 达到上限后放弃")
    log = []
    g = make(log)
    g.start()
    g.handle("我叫张三")
    # 注意：状态会在 confirm / await_name 之间来回切换，
    # 每轮"否认"之后系统会重新问名字，这里补一句名字再继续否认，
    # 直到超过重试上限而放弃。
    for _ in range(8):
        if g.state == EnrollState.IDLE:
            break
        if g.state == EnrollState.CONFIRM:
            g.handle("不是")
        elif g.state == EnrollState.AWAIT_NAME:
            g.handle("我叫张三")
    check("最终放弃", any(k == "giveup" for k, _ in log),
          str([x for x in log if x[0] == "giveup"]))
    check("回到空闲态", g.state == EnrollState.IDLE, g.state)
    check("放弃前没有误注册", not any(k == "enroll" for k, _ in log))


def test_unclear_name():
    hr("5. 没听清名字 -> 重问")
    log = []
    g = make(log)
    g.start()
    g.handle("呃")            # 单字语气词，抽不出名字
    asks = [v for k, v in log if k == "ask"]
    check("再次提问", len(asks) >= 2, f"提问 {len(asks)} 次")
    check("仍在等待名字态", g.state == EnrollState.AWAIT_NAME, g.state)
    # 最终也要放弃
    for _ in range(6):
        if g.state == EnrollState.IDLE:
            break
        g.handle("哦")
    check("持续听不清 -> 放弃", any(k == "giveup" for k, _ in log),
          str([x for x in log if x[0] == "giveup"]))


def test_timeout():
    hr("6. 超时自动放弃")
    log = []
    g = make(log)
    g.timeout = 0.15
    g.start()
    time.sleep(0.25)
    g.handle("我叫张三")
    check("超时后不再消费输入", not g.active, g.state)
    check("没有误注册", not any(k == "enroll" for k, _ in log))


def test_trigger_conditions():
    hr("7. 触发条件：全新环境不追问")
    log = []
    g = make(log)
    # 库为空
    check("库为空时不询问", not g.should_offer(0), f"face_db=0")
    # 库非空
    check("库非空时应当询问", g.should_offer(2), f"face_db=2")
    # 已在流程中不重复询问
    g.start()
    check("流程中不重复询问", not g.should_offer(2), g.state)
    # 关闭开关
    class Off:
        ENABLE_GUIDED_ENROLL = False
        GUIDED_ENROLL_REQUIRE_EXISTING = True
        GUIDED_ENROLL_MAX_RETRIES = 2
    g2 = GuidedEnroll(Off())
    check("开关关闭时不询问", not g2.should_offer(5))


def test_edge_inputs():
    hr("8. 边界输入")
    log = []
    g = make(log)
    g.start()
    check("空输入被消费且不崩", g.handle("") is True)
    check("纯标点不崩", g.handle("。。。") is True)
    check("仍在等待名字态", g.state in (EnrollState.AWAIT_NAME, EnrollState.IDLE),
          g.state)

    # 不可能是名字的词不该被当成名字
    for bad in ["什么", "谁", "不知道"]:
        got = GuidedEnroll._extract_name(bad)
        check(f"{bad!r} 不被当成名字", got is None, f"got={got!r}")


def test_idle_passthrough():
    hr("9. 空闲态不拦截普通对话")
    log = []
    g = make(log)
    check("空闲态 handle 返回 False", g.handle("今天天气不错") is False)
    check("没有产生任何回调", not log, str(log))


def main():
    print("引导注册自检")
    test_happy_path()
    test_name_variants()
    test_confirm_parsing()
    test_retry_limit()
    test_unclear_name()
    test_timeout()
    test_trigger_conditions()
    test_edge_inputs()
    test_idle_passthrough()

    hr("结果汇总")
    for k, v in results.items():
        print(f"  {PASS if v else FAIL} {k}")
    ok = all(results.values())
    print(f"\n{'全部通过 ✓' if ok else '存在失败项 ✗'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
