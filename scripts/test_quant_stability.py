# -*- coding: utf-8 -*-
"""
量化稳定性对比：8bit vs 4bit vs FP16
====================================
发现的问题：8bit 量化下，同一张图同一个问题连问 3 次，会得到
  ① 正常描述  ② 混乱描述  ③ "我看不清图像"
这种输出不稳定（含幻觉）是量化误差的典型症状。

本脚本对同一张图重复提问 N 次，统计：
  · 出现"看不到/看不清"幻觉的次数
  · 答案之间的不一致程度
从而判断哪种量化在 8GB 显存下更可靠。

运行: python scripts/test_quant_stability.py
      python scripts/test_quant_stability.py --bits 4
      python scripts/test_quant_stability.py --bits none
"""
import argparse
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 会出现的"幻觉"式回答
HALLUCINATION = ["无法看到", "看不到", "看不清", "没有图片", "无法查看",
                 "未提供图片", "请上传", "无法识别图像", "没有收到图片"]


def normalize(s: str) -> str:
    return re.sub(r"[\s，。！？、,.!?:：；;（）()\[\]「」]", "", (s or "").lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", default="8", choices=["4", "8", "none"],
                    help="量化位数")
    ap.add_argument("--repeat", type=int, default=5, help="重复次数")
    ap.add_argument("--temp", type=float, default=None,
                    help="覆盖采样温度；0 表示贪心解码（最稳定）")
    args = ap.parse_args()

    from config import VLConfig
    from core.vl_model import VLModel

    cfg = VLConfig()
    if args.bits == "4":
        cfg.LOAD_IN_4BIT, cfg.LOAD_IN_8BIT = True, False
    elif args.bits == "8":
        cfg.LOAD_IN_4BIT, cfg.LOAD_IN_8BIT = False, True
    else:
        cfg.LOAD_IN_4BIT, cfg.LOAD_IN_8BIT = False, False

    if args.temp is not None:
        cfg.TEMPERATURE = args.temp
        # 温度为 0 时必须关掉采样，否则 transformers 会报错/警告
        if args.temp <= 0:
            cfg.DO_SAMPLE = False

    print("=" * 72)
    print(f" 量化稳定性测试  模式={args.bits}bit  重复={args.repeat}次  "
          f"温度={cfg.TEMPERATURE} 采样={cfg.DO_SAMPLE}")
    print("=" * 72)

    model = VLModel(cfg)
    t0 = time.time()
    model.load()
    print(f" 加载耗时 {time.time() - t0:.1f}s")

    tests = [
        ("多物体场景", os.path.join(ROOT, "data", "vision_test2", "多物体场景（重画）.png"),
         "桌子上摆了哪些东西？请逐一说出物体名称。"),
        ("正方形判断", os.path.join(ROOT, "data", "vision_test2", "正方形判断（问宽高）.png"),
         "图中左边那个蓝色图形，它的宽和高是否相等？"),
    ]

    all_halluc = 0
    all_total = 0

    for name, path, q in tests:
        if not os.path.exists(path):
            print(f"  [SKIP] 缺图 {path}")
            continue
        img = Image.open(path).convert("RGB")

        print()
        print("-" * 72)
        print(f" {name}   问题：{q}")
        print("-" * 72)

        answers = []
        halluc = 0
        for i in range(args.repeat):
            t0 = time.time()
            try:
                a = model.generate(text=q, image=img, history=None)
            except Exception as e:
                a = f"<异常 {e}>"
            dt = time.time() - t0
            a = (a or "").strip()
            answers.append(a)
            is_h = any(k in a for k in HALLUCINATION)
            if is_h:
                halluc += 1
            flag = "  ★幻觉" if is_h else ""
            print(f"   第{i+1}次 {dt:5.1f}s: {a[:78].replace(chr(10),' ')}{flag}")

        all_halluc += halluc
        all_total += args.repeat

        # 一致性：归一化后完全相同的答案数量
        cnt = Counter(normalize(a) for a in answers)
        most = cnt.most_common(1)[0]
        print(f"\n   幻觉次数: {halluc}/{args.repeat}")
        print(f"   唯一答案数: {len(cnt)}/{args.repeat}（越少越稳定）")
        if most[1] > 1:
            print(f"   最常见答案出现 {most[1]} 次: {most[0][:60]}")

    print()
    print("=" * 72)
    print(" 结论")
    print("=" * 72)
    print(f"  模式 {args.bits}bit：幻觉 {all_halluc}/{all_total} 次")
    if all_halluc == 0:
        print("  -> 未出现'看不到图片'类幻觉")
    else:
        print("  -> 出现幻觉，输出不可靠")
    return 0


if __name__ == "__main__":
    sys.exit(main())
