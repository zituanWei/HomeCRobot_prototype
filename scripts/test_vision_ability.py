# -*- coding: utf-8 -*-
"""
视觉能力实测
============
用程序生成**带标准答案**的测试图，逐个测模型在各类视觉任务上的表现，
从而得出"哪类物体/任务有优势、哪类有缺陷"的结论，而不是空谈。

测试维度：
  1. OCR（大号文字识别）
  2. 颜色识别
  3. 形状+位置
  4. 计数
  5. 空间关系
  6. 小物体（远处/小尺寸）
  7. 细粒度分类（相似物体区分）

每张图给出标准答案，脚本自动判分。
运行: python scripts/test_vision_ability.py
"""
import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "vision_test")


def _font(size):
    """找一个可用的 TTF 字体"""
    for name in ["msyh.ttc", "simhei.ttf", "arial.ttf", "DejaVuSans.ttf"]:
        for base in [r"C:\Windows\Fonts", "/usr/share/fonts", ""]:
            p = os.path.join(base, name) if base else name
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ======================================================================
#  测试图生成
# ======================================================================
def make_ocr():
    """大号英文数字：清晰大字体"""
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((60, 170), "A7K9", fill=(0, 0, 0), font=_font(140))
    return img, "A7K9", "图里的大号字符是什么？只回答字符本身。"


def make_ocr_small():
    """小号文字：占画面很小"""
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((20, 20), "B3M8", fill=(0, 0, 0), font=_font(20))
    return img, "B3M8", "请读出图中所有的字母和数字，按顺序回答。"


def make_color():
    """颜色识别：三个大色块"""
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 120, 200, 360], fill=(220, 30, 30))     # 红
    d.rectangle([240, 120, 400, 360], fill=(30, 180, 60))    # 绿
    d.rectangle([440, 120, 600, 360], fill=(40, 80, 220))    # 蓝
    return img, "红绿蓝", "从左到右，图中的三个色块分别是什么颜色？只回答颜色名。"


def make_count():
    """计数：7 个黑点"""
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    pts = [(100, 120), (250, 120), (400, 120), (550, 120),
           (175, 300), (325, 300), (475, 300)]
    for x, y in pts:
        d.ellipse([x - 30, y - 30, x + 30, y + 30], fill=(0, 0, 0))
    return img, "7", "图中有几个黑色的圆点？只回答数字。"


def make_spatial():
    """空间关系：红方块在蓝方块左边"""
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 160, 260, 320], fill=(210, 40, 40))    # 红 在左
    d.rectangle([380, 160, 540, 320], fill=(40, 70, 210))    # 蓝 在右
    return img, "左边", "红色的方块在蓝色方块的哪一边？只回答「左边」或「右边」。"


def make_tiny():
    """小物体：右上角一个小三角"""
    img = Image.new("RGB", (640, 480), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.polygon([(590, 30), (620, 30), (605, 60)], fill=(20, 20, 20))
    return img, "三角形", "图中除了背景，还有一个很小的图形，它是什么形状？"


def make_finegrained():
    """细粒度：两个相似图形（正方形 vs 长方形）"""
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 140, 260, 340], fill=(80, 80, 200))     # 正方形 200x200
    d.rectangle([340, 160, 600, 320], fill=(80, 80, 200))    # 长方形 260x160
    return img, "正方形", "图中左边的蓝色图形是什么形状？只回答形状名。"


def make_scene():
    """自然场景模拟：多物体 + 关系"""
    img = Image.new("RGB", (640, 480), (240, 245, 250))
    d = ImageDraw.Draw(img)
    # 桌子
    d.rectangle([40, 340, 600, 380], fill=(150, 100, 60))
    # 杯子（矩形+把手弧）
    d.rectangle([120, 250, 200, 340], fill=(230, 230, 235), outline=(120, 120, 120), width=3)
    d.arc([200, 270, 240, 320], 270, 90, fill=(120, 120, 120), width=4)
    # 书（两个叠起来的矩形）
    d.rectangle([330, 300, 520, 340], fill=(200, 60, 60))
    d.rectangle([345, 265, 505, 300], fill=(60, 130, 200))
    # 手机
    d.rectangle([540, 250, 600, 340], fill=(30, 30, 30))
    return img, "杯子|书|手机", "桌子上摆了哪些东西？请列出物体名称。"


TESTS = [
    ("OCR-大号字符", make_ocr, ["a7k9"]),
    ("OCR-小号字符", make_ocr_small, ["b3m8"]),
    ("颜色识别", make_color, ["红", "绿", "蓝"]),
    ("计数", make_count, ["7", "七"]),
    ("空间关系", make_spatial, ["左"]),
    ("小物体", make_tiny, ["三角"]),
    ("细粒度形状", make_finegrained, ["正方形", "正方"]),
    ("多物体场景", make_scene, ["杯子", "书", "手机"]),
]


def normalize(s: str) -> str:
    return re.sub(r"[\s，。！？、,.!?:：；;（）()\[\]「」]", "", (s or "").lower())


def judge(answer: str, keys: list) -> tuple:
    """判断回答是否命中关键信息；多关键项按命中比例计分"""
    a = normalize(answer)
    hit = sum(1 for k in keys if normalize(k) in a)
    total = len(keys)
    if hit == total:
        return "正确", 1.0
    if hit > 0:
        return "部分正确", hit / total
    return "错误", 0.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------- 生成测试图 ----------
    print("=" * 70)
    print(" 生成测试图")
    print("=" * 70)
    cases = []
    for name, gen, keys in TESTS:
        img, expected, question = gen()
        path = os.path.join(OUT_DIR, f"{name}.png")
        img.save(path)
        cases.append((name, img, question, keys, expected))
        print(f"  {name:<14} 标准答案={expected:<12} -> {os.path.basename(path)}")

    # ---------- 加载模型 ----------
    print()
    print("=" * 70)
    print(" 加载视觉模型（可能需要一会儿）")
    print("=" * 70)
    from core.vl_model import VLModel
    from config import VLConfig
    cfg = VLConfig()
    model = VLModel(cfg)
    t0 = time.time()
    model.load()
    print(f"  模型加载耗时 {time.time() - t0:.1f}s")

    # ---------- 逐个测试 ----------
    print()
    print("=" * 70)
    print(" 逐个测试")
    print("=" * 70)
    rows = []
    for name, img, question, keys, expected in cases:
        t0 = time.time()
        try:
            ans = model.generate(text=question, image=img, history=None)
        except Exception as e:
            ans = f"<异常: {e}>"
        dt = time.time() - t0
        verdict, score = judge(ans, keys)
        ans_clean = ans.strip().replace("\n", " ")[:60]
        print(f"\n  [{name}]  ({dt:.1f}s)")
        print(f"    问: {question}")
        print(f"    答: {ans_clean}")
        print(f"    标准: {expected}  ->  {verdict}")
        rows.append((name, verdict, score, dt, expected, ans_clean))

    # ---------- 汇总 ----------
    print()
    print("=" * 70)
    print(" 汇总")
    print("=" * 70)
    print(f"  {'任务':<16}{'结果':<10}{'得分':<8}{'耗时':<8}")
    print("  " + "-" * 46)
    total = 0.0
    for name, verdict, score, dt, expected, _ in rows:
        total += score
        print(f"  {name:<16}{verdict:<10}{score:<8.2f}{dt:<7.1f}s")
    avg = total / len(rows)
    print()
    print(f"  平均得分: {avg:.2f} / 1.00   总耗时 {sum(r[3] for r in rows):.1f}s")

    # 分类结论
    good = [r[0] for r in rows if r[2] >= 1.0]
    bad = [r[0] for r in rows if r[2] < 1.0]
    print()
    print(f"  完全正确: {', '.join(good) if good else '（无）'}")
    print(f"  有缺陷  : {', '.join(bad) if bad else '（无）'}")
    print()
    print(f"  测试图保留在: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
