# -*- coding: utf-8 -*-
"""
视觉能力实测（修正版）
====================
在 test_vision_ability.py 基础上的改进：

1. **修正等价判定**：模型答"七个"应等同于"7"，之前被误判为部分正确。
   现在支持别名/等价词。
2. **补充"可识别性"判定**：把"小物体也没看到"和"说错形状"区分开，
   两者对应用户场景里的不同后果。
3. **补充细粒度复测**：用更明确的问法复测正方形/长方形，
   判断是"能力不足"还是"问法不够明确"。

运行: python scripts/test_vision_ability2.py
"""
import os
import sys
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "vision_test2")


def _font(size):
    for name in ["msyh.ttc", "simhei.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", name), size)
        except Exception:
            continue
    return ImageFont.load_default()


def normalize(s: str) -> str:
    return re.sub(r"[\s，。！？、,.!?:：；;（）()\[\]「」]", "", (s or "").lower())


# ======================================================================
#  每组测试：(名称, 图片生成, 问题, 等价答案组列表)
#  等价答案组：外层是"必须都命中"，内层是"任一命中即可"
# ======================================================================
def make_count():
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    pts = [(100, 120), (250, 120), (400, 120), (550, 120),
           (175, 300), (325, 300), (475, 300)]
    for x, y in pts:
        d.ellipse([x - 30, y - 30, x + 30, y + 30], fill=(0, 0, 0))
    # 等价答案：阿拉伯数字或中文数字都算对
    return img, "图中有几个黑色的圆点？只回答数字。", [["7", "七"]]


def make_square_vs_rect():
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 140, 260, 340], fill=(80, 80, 200))     # 200x200 正方形
    d.rectangle([340, 160, 600, 320], fill=(80, 80, 200))    # 260x160 长方形
    return (img,
            "图中左边那个蓝色图形，它的宽和高是否相等？",
            [["相等", "一样", "相同", "yes"]])


def make_square_clear():
    """更明确的正方形：给出边长标注对比"""
    img = Image.new("RGB", (640, 480), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # 正方形 180x180
    d.rectangle([80, 150, 260, 330], fill=(80, 80, 200))
    d.text((150, 340), "180", fill=(0, 0, 0), font=_font(24))
    d.text((30, 230), "180", fill=(0, 0, 0), font=_font(24))
    return (img, "图中蓝色图形的宽是180、高是180，它是什么形状？只回答形状名。",
            [["正方形", "正方"]])


def make_tiny_bigger():
    """小物体稍大一点：20x20 改为 36x36，看是不是纯尺寸问题"""
    img = Image.new("RGB", (640, 480), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.polygon([(575, 20), (615, 20), (595, 58)], fill=(20, 20, 20))
    return img, "图中右上角有什么图形？", [["三角"]]


def make_scene():
    """多物体场景：明确画 杯子/书/手机"""
    img = Image.new("RGB", (640, 480), (240, 245, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 340, 600, 385], fill=(150, 100, 60))        # 桌面
    # 杯子
    d.rectangle([120, 245, 205, 340], fill=(235, 235, 240),
                outline=(110, 110, 110), width=3)
    d.arc([205, 265, 250, 320], 270, 90, fill=(110, 110, 110), width=5)
    # 书（两本叠放，明显长方形 + 书脊）
    d.rectangle([330, 300, 540, 340], fill=(200, 60, 60))
    d.rectangle([330, 300, 348, 340], fill=(160, 40, 40))        # 书脊
    d.rectangle([345, 258, 528, 300], fill=(60, 130, 200))
    d.rectangle([345, 258, 363, 300], fill=(40, 100, 170))
    # 手机
    d.rectangle([555, 240, 605, 340], fill=(30, 30, 30))
    d.rectangle([559, 248, 601, 322], fill=(70, 90, 120))
    return (img, "桌子上摆了哪些东西？请逐一说出物体名称。",
            [["杯"], ["书"], ["手机", "电话"]])


TESTS = [
    ("计数（等价判定）", make_count),
    ("正方形判断（问宽高）", make_square_vs_rect),
    ("正方形判断（给尺寸）", make_square_clear),
    ("小物体（稍大36px）", make_tiny_bigger),
    ("多物体场景（重画）", make_scene),
]


def judge(answer: str, groups: list):
    """groups: 外层每项都必须命中，内层任一等价词命中即可"""
    a = normalize(answer)
    hit_groups = 0
    detail = []
    for alts in groups:
        ok = any(normalize(k) in a for k in alts)
        if ok:
            hit_groups += 1
        detail.append(f"{'/'.join(alts)}={'✓' if ok else '✗'}")
    score = hit_groups / len(groups)
    verdict = "正确" if score >= 1.0 else ("部分正确" if score > 0 else "错误")
    return verdict, score, " ".join(detail)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print(" 视觉能力实测（修正评分）")
    print("=" * 72)

    cases = []
    for name, gen in TESTS:
        img, q, groups = gen()
        img.save(os.path.join(OUT_DIR, f"{name}.png"))
        cases.append((name, img, q, groups))
        print(f"  {name:<22} 标准={' 且 '.join('/'.join(g) for g in groups)}")

    print()
    print(" 加载模型...")
    from core.vl_model import VLModel
    from config import VLConfig
    model = VLModel(VLConfig())
    t0 = time.time()
    model.load()
    print(f"  加载 {time.time() - t0:.1f}s")

    print()
    print("=" * 72)
    print(" 逐项测试")
    print("=" * 72)
    rows = []
    for name, img, q, groups in cases:
        t0 = time.time()
        try:
            ans = model.generate(text=q, image=img, history=None)
        except Exception as e:
            ans = f"<异常: {e}>"
        dt = time.time() - t0
        verdict, score, detail = judge(ans, groups)
        print(f"\n  [{name}]  {dt:.1f}s")
        print(f"    问: {q}")
        print(f"    答: {ans.strip()[:110]}")
        print(f"    判定: {verdict}  ({detail})")
        rows.append((name, verdict, score, dt, detail))

    print()
    print("=" * 72)
    print(" 汇总")
    print("=" * 72)
    print(f"  {'任务':<24}{'结果':<10}{'得分':<7}{'耗时':<7}")
    print("  " + "-" * 50)
    for name, verdict, score, dt, detail in rows:
        print(f"  {name:<24}{verdict:<10}{score:<7.2f}{dt:<6.1f}s")
    avg = sum(r[2] for r in rows) / len(rows)
    print(f"\n  平均 {avg:.2f}/1.00")
    print(f"  测试图: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
