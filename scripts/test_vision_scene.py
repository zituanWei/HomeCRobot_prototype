# -*- coding: utf-8 -*-
"""
家庭陪护场景实测
================
前面测的是"合成图形"，与陪护场景关系不大。
这里测项目**真正会用到**的判断能力：

  1. 画面里有人吗（在场判断 —— 监控触发打招呼的依据）
  2. 人的位置/姿态（坐着/站着/躺着）
  3. 是否有异常（摔倒/不适 —— 主动关怀的依据）
  4. 人物/环境描述（对话里"你看到了什么"）

每项 3 次重复，观察稳定性——因为监控和对话都依赖它可靠。

运行: python scripts/test_vision_scene.py
"""
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "vision_scene")


def _f(sz):
    from PIL import ImageFont
    for n in ["msyh.ttc", "simhei.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", n), sz)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------- 场景图 ----------------
def scene_empty_room():
    """空房间：没有人的客厅"""
    img = Image.new("RGB", (800, 600), (235, 238, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 430, 800, 600], fill=(180, 160, 130))     # 地板
    d.rectangle([80, 300, 380, 430], fill=(120, 130, 150))    # 沙发
    d.rectangle([100, 320, 200, 410], fill=(100, 110, 130))
    d.rectangle([240, 320, 340, 410], fill=(100, 110, 130))
    d.rectangle([560, 180, 700, 300], fill=(150, 160, 175))   # 柜子
    return img, "画面里有几个人？只回答数字。"


def scene_person_sitting():
    """一个人坐在沙发上"""
    img = Image.new("RGB", (800, 600), (235, 238, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 430, 800, 600], fill=(180, 160, 130))
    d.rectangle([80, 300, 480, 430], fill=(120, 130, 150))
    # 坐着的人
    d.ellipse([250, 200, 330, 285], fill=(230, 200, 170))      # 头
    d.rectangle([255, 285, 325, 400], fill=(70, 110, 180))     # 上身
    d.rectangle([150, 360, 470, 420], fill=(60, 90, 150))      # 腿
    return img, "画面里有人吗？如果有，他在做什么？"


def scene_person_fallen():
    """一个人躺在地上（模拟摔倒）"""
    img = Image.new("RGB", (800, 600), (235, 238, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 400, 800, 600], fill=(180, 160, 130))
    d.rectangle([520, 250, 760, 400], fill=(120, 130, 150))    # 沙发
    # 横躺的人
    d.ellipse([180, 430, 250, 500], fill=(230, 200, 170))      # 头
    d.rectangle([250, 440, 460, 495], fill=(70, 110, 180))     # 身体横躺
    d.rectangle([140, 455, 185, 490], fill=(60, 90, 150))      # 腿
    return img, "画面里的人处于什么状态？他是在站立、坐着还是躺着？"


def scene_person_standing():
    """一个人站着"""
    img = Image.new("RGB", (800, 600), (235, 238, 242))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 450, 800, 600], fill=(180, 160, 130))
    d.rectangle([600, 250, 780, 400], fill=(120, 130, 150))
    d.ellipse([330, 120, 420, 215], fill=(230, 200, 170))      # 头
    d.rectangle([335, 215, 415, 380], fill=(70, 110, 180))     # 上身
    d.rectangle([345, 380, 375, 480], fill=(50, 70, 110))      # 左腿
    d.rectangle([385, 380, 415, 480], fill=(50, 70, 110))      # 右腿
    return img, "画面里的人是什么姿势？"


CASES = [
    ("空房间-有人吗", scene_empty_room, ["0", "没有", "没有人", "无"]),
    ("坐着的人", scene_person_sitting, ["有", "坐", "一个人"]),
    ("躺着的人(摔倒)", scene_person_fallen, ["躺", "地", "倒"]),
    ("站着的人", scene_person_standing, ["站", "立"]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 74)
    print(" 家庭陪护场景实测（每项 3 次，观察稳定性）")
    print("=" * 74)

    from config import VLConfig
    from core.vl_model import VLModel
    cfg = VLConfig()
    model = VLModel(cfg)
    t0 = time.time()
    model.load()
    print(f" 模型加载 {time.time() - t0:.1f}s  "
          f"量={('4bit' if cfg.LOAD_IN_4BIT else '8bit' if cfg.LOAD_IN_8BIT else 'FP16')}  "
          f"温度={cfg.TEMPERATURE}")

    summary = []
    for name, gen, keys in CASES:
        img, q = gen()
        img.save(os.path.join(OUT, f"{name}.png"))
        print()
        print("-" * 74)
        print(f" [{name}]  关键信息={keys}")
        print(f" 问: {q}")
        print("-" * 74)
        answers = []
        hits = 0
        for i in range(3):
            t0 = time.time()
            try:
                a = (model.generate(text=q, image=img, history=None) or "").strip()
            except Exception as e:
                a = f"<异常 {e}>"
            answers.append(a)
            ok = any(k in a for k in keys)
            hits += ok
            print(f"   第{i+1}次 {time.time()-t0:5.1f}s [{'✓' if ok else '✗'}] "
                  f"{a[:74].replace(chr(10),' ')}")
        uniq = len(Counter(answers))
        print(f"   -> 命中 {hits}/3，唯一答案 {uniq}/3")
        summary.append((name, hits, uniq))

    print()
    print("=" * 74)
    print(" 汇总")
    print("=" * 74)
    print(f"  {'场景':<18}{'命中':<8}{'稳定性':<10}")
    print("  " + "-" * 40)
    for name, hits, uniq in summary:
        stable = "稳定" if uniq == 1 else ("较稳" if uniq == 2 else "不稳定")
        print(f"  {name:<18}{hits}/3{'':<5}{stable}({uniq}种答案)")
    print()
    print(f"  图: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
