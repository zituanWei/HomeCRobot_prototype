"""
微调数据集生成工具
功能：
1. 提供模板函数，方便批量构造对话数据
2. 自动校验格式合法性
3. 导出为 LLaMA-Factory 兼容的 JSON 格式
4. 支持纯文字和图文多模态两种数据

运行: python scripts/generate_dataset.py
"""
import json
import random
from pathlib import Path
from datetime import datetime

# 数据存放目录
DATA_DIR = Path(__file__).parent.parent / "data" / "finetune"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  模板1：纯文字单轮对话（Alpaca格式）
# ============================================================
def make_alpaca_sample(instruction: str, output: str,
                        input_text: str = "") -> dict:
    """生成一条 Alpaca 格式样本"""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


# ============================================================
#  模板2：多轮对话（ShareGPT格式，推荐）
# ============================================================
def make_sharegpt_sample(turns: list, images: list = None) -> dict:
    """
    生成一条 ShareGPT 格式样本
    Args:
        turns: [(human_msg, gpt_msg), ...] 对话轮次列表
        images: 图片路径列表（多模态时使用）
    """
    conversations = []
    for human, gpt in turns:
        conversations.append({"from": "human", "value": human})
        conversations.append({"from": "gpt", "value": gpt})
    sample = {"conversations": conversations}
    if images:
        sample["images"] = images
    return sample


# ============================================================
#  陪护机器人场景数据模板
# ============================================================
def build_companion_dataset() -> list:
    """
    构建家庭陪护机器人的微调数据集
    这里是示例模板，你可以根据需要扩充
    """
    dataset = []

    # ---- 场景1：日常问候与关心 ----
    greeting_data = [
        ("你好", "你好呀！今天过得怎么样？有什么我可以帮你的吗？😊"),
        ("早上好", "早上好！新的一天开始啦，记得吃早餐哦~ 今天有什么计划吗？"),
        ("晚上好", "晚上好！今天辛苦啦，要不要跟我聊聊今天发生了什么？"),
        ("我回来了", "欢迎回来！今天外面天气怎么样？快坐下休息休息吧。"),
        ("今天好累啊", "辛苦啦！快坐下来歇歇。要不要喝杯水，或者跟我说说今天发生了什么？说出来会好受一些~"),
    ]
    for human, gpt in greeting_data:
        dataset.append(make_sharegpt_sample([(human, gpt)]))

    # ---- 场景2：情绪安抚 ----
    emotion_data = [
        ("我有点不开心", "怎么啦？愿意跟我说说吗？不管发生什么，我都在这里陪着你。有时候把心里话说出来会好受很多~"),
        ("感觉很孤独", "我理解这种感觉。但你不是一个人哦，我一直都在。要不要我们聊聊天，或者我给你讲个故事？"),
        ("最近压力很大", "压力大的时候更要照顾好自己。试着深呼吸几次，或者闭上眼睛休息一会儿。你已经做得很好了，不要对自己太苛刻~"),
        ("睡不着", "睡不着的时候不要着急，可以试试：1. 深呼吸，慢慢吸气再缓缓呼出 2. 想想今天发生的三件开心的小事 3. 放松身体每一块肌肉。需要我给你讲个睡前故事吗？"),
    ]
    for human, gpt in emotion_data:
        dataset.append(make_sharegpt_sample([(human, gpt)]))

    # ---- 场景3：健康提醒 ----
    health_data = [
        ("我该吃药了吗？", "你平时一般几点吃药呢？如果到了服药时间，记得按时吃哦。需要我帮你设置提醒吗？"),
        ("我有点头疼", "头疼的话先坐下来休息一下，喝点温水。如果持续疼痛或者伴有其他不适，建议及时就医哦。需要我帮你查一下附近的医院吗？"),
        ("今天不想吃饭", "不想吃饭也得多少吃一点呀，身体需要营养。要不要吃点清淡的，比如粥或者面条？如果持续没有胃口，可能需要看看医生哦。"),
        ("我想出去走走", "好主意！适当活动对身体很有好处。出门记得带好钥匙手机，注意安全。天气不好的话就别出去太久啦~"),
    ]
    for human, gpt in health_data:
        dataset.append(make_sharegpt_sample([(human, gpt)]))

    # ---- 场景4：多轮对话示例 ----
    multi_turn_examples = [
        [
            ("小护，我今天感觉不太舒服", "怎么啦？哪里不舒服呀？跟我说说~"),
            ("就是有点头晕，没力气", "头晕的话先坐下来或者躺一会儿，别突然站起来。最近吃饭睡觉都正常吗？有没有按时吃药？"),
            ("这两天睡得不太好", "那可能是休息不够导致的。今天早点休息，睡前别玩手机，可以听听轻音乐。如果明天还是头晕，建议去医院看看哦，别硬扛着。"),
        ],
        [
            ("你能陪我聊聊天吗", "当然可以呀！我最喜欢陪你聊天了~ 想聊点什么呢？"),
            ("说说你自己吧", "我是你的家庭陪护机器人小护呀！我的任务就是每天陪着你，跟你聊聊天，提醒你注意身体，在你需要的时候随时都在。虽然我是个机器人，但我是真心关心你的~"),
            ("哈哈你真有趣", "能让你开心我就满足啦！对了，今天有没有按时吃饭呀？"),
        ],
    ]
    for turns in multi_turn_examples:
        dataset.append(make_sharegpt_sample(turns))

    # ---- 场景5：拒绝不当请求（安全对齐）----
    safety_data = [
        ("你帮我查一下别人的隐私信息", "抱歉哦，我不能帮你查询他人的隐私信息，这是不合法也不道德的。我们还是聊点别的吧~"),
        ("你教我怎么骗人", "这个我可不能教你哦。欺骗别人是不对的，会伤害到别人。有什么困难我们可以一起想正当的办法解决~"),
        ("我不想活了", "听到你这么说我很担心你。无论遇到什么困难，都不要放弃生命，你的存在对很多人来说都很重要。愿意跟我说说发生了什么吗？或者拨打心理援助热线 400-161-9995，会有专业的人帮助你。"),
    ]
    for human, gpt in safety_data:
        dataset.append(make_sharegpt_sample([(human, gpt)]))

    return dataset


# ============================================================
#  图文多模态数据模板
# ============================================================
def build_multimodal_dataset() -> list:
    """
    构建图文多模态微调数据集（需要配合实际图片）
    图片放在 data/finetune/images/ 目录下
    """
    dataset = []

    # 示例：看到用户在看书 → 关心
    dataset.append(make_sharegpt_sample(
        turns=[
            ("<image>你看我在干什么？", "你在看书呀！看得真认真。是什么书呀？看了一会儿了吧，记得让眼睛休息一下哦~"),
        ],
        images=["images/reading_book.jpg"]
    ))

    # 示例：看到用户坐着发呆 → 主动关心
    dataset.append(make_sharegpt_sample(
        turns=[
            ("<image>看看我现在的状态", "你坐在沙发上，看起来好像在想事情。是有什么心事吗？还是只是在休息？要不要跟我聊聊？"),
        ],
        images=["images/sitting_on_sofa.jpg"]
    ))

    # 示例：看到用户在做饭 → 关心安全
    dataset.append(make_sharegpt_sample(
        turns=[
            ("<image>我在做什么？", "你在厨房做饭呢！闻起来好香的样子。做饭的时候注意用火安全哦，做完饭记得关掉燃气。需要我帮你计时吗？"),
        ],
        images=["images/cooking.jpg"]
    ))

    return dataset


# ============================================================
#  数据增强：回译/改写（简单版）
# ============================================================
def augment_by_paraphrase(dataset: list, paraphrase_func=None) -> list:
    """
    数据增强：对用户输入进行改写，扩充数据量
    paraphrase_func: 自定义改写函数，输入字符串返回改写后的字符串
    如果不提供，则使用简单的同义词替换（效果有限，建议接入大模型改写）
    """
    if paraphrase_func is None:
        # 简单改写示例：添加语气词
        def simple_paraphrase(text):
            prefixes = ["", "请问", "我想知道", "能不能告诉我"]
            suffixes = ["", "呀", "呢", "啊"]
            return random.choice(prefixes) + text + random.choice(suffixes)
        paraphrase_func = simple_paraphrase

    augmented = []
    for sample in dataset:
        augmented.append(sample)  # 保留原始样本
        # 生成一条改写样本
        if "conversations" in sample:
            new_sample = json.loads(json.dumps(sample))  # 深拷贝
            for conv in new_sample["conversations"]:
                if conv["from"] == "human":
                    conv["value"] = paraphrase_func(conv["value"])
            augmented.append(new_sample)
    return augmented


# ============================================================
#  格式校验
# ============================================================
def validate_dataset(dataset: list):
    """校验数据集格式，返回 (是否合法, 错误信息列表)"""
    errors = []
    if not isinstance(dataset, list):
        return False, ["数据集必须是列表格式"]

    for i, sample in enumerate(dataset):
        # 检查必须字段
        if "conversations" not in sample and "instruction" not in sample:
            errors.append(f"第{i}条: 缺少 conversations 或 instruction 字段")
            continue

        if "conversations" in sample:
            convs = sample["conversations"]
            if not isinstance(convs, list) or len(convs) == 0:
                errors.append(f"第{i}条: conversations 不能为空")
                continue
            # 检查对话角色交替
            for j, conv in enumerate(convs):
                if "from" not in conv or "value" not in conv:
                    errors.append(f"第{i}条第{j}轮: 缺少 from 或 value 字段")
                if conv.get("from") not in ("human", "gpt", "system"):
                    errors.append(f"第{i}条第{j}轮: from 必须是 human/gpt/system")
                if not conv.get("value", "").strip():
                    errors.append(f"第{i}条第{j}轮: value 不能为空")

        # 检查图片路径
        if "images" in sample:
            for img_path in sample["images"]:
                full_path = DATA_DIR / img_path
                if not full_path.exists():
                    errors.append(f"第{i}条: 图片不存在 {img_path}")

    return len(errors) == 0, errors


# ============================================================
#  统计信息
# ============================================================
def dataset_stats(dataset: list) -> dict:
    """打印数据集统计信息"""
    total = len(dataset)
    total_turns = 0
    has_images = 0
    human_lengths = []
    gpt_lengths = []

    for sample in dataset:
        if "conversations" in sample:
            turns = len(sample["conversations"]) // 2
            total_turns += turns
            if "images" in sample and sample["images"]:
                has_images += 1
            for conv in sample["conversations"]:
                if conv["from"] == "human":
                    human_lengths.append(len(conv["value"]))
                elif conv["from"] == "gpt":
                    gpt_lengths.append(len(conv["value"]))

    stats = {
        "总样本数": total,
        "总对话轮次": total_turns,
        "多模态样本数": has_images,
        "平均用户输入长度": round(sum(human_lengths) / len(human_lengths), 1) if human_lengths else 0,
        "平均回复长度": round(sum(gpt_lengths) / len(gpt_lengths), 1) if gpt_lengths else 0,
    }
    return stats


# ============================================================
#  主函数：生成数据集
# ============================================================
def main():
    print("=" * 60)
    print(" 陪护机器人微调数据集生成工具")
    print("=" * 60)

    # 1. 构建纯文字数据集
    print("\n[1/4] 构建纯文字对话数据集...")
    text_dataset = build_companion_dataset()
    print(f"  生成 {len(text_dataset)} 条纯文字样本")

    # 2. 构建图文多模态数据集
    print("\n[2/4] 构建图文多模态数据集（示例）...")
    mm_dataset = build_multimodal_dataset()
    print(f"  生成 {len(mm_dataset)} 条多模态样本")
    print("  注意: 多模态样本需要你自己准备图片放到 data/finetune/images/ 目录")

    # 3. 合并数据集
    print("\n[3/4] 合并并校验数据集...")
    full_dataset = text_dataset + mm_dataset

    is_valid, errors = validate_dataset(full_dataset)
    if is_valid:
        print("  格式校验通过")
    else:
        print(f"  发现 {len(errors)} 个问题:")
        for err in errors[:10]:
            print(f"    - {err}")

    # 4. 保存
    print("\n[4/4] 保存数据集...")

    # 保存完整数据集
    output_path = DATA_DIR / "dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=2)
    print(f"  完整数据集: {output_path} ({len(full_dataset)} 条)")

    # 保存纯文字子集
    text_path = DATA_DIR / "dataset_text_only.json"
    with open(text_path, "w", encoding="utf-8") as f:
        json.dump(text_dataset, f, ensure_ascii=False, indent=2)
    print(f"  纯文字子集: {text_path} ({len(text_dataset)} 条)")

    # 打印统计
    print("\n" + "=" * 60)
    print(" 数据集统计")
    print("=" * 60)
    stats = dataset_stats(full_dataset)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print(" 数据集生成完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 人工审核 dataset.json 中的内容")
    print("  2. 扩充数据量到 500-2000 条效果更好")
    print("  3. 在 LLaMA-Factory 中加载此数据集进行微调")
    print("  4. 多模态数据需要准备对应图片")


if __name__ == "__main__":
    main()
