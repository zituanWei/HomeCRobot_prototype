"""
全局配置文件
所有可调参数集中在此，方便统一管理
"""
import os
from pathlib import Path

# ========== 路径配置 ==========
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"              # 图片/录音等临时数据
MODEL_DIR = PROJECT_ROOT / "models"           # 本地模型缓存目录
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ========== 视觉语言模型配置 ==========
class VLConfig:
    # 模型选择（按显存从小到大排列，8GB 显存推荐 3B-4bit）
    # - "Qwen/Qwen2.5-VL-3B-Instruct"  (~6GB FP16 / ~3GB 4bit)  ✅ 推荐
    # - "Qwen/Qwen2-VL-2B-Instruct"    (~4GB FP16 / ~2GB 4bit)  更轻量
    # - "Qwen/Qwen2.5-VL-7B-Instruct"  (~14GB FP16 / ~5GB 4bit) 需关闭其他程序
    MODEL_NAME = r"D:\PyCharm_projects\HomeCRobot\models\Qwen2.5-VL-3B-Instruct"

    # 量化配置：4bit 大幅降低显存，8bit 精度更高但更占显存
    LOAD_IN_4BIT = True
    LOAD_IN_8BIT = False          # 4bit 和 8bit 只能开一个

    # 生成参数
    MAX_NEW_TOKENS = 512          # 回复最大长度
    TEMPERATURE = 0.3             # 创造性：0=确定，1=随机
    TOP_P = 0.9
    DO_SAMPLE = True

    # === 图片分辨率限制（控制显存峰值的关键参数）===
    # Qwen2.5-VL 把图片切成 28x28 的 patch，patch 数量就是视觉 token 数。
    # 一张 1280x720 的图约 1175 个 patch，视觉推理的显存峰值主要来自这里。
    #
    # 为什么必须限制：7B 模型在 8GB 显卡上显存占用已接近上限
    # （实测 7.8/8.2 GB）。视觉推理瞬间申请更多显存时若不够，
    # 在 Windows 上会触发显存换页（变慢几十倍），严重时进程被系统
    # 直接杀掉——表现为「发出消息后一直没有回应」。
    #
    # 换算：max_pixels = 目标patch数 × 28 × 28
    #   不限（1280x720 原始）  -> 约 1175 patch
    #   768 × 28 × 28 = 602112 -> 约  768 patch（省 35%）
    #   512 × 28 × 28 = 401408 -> 约  512 patch（省 56%）
    #
    # 权衡：patch 越少越省显存，但小字与细节识别能力下降。
    #   家庭陪护场景（有没有人 / 什么姿势 / 显眼物体）768 够用；
    #   需要辨认远处小字时应调大，或改用 3B 模型（显存占用减半）。
    IMAGE_MAX_PIXELS = 768 * 28 * 28
    IMAGE_MIN_PIXELS = 4 * 28 * 28

    # 系统提示词：定义陪护机器人人设
    SYSTEM_PROMPT = (
        "你是一个温暖、耐心的家庭陪护机器人，名叫「小护」。\n"
        "\n"
        "【称呼规则（最重要，务必严格遵守）】\n"
        "· 「我」永远指你自己（小护）；「你」永远指和你说话的用户。\n"
        "· 用户的名字是**他的**，不是你的。绝对不能用第一人称「我」去说用户的名字。\n"
        "· 只有收到形如「[当前和你说话的用户叫「某某」]」的提示时，\n"
        "  才说明系统认出了用户，这时才可以用那个名字称呼他。\n"
        "· **没有**收到这类提示时，说明系统没认出这个人：\n"
        "  此时绝不要猜名字、不要编名字，也不要把对话历史里出现过的名字\n"
        "  当成当前用户的名字。被问到「我是谁」又没收到提示时，如实说还不认识他。\n"
        "· 当收到身份提示且用户问「我是谁」时，用第二人称「你是XX」回答，\n"
        "  而不是「我是XX」（那是错的，等于把用户当成了自己）。\n"
        "· 下面示例里的名字只是**格式演示**，不是真实用户，切勿照搬：\n"
        "    收到提示「[当前和你说话的用户叫「张三」]」，用户问：我是谁？\n"
        "    你应该答：你是张三呀，我一直记得你呢。\n"
        "  错误示例（绝对不要这样）：\n"
        "    「我就是张三」  ← 把用户的名字当成了自己的\n"
        "    「你好呀，小护」  ← 用你自己的名字称呼用户\n"
        "\n"
        "【身份信息】\n"
        "· 收到「[当前和你说话的用户叫「某某」]」表示摄像头认出了说话的人，\n"
        "  那是**用户**的名字。\n"
        "· 其他情况下不要刻意提名字，也不要把话题引向身份。\n"
        "· 你是机器人小护，用户是你的家人，两者不能混淆。\n"
        "\n"
        "【回答要求】\n"
        "1. 直接回答用户问的那件事，不要答非所问或转移话题\n"
        "2. 用亲切自然的中文，简短温暖，像家人一样\n"
        "3. 如果不确定或看不到，就直说，不要编造\n"
        "4. 看到用户有异常（摔倒、长时间不动）时主动询问是否需要帮助\n"
        "\n"
        "始终保持友善、尊重和关怀的态度。"
    )

# ========== 摄像头配置 ==========
class CameraConfig:
    CAMERA_INDEX = 0              # 摄像头编号，笔记本内置通常为0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    FPS = 30
    # 自动拍照间隔（秒），设为0则只在用户要求时拍照
    AUTO_CAPTURE_INTERVAL = 0

    # ---- 拍照存储与容量控制 ----
    # 普通拍照（/look、主动拍照等）统一存这里，超限后**先删最早的**
    PHOTO_DIR = DATA_DIR / "photos"
    PHOTO_MAX_MB = 500            # 目录总容量上限（MB），超出后删最早的
    PHOTO_MAX_COUNT = 2000        # 张数上限，超出后删最早的
    PHOTO_JPEG_QUALITY = 95       # JPEG 质量


# ========== 人脸识别配置 ==========
class FaceConfig:
    # ---- 存储布局（按人分目录）----
    #   data/faces/<姓名>/1.jpg, 2.jpg, 3.jpg ...
    #   data/face_database.json 只存特征索引，与目录一一对应
    #
    # 原图**必须保留**，因为：
    #   1) 认不出时能人工看一眼当初注册的是哪张脸
    #   2) 换模型/调分辨率时可以重新提特征，不必让人重拍
    FACE_DIR = DATA_DIR / "faces"
    DB_FILE = DATA_DIR / "face_database.json"

    # ---- 识别阈值 ----
    # SFace 余弦相似度。官方推荐 0.363；越低越容易认出（但也越容易认错）。
    # 实测同一人在不同帧会时认时不认，所以配合「一人多照片」使用。
    # 倾向：0.45=宁可认不出  0.4=折中(当前)  0.363=宁可认出来
    THRESHOLD = 0.4

    # ---- 注册质量门槛（不达标就拒绝，避免存进垃圾特征）----
    REQUIRE_SINGLE_FACE = True    # 画面里必须只有一张脸
    MIN_FACE_RATIO = 0.04         # 人脸面积占画面比例下限（太小=离太远）
    MIN_DET_SCORE = 0.9           # 检测置信度下限
    MIN_SHARPNESS = 60.0          # 拉普拉斯方差下限（越低越模糊）
    MAX_PHOTOS_PER_PERSON = 8     # 每人最多保留几张照片

    # ---- 检测分辨率 ----
    # 320 太小，脸稍小就提不出有效特征；640 只多几毫秒（轮询 1.5s 完全无感）
    DETECT_SIZE = 640

    # ---- 认不出时的"引导注册"（两步确认）----
    # 场景：用户说了身份相关的话（"我是谁""你认识我吗"），但系统没认出他。
    # 流程：主动问名字 -> 复述确认 -> 用户点头后才拍照注册。
    #
    # 为什么必须两步确认：
    #   1) 语音识别名字经常出错（实测「武凉烨」->「五粮液」），
    #      一步到位会把错名字写进人脸库；
    #   2) 若"说个名字就注册"，任何人都能对着摄像头冒充家人。
    ENABLE_GUIDED_ENROLL = True
    # 只在"人脸库里已经有人、但没认出当前这个人"时才问。
    # 库为空 = 全新环境，此时追问会让第一次来的客人很突兀。
    GUIDED_ENROLL_REQUIRE_EXISTING = True
    # 名字确认环节最多让用户重说几次
    GUIDED_ENROLL_MAX_RETRIES = 2

# ========== 语音配置 ==========
class VoiceConfig:
    # --- 总开关 ---
    ENABLE_ASR = True            # 是否启用语音识别（麦克风输入）
    ENABLE_TTS = True             # 是否启用语音合成（扬声器输出）
    # 启动后自动进入语音对话模式（需要 ASR 与 TTS 都可用）
    AUTO_VOICE_MODE = True
    # 语音模式下是否同时保留键盘输入（建议 True，方便调试）
    KEEP_KEYBOARD_IN_VOICE = True

    # --- 音频设备 ---
    # 查看编号: python scripts/calibrate_mic.py --scan
    #   >>> 该命令会让你对着麦克风说话，逐个设备测拾音，是最可靠的选法 <<<
    #
    # ⚠️ 关于蓝牙耳机（重要）：
    #   蓝牙耳机有两个身份，设备列表里会分别出现：
    #     · A2DP（音乐模式）    —— 只有输出，没有麦克风输入
    #     · Hands-Free（通讯模式）—— 有麦克风，但音质明显变差
    #   平时语音通话正常说明 Hands-Free 是工作的，只是程序默认可能挑中
    #   那个"没有麦克风的 A2DP 条目"。
    #   另外：蓝牙耳机断开后这些设备条目会消失、编号会变，
    #   所以不建议写死编号，除非你固定戴同一副耳机。
    #
    # 结论：不要留 None —— 实测 sounddevice 的"系统默认"会挑中
    #       [1] 耳机 (P3 PRO)，而它的 peak 仅 0.00002（几乎没有输入），
    #       结果就是"说话没反应"。
    #
    # 本机实测（扬声器放测试音、逐个设备录回，peak 越大越灵敏）：
    #   [ 8] 麦克风 (Realtek)      0.08092  ★ 正常，推荐
    #   [ 2] 麦克风 (Realtek)      0.01382
    #   [25] 电脑扬声器(回环)        0.00098
    #   [14] 耳机 P3 PRO (WASAPI)  0.00474  仅底噪级
    #   [30] 耳机 P3 PRO Hands-Free 0.00312  仅底噪级
    #   [ 1] 耳机 P3 PRO           0.00002  ← 默认会选它，几乎无信号
    #   [ 0] Microsoft 声音映射器    0.00002
    #
    # ⚠️ 蓝牙耳机麦克风是间歇性的：同一个 Hands-Free 端点在某些时刻
    #    能测到 0.29，另一些时刻只有 0.00002。要用耳机麦必须确保它在
    #    "通讯/Hands-Free"模式并且系统把它设为默认录音设备。
    #    稳妥做法是用内置麦（下面的 8），或先跑 --scan 实测再填。
    MIC_DEVICE = 8                # 麦克风设备编号（None=系统默认，实测会选错）
    SPEAKER_DEVICE = None         # 扬声器设备编号
    # 启动自检：采样这段时间内若峰值低于 MIC_HEALTH_MIN 就警告设备可能有问题
    MIC_HEALTH_MIN = 0.0008       # 正常说话应在 0.01~0.1；低于此值基本是没拾音
    ASR_SAMPLE_RATE = 16000       # 识别采样率（FunASR 固定 16k）
    BLOCK_MS = 50                 # 采集块大小（毫秒）

    # --- VAD 端点检测（决定"什么时候算说完一句"）---
    # ⚠️ 这几个值直接影响识别准确率，实测踩过的坑：
    #
    #  1) MIN_SILENCE 太短（700ms）会把一句话**切成几段分别识别**，
    #     每段都被硬猜成文字 -> 拼起来就是乱码。已调大到 1400ms。
    #
    #  2) 阈值太低（0.002）会让**环境噪声/打字声**触发"开始说话"，
    #     VAD 于是一路累积到 MAX_SPEECH 上限才切断，产生超长音频段
    #     （实测出现过单次推理 RTF=9.2、耗时 52 秒的异常），
    #     而且 FunASR 收到这种段只会回 "empty speech"。
    #     所以阈值必须高于实际底噪，用 calibrate_mic.py 实测后填。
    #
    #  3) PREBUFFER 是"起点前保留多少音频"，太少会吃掉第一个字。
    #
    # 建议：先跑 `python scripts/calibrate_mic.py --scan` 选设备，
    #       再跑 `--device N` 得到适配你麦克风的阈值。
    VAD_START_THRESHOLD = 0.006   # 超过此音量认为开始说话（须高于底噪）
    VAD_END_THRESHOLD = 0.004     # 低于此音量认为可能说完了
    VAD_SOFT_THRESHOLD_RATIO = 0.7  # 软起始阈值 = START × 此比例
    VAD_MIN_SILENCE_MS = 1400     # ★ 静音持续多久算一句话结束
    VAD_PREBUFFER_MS = 600        # ★ 起点前保留多少音频（防第一个字被切掉）
    VAD_MAX_SPEECH_MS = 12000     # 单句最长时长（过长多半是噪声堆积，限制它）
    VAD_IDLE_FLUSH_MS = 1800      # 麦克风安静这么久就收口悬着的语音段
    VAD_POLL_INTERVAL = 0.1       # VAD 线程轮询间隔（秒）

    # --- 打断与自激抑制 ---
    BARGE_IN = True               # 用户开口时是否打断机器人正在说的话
    BARGE_IN_RMS = 0.006          # 判定"用户真的在说话"的音量阈值

    # --- 识别模型 ---
    ASR_MODEL = "paraformer-zh"   # 中文语音识别模型
    ASR_VAD_MODEL = "fsmn-vad"    # 语音端点检测（FunASR 内置）
    ASR_PRELOAD = True            # 启动时后台预热 ASR 模型

    # --- 无效语音过滤（挡咳嗽/清嗓/环境噪声）---
    # 原理：人说话有**基频谐波结构**（周期性），咳嗽/敲击/清嗓是宽带噪声（无周期）。
    # 注意：唱歌同样有基频，所以这一层挡不住唱歌（那需要唤醒词或声纹）。
    MIN_SPEECH_MS = 400           # 短于此时长视为咳嗽/杂音（会误伤"嗯""对"这类单字）
    ENABLE_PERIODICITY_CHECK = True   # 用周期性判断是否像人声
    MIN_PERIODICITY = 0.30        # 自相关峰值下限，低于此视为噪声段

    # --- 识别结果过滤与去重 ---
    # 环境里清嗓常被识别成"嗯"、他人说话会混进来，这里做一层兜底过滤。
    DROP_FILLER_WORDS = True      # 丢弃单个语气词（嗯/啊/哦/呃…）
    FILLER_WORDS = ["嗯", "啊", "哦", "呃", "额", "唉", "哎", "呀", "嘛", "唔",
                    "嗯嗯", "啊啊", "哦哦", "呃呃", "emm", "en"]
    DEDUP_REPEAT = True           # 用户重说同一句时，丢弃重复的那条
    # 相似度阈值：0.8 会误杀正常对话（"今天天气" vs "明天天气" = 0.83），
    # 所以放到 0.85，只拦"几乎一模一样"的重说。
    DEDUP_SIMILARITY = 0.85
    # 去重时间窗（秒）：超过这个间隔就算用户真的想重复说，不再判重。
    # 10 秒够覆盖"说错了立刻重说"，又不会把隔了一会儿的正常重复当纠正。
    DEDUP_WINDOW_SECONDS = 10.0

    # --- TTS 配置 ---
    # 引擎选择（音质 vs 延迟，实测数据见 README「TTS 引擎选择」）：
    #   "edge-tts"  : ★默认。**微软在线语音**（Edge 浏览器的朗读服务），音质最自然。
    #                 代价是合成 1.6~13.9s 且随网络波动——但配合下面的
    #                 自动降级，网络慢时会改用本地引擎先出声，不会哑掉。
    #   "piper"     : 本地神经TTS，合成仅 0.23s、完全离线、不占显存。
    #                 但 medium 档模型音质明显不如 edge-tts（实测对比后确认）。
    #   "sapi"      : Windows 本地语音（**微软 SAPI**），约0.5s，音质机械。
    #
    # 本项目为微软技术栈，默认与兜底均为微软语音：
    #   edge-tts（微软在线） -> 慢时降级 -> piper（开源，仅作离线兜底）
    #   如需更正式的微软方案，可升级到 Azure AI Speech（见 modules/tts.py）。
    TTS_ENGINE = "piper"

    # --- Piper 本地引擎（推荐）---
    # 模型下载：见 README「本地 TTS 部署」一节，或运行
    #   python scripts/download_piper.py
    PIPER_MODEL = "models/piper/zh_CN-huayan-medium.onnx"
    PIPER_USE_CUDA = False        # 用 CPU 即可（RTF 0.06，比实时快十几倍）
    PIPER_LENGTH_SCALE = 1.0      # 语速，越大越慢
    PIPER_NOISE_SCALE = 0.667     # 音色变化程度
    PIPER_NOISE_W = 0.8           # 韵律变化程度

    TTS_VOICE = "zh-CN-XiaoxiaoNeural"  # edge-tts 音色
    TTS_RATE = "+0%"              # 语速（edge-tts 专用）
    TTS_VOLUME = "+0%"            # 音量（edge-tts 专用）
    TTS_MAX_CHARS = 200           # 超过此长度截断后再播报（避免念一大段）
    TTS_CACHE_SIZE = 64           # 合成缓存条数（同一句话不重复联网合成）

    # --- 预合成常用语（降低"说了话却要等十几秒"的体验问题）---
    # 启动后在后台把高频回复先合成进缓存，命中时**零延迟**。
    # 这是在不牺牲 edge-tts 音质的前提下，唯一能压缩等待的手段。
    TTS_PRELOAD_PHRASES = True    # 启动后后台预合成
    TTS_PRELOAD_DELAY = 8.0       # 等主流程稳定多久后开始（秒），避免拖慢启动

    # 流式播放：edge-tts 是分块返回音频的，边收边播能降低"首字延迟"
    # 注意：edge-tts 的首块延迟取决于服务端合成速度，网络慢时可能十几秒，
    #       流式并不能消除它——真正要低延迟得用本地引擎（见下方自动降级）。
    TTS_STREAMING = True
    # 首块超时（秒）：合成迟迟不来时放弃流式
    TTS_STREAM_FIRST_TIMEOUT = 20.0

    # ★ 自动降级：在线引擎慢到无法忍受时，自动改用本地引擎先出声
    #   "piper" = 本地神经语音（离线、约0.2s、音质尚可）★ 首选兜底
    #   "sapi"  = Windows 本地语音（离线、约0.5s、音质机械）
    #   None    = 不降级，一直等在线合成（如果你只想要 edge-tts 音质）
    #
    # ⚠️ 超时值需要按你的网络调：
    #   实测某网络下 edge-tts 几乎每次都要 7 秒以上，用旧值（3+4）
    #   会导致**每轮都降级到 piper**，等于听不到 edge-tts 的音质。
    #   这里放宽到 6+8=14 秒：给在线合成更多机会，
    #   真慢到 14 秒才用本地引擎兜底（保证不失声）。
    #
    #   调参参考：
    #     TTS_FALLBACK_AFTER=2, TTS_SYNTH_WAIT=3   -> 响应快，常听到 piper
    #     TTS_FALLBACK_AFTER=6, TTS_SYNTH_WAIT=8   -> 当前值，偏向 edge-tts
    #     TTS_FALLBACK_ENGINE=None                 -> 死等 edge-tts（可能哑很久）
    TTS_FALLBACK_ENGINE = "piper"
    # 阶段1：流式首块最多等这么久（超时则改用整句合成）
    TTS_FALLBACK_AFTER = 6.0
    # 阶段2：整句合成最多再等这么久（超时则降级到本地引擎先出声）
    TTS_SYNTH_WAIT = 8.0

    # Windows SAPI 本地引擎参数
    SAPI_RATE = 0                 # 语速 -10~10（0 为正常）
    SAPI_VOLUME = 100             # 音量 0~100

# ========== Web 界面配置 ==========
class WebConfig:
    # ---- 机器人头像 ----
    # 支持两种：
    #   1) emoji / 单字  -> 直接显示（默认的 🐋 是"小鲸鱼"，不是鱼）
    #   2) 图片文件      -> 在网页上点头像上传，存到下面这个路径后自动使用
    #      也可以手动把图片命名为 avatar.png 放进 data/ 目录
    AVATAR_FILE = DATA_DIR / "avatar.png"
    AVATAR_DEFAULT = "🐋"
    AVATAR_MAX_MB = 5              # 上传图片大小上限（MB）

    # ---- 回复呈现 ----
    # True = 打字机效果逐字显示（观感更接近真人说话）
    # 说明：这是前端呈现效果，不是真正的模型流式输出；
    #       真正的流式需要改 core/vl_model.py，改动较大故未采用。
    TYPEWRITER = True
    TYPEWRITER_INTERVAL = 0.018    # 每字间隔（秒）

    # ---- 服务 ----
    HOST = "127.0.0.1"
    PORT = 8000

# ========== 对话配置 ==========
class ConversationConfig:
    MAX_HISTORY_TURNS = 25        # 保留最近多少轮对话上下文
    HISTORY_FILE = DATA_DIR / "conversation_history.json"

# ========== 后台视觉监控配置 ==========
class MonitorConfig:
    ENABLE_MONITOR = True          # 是否启用后台视觉监控
    INTERVAL_SECONDS = 60          # VL 深度分析间隔（秒），建议30-120，太频繁费GPU
    INACTIVE_THRESHOLD_SECONDS = 1800  # 久坐不动提醒阈值（30分钟）
    NO_PERSON_THRESHOLD_COUNT = 5  # 连续N次没看到人则主动询问
    PROACTIVE_COOLDOWN_SECONDS = 300  # 两次主动消息最小间隔（5分钟），避免打扰

    # ---------------- 实时"有人出现"检测 ----------------
    # 用轻量的 YuNet 人脸检测做高频轮询（毫秒级、不占 GPU），
    # 只有"没人 -> 有人"这种状态变化时才触发识别与打招呼，
    # 因此可以把间隔调得很短，做到接近实时的反应。
    ENABLE_PRESENCE_WATCH = True
    PRESENCE_INTERVAL_SECONDS = 1.5   # 人脸存在性检测间隔（秒）
    PRESENCE_ABSENT_SECONDS = 8.0     # 连续多久没人，才算"离开了"
    PRESENCE_MIN_FACE_RATIO = 0.02    # 人脸面积占画面比例下限（过滤远处误检）

    # ---------------- 主动打招呼 ----------------
    ENABLE_GREETING = True            # 检测到人出现时主动打招呼
    GREETING_COOLDOWN_SECONDS = 120   # 同一个人两次打招呼的最小间隔
    GREETING_UNKNOWN_COOLDOWN = 180   # 陌生人问候间隔（可稍长，避免对着路人反复说）
    GREETING_USE_VL = True            # 打招呼时是否调用 VL 看场景（更自然，但更慢更费GPU）
    GREETING_WITH_TTS = True          # 打招呼时是否语音播报

    # ---------------- 异常与久坐（沿用 VL 分析）----------------
    ENABLE_ABNORMAL_DETECT = True     # 是否启用摔倒/不适等异常检测
