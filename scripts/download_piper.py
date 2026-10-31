"""
Piper 本地 TTS 模型下载
=======================
Piper 是完全离线的神经语音合成，实测（CPU）：
    模型加载 2.25s（一次性）
    合成 0.07~0.27s   RTF 0.057（比实时快约 17 倍）
对比 edge-tts 在线的 2.5~13s，延迟低两个数量级。

模型体积约 60MB，下载到 models/piper/ 目录。

用法:
    python scripts/download_piper.py                 # 下载默认中文音色
    python scripts/download_piper.py --voice xiao_ya # 换一个音色
    python scripts/download_piper.py --list          # 列出可选音色
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 可选中文音色（repo_id, 文件名前缀, 说明）
VOICES = {
    "huayan": (
        "csukuangfj/vits-piper-zh_CN-huayan-medium",
        "zh_CN-huayan-medium",
        "华言女声，medium 质量，60MB —— 默认推荐",
    ),
    "xiao_ya": (
        "csukuangfj2/vits-piper-zh_CN-xiao_ya-medium",
        "zh_CN-xiao_ya-medium",
        "小雅女声，medium 质量",
    ),
}


def main():
    parser = argparse.ArgumentParser(description="下载 Piper 本地 TTS 模型")
    parser.add_argument("--voice", default="huayan", choices=list(VOICES.keys()),
                        help="要下载的音色")
    parser.add_argument("--list", action="store_true", help="列出可选音色")
    parser.add_argument("--dir", default=None, help="下载目录（默认 models/piper）")
    args = parser.parse_args()

    if args.list:
        print("可选中文音色：")
        for k, (repo, prefix, desc) in VOICES.items():
            print(f"  {k:<10} {desc}")
            print(f"             repo={repo}")
        return 0

    from config import PROJECT_ROOT
    out_dir = args.dir or os.path.join(PROJECT_ROOT, "models", "piper")
    os.makedirs(out_dir, exist_ok=True)

    repo_id, prefix, desc = VOICES[args.voice]
    print("=" * 62)
    print(f" 下载 Piper 中文音色: {args.voice}")
    print(f" {desc}")
    print("=" * 62)
    print(f" 目标目录: {out_dir}")

    # 优先用 hf-mirror（国内可达），并走 huggingface_hub 的请求头
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[FAIL] 需要 huggingface_hub：pip install huggingface_hub")
        return 1

    files = [f"{prefix}.onnx", f"{prefix}.onnx.json"]
    for fname in files:
        dst = os.path.join(out_dir, fname)
        if os.path.exists(dst) and os.path.getsize(dst) > 200:
            print(f"  [跳过] {fname} 已存在 ({os.path.getsize(dst)/1024/1024:.1f}MB)")
            continue
        print(f"  [下载] {fname} ...")
        try:
            path = hf_hub_download(repo_id=repo_id, filename=fname)
            shutil.copy(path, dst)
            print(f"         OK {os.path.getsize(dst)/1024/1024:.1f}MB")
        except Exception as e:
            print(f"         FAIL {type(e).__name__}: {e}")
            print(f"         可手动下载: https://hf-mirror.com/{repo_id}/resolve/main/{fname}")
            return 1

    print()
    print("=" * 62)
    print(" 完成！在 config.py 里设置：")
    print("=" * 62)
    print(f'    TTS_ENGINE = "piper"')
    print(f'    PIPER_MODEL = "models/piper/{prefix}.onnx"')
    print()
    print(" 验证：")
    print("    python scripts/test_tts_latency.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
