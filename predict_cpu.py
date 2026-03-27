"""
Video Swin Transformer — CPU 模式视频行为识别推理脚本
=====================================================
用法:
    python predict_cpu.py \
        --video demo_video.mp4 \
        --config Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
        --checkpoint checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
        --label Video-Swin-Transformer/demo/label_map_k400.txt \
        --top-k 5
"""

import argparse
import os
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Swin Transformer — CPU 推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="待识别的视频文件路径（.mp4/.avi 等）",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="模型配置文件路径（.py），例如 "
        "Video-Swin-Transformer/configs/recognition/swin/"
        "swin_tiny_patch244_window877_kinetics400_1k.py",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="权重文件路径（.pth），例如 "
        "checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="标签映射文件路径（每行一个类别名），例如 "
        "Video-Swin-Transformer/demo/label_map_k400.txt",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="输出 Top-K 结果（默认: 5）",
    )
    parser.add_argument(
        "--use-frames",
        action="store_true",
        default=False,
        help="若视频路径为帧目录，请启用此选项",
    )
    return parser.parse_args()


def check_files(args):
    """检查必要文件是否存在"""
    missing = []
    for path, name in [
        (args.video, "视频文件"),
        (args.config, "配置文件"),
        (args.checkpoint, "权重文件"),
        (args.label, "标签文件"),
    ]:
        if not os.path.exists(path):
            missing.append(f"  ✗ {name}: {path}")
    if missing:
        print("[错误] 以下文件不存在，请检查路径：")
        for m in missing:
            print(m)
        sys.exit(1)


def load_labels(label_path):
    """加载标签映射文件"""
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def print_banner(args):
    print("=" * 60)
    print("  Video Swin Transformer — CPU 推理")
    print(f"  视频: {args.video}")
    print(f"  设备: cpu")
    print("=" * 60)
    print()


def run_inference(args):
    """在 CPU 上执行视频行为识别推理"""
    # 将 Video-Swin-Transformer 加入模块搜索路径
    vst_dir = os.path.join(os.path.dirname(__file__), "Video-Swin-Transformer")
    if vst_dir not in sys.path:
        sys.path.insert(0, vst_dir)

    try:
        from mmcv import Config
        from mmaction.apis import inference_recognizer, init_recognizer
    except ImportError as e:
        print(f"[错误] 依赖未安装: {e}")
        print("请先运行: bash setup_env.sh")
        sys.exit(1)

    device = torch.device("cpu")

    # ── 步骤 1：加载模型 ──────────────────────────────────────
    print("[步骤 1/3] 加载模型...")
    cfg = Config.fromfile(args.config)
    model = init_recognizer(cfg, args.checkpoint, device=device, use_frames=args.use_frames)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,}")
    print(f"  设备: {next(model.parameters()).device}")
    print()

    # ── 步骤 2：读取视频基本信息 ──────────────────────────────
    print("[步骤 2/3] 预处理视频...")
    if not args.use_frames:
        try:
            import decord
            video_reader = decord.VideoReader(args.video)
            n_frames = len(video_reader)
            fps = video_reader.get_avg_fps()
            h, w = video_reader[0].shape[:2]
            print(f"  视频信息: {n_frames} 帧, {fps:.1f} fps, 分辨率 {w}x{h}")
        except Exception:
            print("  （无法读取视频元信息，跳过）")
    print()

    # ── 步骤 3：执行推理 ──────────────────────────────────────
    print("[步骤 3/3] 执行推理（CPU 模式，耗时较长，请耐心等待）...")
    labels = load_labels(args.label)

    t0 = time.time()
    with torch.no_grad():
        results = inference_recognizer(
            model,
            args.video,
            args.label,
            use_frames=args.use_frames,
        )
    elapsed = time.time() - t0
    print(f"  推理耗时: {elapsed:.2f} 秒")
    print()

    # ── 输出结果 ──────────────────────────────────────────────
    top_k = min(args.top_k, len(results))
    print("=" * 60)
    print(f"  Top-{top_k} 行为识别结果（CPU 模式）")
    print("=" * 60)
    for i, (label_name, score) in enumerate(results[:top_k], 1):
        print(f"  #{i:<3} {label_name:<30} 得分: {score:>8.4f}")
    print()

    return results


def main():
    args = parse_args()
    print_banner(args)
    check_files(args)
    run_inference(args)


if __name__ == "__main__":
    main()
