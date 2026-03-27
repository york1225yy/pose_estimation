"""
Video Swin Transformer — GPU 模式视频行为识别推理脚本
=====================================================
用法:
    python predict_gpu.py \
        --video demo_video.mp4 \
        --config Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
        --checkpoint checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
        --label Video-Swin-Transformer/demo/label_map_k400.txt \
        --device cuda:0 \
        --top-k 5

注意:
    - 需要 CUDA 10.1+ 和兼容版本的 PyTorch（GPU 版）
    - 若无 GPU，脚本将自动回退到 CPU 并给出提示
"""

import argparse
import os
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Swin Transformer — GPU 推理",
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
        "--device",
        type=str,
        default="cuda:0",
        help="计算设备（默认: cuda:0）。多卡环境可指定 cuda:1、cuda:2 等",
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


def resolve_device(requested_device: str) -> torch.device:
    """
    解析并验证计算设备。
    若请求 CUDA 但 CUDA 不可用，自动回退到 CPU 并打印警告。
    """
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[警告] CUDA 不可用，自动回退到 CPU 模式。")
            print("       如需使用 GPU，请安装 CUDA 版 PyTorch：")
            print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print()
            return torch.device("cpu")

        # 解析 GPU 编号
        if ":" in requested_device:
            gpu_id = int(requested_device.split(":")[1])
        else:
            gpu_id = 0

        n_gpus = torch.cuda.device_count()
        if gpu_id >= n_gpus:
            print(f"[警告] 指定 GPU {gpu_id} 不存在（共 {n_gpus} 个 GPU），使用 cuda:0。")
            gpu_id = 0

        device = torch.device(f"cuda:{gpu_id}")
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"  GPU 信息: {props.name}，显存 {vram_gb:.1f} GB")
        return device

    return torch.device(requested_device)


def load_labels(label_path):
    """加载标签映射文件"""
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def print_banner(args, actual_device: torch.device):
    device_str = str(actual_device)
    mode = "GPU 加速" if device_str.startswith("cuda") else "CPU（回退）"
    print("=" * 60)
    print("  Video Swin Transformer — GPU 推理")
    print(f"  视频: {args.video}")
    print(f"  设备: {device_str}  [{mode}]")
    print("=" * 60)
    print()


def run_inference(args):
    """在 GPU（或回退至 CPU）上执行视频行为识别推理"""
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

    # 解析设备
    device = resolve_device(args.device)
    print_banner(args, device)

    # ── 步骤 1：加载模型 ──────────────────────────────────────
    print("[步骤 1/3] 加载模型...")
    cfg = Config.fromfile(args.config)
    model = init_recognizer(cfg, args.checkpoint, device=device, use_frames=args.use_frames)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    actual_device = next(model.parameters()).device
    print(f"  模型参数量: {n_params:,}")
    print(f"  模型设备:   {actual_device}")
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
    mode_str = "GPU 加速" if str(actual_device).startswith("cuda") else "CPU"
    print(f"[步骤 3/3] 执行推理（{mode_str} 模式）...")
    labels = load_labels(args.label)

    # 同步 CUDA 以精确计时
    if str(actual_device).startswith("cuda"):
        torch.cuda.synchronize(actual_device)

    t0 = time.time()
    with torch.no_grad():
        results = inference_recognizer(
            model,
            args.video,
            args.label,
            use_frames=args.use_frames,
        )

    if str(actual_device).startswith("cuda"):
        torch.cuda.synchronize(actual_device)

    elapsed = time.time() - t0
    print(f"  推理耗时: {elapsed:.2f} 秒")
    print()

    # ── 输出结果 ──────────────────────────────────────────────
    top_k = min(args.top_k, len(results))
    print("=" * 60)
    print(f"  Top-{top_k} 行为识别结果（{mode_str} 模式）")
    print("=" * 60)
    for i, (label_name, score) in enumerate(results[:top_k], 1):
        print(f"  #{i:<3} {label_name:<30} 得分: {score:>8.4f}")
    print()

    return results


def main():
    args = parse_args()
    check_files(args)
    run_inference(args)


if __name__ == "__main__":
    main()
