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
        --top-k 5 \
        --output result_gpu.mp4

注意:
    - 需要 CUDA 10.1+ 和兼容版本的 PyTorch（GPU 版）
    - 若无 GPU，脚本将自动回退到 CPU 并给出提示
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
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
    parser.add_argument(
        "--output",
        type=str,
        default="result_gpu.mp4",
        help="输出带标注的视频文件路径（默认: result_gpu.mp4）",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="实时显示标注窗口（需要图形界面，默认关闭）",
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
    print(f"  视频:   {args.video}")
    print(f"  设备:   {device_str}  [{mode}]")
    print(f"  输出:   {args.output}")
    print(f"  显示:   {'是' if args.display else '否'}")
    print("=" * 60)
    print()


def draw_results_on_frame(frame, results, top_k, elapsed, device_label="GPU"):
    """在帧上绘制 Top-K 识别结果、置信度条和推理耗时"""
    h, w = frame.shape[:2]

    # 半透明背景面板
    overlay = frame.copy()
    panel_h = 36 + top_k * 34 + 40
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # 标题行
    title = f"Video Swin Transformer [{device_label}]  推理耗时: {elapsed:.2f}s"
    cv2.putText(frame, title, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)

    top_k = min(top_k, len(results))
    for rank, (label_name, score) in enumerate(results[:top_k]):
        y_base = 52 + rank * 34
        bar_max_w = int(w * 0.45)
        bar_w = int(bar_max_w * min(score, 1.0))

        # 颜色：第1名金色，其余蓝绿渐变
        if rank == 0:
            color = (0, 215, 255)
        else:
            g = max(80, 200 - rank * 30)
            color = (0, g, 180)

        # 置信度背景条（灰）
        cv2.rectangle(frame, (12, y_base - 18), (12 + bar_max_w, y_base + 4),
                      (60, 60, 60), -1)
        # 置信度彩色条
        if bar_w > 0:
            cv2.rectangle(frame, (12, y_base - 18), (12 + bar_w, y_base + 4),
                          color, -1)

        # 标签文字
        text = f"#{rank+1} {label_name}  {score:.4f}"
        cv2.putText(frame, text, (18 + bar_max_w + 8, y_base),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return frame


def save_annotated_video(video_path, results, top_k, elapsed, output_path,
                         display=False, device_label="GPU"):
    """读取原始视频，叠加识别结果后写入新文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[警告] 无法打开视频文件: {video_path}，跳过视频保存")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[可视化] 正在生成标注视频: {output_path}")
    print(f"  分辨率: {width}x{height}, FPS: {fps:.1f}, 总帧数: {total}")

    top_k_clamped = min(top_k, len(results))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = draw_results_on_frame(frame, results, top_k_clamped, elapsed, device_label)
        out.write(frame)
        if display:
            cv2.imshow(f"Video Swin Transformer — {device_label}", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
                break
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  已处理: {frame_idx}/{total} 帧", end="\r")

    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()

    print(f"\n  标注视频已保存至: {output_path}")


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

    # ── 可视化并保存 ──────────────────────────────────────────
    device_label = "GPU" if str(actual_device).startswith("cuda") else "CPU"
    if not args.use_frames:
        save_annotated_video(
            video_path=args.video,
            results=results,
            top_k=args.top_k,
            elapsed=elapsed,
            output_path=args.output,
            display=args.display,
            device_label=device_label,
        )
    else:
        print("[提示] 帧目录模式暂不支持视频保存，跳过。")

    return results


def main():
    args = parse_args()
    check_files(args)
    run_inference(args)


if __name__ == "__main__":
    main()
