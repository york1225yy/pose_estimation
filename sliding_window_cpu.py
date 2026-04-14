"""
Video Swin Transformer — CPU 模式滑动窗口连续行为识别脚本
=========================================================
原理：
    维护一个长度为 sample_length（由模型配置自动确定）的帧队列（滑动窗口）。
    每次收集到 --stride 帧新帧后，对窗口内的所有帧执行一次推理，并将
    最新 Top-K 结果叠加到输出视频的每一帧上。随着视频帧的变化，识别结果
    会动态更新，实现连续行为识别。

用法:
    python sliding_window_cpu.py \
        --video demo_video.mp4 \
        --config Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
        --checkpoint checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
        --label Video-Swin-Transformer/demo/label_map_k400.txt \
        --stride 8 \
        --top-k 5 \
        --threshold 0.01 \
        --output result_sliding_cpu.mp4

参数说明:
    --stride   每隔多少帧推理一次（越小越密集，CPU 负担越重，默认 8）
    --top-k    显示 Top-K 结果（默认 5）
    --threshold 分数低于该阈值的标签不显示（默认 0.01）
"""

import argparse
import os
import sys
import time
from collections import deque
from operator import itemgetter

import cv2
import numpy as np
import torch
from mmcv.parallel import collate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Swin Transformer — CPU 滑动窗口连续行为识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="待识别视频文件路径")
    parser.add_argument("--config", required=True, help="模型配置文件路径（.py）")
    parser.add_argument("--checkpoint", required=True, help="权重文件路径（.pth）")
    parser.add_argument("--label", required=True, help="标签映射文件路径（每行一个类别名）")
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="推理步长：每收集到 stride 帧新帧后执行一次推理（默认: 8）",
    )
    parser.add_argument("--top-k", type=int, default=5, help="显示 Top-K 结果（默认: 5）")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="分数低于该阈值的标签不显示（默认: 0.01）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result_sliding_cpu.mp4",
        help="输出视频文件路径（默认: result_sliding_cpu.mp4）",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="实时显示标注窗口（需要图形界面，默认关闭）",
    )
    return parser.parse_args()


def check_files(args):
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
        print("[错误] 以下文件不存在：")
        for m in missing:
            print(m)
        sys.exit(1)


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_pipeline(model_cfg):
    """从模型配置中提取测试 pipeline，去掉解码相关步骤，返回 (pipeline, sample_length, data_tmpl)"""
    from mmaction.datasets.pipelines import Compose

    EXCLUDED_STEPS = [
        "OpenCVInit", "OpenCVDecode",
        "DecordInit", "DecordDecode",
        "PyAVInit", "PyAVDecode",
        "RawFrameDecode", "FrameSelector",
    ]

    pipeline_cfg = model_cfg.data.test.pipeline
    pipeline_ = list(pipeline_cfg)
    sample_length = 0
    data_tmpl = dict(img_shape=None, modality="RGB", label=-1)

    for step in pipeline_cfg:
        if "SampleFrames" in step["type"]:
            sample_length = step["clip_len"] * step["num_clips"]
            data_tmpl["num_clips"] = step["num_clips"]
            data_tmpl["clip_len"] = step["clip_len"]
            pipeline_.remove(step)
        if step["type"] in EXCLUDED_STEPS:
            pipeline_.remove(step)

    assert sample_length > 0, "未能从配置中解析 sample_length，请检查配置文件中 SampleFrames 设置"
    return Compose(pipeline_), sample_length, data_tmpl


def infer_on_window(model, test_pipeline, data_tmpl, frame_window, device):
    """对当前帧窗口执行一次推理，返回 (label, score) 排序列表"""
    cur_data = data_tmpl.copy()
    if cur_data["img_shape"] is None:
        cur_data["img_shape"] = frame_window[0].shape[:2]
    cur_data["imgs"] = list(np.array(frame_window))
    cur_data = test_pipeline(cur_data)
    cur_data = collate([cur_data], samples_per_gpu=1)
    # CPU 模式不做 scatter
    with torch.no_grad():
        scores = model(return_loss=False, **cur_data)[0]
    return scores


def draw_overlay(frame, results, threshold, top_k):
    """在帧上绘制 Top-K 识别结果"""
    shown = 0
    for rank, (label_name, score) in enumerate(results):
        if score < threshold or shown >= top_k:
            break
        text = f"#{rank + 1} {label_name}  {score:.4f}"
        y = 32 + shown * 30
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 1, cv2.LINE_AA)
        shown += 1
    if shown == 0:
        cv2.putText(frame, "Waiting for action ...", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 128), 1, cv2.LINE_AA)
    return frame


def run(args):
    # ── 初始化 ────────────────────────────────────────────────
    vst_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Video-Swin-Transformer")
    if vst_dir not in sys.path:
        sys.path.insert(0, vst_dir)

    try:
        from mmcv import Config
        from mmaction.apis import init_recognizer
    except ImportError as e:
        print(f"[错误] 依赖未安装: {e}\n请先运行: bash setup_env.sh")
        sys.exit(1)

    device = torch.device("cpu")
    labels = load_labels(args.label)

    print("=" * 62)
    print("  Video Swin Transformer — CPU 滑动窗口连续行为识别")
    print(f"  视频:   {args.video}")
    print(f"  步长:   每 {args.stride} 帧推理一次")
    print(f"  输出:   {args.output}")
    print("=" * 62)

    # ── 加载模型 ──────────────────────────────────────────────
    print("\n[1/3] 加载模型...")
    cfg = Config.fromfile(args.config)
    model = init_recognizer(cfg, args.checkpoint, device=device)
    model.eval()
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

    test_pipeline, sample_length, data_tmpl = build_pipeline(model.cfg)
    print(f"  滑动窗口长度: {sample_length} 帧")

    # ── 打开视频 ──────────────────────────────────────────────
    print("\n[2/3] 打开视频...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  分辨率: {width}x{height}  FPS: {fps:.1f}  总帧数: {total}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"[错误] 无法创建输出视频: {args.output}")
        cap.release()
        sys.exit(1)

    if args.display:
        cv2.namedWindow("滑动窗口行为识别 [CPU]", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("滑动窗口行为识别 [CPU]", min(width, 1280), min(height, 720))

    # ── 逐帧处理 ──────────────────────────────────────────────
    print("\n[3/3] 开始逐帧处理（CPU 模式，请耐心等待）...")
    frame_queue = deque(maxlen=sample_length)   # 滑动窗口
    current_results = []                         # 最新推理结果
    frame_idx = 0
    new_frame_count = 0                          # 自上次推理后新加入的帧数
    infer_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # BGR -> RGB 放入窗口
            frame_rgb = frame[:, :, ::-1].copy()
            frame_queue.append(frame_rgb)
            new_frame_count += 1

            # 窗口满后，每 stride 帧推理一次
            if len(frame_queue) == sample_length and new_frame_count >= args.stride:
                scores = infer_on_window(model, test_pipeline, data_tmpl,
                                         frame_queue, device)
                scores_tuples = list(zip(labels, scores))
                current_results = sorted(scores_tuples, key=itemgetter(1), reverse=True)
                new_frame_count = 0
                infer_count += 1

            # 绘制当前结果
            annotated = draw_overlay(frame.copy(), current_results, args.threshold, args.top_k)
            writer.write(annotated)

            if args.display:
                cv2.imshow("滑动窗口行为识别 [CPU]", annotated)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                elapsed = time.time() - t_start
                print(f"  进度: {frame_idx}/{total}  推理次数: {infer_count}  "
                      f"已用时: {elapsed:.1f}s", end="\r")
    finally:
        cap.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n  处理完成: 共 {frame_idx} 帧，推理 {infer_count} 次，耗时 {elapsed:.1f}s")
    print(f"  输出已保存: {args.output}")


def main():
    args = parse_args()
    check_files(args)
    run(args)


if __name__ == "__main__":
    main()
