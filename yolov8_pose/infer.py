#!/usr/bin/env python3
"""
YOLOv8-Pose 推理脚本
====================
支持：
  - 单张图片推理
  - 文件夹批量推理
  - 视频推理
输出：
  - 控制台逐帧打印所有几何参数
  - 结果保存为带骨骼 + 坐姿标注的图片 / 视频
  - CSV 参数日志

用法示例：
  python infer.py --source image.jpg
  python infer.py --source ./images/
  python infer.py --source video.mp4
  python infer.py --source video.mp4 --model yolov8m-pose.pt --device cuda:0
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 上层目录加入 path，以便导入 utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from utils.posture_rules import PostureClassifier, PostureParams
from utils.visualization import PoseVisualizer

# ── 支持的文件扩展名 ──
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8-Pose 驾驶员坐姿推理")
    parser.add_argument("--source", type=str, required=True,
                        help="输入源：图片路径 / 文件夹路径 / 视频路径")
    parser.add_argument("--model", type=str, default="yolov8s-pose.pt",
                        help="YOLOv8-Pose 模型权重路径 (默认: yolov8s-pose.pt)")
    parser.add_argument("--device", type=str, default="",
                        help="推理设备：cpu / cuda / cuda:0 (默认自动)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="检测置信度阈值 (默认: 0.5)")
    parser.add_argument("--kp-conf", type=float, default=0.4,
                        help="关键点置信度阈值 (默认: 0.4)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--output", type=str, default="output",
                        help="输出目录 (默认: output)")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹窗显示结果")
    parser.add_argument("--no-params", action="store_true",
                        help="不在画面上叠加参数面板")
    parser.add_argument("--save-csv", action="store_true", default=True,
                        help="保存参数 CSV（默认开启）")
    return parser.parse_args()


def process_frame(frame: np.ndarray,
                  model: YOLO,
                  classifier: PostureClassifier,
                  visualizer: PoseVisualizer,
                  args,
                  frame_idx: int = 0,
                  csv_writer=None,
                  source_name: str = ""):
    """处理单帧：检测 → 分类 → 可视化 → 打印 → CSV"""
    results = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                            verbose=False, device=args.device or None)

    annotated = frame.copy()
    result = results[0]

    if result.keypoints is None or len(result.keypoints) == 0:
        print(f"[帧 {frame_idx:06d}] 未检测到人体")
        return annotated

    keypoints_data = result.keypoints.data.cpu().numpy()   # (N, 17, 3)
    boxes = result.boxes

    for pid in range(len(keypoints_data)):
        kp_xyc = keypoints_data[pid]            # (17, 3) -> x, y, conf
        kp_xy = kp_xyc[:, :2]                   # (17, 2)
        kp_conf = kp_xyc[:, 2]                  # (17,)

        # 获取检测框
        bbox = None
        if boxes is not None and pid < len(boxes):
            bbox = boxes.xyxy[pid].cpu().numpy().tolist()

        # 坐姿分类
        params: PostureParams = classifier.classify(kp_xy, kp_conf)

        # 控制台输出
        print(f"[帧 {frame_idx:06d}] 人 {pid} ({source_name})")
        print(params.to_print_str())

        # CSV
        if csv_writer is not None:
            row = params.to_dict()
            row["帧序号"] = frame_idx
            row["人序号"] = pid
            row["来源"] = source_name
            csv_writer.writerow(row)

        # 可视化
        annotated = visualizer.draw(annotated, kp_xy, kp_conf, params,
                                    bbox=bbox, person_idx=pid)

    return annotated


def infer_image(image_path: str, model, classifier, visualizer, args,
                csv_writer=None):
    """推理单张图片"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        return

    name = Path(image_path).name
    print(f"\n{'='*60}")
    print(f"处理图片: {name}")
    print(f"{'='*60}")

    annotated = process_frame(frame, model, classifier, visualizer, args,
                              frame_idx=0, csv_writer=csv_writer,
                              source_name=name)

    # 保存
    out_path = os.path.join(args.output, f"result_{name}")
    cv2.imwrite(out_path, annotated)
    print(f"结果已保存: {out_path}")

    # 显示
    if not args.no_show:
        cv2.imshow("YOLOv8-Pose", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def infer_folder(folder_path: str, model, classifier, visualizer, args,
                 csv_writer=None):
    """推理文件夹中的所有图片"""
    files = sorted([
        f for f in Path(folder_path).iterdir()
        if f.suffix.lower() in IMG_EXTS
    ])
    if not files:
        print(f"[WARN] 文件夹中无图片文件: {folder_path}")
        return

    print(f"\n找到 {len(files)} 张图片")
    for idx, fp in enumerate(files):
        infer_image(str(fp), model, classifier, visualizer, args,
                    csv_writer=csv_writer)


def infer_video(video_path: str, model, classifier, visualizer, args,
                csv_writer=None):
    """推理视频"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    name = Path(video_path).stem
    out_path = os.path.join(args.output, f"result_{name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"\n{'='*60}")
    print(f"处理视频: {Path(video_path).name}")
    print(f"分辨率: {w}x{h}, FPS: {fps:.1f}, 总帧数: {total}")
    print(f"{'='*60}")

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = process_frame(frame, model, classifier, visualizer, args,
                                  frame_idx=frame_idx, csv_writer=csv_writer,
                                  source_name=Path(video_path).name)
        writer.write(annotated)

        if not args.no_show:
            cv2.imshow("YOLOv8-Pose", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("用户中断")
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - t_start
            speed = frame_idx / elapsed if elapsed > 0 else 0
            print(f"[进度] {frame_idx}/{total} 帧, {speed:.1f} FPS")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n视频处理完成: {frame_idx} 帧, 耗时 {elapsed:.1f}s, "
          f"平均 {frame_idx / elapsed:.1f} FPS")
    print(f"输出视频: {out_path}")


def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    print(f"加载模型: {args.model}")
    model = YOLO(args.model)
    print(f"模型加载完成 ✓")

    # 初始化分类器和可视化器
    classifier = PostureClassifier(conf_threshold=args.kp_conf)
    visualizer = PoseVisualizer(
        conf_threshold=args.kp_conf,
        show_params=not args.no_params,
    )

    # CSV 日志
    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_path = os.path.join(args.output, "posture_log.csv")
        csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
        fieldnames = [
            "帧序号", "人序号", "来源",
            "躯干前倾角(度)", "躯干后仰", "头前倾角(度)", "头侧倾角(度)",
            "躯干旋转角(度)", "肩膀抬起量(归一化)", "手臂抬起",
            "头后仰角(度)", "侧向倾斜量", "Posture",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    # 判断输入类型
    source = args.source
    source_path = Path(source)

    try:
        if source_path.is_dir():
            infer_folder(source, model, classifier, visualizer, args, csv_writer)
        elif source_path.is_file():
            ext = source_path.suffix.lower()
            if ext in IMG_EXTS:
                infer_image(source, model, classifier, visualizer, args, csv_writer)
            elif ext in VID_EXTS:
                infer_video(source, model, classifier, visualizer, args, csv_writer)
            else:
                print(f"[ERROR] 不支持的文件格式: {ext}")
                sys.exit(1)
        else:
            print(f"[ERROR] 输入源不存在: {source}")
            sys.exit(1)
    finally:
        if csv_file:
            csv_file.close()
            print(f"参数日志已保存: {os.path.join(args.output, 'posture_log.csv')}")


if __name__ == "__main__":
    main()
