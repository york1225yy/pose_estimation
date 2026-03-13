"""
视频文件推理脚本 - 无需 UCF24 数据集
用法：
    python infer_video.py -cf weights/model_checkpoint/config.yaml -v input.mp4 -o output.mp4
    python infer_video.py -cf weights/model_checkpoint/config.yaml -v input.mp4 --device cpu
    python infer_video.py -cf weights/model_checkpoint/config.yaml --camera 0   # 摄像头
"""

import argparse
import os
import sys

import cv2
import torch
import torchvision.transforms.functional as FT
from PIL import Image

from model.TSN.YOWOv3 import build_yowov3
from utils.box import draw_bounding_box, non_max_suppression
from utils.build_config import build_config


class VideoTransform:
    """将单帧 PIL Image 转换为模型输入 tensor。"""
    def __init__(self, img_size: int):
        self.img_size = img_size
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std  = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def __call__(self, frame_bgr: "np.ndarray") -> torch.Tensor:
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img = img.resize((self.img_size, self.img_size))
        t   = FT.to_tensor(img)          # [3, H, W], float32, 0-1
        t   = (t - self.mean) / self.std
        return t


def infer_video(config, video_source, output_path, device,
                conf_thresh, iou_thresh, show):
    # ── 构建模型 ──────────────────────────────────────────────────────────
    model = build_yowov3(config)
    model.to(device)
    model.eval()

    mapping    = config["idx2name"]
    img_size   = config["img_size"]
    clip_len   = config.get("clip_length", 16)
    transform  = VideoTransform(img_size)

    # ── 打开视频源 ─────────────────────────────────────────────────────────
    is_camera = isinstance(video_source, int)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {video_source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── 初始化输出写入器 ───────────────────────────────────────────────────
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps,
                                 (img_size, img_size))
        print(f"输出保存到: {output_path}")

    print(f"设备: {device} | clip_length: {clip_len} | img_size: {img_size}")
    print(f"conf_threshold: {conf_thresh} | iou_threshold: {iou_thresh}")
    print("按 q 退出" if show else "处理中...")

    frame_buf = []   # 滑动窗口缓存 clip_len 帧

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_tensor = transform(frame)
            frame_buf.append(frame_tensor)
            if len(frame_buf) > clip_len:
                frame_buf.pop(0)
            if len(frame_buf) < clip_len:
                continue   # 缓冲区未满时跳过推理

            # [C, T, H, W] -> [1, C, T, H, W]
            clip = torch.stack(frame_buf, dim=1).unsqueeze(0).to(device)

            outputs = model(clip)
            dets    = non_max_suppression(outputs,
                                          conf_threshold=conf_thresh,
                                          iou_threshold=iou_thresh)[0]

            vis = cv2.resize(frame, (img_size, img_size))
            draw_bounding_box(vis, dets[:, :4], dets[:, 5], dets[:, 4], mapping)

            if writer:
                writer.write(vis)
            if show:
                cv2.imshow("YOWOv3 inference", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("完成。")


def main():
    parser = argparse.ArgumentParser(description="YOWOv3 视频推理（无需数据集）")
    parser.add_argument("-cf", "--config",   required=True,
                        help="config 文件路径，如 weights/model_checkpoint/config.yaml")
    parser.add_argument("-v",  "--video",    default=None,
                        help="输入视频文件路径（.mp4 / .avi 等）")
    parser.add_argument("--camera",          type=int, default=None,
                        help="摄像头编号（0 = 默认摄像头），与 -v 二选一")
    parser.add_argument("-o", "--output",    default=None,
                        help="输出视频路径（不指定则不保存）")
    parser.add_argument("--device",          default=None,
                        help="cuda / cpu（默认自动检测）")
    parser.add_argument("--conf",            type=float, default=0.4,
                        help="置信度阈值（默认 0.4）")
    parser.add_argument("--iou",             type=float, default=0.5,
                        help="NMS IoU 阈值（默认 0.5）")
    parser.add_argument("--no-show",         action="store_true",
                        help="不弹出可视化窗口（仅保存文件时使用）")
    args = parser.parse_args()

    if args.video is None and args.camera is None:
        parser.error("请指定 -v <视频文件> 或 --camera <编号>")

    config = build_config(args.config)

    # 自动选择设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_source = args.camera if args.camera is not None else args.video
    show         = not args.no_show

    infer_video(config, video_source, args.output, device,
                args.conf, args.iou, show)


if __name__ == "__main__":
    main()
