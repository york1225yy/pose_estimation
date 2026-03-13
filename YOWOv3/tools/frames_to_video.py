"""
UCF24 图片帧 → 视频 批量转换脚本

UCF24 目录结构：
    <data_root>/rgb-images/<ClassName>/<VideoName>/00001.jpg
                                                   00002.jpg
                                                   ...

用法示例：
    # 转换整个数据集（默认输出到 data_root/videos/）
    python tools/frames_to_video.py --data_root /path/to/ucf24

    # 指定输出目录和帧率
    python tools/frames_to_video.py --data_root /path/to/ucf24 --output /path/to/out --fps 25

    # 只转换某个类别
    python tools/frames_to_video.py --data_root /path/to/ucf24 --class_filter Basketball

    # 使用多进程加速
    python tools/frames_to_video.py --data_root /path/to/ucf24 --workers 8
"""

import argparse
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


def natural_sort_key(s: str):
    """按文件名中的数字自然排序，确保 00009 < 00010。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def frames_to_video(frames_dir: Path, output_path: Path, fps: float) -> Tuple[str, bool, str]:
    """
    将单个 frames_dir 下的图片序列合并为 output_path 视频。
    返回 (video_name, success, message)。
    """
    # 收集图片，支持常见格式
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted(
        [f for f in frames_dir.iterdir() if f.suffix.lower() in exts],
        key=lambda p: natural_sort_key(p.name),
    )

    if not frames:
        return str(frames_dir), False, "no images found"

    # 读第一帧获取尺寸
    first = cv2.imread(str(frames[0]))
    if first is None:
        return str(frames_dir), False, f"cannot read {frames[0].name}"
    h, w = first.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        return str(frames_dir), False, "VideoWriter failed to open"

    for f in frames:
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        writer.write(frame)

    writer.release()
    return str(frames_dir), True, f"{len(frames)} frames -> {output_path}"


def collect_tasks(data_root: Path, output_root: Path, fps: float,
                  class_filter: Optional[str]) -> List[Tuple[Path, Path, float]]:
    """
    遍历 data_root/rgb-images/<Class>/<Video>/ 生成任务列表。
    若不存在 rgb-images 子目录，则直接把 data_root 当作含多层子目录的根目录：
        data_root/<Class>/<Video>/
    """
    rgb_root = data_root / "rgb-images"
    search_root = rgb_root if rgb_root.is_dir() else data_root

    tasks = []
    for class_dir in sorted(search_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_filter and class_dir.name != class_filter:
            continue
        for video_dir in sorted(class_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            rel = video_dir.relative_to(search_root)   # <Class>/<Video>
            out_file = output_root / rel.parent / (rel.name + ".mp4")
            tasks.append((video_dir, out_file, fps))

    return tasks


def main():
    parser = argparse.ArgumentParser(description="UCF24 帧序列批量转视频")
    parser.add_argument("--data_root", required=True,
                        help="UCF24 根目录（包含 rgb-images/ 子目录，或直接是 <Class>/<Video>/ 结构）")
    parser.add_argument("--output", default=None,
                        help="输出根目录（默认：data_root/videos/）")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="输出视频帧率（默认 25）")
    parser.add_argument("--class_filter", default=None,
                        help="只转换指定类别，例如 Basketball")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行进程数（默认 4）")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已存在的视频文件")
    args = parser.parse_args()

    data_root   = Path(args.data_root).resolve()
    output_root = Path(args.output).resolve() if args.output else data_root / "videos"

    print(f"数据根目录: {data_root}")
    print(f"输出目录:   {output_root}")
    print(f"帧率:       {args.fps} FPS | 进程数: {args.workers}")

    tasks = collect_tasks(data_root, output_root, args.fps, args.class_filter)
    if not tasks:
        print("未找到任何视频帧目录，请检查 --data_root 路径。")
        return

    if args.skip_existing:
        tasks = [(d, o, f) for d, o, f in tasks if not o.exists()]

    print(f"共 {len(tasks)} 个视频待转换\n")

    success_count = 0
    fail_count    = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(frames_to_video, d, o, f): (d, o) for d, o, f in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            src_dir, out_path = futures[future]
            try:
                _, ok, msg = future.result()
                status = "✓" if ok else "✗"
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
                print(f"[{i:4d}/{len(tasks)}] {status} {msg}")
            except Exception as e:
                fail_count += 1
                print(f"[{i:4d}/{len(tasks)}] ✗ {src_dir} -> ERROR: {e}")

    print(f"\n完成：成功 {success_count}，失败 {fail_count}")
    print(f"视频已保存至: {output_root}")


if __name__ == "__main__":
    main()
