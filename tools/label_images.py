#!/usr/bin/env python3
"""
图片快速标注工具
================
键盘快捷键：
  [1]  → 标准驾驶坐姿  (Normal Driving)
  [2]  → 严重前倾      (Heavy Lean Fwd)
  [3]  → 后仰放松      (Recline Relax)
  [4]  → 低头玩手机    (Head Down Phone)
  [5]  → 不满足任何一类 (Skip)
  [→ / d]  → 跳过（本次不分类，稍后再标）
  [← / a]  → 返回上一张
  [q / ESC] → 保存进度并退出

用法：
  python label_images.py --source /path/to/images
  python label_images.py --source /path/to/images --out /path/to/output
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ── 分类定义 ──────────────────────────────────────────────────
CLASSES = {
    ord("1"): ("1_normal_driving",   "Normal Driving",   (0, 255, 0)),
    ord("2"): ("2_heavy_lean_fwd",   "Heavy Lean Fwd",   (0, 128, 255)),
    ord("3"): ("3_recline_relax",    "Recline Relax",    (255, 255, 0)),
    ord("4"): ("4_head_down_phone",  "Head Down Phone",  (0, 80, 255)),
    ord("5"): ("5_skip",             "Skip",             (120, 120, 120)),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# 窗口最大显示尺寸
MAX_W, MAX_H = 1280, 800


def parse_args():
    parser = argparse.ArgumentParser(description="图片快速标注工具")
    parser.add_argument("--source", type=str, required=True,
                        help="待标注图片所在目录")
    parser.add_argument("--out", type=str, default=None,
                        help="输出根目录（默认在 source 同级创建 labeled/ 文件夹）")
    return parser.parse_args()


def collect_images(source: Path) -> list[Path]:
    """只收集 source 目录第一层（未分类）的图片，子目录忽略"""
    return sorted([
        f for f in source.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    ])


def load_labels(out_root: Path) -> dict[str, str]:
    """
    扫描 out_root 各子目录，返回 {filename: class_dir_name}
    用于判断哪些图片已经标注过。
    """
    labeled = {}
    for cls_dir, _, _ in CLASSES.values():
        d = out_root / cls_dir
        if d.exists():
            for f in d.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    labeled[f.name] = cls_dir
    return labeled


def resize_for_display(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def draw_overlay(img: np.ndarray, filename: str, idx: int, total: int,
                 already_labeled: str | None) -> np.ndarray:
    """在图片上叠加提示信息"""
    disp = img.copy()
    h, w = disp.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 半透明顶部状态栏
    overlay = disp.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, disp, 0.4, 0, disp)

    # 文件名 + 进度
    cv2.putText(disp, f"[{idx+1}/{total}] {filename}",
                (8, 22), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if already_labeled:
        # 已标注：显示已有标签
        cls_name = next(
            (name for _, (d, name, _) in enumerate(CLASSES.values()) if d == already_labeled),
            already_labeled
        )
        cv2.putText(disp, f"Already labeled: {cls_name}  (press key to re-label)",
                    (8, 50), font, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(disp, "1:Normal  2:HeavyLean  3:Recline  4:HeadDown  5:Skip  |  a:Prev  d:Next  q:Quit",
                    (8, 50), font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # 半透明底部：分类颜色提示条
    bar_h = 22
    n = len(CLASSES)
    bw = w // n
    overlay2 = disp.copy()
    for i, (key, (d, name, color)) in enumerate(CLASSES.items()):
        x1, x2 = i * bw, (i + 1) * bw
        cv2.rectangle(overlay2, (x1, h - bar_h), (x2, h), color, -1)
    cv2.addWeighted(overlay2, 0.55, disp, 0.45, 0, disp)
    for i, (key, (d, name, color)) in enumerate(CLASSES.items()):
        x1 = i * bw
        label_txt = f"{chr(key)}:{name}"
        cv2.putText(disp, label_txt, (x1 + 4, h - 5),
                    font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return disp


def main():
    args = parse_args()
    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"[ERROR] 路径不存在或不是目录: {source}")
        sys.exit(1)

    out_root = Path(args.out).resolve() if args.out else source.parent / "labeled"
    out_root.mkdir(parents=True, exist_ok=True)
    for _, (cls_dir, _, _) in CLASSES.items():
        (out_root / cls_dir).mkdir(exist_ok=True)

    print(f"源目录:   {source}")
    print(f"输出目录: {out_root}")

    images = collect_images(source)
    if not images:
        print("[WARN] 未找到任何图片")
        sys.exit(0)
    print(f"找到 {len(images)} 张图片")

    # 已标注的文件名 → 目标目录
    labeled_map = load_labels(out_root)

    cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)

    idx = 0
    stats = {d: 0 for _, (d, _, _) in CLASSES.items()}
    # 刷新当前已标注数量
    for cls_dir in stats:
        p = out_root / cls_dir
        stats[cls_dir] = len(list(p.glob("*"))) if p.exists() else 0

    while 0 <= idx < len(images):
        img_path = images[idx]
        filename = img_path.name
        already = labeled_map.get(filename)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] 无法读取: {img_path}")
            idx += 1
            continue

        disp = resize_for_display(img)
        disp = draw_overlay(disp, filename, idx, len(images), already)
        cv2.imshow("Labeler", disp)
        cv2.setWindowTitle("Labeler",
                           f"Labeler  [{idx+1}/{len(images)}]  "
                           + "  ".join(f"{n}:{stats[d]}" for _, (d, n, _) in CLASSES.items()))

        key = cv2.waitKey(0) & 0xFF

        # 退出
        if key in (ord("q"), 27):
            print("\n退出标注")
            break

        # 向后跳（不标注）
        if key in (ord("d"), 83, 0xFF & ord("\t")):  # d / → / Tab
            idx += 1
            continue

        # 向前
        if key in (ord("a"), 81):  # a / ←
            idx = max(0, idx - 1)
            continue

        # 分类操作
        if key in CLASSES:
            cls_dir, cls_name, color = CLASSES[key]
            dst = out_root / cls_dir / filename

            # 如果之前已标注到其他类别，先从旧目录移除
            if already and already != cls_dir:
                old_file = out_root / already / filename
                if old_file.exists():
                    old_file.unlink()
                    stats[already] = max(0, stats[already] - 1)

            # 复制（保留源文件，方便反复标注/纠错）
            shutil.copy2(str(img_path), str(dst))
            labeled_map[filename] = cls_dir
            stats[cls_dir] += 1

            # 简短反馈
            feedback = disp.copy()
            h, w = feedback.shape[:2]
            overlay = feedback.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.25, feedback, 0.75, 0, feedback)
            cv2.putText(feedback, f"-> {cls_name}",
                        (w // 4, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
            cv2.imshow("Labeler", feedback)
            cv2.waitKey(300)

            idx += 1
            continue

    cv2.destroyAllWindows()

    # 最终统计
    print("\n" + "=" * 50)
    print("标注统计:")
    total_labeled = 0
    for _, (cls_dir, cls_name, _) in CLASSES.items():
        n = len(list((out_root / cls_dir).glob("*")))
        print(f"  {cls_name:20s}: {n} 张")
        total_labeled += n
    print(f"  {'合计':20s}: {total_labeled} / {len(images)} 张")
    print(f"\n输出目录: {out_root}")


if __name__ == "__main__":
    main()
