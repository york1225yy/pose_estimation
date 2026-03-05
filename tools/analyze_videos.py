#!/usr/bin/env python3
"""
视频参数批量提取 + 统计分析脚本
================================
对 data/ 目录下的 4 个标注视频逐帧运行 YOLO11-Pose，
提取所有几何参数并保存到统一 CSV，然后打印各类别的参数统计范围。

用法：
  cd /workspaces/pose_estimation
  python tools/analyze_videos.py
  python tools/analyze_videos.py --data data --model yolo11s-pose.pt --sample 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from utils.posture_rules import PostureClassifier

# 视频名 → 真实类别标签（取文件名第1个下划线前的数字编号作为 gt_class）
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

FIELD_ORDER = [
    "video", "frame",
    "gt_class",
    "trunk_tilt", "trunk_backward",
    "head_forward_angle", "head_backward_angle",
    "head_side_angle",
    "trunk_rotate_angle",
    "shoulder_lift_norm",
    "lateral_lean",
    "arm_raised",
    "pred_posture",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",  default="data",  help="视频目录 (default: data)")
    p.add_argument("--model", default="yolo11s-pose.pt", help="YOLO11-Pose 权重")
    p.add_argument("--out",   default="output/analysis", help="输出目录")
    p.add_argument("--conf",  type=float, default=0.45, help="检测置信度")
    p.add_argument("--kp-conf", type=float, default=0.35, help="关键点置信度")
    p.add_argument("--sample", type=int, default=1,
                   help="每隔N帧采样一帧 (default: 1 = 每帧)")
    p.add_argument("--device", default="", help="cpu / cuda")
    return p.parse_args()


def extract_video(video_path: Path, model: YOLO, classifier: PostureClassifier,
                  args, gt_class: str) -> list[dict]:
    """逐帧提取参数，返回行列表"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] 无法打开: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"  总帧数={total}  FPS={fps:.1f}  采样间隔={args.sample}")

    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.sample != 0:
            frame_idx += 1
            continue

        results = model.predict(frame, conf=args.conf, imgsz=640,
                                verbose=False,
                                device=args.device or None)

        # 只取置信度最高的人
        best_params = None
        best_score = -1.0
        for r in results:
            if r.keypoints is None or len(r.keypoints.xy) == 0:
                continue
            for i, (kps_xy, kps_conf) in enumerate(
                    zip(r.keypoints.xy, r.keypoints.conf)):
                kp = kps_xy.cpu().numpy()
                cf = kps_conf.cpu().numpy()
                mean_cf = float(np.mean(cf))
                if mean_cf > best_score:
                    best_score = mean_cf
                    best_params = classifier.classify(kp, cf)

        if best_params is None:
            frame_idx += 1
            continue

        row = {
            "video":        video_path.name,
            "frame":        frame_idx,
            "gt_class":     gt_class,
            "trunk_tilt":       round(best_params.trunk_tilt, 2),
            "trunk_backward":   int(best_params.trunk_backward),
            "head_forward_angle":  round(best_params.head_forward_angle, 2),
            "head_backward_angle": round(best_params.head_backward_angle, 2),
            "head_side_angle":     round(best_params.head_side_angle, 2),
            "trunk_rotate_angle":  round(best_params.trunk_rotate_angle, 2),
            "shoulder_lift_norm":  round(best_params.shoulder_lift_norm, 3),
            "lateral_lean":        round(best_params.lateral_lean, 3),
            "arm_raised":          int(best_params.arm_raised),
            "pred_posture":        best_params.posture_label,
        }
        rows.append(row)
        frame_idx += 1

    cap.release()
    return rows


def percentile_row(arr: np.ndarray, label: str) -> str:
    if len(arr) == 0:
        return f"  {label:28s}: N/A"
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    mn, mx = arr.min(), arr.max()
    return (f"  {label:28s}: "
            f"min={mn:7.2f}  p5={p5:7.2f}  p25={p25:7.2f}  "
            f"median={p50:7.2f}  p75={p75:7.2f}  p95={p95:7.2f}  max={mx:7.2f}")


def analyze(csv_path: Path):
    """读取 CSV，按 gt_class 分组打印参数统计，并返回各类别中位数字典"""
    import csv as _csv
    rows_by_class: dict[str, list[dict]] = {}
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            cls = row["gt_class"]
            rows_by_class.setdefault(cls, []).append(row)

    numeric_cols = [
        "trunk_tilt", "trunk_backward",
        "head_forward_angle", "head_backward_angle",
        "head_side_angle", "trunk_rotate_angle",
        "shoulder_lift_norm", "lateral_lean",
    ]

    summary: dict[str, dict[str, float]] = {}

    print("\n" + "=" * 80)
    print("  参数统计分析（按真实类别）")
    print("=" * 80)
    for cls in sorted(rows_by_class.keys()):
        rows = rows_by_class[cls]
        n = len(rows)
        # 预测准确率
        correct = sum(1 for r in rows if r["pred_posture"].lower().replace(" ", "_")
                      in cls.lower().replace(" ", "_") or True)  # placeholder
        # 实际预测准确率
        pred_counts: dict[str, int] = {}
        for r in rows:
            pred_counts[r["pred_posture"]] = pred_counts.get(r["pred_posture"], 0) + 1
        top_pred = max(pred_counts, key=lambda k: pred_counts[k])

        print(f"\n【{cls}】  有效帧={n}  当前规则最多预测为: {top_pred} ({pred_counts[top_pred]}/{n})")
        summary[cls] = {}
        for col in numeric_cols:
            arr = np.array([float(r[col]) for r in rows])
            print(percentile_row(arr, col))
            summary[cls][col] = {
                "min":    float(arr.min()),
                "p5":     float(np.percentile(arr, 5)),
                "p25":    float(np.percentile(arr, 25)),
                "median": float(np.median(arr)),
                "p75":    float(np.percentile(arr, 75)),
                "p95":    float(np.percentile(arr, 95)),
                "max":    float(arr.max()),
            }

    # ── 给出阈值建议 ──
    print("\n" + "=" * 80)
    print("  阈值建议（基于各类别 p5/p95）")
    print("=" * 80)

    def _get(cls, col, stat):
        return summary.get(cls, {}).get(col, {}).get(stat, None)

    # head_forward_angle: HeadDownPhone vs others
    hdf_phone_p5  = _get("4_head_down_phone",  "head_forward_angle", "p5")
    hdf_normal_p95 = _get("1_normal_driving", "head_forward_angle", "p95")
    hdf_lean_p95   = _get("2_heavy_lean_fwd", "head_forward_angle", "p95")
    hdf_recline_p95= _get("3_recline_relax",  "head_forward_angle", "p95")
    others_hdf_max = max(x for x in [hdf_normal_p95, hdf_lean_p95, hdf_recline_p95] if x is not None)
    if hdf_phone_p5 is not None:
        suggested_hdf = (hdf_phone_p5 + others_hdf_max) / 2
        print(f"\n  head_forward_angle 阈值 (HeadDownPhone 触发):")
        print(f"    HeadDownPhone p5={hdf_phone_p5:.1f}°   其他类别 max(p95)={others_hdf_max:.1f}°")
        print(f"    建议阈值 → {suggested_hdf:.1f}°  (当前: 25°)")

    # trunk_tilt: HeavyLeanFwd vs normal
    tt_lean_p5   = _get("2_heavy_lean_fwd", "trunk_tilt", "p5")
    tt_normal_p95= _get("1_normal_driving",  "trunk_tilt", "p95")
    tt_recline_p95= _get("3_recline_relax",  "trunk_tilt", "p95")
    if tt_lean_p5 is not None and tt_normal_p95 is not None:
        others_tt_max = max(x for x in [tt_normal_p95, tt_recline_p95] if x is not None)
        suggested_tt = (tt_lean_p5 + others_tt_max) / 2
        print(f"\n  trunk_tilt 阈值 (HeavyLeanFwd 触发):")
        print(f"    HeavyLeanFwd p5={tt_lean_p5:.1f}°   Normal p95={tt_normal_p95:.1f}°  Recline p95={tt_recline_p95:.1f}°")
        print(f"    建议阈值 → {suggested_tt:.1f}°  (当前: 15°)")

    # trunk_tilt 上界 + head_backward: Recline
    tt_recline_p95_v = _get("3_recline_relax", "trunk_tilt", "p95")
    hb_recline_p5    = _get("3_recline_relax", "head_backward_angle", "p5")
    hb_recline_p95   = _get("3_recline_relax", "head_backward_angle", "p95")
    hb_normal_p95    = _get("1_normal_driving", "head_backward_angle", "p95")
    if tt_recline_p95_v is not None:
        print(f"\n  trunk_tilt 上界 (Recline Relax):")
        print(f"    Recline trunk_tilt p95={tt_recline_p95_v:.1f}°  (< 此值即可认为未前倾)")
        print(f"    建议阈值 → {tt_recline_p95_v+2:.1f}°  (当前: 12°)")
    if hb_recline_p5 is not None:
        print(f"\n  head_backward_angle 范围 (Recline Relax):")
        print(f"    Recline p5={hb_recline_p5:.1f}°  p95={hb_recline_p95:.1f}°")
        print(f"    Normal  p95={hb_normal_p95:.1f}°")
        print(f"    建议范围 → [{max(0, hb_recline_p5-2):.1f}°, {hb_recline_p95+3:.1f}°]  (当前: [10°, 20°])")

    print("\n" + "=" * 80)
    return summary


def main():
    args = parse_args()
    data_dir = ROOT / args.data
    out_dir  = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted([v for v in data_dir.iterdir()
                     if v.is_file() and v.suffix.lower() in VIDEO_EXTS])
    if not videos:
        print(f"[ERROR] {data_dir} 下未找到视频文件")
        sys.exit(1)

    print(f"加载模型: {args.model}")
    model_path = ROOT / "yolo11_pose" / args.model
    if not model_path.exists():
        model_path = Path(args.model)   # 尝试直接使用（会自动下载）
    model = YOLO(str(model_path))
    classifier = PostureClassifier(conf_threshold=args.kp_conf)

    csv_path = out_dir / "all_frames.csv"
    all_rows: list[dict] = []

    for v in videos:
        # 取文件名（不含后缀）作为 gt_class
        gt_class = v.stem   # e.g. "1_normal_driving"
        print(f"\n处理视频: {v.name}  gt_class={gt_class}")
        rows = extract_video(v, model, classifier, args, gt_class)
        all_rows.extend(rows)
        print(f"  提取帧数: {len(rows)}")

    # 写 CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELD_ORDER)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n已保存 CSV → {csv_path}  (共 {len(all_rows)} 行)")

    # 统计分析
    summary = analyze(csv_path)

    # 同时将 summary 以可读形式写入 txt
    import json
    summary_path = out_dir / "param_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"统计摘要 → {summary_path}")


if __name__ == "__main__":
    main()
