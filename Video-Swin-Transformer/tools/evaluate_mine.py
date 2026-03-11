#!/usr/bin/env python3
"""
Video Swin Transformer 扩展评估脚本。

在标准 top-1/top-3/mean_class_accuracy 基础上，额外输出与 ST-GCN 对齐的
逐类别 Precision / Recall / F1，以及 Macro / Weighted 平均和混淆矩阵。

用法示例::

    # 评估验证集（默认，需要 data/annotations/val.csv 已存在）
    python tools/evaluate_mine.py \\
        configs/recognition/swin/swin_base_drive_activity.py \\
        work_dirs/swin_base_drive_activity/best_top1_acc.pth

    # 直接指定标注 CSV（无需提前运行 prepare_annotations.py）
    python tools/evaluate_mine.py \\
        configs/recognition/swin/swin_base_drive_activity.py \\
        work_dirs/swin_base_drive_activity/best_top1_acc.pth \\
        --ann-file data/activity_label/midlevel.chunks_90.csv

    # 指定评估 test 集
    python tools/evaluate_mine.py \\
        configs/recognition/swin/swin_base_drive_activity.py \\
        work_dirs/swin_base_drive_activity/best_top1_acc.pth \\
        --split test

    # 保存结果到 JSON
    python tools/evaluate_mine.py \\
        configs/recognition/swin/swin_base_drive_activity.py \\
        work_dirs/swin_base_drive_activity/best_top1_acc.pth \\
        --out results/eval_results.json
"""

import argparse
import json
import os
import os.path as osp
import sys
import warnings

import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# 确保 mmaction2 包可以被找到
sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import register_module_hooks

try:
    from mmcv.engine import single_gpu_test
except (ImportError, ModuleNotFoundError):
    from mmaction.apis import single_gpu_test

# 驾驶活动5类标签（与 DriveActivityDataset 中 TARGET_CLASSES 顺序一致）
CLASS_NAMES = [
    'sitting_still',
    'eating',
    'fetching_an_object',
    'placing_an_object',
    'reading_magazine',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Video Swin 扩展评估：Top-K + Per-class Precision/Recall/F1')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型权重文件路径（.pth）')
    parser.add_argument(
        '--split',
        choices=['val', 'test'],
        default='val',
        help='评估的数据集划分（默认: val）',
    )
    parser.add_argument(
        '--out',
        default=None,
        help='将结果保存为 JSON 文件的路径（可选）',
    )
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='是否融合 Conv 和 BN（轻微加速推理）',
    )
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob'],
        default='score',
        help='多 clip 聚合方式（默认: score）',
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='覆盖配置项，格式: key=value',
    )
    parser.add_argument(
        '--topk',
        type=int,
        nargs='+',
        default=[1, 3],
        help='Top-K 准确率的 K 值（默认: 1 3）',
    )
    parser.add_argument(
        '--ann-file',
        default=None,
        help='直接覆盖配置中的标注 CSV 路径（绝对路径或相对于 CWD）。'
             '可传入原始 midlevel.chunks_90.csv，无需提前运行 prepare_annotations.py。',
    )
    parser.add_argument(
        '--data-prefix',
        default=None,
        help='覆盖配置中的视频根目录（data_prefix）。'
             '例如: --data-prefix /root/autodl-tmp/pose_estimation/data/video',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='推理 batch size，即每次送入 GPU 的视频片段数（默认: 8）。'
             '根据 GPU 显存调整，RTX 4090 可设 16~32。',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='DataLoader 并行读取的 worker 数（默认: 4）',
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='推理时使用 FP16 半精度（加速约 30%%，显存减半）',
    )
    args = parser.parse_args()
    return args


def turn_off_pretrained(cfg):
    """递归关闭所有 pretrained 权重加载，避免推理时重复下载。"""
    if 'pretrained' in cfg:
        cfg.pretrained = None
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def top_k_accuracy(scores, labels, topk=(1,)):
    """计算 Top-K 准确率（与 mmaction 实现一致）。"""
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        res.append(float(match_array.sum() / match_array.shape[0]))
    return res


def mean_class_accuracy(scores, labels):
    """逐类召回率均值（= Video Swin mean_class_accuracy = ST-GCN recall_macro）。"""
    pred = np.argmax(scores, axis=1)
    num_classes = scores.shape[1]
    per_class_recall = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        per_class_recall.append((pred[mask] == c).mean())
    return float(np.mean(per_class_recall))


def print_separator(char='=', width=70):
    print(char * width)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # 设置 average_clips
    if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
        cfg.model.setdefault('test_cfg',
                             dict(average_clips=args.average_clips))
    else:
        if cfg.model.get('test_cfg') is not None:
            cfg.model.test_cfg.average_clips = args.average_clips
        else:
            cfg.test_cfg.average_clips = args.average_clips

    # max_testing_views 要求 DataLoader batch_size==1（它内部自己做 batch 分割）
    # val/test pipeline 只有 num_clips=1，无需此限制，batch_size>1 时直接移除
    if args.batch_size > 1:
        if cfg.model.get('test_cfg') is not None and \
                'max_testing_views' in cfg.model.test_cfg:
            cfg.model.test_cfg.pop('max_testing_views')
        if cfg.get('test_cfg') is not None and \
                'max_testing_views' in cfg.test_cfg:
            cfg.test_cfg.pop('max_testing_views')

    cfg.setdefault('module_hooks', [])

    # ---------- 选择数据集划分 ----------
    if args.split == 'val':
        dataset_cfg = cfg.data.val
    else:
        dataset_cfg = cfg.data.test

    # 允许通过 --ann-file 覆盖配置中的标注文件路径
    if args.ann_file is not None:
        dataset_cfg.ann_file = osp.abspath(args.ann_file)

    # 允许通过 --data-prefix 覆盖视频根目录
    if args.data_prefix is not None:
        dataset_cfg.data_prefix = osp.abspath(args.data_prefix)
    elif not osp.isabs(dataset_cfg.data_prefix) and not osp.exists(dataset_cfg.data_prefix):
        # 自动尝试相对于仓库根目录解析 data_prefix
        repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
        candidate_prefix = osp.join(repo_root, dataset_cfg.data_prefix)
        if osp.exists(candidate_prefix):
            dataset_cfg.data_prefix = candidate_prefix

    # 若 ann_file 是相对路径，尝试相对于配置文件目录解析
    if not osp.isabs(dataset_cfg.ann_file) and not osp.exists(dataset_cfg.ann_file):
        cfg_dir = osp.dirname(osp.abspath(args.config))
        candidate = osp.join(cfg_dir, dataset_cfg.ann_file)
        if osp.exists(candidate):
            dataset_cfg.ann_file = candidate

    # 若仍然找不到，报清晰的错误
    if not osp.exists(dataset_cfg.ann_file):
        raise FileNotFoundError(
            f"找不到标注文件: {dataset_cfg.ann_file}\n"
            "请通过 --ann-file 指定路径，例如:\n"
            "  --ann-file data/activity_label/midlevel.chunks_90.csv\n"
            "或先运行: python tools/data/drive_activity/prepare_annotations.py"
        )

    print(f"[数据集] 使用{'验证' if args.split == 'val' else '测试'}集: {dataset_cfg.ann_file}")
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)

    # DataLoader：batch_size 由 --batch-size 控制，默认 8
    dataloader_cfg = dict(
        videos_per_gpu=args.batch_size,
        workers_per_gpu=args.num_workers,
        dist=False,
        shuffle=False,
        pin_memory=True,
    )
    data_loader = build_dataloader(dataset, **dataloader_cfg)
    print(f"[DataLoader] batch_size={args.batch_size}, num_workers={args.num_workers}")

    # ---------- 构建模型 ----------
    turn_off_pretrained(cfg.model)
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        register_module_hooks(model, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None or args.fp16:
        wrap_fp16_model(model)
        print("[模型] 已启用 FP16 半精度推理")

    ckpt_path = osp.abspath(args.checkpoint)
    if not osp.exists(ckpt_path):
        # 回退1：相对于配置文件所在目录查找（适合 work_dirs 在 Video-Swin-Transformer/ 下的情形）
        cfg_dir = osp.dirname(osp.abspath(args.config))
        # 向上找到 Video-Swin-Transformer 根目录（config 在 configs/ 下面，根目录在上两级）
        swin_root = osp.abspath(osp.join(cfg_dir, '..', '..', '..'))
        candidate = osp.join(swin_root, args.checkpoint)
        if osp.exists(candidate):
            ckpt_path = candidate
        else:
            # 回退2：相对于脚本目录的上一级（Video-Swin-Transformer 根目录）
            script_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
            candidate2 = osp.join(script_root, args.checkpoint)
            if osp.exists(candidate2):
                ckpt_path = candidate2
            else:
                raise FileNotFoundError(
                    f"找不到权重文件，已尝试以下路径：\n"
                    f"  1. {osp.abspath(args.checkpoint)}\n"
                    f"  2. {candidate}\n"
                    f"  3. {candidate2}\n"
                    f"请确认权重文件路径，例如:\n"
                    f"  Video-Swin-Transformer/work_dirs/swin_base_drive_activity/best_top1_acc_epoch_23.pth"
                )
    print(f"[模型] 加载权重: {ckpt_path}")
    load_checkpoint(model, ckpt_path, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDataParallel(model, device_ids=[0])

    # ---------- 推理 ----------
    n_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    print(f"\n[推理] 开始在 {args.split} 集上推理，共 {len(dataset)} 条样本"
          f"，batch_size={args.batch_size}，共约 {n_batches} 个 batch...")
    outputs = single_gpu_test(model, data_loader)  # list of np.ndarray, shape (num_classes,)

    # ---------- 整理 GT 标签 ----------
    gt_labels = np.array([info['label'] for info in dataset.video_infos])
    scores = np.stack(outputs, axis=0)          # (N, num_classes)
    pred_labels = np.argmax(scores, axis=1)     # (N,)

    num_classes = scores.shape[1]
    class_names = CLASS_NAMES[:num_classes]

    # ---------- 指标计算 ----------
    print_separator()
    print(f"{'评估结果':^70}")
    print_separator()

    # 1. Top-K Accuracy（与 Video Swin 原始指标一致）
    topk_vals = top_k_accuracy(scores, gt_labels, topk=args.topk)
    print("\n【Top-K Accuracy（Video Swin 原始指标）】")
    for k, acc in zip(args.topk, topk_vals):
        print(f"  Top-{k} Accuracy : {acc:.4f}  ({acc*100:.2f}%)")

    # 2. Mean Class Accuracy（= recall_macro，可与 ST-GCN 直接比较）
    mca = mean_class_accuracy(scores, gt_labels)
    print(f"\n【Mean Class Accuracy = Macro-Recall（可与 ST-GCN recall_macro 直接比）】")
    print(f"  Mean Class Accuracy : {mca:.4f}  ({mca*100:.2f}%)")

    # 3. Overall Accuracy（= Top-1，验证一致）
    overall_acc = accuracy_score(gt_labels, pred_labels)

    # 4. Macro / Weighted 平均（与 ST-GCN 对齐）
    precision_macro   = precision_score(gt_labels, pred_labels, average='macro',    zero_division=0)
    recall_macro      = recall_score   (gt_labels, pred_labels, average='macro',    zero_division=0)
    f1_macro          = f1_score       (gt_labels, pred_labels, average='macro',    zero_division=0)
    precision_weighted= precision_score(gt_labels, pred_labels, average='weighted', zero_division=0)
    recall_weighted   = recall_score   (gt_labels, pred_labels, average='weighted', zero_division=0)
    f1_weighted       = f1_score       (gt_labels, pred_labels, average='weighted', zero_division=0)

    print(f"\n【整体指标汇总（可与 ST-GCN 横向对比）】")
    print(f"  {'指标':<25} {'Macro':>10} {'Weighted':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Accuracy (Top-1)':<25} {overall_acc:>10.4f}  (weighted N/A)")
    print(f"  {'Precision':<25} {precision_macro:>10.4f} {precision_weighted:>10.4f}")
    print(f"  {'Recall (= MCA)':<25} {recall_macro:>10.4f} {recall_weighted:>10.4f}")
    print(f"  {'F1 Score':<25} {f1_macro:>10.4f} {f1_weighted:>10.4f}")

    # 5. 逐类别详细报告（与 ST-GCN classification_report 格式相同）
    print(f"\n【逐类别 Precision / Recall / F1（与 ST-GCN 格式一致）】")
    report = classification_report(
        gt_labels, pred_labels,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    print(report)

    # 6. 混淆矩阵
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(num_classes)))
    print("【混淆矩阵】（行=真实标签，列=预测标签）")
    short_names = [n[:12] for n in class_names]
    header = " " * 14 + "  ".join(f"{n:>12s}" for n in short_names)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>12d}" for v in row)
        print(f"{short_names[i]:>12s}  {row_str}")

    # ---------- 保存结果 ----------
    eval_results = {
        'split': args.split,
        'checkpoint': ckpt_path,
        'num_samples': int(len(dataset)),
        'num_classes': num_classes,
        'class_names': class_names,
        'top_k_accuracy': {f'top{k}': float(v) for k, v in zip(args.topk, topk_vals)},
        'mean_class_accuracy': float(mca),
        'overall_accuracy': float(overall_acc),
        'macro': {
            'precision': float(precision_macro),
            'recall': float(recall_macro),
            'f1': float(f1_macro),
        },
        'weighted': {
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1': float(f1_weighted),
        },
        'confusion_matrix': cm.tolist(),
    }

    if args.out:
        os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print(f"\n[保存] 结果已写入: {args.out}")

    print_separator()
    print("完成。")


if __name__ == '__main__':
    main()
