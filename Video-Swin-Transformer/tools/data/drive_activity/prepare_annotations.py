#!/usr/bin/env python3
"""
准备驾驶舱活动数据集的训练/验证 CSV。

用法::

    python tools/data/drive_activity/prepare_annotations.py \\
        --csv  data/activity_label/midlevel.chunks_90.csv \\
        --out  data/annotations \\
        --val_ratio 0.2 \\
        --seed 42

输出文件::

    data/annotations/train.csv
    data/annotations/val.csv

CSV 格式与原始 midlevel.chunks_90.csv 相同，但只保留5个目标类别，
并按 participant_id 进行 train/val 划分（避免同一参与者同时出现在
train 和 val 中）。
"""
import argparse
import os
import random

import pandas as pd

TARGET_CLASSES = [
    'sitting_still',
    'eating',
    'fetching_an_object',
    'placing_an_object',
    'reading_magazine',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='生成驾驶活动数据集的 train/val CSV 划分')
    parser.add_argument('--csv',
                        default='data/activity_label/midlevel.chunks_90.csv',
                        help='原始 midlevel.chunks_90.csv 路径')
    parser.add_argument('--out',
                        default='data/annotations',
                        help='输出目录')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集中参与者比例（按 participant_id 划分）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.csv)

    # 只保留5类目标活动
    df = df[df['activity'].isin(TARGET_CLASSES)].reset_index(drop=True)
    print(f'过滤后共 {len(df)} 条样本')

    # 按 participant_id 划分（使相同的人不同时出现在 train / val）
    participants = sorted(df['participant_id'].unique().tolist())
    random.shuffle(participants)
    n_val = max(1, int(len(participants) * args.val_ratio))
    val_pids = set(participants[:n_val])
    train_pids = set(participants[n_val:])

    train_df = df[df['participant_id'].isin(train_pids)].reset_index(drop=True)
    val_df   = df[df['participant_id'].isin(val_pids)].reset_index(drop=True)

    print(f'Train: {len(train_df)} 样本, 参与者: {sorted(train_pids)}')
    print(f'Val:   {len(val_df)} 样本, 参与者: {sorted(val_pids)}')

    # 打印各类别分布
    print('\nTrain 类别分布:')
    print(train_df['activity'].value_counts().to_string())
    print('\nVal 类别分布:')
    print(val_df['activity'].value_counts().to_string())

    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.out, 'train.csv')
    val_path   = os.path.join(args.out, 'val.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f'\n已保存:')
    print(f'  {train_path}')
    print(f'  {val_path}')


if __name__ == '__main__':
    main()
