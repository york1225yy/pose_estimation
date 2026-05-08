# Copyright (c) OpenMMLab. All rights reserved.
"""
自定义数据集文件列表生成脚本

用法:
    cd /workspaces/pose_estimation/mmaction2
    python tools/data/build_custom_filelist.py \
        --video-root data/my_dataset/videos \
        --output-dir data/my_dataset \
        --val-ratio 0.2 \
        --seed 42

目录结构要求:
    data/my_dataset/videos/
    ├── class_A/
    │   ├── video_001.mp4
    │   └── video_002.mp4
    ├── class_B/
    │   └── video_001.mp4
    └── class_C/
        └── video_001.mp4

输出文件:
    data/my_dataset/train.txt   — 训练集标注文件
    data/my_dataset/val.txt     — 验证集标注文件
    data/my_dataset/annotations/label_map.txt  — 类别名称映射文件

标注文件格式 (VideoDataset):
    <视频相对路径> <类别标签整数(0-based)>
    例: basketball/clip_001.mp4 0
"""

import argparse
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(
        description='为自定义视频数据集生成 MMAction2 格式的文件列表')
    parser.add_argument(
        '--video-root',
        type=str,
        default='data/my_dataset/videos',
        help='视频文件根目录，下一级为各类别子目录')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/my_dataset',
        help='输出 train.txt / val.txt 的目录')
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='验证集比例，取值范围 (0, 1)，默认 0.2 即 20%% 用于验证')
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子，保证划分结果可复现')
    parser.add_argument(
        '--ext',
        nargs='+',
        default=['.mp4', '.avi', '.mkv', '.mov', '.wmv'],
        help='支持的视频文件后缀（空格分隔），默认: .mp4 .avi .mkv .mov .wmv')
    parser.add_argument(
        '--save-label-map',
        action='store_true',
        default=True,
        help='是否保存 label_map.txt 类别映射文件（默认开启）')
    return parser.parse_args()


def main():
    args = parse_args()

    video_root = args.video_root
    output_dir = args.output_dir
    val_ratio  = args.val_ratio
    supported_ext = set(e.lower() if e.startswith('.') else f'.{e.lower()}'
                        for e in args.ext)

    # 参数校验
    if not os.path.isdir(video_root):
        raise FileNotFoundError(
            f'视频根目录不存在: {video_root}\n'
            f'请先将视频按类别整理到对应子目录中。')
    if not (0 < val_ratio < 1):
        raise ValueError(f'--val-ratio 必须在 (0, 1) 之间，当前值: {val_ratio}')

    random.seed(args.seed)

    # 扫描类别子目录（按字母顺序排序，保证标签编号稳定）
    # 过滤以 '.' 开头的隐藏目录（如 .ipynb_checkpoints）
    classes = sorted([
        d for d in os.listdir(video_root)
        if os.path.isdir(os.path.join(video_root, d)) and not d.startswith('.')
    ])

    if not classes:
        raise RuntimeError(
            f'在 {video_root} 下未找到任何子目录。\n'
            f'请确保目录结构为: {video_root}/<class_name>/<video_file>')

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    print(f'发现 {len(classes)} 个类别:')
    for cls, idx in class_to_idx.items():
        print(f'  [{idx:3d}] {cls}')

    # 扫描所有视频文件
    all_samples = []
    missing_videos = []
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = os.path.join(video_root, cls_name)
        found = 0
        for fname in sorted(os.listdir(cls_dir)):
            if os.path.splitext(fname)[1].lower() in supported_ext:
                rel_path = os.path.join(cls_name, fname)
                all_samples.append((rel_path, cls_idx))
                found += 1
        if found == 0:
            missing_videos.append(cls_name)

    if missing_videos:
        print(f'\n警告: 以下类别目录中未找到视频文件: {missing_videos}')
        print(f'支持的格式: {supported_ext}')

    if not all_samples:
        raise RuntimeError('未找到任何视频文件，请检查视频格式或 --ext 参数。')

    print(f'\n共找到 {len(all_samples)} 个视频样本')

    # 按类别分层划分，保证每个类别在训练/验证集中的比例一致
    class_samples = {idx: [] for idx in class_to_idx.values()}
    for path, label in all_samples:
        class_samples[label].append((path, label))

    train_samples = []
    val_samples   = []
    for label, samples in class_samples.items():
        random.shuffle(samples)
        n_val = max(1, int(len(samples) * val_ratio))
        val_samples.extend(samples[:n_val])
        train_samples.extend(samples[n_val:])

    # 打乱顺序
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    # 写入输出文件
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train.txt')
    val_path   = os.path.join(output_dir, 'val.txt')

    with open(train_path, 'w', encoding='utf-8') as f:
        for path, label in train_samples:
            f.write(f'{path} {label}\n')

    with open(val_path, 'w', encoding='utf-8') as f:
        for path, label in val_samples:
            f.write(f'{path} {label}\n')

    # 保存类别映射文件
    if args.save_label_map:
        anno_dir = os.path.join(output_dir, 'annotations')
        os.makedirs(anno_dir, exist_ok=True)
        label_map_path = os.path.join(anno_dir, 'label_map.txt')
        with open(label_map_path, 'w', encoding='utf-8') as f:
            for cls_name in classes:
                f.write(f'{cls_name}\n')
        print(f'类别映射已保存至: {label_map_path}')

    print(f'\n划分结果:')
    print(f'  训练集: {len(train_samples)} 条 → {train_path}')
    print(f'  验证集: {len(val_samples)} 条  → {val_path}')
    print(f'\n示例 train.txt 前5行:')
    for path, label in train_samples[:5]:
        print(f'  {path} {label}')

    print('\n完成！请在配置文件中设置:')
    print(f'  data_root      = "{video_root}"')
    print(f'  ann_file_train = "{train_path}"')
    print(f'  ann_file_val   = "{val_path}"')
    print(f'  cls_head.num_classes = {len(classes)}')


if __name__ == '__main__':
    main()
