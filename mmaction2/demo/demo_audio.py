# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================
# demo_audio.py —— 【音频模态行为识别】脚本
#
# 功能：对预先提取大的音频特征文件（.npy）进行行为识别。
#       输入不是视频文件而是 numpy 格式的音频特征。
#
# 与 demo.py 的区别：
#   demo.py      → 输入 RGB 视频，使用视觉特征识别
#   demo_audio.py→ 输入音频特征（.npy），使用音频模态识别
# ============================================================

import argparse
from operator import itemgetter  # 用于按分数排序

import torch
from mmengine import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='MMAction2 demo')

    # ── 4 个必须位置参数 ──
    parser.add_argument('config',     help='模型配置文件 (.py)')
    parser.add_argument('checkpoint', help='模型权重 (.pth)或 URL')
    parser.add_argument('audio',      help='音频特征文件（必须是 .npy 格式）')
    parser.add_argument('label',      help='标签文件（每行一类）')

    # ── 可选参数 ──
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default={},
        help='动态覆盖配置字段，格式: key=value')
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='推理设备')

    args = parser.parse_args()
    return args


def main():
    """加载模型 → 对音频特征推理 → 输出 Top-5 结果。"""
    args   = parse_args()
    device = torch.device(args.device)  # 转换设备字符串为 torch.device

    # 加载配置并应用命令行覆盖
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # 初始化模型
    model = init_recognizer(cfg, args.checkpoint, device=device)

    # 仅支持 .npy 音频特征和输入格式校验
    if not args.audio.endswith('.npy'):
        raise NotImplementedError('当前只支持预先提取的 .npy 音频特征文件')

    # 执行推理，返回 ActionDataSample 对象
    pred_result = inference_recognizer(model, args.audio)

    # 提取各类别的预测分数
    pred_scores = pred_result.pred_score.tolist()

    # 按分数从高到低排序，取 Top-5
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label   = score_sorted[:5]

    # 读取标签文件
    labels  = open(args.label).readlines()
    labels  = [x.strip() for x in labels]

    # 将 class_index 映射为类名
    # 加载配置并应用命令行覆盖
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # 初始化模型
    model = init_recognizer(cfg, args.checkpoint, device=device)

    # 仅支持 .npy 音频特征和输入格式校验
    if not args.audio.endswith('.npy'):
        raise NotImplementedError('当前只支持预先提取的 .npy 音频特征文件')

    # 执行推理，返回 ActionDataSample 对象
    pred_result = inference_recognizer(model, args.audio)

    # 提取各类别的预测分数
    pred_scores = pred_result.pred_score.tolist()

    # 按分数从高到低排序，取 Top-5
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    top5_label   = score_sorted[:5]

    # 读取标签文件
    labels  = open(args.label).readlines()
    labels  = [x.strip() for x in labels]

    # 将 class_index 映射为类名
    results = [(labels[k[0]], k[1]) for k in top5_label]

    print('The top-5 labels with corresponding scores are:')
    for result in results:
        print(f'{result[0]}: ', result[1])


if __name__ == '__main__':
    main()
