# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================
# demo.py —— 【短视频 / 片段级行为识别】推理脚本
#
# 功能：对一段完整视频（或原始帧目录）执行一次整体推理，输出 Top-5 动作类别
#       及对应置信度，并可选择将结果渲染成带标签的视频/GIF。
#
# 与其他 demo 的区别：
#   - demo.py           → 短视频整体推理，输出 Top-5 标签（本文件）
#   - long_video_demo.py→ 长视频滑窗推理，逐帧输出当前窗口预测结果
#   - webcam_demo.py    → 摄像头实时推理
#   - demo_skeleton.py  → 先检测人体姿态关键点，再做骨骼点行为识别
#   - demo_audio.py     → 基于音频特征（.npy）的行为识别
#   - demo_inferencer.py→ 新式高级 API，支持批量输入输出
# ============================================================

import argparse
import os.path as osp
from operator import itemgetter       # 用于按得分排序
from typing import Optional, Tuple   # 类型注解

from mmengine import Config, DictAction  # Config：读取 .py 配置文件；DictAction：命令行传字典

from mmaction.apis import inference_recognizer, init_recognizer  # 核心推理 API
from mmaction.visualization import ActionVisualizer              # 结果可视化工具


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='MMAction2 demo')

    # ── 四个必须的位置参数（按顺序填写，无需 -- 前缀）──
    parser.add_argument('config',     help='模型配置文件路径 (.py)')
    parser.add_argument('checkpoint', help='模型权重文件路径或 URL (.pth)')
    parser.add_argument('video',      help='输入视频文件路径/URL，或原始帧目录')
    parser.add_argument('label',      help='标签文件路径（每行一个类别名）')

    # ── 可选参数 ──
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,   # 允许用 key=value 形式传入，自动解析为 dict
        help='动态覆盖配置文件中的字段，格式: key=value')
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='推理设备，默认 cuda:0；无 GPU 可改为 cpu')
    parser.add_argument(
        '--fps', default=30, type=int,
        help='输出视频帧率（当输入是原始帧目录时使用）')
    parser.add_argument(
        '--font-scale', default=None, type=float,
        help='输出视频中标签文字的字体大小缩放比')
    parser.add_argument(
        '--font-color', default='white',
        help='输出视频中标签文字颜色，默认白色')
    parser.add_argument(
        '--target-resolution', nargs=2, default=None, type=int,
        help='目标分辨率 (width height)，-1 表示等比缩放对应维度')
    parser.add_argument(
        '--out-filename', default=None,
        help='输出文件名；扩展名为 .gif 则导出 GIF，否则导出视频；不指定则不保存')

    args = parser.parse_args()
    return args


def get_output(
    video_path: str,
    out_filename: str,
    data_sample: str,
    labels: list,
    fps: int = 30,
    font_scale: Optional[str] = None,
    font_color: str = 'white',
    target_resolution: Optional[Tuple[int]] = None,
) -> None:
    """将推理结果渲染成带标签的视频或 GIF。

    使用 moviepy 库完成视频读取与合成；
    使用 ActionVisualizer 在每帧上叠加预测标签文字。

    Args:
        video_path:        输入视频路径
        out_filename:      输出文件名（.mp4 或 .gif）
        data_sample:       inference_recognizer 返回的 ActionDataSample 对象
        labels:            所有类别名称列表
        fps:               输出视频帧率
        font_scale:        文字大小缩放，None 则使用默认值
        font_color:        文字颜色字符串
        target_resolution: 输出分辨率 (w, h)，None 则保持原始分辨率
    """
    # 暂不支持 HTTP/HTTPS 视频 URL（需要先下载）
    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError

    # 根据输出扩展名决定生成视频还是 GIF
    out_type = 'gif' if osp.splitext(out_filename)[1] == '.gif' else 'video'

    # 初始化可视化器，并设置类别元信息（类别名列表）
    visualizer = ActionVisualizer()
    visualizer.dataset_meta = dict(classes=labels)

    # 构造文字渲染配置
    text_cfg = {'colors': font_color}
    if font_scale is not None:
        text_cfg.update({'font_sizes': font_scale})  # 仅在用户指定时覆盖默认字体大小

    # 调用可视化器：将预测结果叠加到视频帧上并写出
    visualizer.add_datasample(
        out_filename,             # 输出文件名
        video_path,               # 原始视频路径
        data_sample,              # 模型预测结果（含 pred_score、pred_label 等）
        draw_pred=True,           # 绘制预测标签
        draw_gt=False,            # 不绘制真实标签（推理阶段无 GT）
        text_cfg=text_cfg,        # 文字样式
        fps=fps,                  # 输出帧率
        out_type=out_type,        # 输出类型：video / gif
        out_path=osp.join('demo', out_filename),  # 实际保存路径
        target_resolution=target_resolution)      # 目标分辨率


def main():
    """主函数：加载模型 → 推理 → 打印结果 → 可选保存视频。"""
    args = parse_args()

    # 读取配置文件，然后用命令行参数覆盖其中的字段
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 根据配置和权重文件初始化识别器模型（含加载权重到指定设备）
    model = init_recognizer(cfg, args.checkpoint, device=args.device)

    # 对输入视频执行一次完整推理，返回 ActionDataSample 对象
    pred_result = inference_recognizer(model, args.video)

    # 取出各类别的预测分数列表（长度 = 类别数，如 400）
    pred_scores = pred_result.pred_score.tolist()

    # 将 (class_index, score) 打包成元组列表，再按分数从高到低排序
    score_tuples = tuple(zip(range(len(pred_scores)), pred_scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)

    # 取 Top-5 结果
    top5_label = score_sorted[:5]

    # 读取标签文件，每行一个类别名，去掉首尾空白
    labels = open(args.label).readlines()
    labels = [x.strip() for x in labels]

    # 将 class_index 映射为类别名称
    results = [(labels[k[0]], k[1]) for k in top5_label]

    # 打印 Top-5 预测结果
    print('The top-5 labels with corresponding scores are:')
    for result in results:
        print(f'{result[0]}: ', result[1])

    # 如果指定了输出文件名，则渲染带标签的视频
    if args.out_filename is not None:

        # 校验 target_resolution 参数的合法性（-1 代表等比缩放该维度）
        if args.target_resolution is not None:
            if args.target_resolution[0] == -1:
                assert isinstance(args.target_resolution[1], int)
                assert args.target_resolution[1] > 0
            if args.target_resolution[1] == -1:
                assert isinstance(args.target_resolution[0], int)
                assert args.target_resolution[0] > 0
            args.target_resolution = tuple(args.target_resolution)

        # 调用 get_output 生成带标签视频
        get_output(
            args.video,
            args.out_filename,
            pred_result,
            labels,
            fps=args.fps,
            font_scale=args.font_scale,
            font_color=args.font_color,
            target_resolution=args.target_resolution)


if __name__ == '__main__':
    main()
