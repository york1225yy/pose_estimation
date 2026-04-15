# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================
# long_video_demo.py —— 【长视频滑窗推理】脚本
#
# 功能：对任意长度的视频采用滑动窗口策略逐帧进行行为识别，
#       并实时将预测结果叠加到视频上（或保存为 JSON）。
#
# 与 demo.py 的区别：
#   demo.py         → 对整段视频做一次整体推理，适合<1min 的短片
#   long_video_demo → 滑动窗口逗帧推理，适合任意长度视频
#
# stride 参数说明（重要！）：
#   stride 是 0~1 之间的小数，表示窗口每次滑动帧数占窗口大小的比例。
#   例： clip_len=32, stride=0.5 → 每次滑动 32*0.5=16 帧
#   设为 0（默认）→ 每次仅滑动 1 帧（最密集）
#   设为 1.0 → 窗口无重叠（最快）
#   不要传 16 这样的整数！会导致滑动帧数远超过队列大小而崩溃
# ============================================================

import argparse
import json
import random
from collections import deque     # 双端队列，用于维护滑动窗口内的帧缓冲
from operator import itemgetter   # 按元组第 N 个元素排序

import cv2                        # OpenCV：视频读/写及帧上绘制文字
import mmengine
import numpy as np
import torch
from mmengine import Config, DictAction
from mmengine.dataset import Compose  # 将多个预处理 transform 组合成 pipeline

from mmaction.apis import inference_recognizer, init_recognizer

# ── OpenCV 绘制文字相关常量 ──
FONTFACE  = cv2.FONT_HERSHEY_COMPLEX_SMALL  # 字体样式
FONTSCALE = 1      # 字体大小倍率
THICKNESS = 1      # 线宽
LINETYPE  = 1      # 线型类型

# 在构建 demo pipeline 时需要剪除的预处理步骤名称，
# 这些步骤负责打开视频文件并解码，在 demo 中我们直接操作内存帧数组来替代
EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='MMAction2 predict different labels in a long video demo')

    # ── 5 个必须位置参数（按顺序填写，不要加 -- 前缀）──
    parser.add_argument('config',      help='模型配置文件 (.py)')
    parser.add_argument('checkpoint',  help='模型权重文件 (.pth)或 URL')
    parser.add_argument('video_path',  help='输入视频路径')
    parser.add_argument('label',       help='标签文件（每行一个类名）')
    parser.add_argument('out_file',    help='输出文件路径（.mp4 输出视频，.json 输出 JSON）')

    # ── 可选参数 ──
    parser.add_argument(
        '--input-step', type=int, default=1,
        help='输入采样步长：每隔多少帧从备份帧中随机抽取一帧入队列')
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='推理设备')
    parser.add_argument(
        '--threshold', type=float, default=0.01,
        help='预测分数阈值，低于此阈值的类别不会显示')
    parser.add_argument(
        '--stride', type=float, default=0,
        help=(
            '窗口滑动比例（0~1之间的小数）。'
            '实际滑动帧数 = stride * sample_length。'
            '设为 0 表示每次滑动 1 帧（最密集）。'
            '示例： stride=0.5 并 clip_len=32 → 每次滑动 16 帧'))
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction, default={},
        help='动态覆盖配置字段，格式: key=value')
    parser.add_argument(
        '--label-color', nargs='+', type=int, default=(255, 255, 255),
        help='输出视频中标签文字的 BGR 颜色，默认白色')
    parser.add_argument(
        '--msg-color', nargs='+', type=int, default=(128, 128, 128),
        help='输出视频中提示消息文字的 BGR 颜色，默认灰色')

    args = parser.parse_args()
    return args


def show_results_video(result_queue, text_info, thr, msg, frame,
                       video_writer, label_color=(255, 255, 255),
                       msg_color=(128, 128, 128)):
    """将当前窗口的识别结果叠加到视频帧上并写入输出文件。

    Args:
        result_queue: 存储最新推理结果的队列（容量为 1）
        text_info:    上一帧已绘制的文字信息，当没有新结果时复用
        thr:          分数阈值
        msg:          还没收集到足够帧时显示的提示文字
        frame:        当前 BGR 视频帧（numpy 数组）
        video_writer: OpenCV VideoWriter 对象
    """
    if len(result_queue) != 0:
        # 有新的推理结果到来：清空上一帧文字信息，重新绘制
        text_info = {}
        results = result_queue.popleft()   # 弹出最新结果
        for i, result in enumerate(results):
            selected_label, score = result
            if score < thr:
                break  # 分数低于阈值则不显示
            location = (0, 40 + i * 20)    # 窗口内每行标签的屏幕位置
            text = selected_label + ': ' + str(round(score, 2))
            text_info[location] = text
            # 在帧上用 OpenCV 绘制文字
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                        label_color, THICKNESS, LINETYPE)
    elif len(text_info):
        # 没有新结果但上帧有文字：复制上帧的标签（防止闪烁）
        for location, text in text_info.items():
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                        label_color, THICKNESS, LINETYPE)
    else:
        # 还没有任何结果：显示等待提示文字
        cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE,
                    msg_color, THICKNESS, LINETYPE)
    video_writer.write(frame)   # 将当前帧写入输出视频
    return text_info


def get_results_json(result_queue, text_info, thr, msg, ind, out_json):
    """将当前窗口的识别结果写入 JSON 字典。

    Args:
        result_queue: 存储推理结果的队列
        text_info:    上一帧保存的文字信息（没有新结果时复用）
        thr:          分数阈值
        msg:          提示消息
        ind:          当前帧编号（JSON key）
        out_json:     输出 JSON 字典
    """
    if len(result_queue) != 0:
        text_info = {}
        results = result_queue.popleft()
        for i, result in enumerate(results):
            selected_label, score = result
            if score < thr:
                break
            # JSON 中用序号作为 key，内容是 "类名: 分数"
            text_info[i + 1] = selected_label + ': ' + str(round(score, 2))
        out_json[ind] = text_info
    elif len(text_info):
        out_json[ind] = text_info   # 复用上帧结果
    else:
        out_json[ind] = msg         # 还在等待阶段
    return text_info, out_json


def show_results(model, data, label, args):
    """主循环：逆帧读取视频 → 维护滑动窗口 → 触发推理 → 写入结果。"""
    # 滑动窗口队列：maxlen=sample_length，满后自动弹出最旧帧
    frame_queue  = deque(maxlen=args.sample_length)
    # 结果队列：容量为 1，仅保留最新一次推理结果
    result_queue = deque(maxlen=1)

    # 使用 OpenCV 打开视频文件
    cap          = cv2.VideoCapture(args.video_path)
    num_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 总帧数
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 帧宽
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高
    fps          = cap.get(cv2.CAP_PROP_FPS)                 # 帧率

    msg       = 'Preparing action recognition ...'  # 等待提示文字
    text_info = {}    # 当前帧要展示的标签信息
    out_json  = {}    # JSON 输出用字典

    # 设置视频编码格式和输出 VideoWriter
    fourcc      = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size  = (frame_width, frame_height)
    ind         = 0
    # 如果输出为 JSON 则不需要 VideoWriter
    video_writer = None if args.out_file.endswith('.json') \
        else cv2.VideoWriter(args.out_file, fourcc, fps, frame_size)

    prog_bar     = mmengine.ProgressBar(num_frames)  # 显示进度条
    backup_frames = []  # 每个滑动步长内收集的帧备份

    while ind < num_frames:
        ind += 1
        prog_bar.update()             # 更新进度条
        ret, frame = cap.read()       # 读取一帧 BGR 图像
        if frame is None:
            continue                  # 解码失败则跳过

        # 将 BGR 帧转为 RGB，并添加到备份列表
        backup_frames.append(np.array(frame)[:, :, ::-1])

        if ind == args.sample_length:
            # 运行到第一个窗口大小时，直接将备份帧全部添加到窗口，快速得到第一次预测
            frame_queue.extend(backup_frames)
            backup_frames = []
        elif ((len(backup_frames) == args.input_step
               and ind > args.sample_length) or ind == num_frames):
            # 每雔 input_step 帧或到达末帧时：从备份帧中随机取一帧入队列
            chosen_frame = random.choice(backup_frames)
            backup_frames = []
            frame_queue.append(chosen_frame)  # deque 满则自动 popleft

        # 尝试对当前窗口做推理
        ret, scores = inference(model, data, args, frame_queue)

        if ret:
            # 推理成功：将 label+score 打包并按分数排序，取 Top-5
            num_selected_labels = min(len(label), 5)
            scores_tuples = tuple(zip(label, scores))
            scores_sorted = sorted(scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]
            result_queue.append(results)  # 将结果添加到结果队列

        # 根据输出格式分别处理结果
        if args.out_file.endswith('.json'):
            text_info, out_json = get_results_json(
                result_queue, text_info, args.threshold, msg, ind, out_json)
        else:
            text_info = show_results_video(
                result_queue, text_info, args.threshold, msg, frame,
                video_writer, args.label_color, args.msg_color)

    # 释放资源
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # 输出 JSON 文件
    if args.out_file.endswith('.json'):
        with open(args.out_file, 'w') as js:
            json.dump(out_json, js)


def inference(model, data, args, frame_queue):
    """对当前滑动窗口内的帧序列执行一次推理。

    Returns:
        (False, None): 窗口内帧数不足，等待更多帧
        (True, scores): 推理成功，返回各类别的分数列表
    """
    # 窗口内帧数不够时直接返回，不推理
    if len(frame_queue) != args.sample_length:
        return False, None

    cur_windows = list(np.array(frame_queue))  # 将队列转为 numpy 数组列表

    # 第一次推理时初始化图像尺寸（后续使用相同尺寸）
    if data['img_shape'] is None:
        data['img_shape'] = frame_queue[0].shape[:2]  # (height, width)

    # 构建当前窗口的输入数据字典
    cur_data = data.copy()
    cur_data.update(dict(
        array=cur_windows,                   # 帧 RGB 数组列表
        modality='RGB',                       # 模态
        frame_inds=np.arange(args.sample_length)  # 帧标签（对应 pipeline 输入）
    ))

    # 调用 inference_recognizer 并提供预先构建好的 pipeline
    result = inference_recognizer(model, cur_data, test_pipeline=args.test_pipeline)
    scores = result.pred_score.tolist()  # 各类别预测分数列表

    # stride>0 时主动弹出多个帧，实现大步距滑动
    if args.stride > 0:
        pred_stride = int(args.sample_length * args.stride)  # 计算实际滑动帧数
        pred_stride = min(pred_stride, len(frame_queue))     # 防止超过队列大小导致崩溃
        for _ in range(pred_stride):
            frame_queue.popleft()    # 主动弹出旧帧
    # stride=0 时 deque 的 maxlen 机制已自动 popleft（添加新帧时）

    return True, scores


def main():
    """主函数：加载模型并构建滑动窗口 pipeline，然后启动长视频识别。"""
    args = parse_args()
    args.device = torch.device(args.device)  # 将字符串设备名转成 torch.device

    # 加载配置并和并命令行额外选项
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.cfg_options)

    # 初始化识别模型
    model = init_recognizer(cfg, args.checkpoint, device=args.device)

    # 构建输入数据模板（img_shape 一开始为 None，第一次推理时填充）
    data = dict(img_shape=None, modality='RGB', label=-1)

    # 读取标签文件，每元素为一个类名字符串
    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]

    # ── 将模型的 test_pipeline 改造为适合内存帧数组输入的 pipeline ──
    cfg = model.cfg
    sample_length = 0
    pipeline  = cfg.test_pipeline
    pipeline_ = pipeline.copy()    # 拷贝一份用于修改

    for step in pipeline:
        if 'SampleFrames' in step['type']:
            # 提取窗口大小 = clip_len * num_clips
            sample_length        = step['clip_len'] * step['num_clips']
            data['num_clips']    = step['num_clips']
            data['clip_len']     = step['clip_len']
            pipeline_.remove(step)   # 去掉文件采样步骤（我们已从内存获得帧）
        if step['type'] in EXCLUED_STEPS:
            pipeline_.remove(step)   # 去掉文件无式解码步骤

    # 在 pipeline 开头插入 ArrayDecode：将内存 numpy 帧数组转为模型输入格式
    # 注意：必须插在索引 0（即第一位），确保 Resize、ThreeCrop 在它之后执行
    pipeline_.insert(0, dict(type='ArrayDecode'))
    test_pipeline = Compose(pipeline_)  # 组合成可调用的 pipeline

    assert sample_length > 0, '配置中未找到 SampleFrames，请检查配置文件'
    args.sample_length = sample_length
    args.test_pipeline = test_pipeline

    # 启动长视频识别主循环
    show_results(model, data, label, args)


if __name__ == '__main__':
    main()
