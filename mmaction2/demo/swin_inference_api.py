"""
swin_inference_api.py —— Video Swin Transformer 行为识别 Python API 推理脚本
=========================================================================
直接在此文件顶部修改 CONFIG 区域的路径和参数，然后运行：
    python swin_inference_api.py

无需每次在命令行手动输入长路径，适合反复调试使用。

支持三种推理模式（修改 MODE 变量切换）：
    "short"  —— 对整段视频做一次整体推理，输出 Top-K 标签
    "long"   —— 滑窗逐帧推理，适合长视频，逐帧保存结果
    "batch"  —— 批量推理多个视频目录下的所有 .mp4 文件

环境要求：Python 3.8 / PyTorch 2.0 / mmcv 2.2.0 / mmaction2 1.x
"""

# ============================================================
#  ★★★ 用户配置区：修改这里即可，不需要改下方代码 ★★★
# ============================================================

# ---------- 路径配置 ----------
# mmaction2 根目录（绝对路径）
MMACTION2_ROOT = '/root/autodl-tmp/pose_estimation/mmaction2'

# 模型配置文件路径（相对于 MMACTION2_ROOT）
CONFIG_FILE = 'configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py'

# 模型权重文件路径（相对于 MMACTION2_ROOT）
CHECKPOINT_FILE = 'checkpoints/swin_tiny_kinetics400.pth'

# 标签文件路径（相对于 MMACTION2_ROOT）
LABEL_FILE = 'tools/data/kinetics/label_map_k400.txt'

# ---------- 输入配置 ----------
# 推理模式："short" / "long" / "batch"
MODE = 'short'

# [short/long 模式] 输入视频文件路径（相对于 MMACTION2_ROOT）
INPUT_VIDEO = 'demo/demo.mp4'

# [batch 模式] 批量推理的视频目录（相对于 MMACTION2_ROOT），会扫描所有 .mp4
BATCH_VIDEO_DIR = 'demo/'

# ---------- 输出配置 ----------
# 结果输出目录（相对于 MMACTION2_ROOT）
OUTPUT_DIR = 'outputs'

# [short 模式] 输出视频文件名（None 则不保存视频，仅打印结果）
OUTPUT_VIDEO_NAME = 'result_short.mp4'

# [long 模式] 输出文件名（.mp4 保存视频，.json 保存 JSON）
OUTPUT_LONG_NAME = 'result_long.mp4'

# ---------- 推理参数 ----------
# 推理设备："cuda:0" / "cuda:1" / "cpu"
DEVICE = 'cuda:0'

# 显示 Top-K 个预测结果
TOP_K = 5

# [long 模式] 分数显示阈值（低于此值的类别不显示）
LONG_THRESHOLD = 0.4

# [long 模式] 窗口滑动比例（0~1 之间的小数）
#   0   → 每次滑动 1 帧（最密集，最慢）
#   0.5 → 每次滑动 clip_len * 0.5 帧（推荐）
#   1.0 → 窗口无重叠（最快）
LONG_STRIDE = 0.5

# ============================================================
#  以下为核心逻辑，通常无需修改
# ============================================================

import json
import os
import random
import sys
from collections import deque
from operator import itemgetter
from pathlib import Path

import cv2
import numpy as np
import torch

# 将 mmaction2 根目录加入 Python 路径，确保优先使用本地源码
sys.path.insert(0, MMACTION2_ROOT)
os.chdir(MMACTION2_ROOT)  # 切换工作目录，使相对路径生效

from mmengine import Config
from mmengine.dataset import Compose

from mmaction.apis import inference_recognizer, init_recognizer


# -------- 工具函数 --------

def load_labels(label_file: str) -> list:
    """读取标签文件，返回类别名称列表（index 对应行号）。"""
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def build_model(config_file: str, checkpoint_file: str, device: str):
    """根据配置文件和权重文件初始化识别模型。"""
    cfg = Config.fromfile(config_file)
    # 禁止自动从网络下载 backbone 预训练权重（推理阶段不需要）
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'backbone'):
        cfg.model.backbone.pretrained = None
    model = init_recognizer(cfg, checkpoint_file, device=device)
    print(f'[OK] 模型加载完成，设备: {device}')
    return model


def get_topk_results(pred_scores: list, labels: list, k: int = 5) -> list:
    """将预测分数转换为 Top-K (类别名称, 分数) 列表。"""
    score_tuples = list(zip(labels, pred_scores))
    score_sorted = sorted(score_tuples, key=lambda x: x[1], reverse=True)
    return score_sorted[:k]


def print_topk(results: list, title: str = ''):
    """格式化打印 Top-K 预测结果。"""
    if title:
        print(f'\n{"=" * 50}')
        print(f'  {title}')
        print(f'{"=" * 50}')
    for rank, (label, score) in enumerate(results, 1):
        bar = '█' * int(score * 30)
        print(f'  #{rank:2d}  {label:<30s}  {score:.4f}  {bar}')


# -------- 短视频整体推理 --------

def run_short_inference(model, video_path: str, labels: list,
                        top_k: int = 5, out_video: str = None):
    """对整段视频做一次整体推理（适合 <1min 的短视频/片段）。

    Args:
        model:      已初始化的识别模型
        video_path: 输入视频路径
        labels:     类别名称列表
        top_k:      显示前 K 个结果
        out_video:  输出视频路径（None 则不保存）
    """
    print(f'\n[short 模式] 推理视频: {video_path}')

    # 对整个视频做一次推理，返回 ActionDataSample
    pred_result = inference_recognizer(model, video_path)

    # 提取各类别分数并获取 Top-K
    pred_scores = pred_result.pred_score.tolist()
    results = get_topk_results(pred_scores, labels, top_k)
    print_topk(results, title=f'Top-{top_k} 识别结果')

    # 可选：保存带标签叠加的输出视频
    if out_video is not None:
        try:
            from mmaction.visualization import ActionVisualizer
            os.makedirs(os.path.dirname(out_video) or '.', exist_ok=True)
            visualizer = ActionVisualizer()
            visualizer.dataset_meta = dict(classes=labels)
            # 注意：add_datasample 会将输出写到 out_path 指定的路径
            visualizer.add_datasample(
                os.path.basename(out_video),
                video_path,
                pred_result,
                draw_pred=True,
                draw_gt=False,
                text_cfg={'colors': 'white'},
                fps=30,
                out_type='video',
                out_path=out_video)
            print(f'[OK] 输出视频已保存: {out_video}')
        except Exception as e:
            print(f'[WARN] 保存视频失败（可能缺少 moviepy）: {e}')

    return results


# -------- 长视频滑窗推理 --------

# 需要从 test_pipeline 中移除的文件解码步骤（由内存帧数组替代）
_EXCLUDED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode',
    'PyAVInit', 'PyAVDecode', 'RawFrameDecode'
]


def build_long_video_pipeline(model):
    """将模型的 test_pipeline 改造为接收内存帧数组的 pipeline。

    原始 pipeline 中的文件解码步骤被移除，改为 ArrayDecode，
    这样可以直接传入从 OpenCV 读取的 numpy 帧数组。

    Returns:
        (test_pipeline, sample_length, num_clips, clip_len)
    """
    cfg = model.cfg
    pipeline = cfg.test_pipeline
    pipeline_ = list(pipeline)  # 拷贝一份用于修改
    sample_length = 0
    num_clips = 1
    clip_len = 32

    for step in list(pipeline):
        if 'SampleFrames' in step['type']:
            # 提取窗口大小参数
            sample_length = step['clip_len'] * step['num_clips']
            num_clips = step['num_clips']
            clip_len  = step['clip_len']
            pipeline_.remove(step)   # 移除文件采样步骤
        if step['type'] in _EXCLUDED_STEPS:
            pipeline_.remove(step)   # 移除文件解码步骤

    # 在 pipeline 开头插入 ArrayDecode，接收内存帧数组
    pipeline_.insert(0, dict(type='ArrayDecode'))
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0, '未在 test_pipeline 中找到 SampleFrames，请检查配置文件'
    return test_pipeline, sample_length, num_clips, clip_len


def _infer_window(model, data_template, frame_queue, sample_length,
                  test_pipeline, stride):
    """对当前滑动窗口内的帧序列做一次推理。

    Returns:
        (True, scores): 推理成功，scores 为各类别分数列表
        (False, None):  帧数不足，跳过
    """
    if len(frame_queue) != sample_length:
        return False, None

    cur_windows = list(np.array(frame_queue))

    # 第一次推理时确定图像尺寸
    cur_data = data_template.copy()
    if cur_data['img_shape'] is None:
        cur_data['img_shape'] = frame_queue[0].shape[:2]
    cur_data.update(dict(
        array=cur_windows,
        modality='RGB',
        frame_inds=np.arange(sample_length)
    ))

    result = inference_recognizer(model, cur_data, test_pipeline=test_pipeline)
    scores = result.pred_score.tolist()

    # 根据 stride 主动弹出旧帧，实现大步滑动
    if stride > 0:
        pred_stride = int(sample_length * stride)
        pred_stride = min(pred_stride, len(frame_queue))  # 防止越界
        for _ in range(pred_stride):
            frame_queue.popleft()
    # stride=0 时 deque 的 maxlen 机制会自动弹出最旧帧

    return True, scores


def run_long_inference(model, video_path: str, labels: list,
                       out_file: str, threshold: float = 0.4,
                       stride: float = 0.5, input_step: int = 1):
    """对长视频进行滑窗逐帧推理。

    Args:
        model:      已初始化的识别模型
        video_path: 输入视频路径
        labels:     类别名称列表
        out_file:   输出文件路径（.mp4 输出视频，.json 输出 JSON）
        threshold:  分数显示阈值
        stride:     滑动窗口步长比例（0~1）
        input_step: 每隔多少帧从备份帧中随机取一帧入队
    """
    print(f'\n[long 模式] 推理视频: {video_path}')
    print(f'  stride={stride}, threshold={threshold}')

    # 构建适配内存帧数组的 pipeline
    test_pipeline, sample_length, _, _ = build_long_video_pipeline(model)
    data_template = dict(img_shape=None, modality='RGB', label=-1,
                         num_clips=1, clip_len=sample_length)

    # 初始化滑动窗口队列和结果队列
    frame_queue  = deque(maxlen=sample_length)
    result_queue = deque(maxlen=1)

    # 打开视频文件
    cap          = cv2.VideoCapture(video_path)
    num_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f'  视频信息: {num_frames} 帧, {frame_width}x{frame_height}, {fps:.1f}fps')

    # 准备输出
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = (None if out_file.endswith('.json')
                    else cv2.VideoWriter(out_file, fourcc, fps,
                                         (frame_width, frame_height)))
    out_json    = {}
    text_info   = {}
    backup      = []
    ind         = 0

    # 字体参数（用于在视频帧上绘制文字）
    FONT  = cv2.FONT_HERSHEY_COMPLEX_SMALL
    SCALE = 1
    WHITE = (255, 255, 255)
    GRAY  = (128, 128, 128)

    print(f'  处理中，共 {num_frames} 帧...')
    while ind < num_frames:
        ind += 1
        ret, frame = cap.read()
        if frame is None:
            continue

        # OpenCV 读取的是 BGR，转换为 RGB 后入队
        backup.append(np.array(frame)[:, :, ::-1])

        if ind == sample_length:
            # 第一次填满窗口：直接批量入队
            frame_queue.extend(backup)
            backup = []
        elif (len(backup) == input_step and ind > sample_length) or ind == num_frames:
            # 每隔 input_step 帧随机取一帧入队
            frame_queue.append(random.choice(backup))
            backup = []

        # 对当前窗口做推理
        ok, scores = _infer_window(
            model, data_template, frame_queue, sample_length,
            test_pipeline, stride)

        if ok:
            # 按分数排序取 Top-5
            paired = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
            result_queue.append(paired[:min(len(labels), 5)])

        # 将结果写入视频帧或 JSON
        if out_file.endswith('.json'):
            # JSON 模式：记录每帧的识别结果
            if result_queue:
                frame_result = {}
                for i, (lbl, sc) in enumerate(result_queue[0]):
                    if sc < threshold:
                        break
                    frame_result[i + 1] = f'{lbl}: {sc:.2f}'
                out_json[ind] = frame_result if frame_result else 'waiting...'
        else:
            # 视频模式：在当前 BGR 帧上叠加文字后写入
            if result_queue:
                text_info = {}
                for i, (lbl, sc) in enumerate(list(result_queue)[0]):
                    if sc < threshold:
                        break
                    pos = (0, 40 + i * 22)
                    text = f'{lbl}: {sc:.2f}'
                    text_info[pos] = text
                    cv2.putText(frame, text, pos, FONT, SCALE, WHITE, 1, 1)
            elif text_info:
                for pos, text in text_info.items():
                    cv2.putText(frame, text, pos, FONT, SCALE, WHITE, 1, 1)
            else:
                cv2.putText(frame, 'Preparing...', (0, 40), FONT, SCALE, GRAY, 1, 1)
            video_writer.write(frame)

    cap.release()
    if video_writer:
        video_writer.release()

    if out_file.endswith('.json'):
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(f'[OK] 输出已保存: {out_file}')


# -------- 批量推理 --------

def run_batch_inference(model, video_dir: str, labels: list, top_k: int = 5):
    """扫描目录下所有 .mp4 文件并逐一进行整体推理。

    Args:
        model:     已初始化的识别模型
        video_dir: 视频目录路径
        labels:    类别名称列表
        top_k:     每个视频显示前 K 个结果
    Returns:
        dict: {视频文件名 -> Top-K 结果列表}
    """
    video_paths = list(Path(video_dir).glob('*.mp4'))
    if not video_paths:
        print(f'[WARN] 目录 {video_dir} 下没有找到 .mp4 文件')
        return {}

    all_results = {}
    for i, vp in enumerate(sorted(video_paths), 1):
        print(f'\n[{i}/{len(video_paths)}] 处理: {vp.name}')
        try:
            pred_result = inference_recognizer(model, str(vp))
            pred_scores = pred_result.pred_score.tolist()
            results     = get_topk_results(pred_scores, labels, top_k)
            print_topk(results, title=vp.name)
            all_results[vp.name] = [(lbl, float(f'{sc:.4f}')) for lbl, sc in results]
        except Exception as e:
            print(f'  [ERROR] {vp.name} 推理失败: {e}')
            all_results[vp.name] = []

    return all_results


# ============================================================
#  主程序入口
# ============================================================

def main():
    print('=' * 60)
    print('  Video Swin Transformer 行为识别 —— Python API 推理')
    print('=' * 60)
    print(f'  模式   : {MODE}')
    print(f'  配置   : {CONFIG_FILE}')
    print(f'  权重   : {CHECKPOINT_FILE}')
    print(f'  设备   : {DEVICE}')
    print()

    # 构建绝对路径
    config_path = os.path.join(MMACTION2_ROOT, CONFIG_FILE)
    ckpt_path   = os.path.join(MMACTION2_ROOT, CHECKPOINT_FILE)
    label_path  = os.path.join(MMACTION2_ROOT, LABEL_FILE)
    video_path  = os.path.join(MMACTION2_ROOT, INPUT_VIDEO)
    out_dir     = os.path.join(MMACTION2_ROOT, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # 加载标签
    labels = load_labels(label_path)
    print(f'[OK] 已加载 {len(labels)} 个类别标签')

    # 初始化模型（只加载一次，三种模式共用）
    model = build_model(config_path, ckpt_path, DEVICE)

    # ---- 按 MODE 分支执行 ----
    if MODE == 'short':
        out_video = os.path.join(out_dir, OUTPUT_VIDEO_NAME) if OUTPUT_VIDEO_NAME else None
        run_short_inference(model, video_path, labels, TOP_K, out_video)

    elif MODE == 'long':
        out_file = os.path.join(out_dir, OUTPUT_LONG_NAME)
        run_long_inference(
            model, video_path, labels,
            out_file=out_file,
            threshold=LONG_THRESHOLD,
            stride=LONG_STRIDE)

    elif MODE == 'batch':
        batch_dir = os.path.join(MMACTION2_ROOT, BATCH_VIDEO_DIR)
        all_results = run_batch_inference(model, batch_dir, labels, TOP_K)
        # 保存批量结果到 JSON
        json_out = os.path.join(out_dir, 'batch_results.json')
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f'\n[OK] 批量结果已保存: {json_out}')

    else:
        print(f'[ERROR] 未知 MODE: {MODE}，请设置为 short / long / batch')
        sys.exit(1)

    print('\n[完成] 推理结束。')


if __name__ == '__main__':
    main()
