# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================
# demo_inferencer.py —— 【新式高级 Inferencer API 推理】脚本
#
# 功能：使用 MMAction2Inferencer 高级封装 API 进行推理，
#       支持按模型名称自动下载权重、批量输入和多任务联合推理。
#
# 与 demo.py 的区别：
#   demo.py           → 手动传入 config/checkpoint 路径，适合定制化调试
#   demo_inferencer.py→ 传入模型名称自动下载，支持批量、联合推理
# ============================================================
from argparse import ArgumentParser

# MMAction2Inferencer 是新式高级 API，封装了模型初始化、输入预处理、推理和结果输出
from mmaction.apis.inferencers import MMAction2Inferencer


def parse_args():
    """解析命令行参数，并将其分为初始化参数和调用参数两组。"""
    parser = ArgumentParser()

    # ── 必须参数 ──
    parser.add_argument(
        'inputs', type=str,
        help='输入视频文件路径或原始帧目录路径')

    # ── 输出参数 ──
    parser.add_argument(
        '--vid-out-dir', type=str, default='',
        help='视频输出目录，不指定则不保存视频')

    # ── 模型相关参数（初始化时使用） ──
    parser.add_argument(
        '--rec', type=str, default=None,
        help='行为识别模型：配置文件路径 或 metafile 中定义的模型名称')
    parser.add_argument(
        '--rec-weights', type=str, default=None,
        help='自定义权重文件路径；缺省时如果 --rec 是模型名则自动从 metafile 下载')
    parser.add_argument(
        '--label-file', type=str, default=None,
        help='标签文件，缺省时使用模型内置标签')
    parser.add_argument(
        '--device', type=str, default=None,
        help='推理设备，缺省自动选择可用设备')

    # ── 推理调用参数 ──
    parser.add_argument(
        '--batch-size', type=int, default=1,
        help='批量推理大小')
    parser.add_argument(
        '--show', action='store_true',
        help='在弹出窗口中实时显示推理结果')
    parser.add_argument(
        '--print-result', action='store_true',
        help='将推理结果打印到控制台')
    parser.add_argument(
        '--pred-out-file', type=str, default='',
        help='推理结果保存文件（如 result.json）')

    # 将命令行参数转换为字典
    call_args = vars(parser.parse_args())

    # 将参数分为两组：
    #   init_args  → 传给 MMAction2Inferencer.__init__（模型加载相关）
    #   call_args  → 传给 MMAction2Inferencer.__call__（推理相关）
    init_kws  = ['rec', 'rec_weights', 'device', 'label_file']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)  # 从 call_args 中提取并删除

    return init_args, call_args


def main():
    """主函数：初始化 Inferencer 并执行推理。"""
    init_args, call_args = parse_args()

    # 初始化 Inferencer：内部会加载模型和权重
    mmaction2 = MMAction2Inferencer(**init_args)

    # 执行推理：内部处理输入、批量推理、结果保存ll__（推理相关）
    init_kws  = ['rec', 'rec_weights', 'device', 'label_file']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)  # 从 call_args 中提取并删除

    return init_args, call_args


def main():
    """主函数：初始化 Inferencer 并执行推理。"""
    init_args, call_args = parse_args()

    # 初始化 Inferencer：内部会加载模型和权重
    mmaction2 = MMAction2Inferencer(**init_args)

    # 执行推理：内部处理输入、批量推理、结果保存
    mmaction2(**call_args)


if __name__ == '__main__':
    main()
