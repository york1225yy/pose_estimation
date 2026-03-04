#!/usr/bin/env python3
"""
YOLO11-Pose ONNX 导出脚本
=========================
将 YOLO11-Pose PyTorch 模型导出为 ONNX 格式，
支持自定义输入尺寸、opset 版本、动态 batch 等参数。

用法示例：
  python export_onnx.py
  python export_onnx.py --model yolo11s-pose.pt --imgsz 640 --opset 11
  python export_onnx.py --model yolo11m-pose.pt --imgsz 416 --dynamic --simplify
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11-Pose ONNX 导出")
    parser.add_argument("--model", type=str, default="yolo11s-pose.pt",
                        help="PyTorch 模型路径 (默认: yolo11s-pose.pt)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="导出输入尺寸 (默认: 640)")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset 版本 (默认: 11, TDA4 推荐 11)")
    parser.add_argument("--dynamic", action="store_true",
                        help="启用动态 batch 维度")
    parser.add_argument("--simplify", action="store_true",
                        help="使用 onnx-simplifier 简化模型")
    parser.add_argument("--half", action="store_true",
                        help="导出 FP16 模型")
    parser.add_argument("--device", type=str, default="cpu",
                        help="导出设备 (默认: cpu)")
    parser.add_argument("--output", type=str, default=None,
                        help="自定义输出路径（默认与 .pt 同目录）")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"{'='*60}")
    print(f"YOLO11-Pose ONNX 导出")
    print(f"{'='*60}")
    print(f"  模型:     {args.model}")
    print(f"  输入尺寸: {args.imgsz}")
    print(f"  opset:    {args.opset}")
    print(f"  动态batch: {args.dynamic}")
    print(f"  简化:     {args.simplify}")
    print(f"  FP16:     {args.half}")
    print(f"  设备:     {args.device}")
    print()

    # 加载模型
    model = YOLO(args.model)

    # 导出
    export_kwargs = dict(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
        half=args.half,
        device=args.device,
    )

    export_path = model.export(**export_kwargs)
    print(f"\nONNX 模型已导出: {export_path}")

    # 验证 ONNX 模型有效性
    try:
        import onnx
        onnx_model = onnx.load(str(export_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过 ✓")

        # 打印模型信息
        graph = onnx_model.graph
        print(f"\n模型信息:")
        print(f"  输入: {[inp.name for inp in graph.input]}")
        for inp in graph.input:
            shape = [d.dim_value if d.dim_value else d.dim_param
                     for d in inp.type.tensor_type.shape.dim]
            print(f"    {inp.name}: {shape}")

        print(f"  输出: {[out.name for out in graph.output]}")
        for out in graph.output:
            shape = [d.dim_value if d.dim_value else d.dim_param
                     for d in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: {shape}")

    except ImportError:
        print("[WARN] onnx 库未安装，跳过验证 (pip install onnx)")
    except Exception as e:
        print(f"[WARN] ONNX 验证异常: {e}")

    print(f"\n{'='*60}")
    print(f"导出完成！可用于 TDA4 / ONNX Runtime 部署")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
