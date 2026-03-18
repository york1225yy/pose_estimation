"""
YOWOv3 ONNX 导出脚本
用法（在 YOWOv3/ 目录下运行）：
    python export_onnx.py
或指定权重与配置：
    python export_onnx.py --weights weights/checkpoint/ema_epoch_7.pth \
                          --config  weights/checkpoint/config.yaml   \
                          --output  yowov3_C23.onnx                  \
                          --opset   12
"""

import os
import sys
import argparse

import torch
import yaml

# 确保在 YOWOv3/ 目录下运行，使 import 路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.TSN.YOWOv3 import build_yowov3


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOWOv3 EMA checkpoint to ONNX")
    parser.add_argument("--weights", type=str,
                        default="weights/checkpoint/ema_epoch_7.pth",
                        help="Path to ema_epoch_7.pth (raw state_dict)")
    parser.add_argument("--config", type=str,
                        default="weights/checkpoint/config.yaml",
                        help="Path to the config.yaml used when training this checkpoint")
    parser.add_argument("--output", type=str,
                        default="yowov3_C23.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version (default 12)")
    return parser.parse_args()


def export(weights: str, config_path: str, output: str, opset: int):
    # ── 1. 加载 config ──────────────────────────────────────────────────────
    print(f"[1/4] Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # 将 pretrain_path 指向 EMA 权重（覆盖 config 中的 null）
    config["pretrain_path"] = weights

    print(f"      backbone2D    : {config['backbone2D']}")
    print(f"      backbone3D    : {config['backbone3D']}")
    print(f"      num_classes   : {config['num_classes']}")
    print(f"      img_size      : {config['img_size']}")
    print(f"      clip_length   : {config['clip_length']}")
    print(f"      interchannels : {config['interchannels']}")

    # ── 2. 构建模型并加载权重 ────────────────────────────────────────────────
    print(f"\n[2/4] Building model & loading weights: {weights}")
    model = build_yowov3(config)
    model.eval()

    device = torch.device("cpu")  # ONNX 导出建议在 CPU 上进行
    model.to(device)

    # ── 3. 创建 dummy 输入 ──────────────────────────────────────────────────
    img_size    = config["img_size"]
    clip_length = config["clip_length"]
    dummy_input = torch.randn(1, 3, clip_length, img_size, img_size, device=device)

    print(f"      dummy input shape: {list(dummy_input.shape)}")

    # ── 4. 导出 ONNX ─────────────────────────────────────────────────────────
    print(f"\n[3/4] Exporting to ONNX (opset={opset}): {output}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output,
            verbose=False,
            opset_version=opset,
            input_names=["clips"],        # [B, 3, T, H, W]
            output_names=["detections"],  # [B, 4+num_classes, num_anchors]
            do_constant_folding=True,
            dynamic_axes={
                "clips":      {0: "batch_size"},
                "detections": {0: "batch_size"},
            },
            dynamo=False,
        )

    file_size = os.path.getsize(output) / 1024 / 1024
    print(f"      Export done! File size: {file_size:.1f} MB")

    # ── 5. 验证 ONNX 模型 ────────────────────────────────────────────────────
    print(f"\n[4/4] Verifying ONNX model ...")
    try:
        import onnx
        model_onnx = onnx.load(output)
        onnx.checker.check_model(model_onnx)
        print("      ONNX model check passed!")
    except ImportError:
        print("      [SKIP] onnx package not installed, run: pip install onnx")

    print(f"\nDone! ONNX model saved to: {os.path.abspath(output)}")
    print("\nONNX 输入/输出说明:")
    print(f"  Input  'clips'      : [batch, 3, {clip_length}, {img_size}, {img_size}]  float32")
    print(f"  Output 'detections' : [batch, {4 + config['num_classes']}, num_anchors]  float32")
    print(f"  输出格式: 前4列为 [cx, cy, w, h]（像素坐标），后{config['num_classes']}列为各类别置信度(sigmoid后)")


if __name__ == "__main__":
    args = parse_args()
    export(
        weights=args.weights,
        config_path=args.config,
        output=args.output,
        opset=args.opset,
    )
