#!/usr/bin/env bash
# run_demo.sh — 一键下载权重并测试 CPU/GPU 视频行为识别
# 使用方法: bash run_demo.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py"
CHECKPOINT="checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth"
LABEL="Video-Swin-Transformer/demo/label_map_k400.txt"
DEMO_VIDEO="demo_video.mp4"
CKPT_URL="https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth"

echo "============================================================"
echo "  Video Swin Transformer — 一键测试脚本"
echo "============================================================"
echo ""

# ── 步骤 1：下载权重文件 ──────────────────────────────────────
echo "[步骤 1/4] 检查权重文件..."
mkdir -p checkpoints
if [ -f "$CHECKPOINT" ]; then
    echo "  权重文件已存在，跳过下载: $CHECKPOINT"
else
    echo "  正在下载 Swin-T Kinetics-400 权重（约 110 MB）..."
    echo "  URL: $CKPT_URL"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$CHECKPOINT" "$CKPT_URL"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$CHECKPOINT" "$CKPT_URL"
    else
        echo "[错误] 请安装 wget 或 curl 后重试"
        exit 1
    fi
    echo "  权重下载完成: $CHECKPOINT"
fi
echo ""

# ── 步骤 2：准备测试视频 ──────────────────────────────────────
echo "[步骤 2/4] 准备测试视频..."
if [ -f "$DEMO_VIDEO" ]; then
    echo "  测试视频已存在: $DEMO_VIDEO"
elif [ -f "Video-Swin-Transformer/demo/demo.mp4" ]; then
    cp "Video-Swin-Transformer/demo/demo.mp4" "$DEMO_VIDEO"
    echo "  已复制 demo 视频: $DEMO_VIDEO"
else
    echo "  [警告] 未找到内置 demo 视频，尝试下载示例视频..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$DEMO_VIDEO" \
            "https://github.com/SwinTransformer/Video-Swin-Transformer/raw/master/demo/demo.mp4" 2>/dev/null || true
    fi
    if [ ! -f "$DEMO_VIDEO" ]; then
        echo "  [错误] 无法获取测试视频，请手动将视频文件命名为 demo_video.mp4 放入当前目录"
        exit 1
    fi
fi
echo ""

# ── 步骤 3：CPU 推理 ──────────────────────────────────────────
echo "[步骤 3/4] CPU 模式推理..."
echo "------------------------------------------------------------"
python predict_cpu.py \
    --video "$DEMO_VIDEO" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --label "$LABEL" \
    --top-k 5
echo ""

# ── 步骤 4：GPU 推理（若 GPU 可用） ──────────────────────────
echo "[步骤 4/4] GPU 模式推理..."
echo "------------------------------------------------------------"
GPU_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$GPU_AVAILABLE" = "True" ]; then
    python predict_gpu.py \
        --video "$DEMO_VIDEO" \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --label "$LABEL" \
        --device cuda:0 \
        --top-k 5
else
    echo "  [提示] 当前环境无可用 GPU，GPU 推理脚本将自动回退到 CPU 模式..."
    python predict_gpu.py \
        --video "$DEMO_VIDEO" \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --label "$LABEL" \
        --device cuda:0 \
        --top-k 5
fi

echo ""
echo "============================================================"
echo "  测试完成！"
echo "============================================================"
