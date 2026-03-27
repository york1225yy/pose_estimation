#!/usr/bin/env bash
# setup_env.sh — Video Swin Transformer 环境一键安装脚本
# 使用方法: bash setup_env.sh
set -e

echo "============================================================"
echo "  Video Swin Transformer 环境安装"
echo "============================================================"

# ── 检测 Python 版本 ─────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
PYTHON_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/5] Python 版本: $PYTHON_VER (路径: $PYTHON)"

# ── 升级 pip ─────────────────────────────────────────────────
echo "[2/5] 升级 pip..."
$PYTHON -m pip install --upgrade pip -q

# ── 检测 CUDA 并安装 PyTorch ─────────────────────────────────
echo "[3/5] 检测 CUDA 环境并安装 PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    echo "      检测到 NVIDIA GPU (驱动: $CUDA_VER)"
    echo "      安装 PyTorch GPU 版（CUDA 11.8）..."
    $PYTHON -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu118 -q
else
    echo "      未检测到 NVIDIA GPU，安装 PyTorch CPU 版..."
    $PYTHON -m pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu -q
fi

# ── 安装 mmcv-full ────────────────────────────────────────────
echo "[4/5] 安装 mmcv-full..."
TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__.split('+')[0])")
echo "      PyTorch 版本: $TORCH_VER"
$PYTHON -m pip install openmim -q
$PYTHON -m mim install "mmcv-full>=1.3.8,<1.6.0" -q || \
    $PYTHON -m pip install mmcv-full \
        -f https://download.openmmlab.com/mmcv/dist/cpu/torch${TORCH_VER}/index.html -q

# ── 安装项目依赖 ──────────────────────────────────────────────
echo "[5/5] 安装 Video-Swin-Transformer 项目依赖..."
cd Video-Swin-Transformer
$PYTHON -m pip install -e . -q
cd ..

$PYTHON -m pip install decord einops timm -q

echo ""
echo "============================================================"
echo "  安装完成！"
echo ""
echo "  快速测试:"
echo "    bash run_demo.sh"
echo "============================================================"
