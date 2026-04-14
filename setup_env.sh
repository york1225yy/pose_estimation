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

# ── 升级 pip 及基础工具 ───────────────────────────────────────
echo "[2/5] 升级 pip 及基础工具..."
# Python 3.11+ 某些发行版 pkg_resources 不内置，需同时在系统层和 pip 层安装
# apt 安装确保系统 Python 有 pkg_resources（用于 pip 隔离构建环境内）
if command -v apt-get &> /dev/null; then
    apt-get install -y python3-setuptools python3-pkg-resources -q 2>/dev/null || true
fi
$PYTHON -m pip install --upgrade pip "setuptools>=68.0" wheel -q

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

# ── 安装 mmcv ────────────────────────────────────────────────
echo "[4/5] 安装 mmcv..."
TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__.split('+')[0])")
TORCH_MAJOR=$($PYTHON -c "import torch; print(torch.__version__.split('.')[0])")
TORCH_MINOR=$($PYTHON -c "import torch; print(torch.__version__.split('.')[1])")
echo "      PyTorch 版本: $TORCH_VER（主版本: $TORCH_MAJOR）"

$PYTHON -m pip install openmim -q

if [ "$TORCH_MAJOR" -ge 2 ]; then
    # PyTorch 2.x：使用 mmcv 2.x（统一包，无需从源码编译 mmcv-full）
    echo "      检测到 PyTorch 2.x，安装 mmcv 2.x + mmengine..."
    $PYTHON -m pip install mmengine -q

    # ---------- 尝试顺序：预编译 wheel → openmim → pip 直接安装 ----------
    # 1) 尝试 OpenMMLab 官方预编译 wheel（cu118/torch2.x）
    CUDA_TAG="cu118"
    # 从 torch 2.0 到 2.4 有官方 wheel，2.5+ 可能没有，通配尝试
    INSTALL_OK=0
    for TRY_TORCH in "${TORCH_MAJOR}.${TORCH_MINOR}" "2.4" "2.3" "2.2" "2.1" "2.0"; do
        WHEEL_URL="https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TRY_TORCH}/index.html"
        echo "      尝试预编译 wheel (torch${TRY_TORCH})..."
        if $PYTHON -m pip install "mmcv>=2.0.0" -f "${WHEEL_URL}" -q 2>/dev/null; then
            INSTALL_OK=1
            echo "      ✓ 从 wheel 安装成功（torch${TRY_TORCH} 兼容）"
            break
        fi
    done

    # 2) 若预编译 wheel 均失败，尝试 openmim
    if [ "$INSTALL_OK" -eq 0 ]; then
        echo "      预编译 wheel 未找到，尝试 mim install..."
        if $PYTHON -m mim install "mmcv>=2.0.0" -q 2>/dev/null; then
            INSTALL_OK=1
            echo "      ✓ mim install 成功"
        fi
    fi

    # 3) 最终回退：MMCV_WITH_OPS=0 + --no-build-isolation 从源码编译
    # MMCV_WITH_OPS=0：跳过 CUDA 自定义算子编译（Video Swin 推理不需要它）
    # --no-build-isolation：禁止 pip 创建隔离构建环境，直接使用当前环境的
    #   setuptools/pkg_resources，彻底解决 Python 3.11 pkg_resources 缺失问题
    if [ "$INSTALL_OK" -eq 0 ]; then
        echo "      回退：MMCV_WITH_OPS=0 源码编译（跳过 CUDA 算子，请稍候）..."
        MMCV_WITH_OPS=0 $PYTHON -m pip install "mmcv>=2.0.0,<2.2.0" \
            --no-build-isolation -q
    fi
else
    # PyTorch 1.x：使用旧版 mmcv-full 1.x
    echo "      检测到 PyTorch 1.x，安装 mmcv-full 1.x..."
    $PYTHON -m mim install "mmcv-full>=1.3.8,<1.6.0" -q || \
        $PYTHON -m pip install mmcv-full \
            -f https://download.openmmlab.com/mmcv/dist/cpu/torch${TORCH_VER}/index.html -q
fi

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
