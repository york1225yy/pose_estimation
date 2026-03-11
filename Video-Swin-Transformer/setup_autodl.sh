#!/bin/bash
# =============================================================
# 驾驶舱活动识别 - 一键安装与训练脚本（AutoDL RTX 4090）
# 使用前请将此脚本放在 Video-Swin-Transformer/ 目录下执行
# =============================================================
set -e

# ---------- 颜色输出 ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------- 1. 环境检查 ----------
info "检查 CUDA / Python 环境..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" || error "PyTorch 未安装"
python -c "import torch; assert torch.cuda.is_available(), 'No GPU'" || error "未检测到 GPU"

# ---------- 2. 安装依赖 ----------
info "安装依赖..."

# 先安装基础依赖（不含 mmcv）
pip install decord pandas scikit-learn tensorboard -q

# 检测 PyTorch 和 CUDA 版本以选择正确的预编译 mmcv-full
TORCH_VER=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))" 2>/dev/null)
CUDA_FULL=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
# 将 CUDA 版本格式转为 cuXXX，如 12.1 -> cu121，11.8 -> cu118
CUDA_TAG=$(python -c "
import torch, re
v = torch.version.cuda           # e.g. '12.1', '11.8'
major, minor = v.split('.')[:2]
# 只取次版本号第一位：12.1 -> cu121, 11.8 -> cu118
print('cu' + major + minor[0])
" 2>/dev/null)

info "检测到环境: PyTorch=${TORCH_VER}, CUDA=${CUDA_FULL} (tag=${CUDA_TAG})"

MMCV_INSTALLED=0

# 策略 1：从 OpenMMLab 官方预编译源安装（避免源码编译，解决 THC/THC.h 问题）
if [ -n "$CUDA_TAG" ] && [ -n "$TORCH_VER" ]; then
    MMCV_URL="https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TORCH_VER}.0/index.html"
    info "尝试从预编译源安装 mmcv-full: ${MMCV_URL}"
    pip install "mmcv-full>=1.3.9,<1.8.0" -f "$MMCV_URL" -q && MMCV_INSTALLED=1 || true
fi

# 策略 2：openmim 自动解析兼容版本
if [ "$MMCV_INSTALLED" -eq 0 ]; then
    warn "策略1失败，尝试通过 openmim 安装..."
    pip install openmim -q
    mim install "mmcv-full>=1.3.9,<1.8.0" -q && MMCV_INSTALLED=1 || true
fi

# 策略 3：强制降级 PyTorch 为 1.13 + cu117（最后手段，会重新安装 torch）
if [ "$MMCV_INSTALLED" -eq 0 ]; then
    warn "==================================================================="
    warn "自动安装失败。请手动选择以下方案之一："
    warn ""
    warn "方案 A（推荐）：在 AutoDL 上选择 PyTorch ≤ 2.0 + CUDA 11.8 镜像，"
    warn "  然后执行："
    warn "  pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html"
    warn ""
    warn "方案 B：使用 mim 安装："
    warn "  pip install openmim && mim install 'mmcv-full>=1.3.9,<1.8.0'"
    warn ""
    warn "方案 C（CUDA 12.x + PyTorch 2.1+）："
    warn "  pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"
    warn "  注：mmcv 2.x API 与本代码不完全兼容，需额外适配"
    warn "==================================================================="
    error "请手动安装 mmcv-full 后重新运行此脚本"
fi

info "mmcv-full 安装成功"

# 安装本项目
pip install -e . -q

# ---------- 3. 数据目录结构检查 ----------
info "检查数据目录结构..."
echo ""
echo "  期望的数据目录结构："
echo "  Video-Swin-Transformer/"
echo "  ├── data/"
echo "  │   ├── video/          <- 所有视频（按 vp1/, vp2/ 子目录存放 .mp4）"
echo "  │   │   ├── vp1/"
echo "  │   │   │   ├── run1b_2018-05-29-14-02-47.ids_1.mp4"
echo "  │   │   │   └── ..."
echo "  │   │   └── vp2/ ..."
echo "  │   ├── activity_label/"
echo "  │   │   └── midlevel.chunks_90.csv"
echo "  │   └── annotations/    <- 由脚本自动生成"
echo "  └── pretrained/"
echo "      └── swin_base_patch244_window877_kinetics400_1k.pth"
echo ""

if [ ! -d "data/video" ]; then
    warn "data/video 目录不存在，请先上传视频数据！"
    warn "示例: mkdir -p data/video/vp1 && cp /path/to/vp1/*.mp4 data/video/vp1/"
fi

if [ ! -f "data/activity_label/midlevel.chunks_90.csv" ]; then
    warn "data/activity_label/midlevel.chunks_90.csv 不存在，请上传标签文件！"
fi

# ---------- 4. 生成 train/val 标注 ----------
if [ -f "data/activity_label/midlevel.chunks_90.csv" ]; then
    info "生成 train/val CSV 标注..."
    mkdir -p data/annotations
    python tools/data/drive_activity/prepare_annotations.py \
        --csv  data/activity_label/midlevel.chunks_90.csv \
        --out  data/annotations \
        --val_ratio 0.2 \
        --seed 42
    info "Train/Val 标注已生成到 data/annotations/"
else
    warn "跳过标注生成（标签文件不存在）"
fi

# ---------- 5. 下载预训练权重 ----------
mkdir -p pretrained
CKPT="pretrained/swin_base_patch244_window877_kinetics400_1k.pth"
if [ ! -f "$CKPT" ]; then
    info "下载 Kinetics-400 预训练权重..."
    wget -q --show-progress \
        "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth" \
        -O "$CKPT" || warn "权重下载失败，将从头训练（建议手动下载后放到 pretrained/）"
else
    info "预训练权重已存在: $CKPT"
fi

echo ""
info "安装与准备完成！"
info "开始训练，请执行："
echo ""
echo "    bash tools/train_drive_activity.sh"
echo ""
