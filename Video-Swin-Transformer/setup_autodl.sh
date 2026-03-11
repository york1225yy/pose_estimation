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
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html -q || \
pip install mmcv-full -q
pip install decord pandas scikit-learn tensorboard -q
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
