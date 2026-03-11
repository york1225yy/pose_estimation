#!/bin/bash
# =============================================================
# 驾驶舱活动识别 - 训练脚本（RTX 4090 单卡）
# 在 Video-Swin-Transformer/ 目录下执行：
#   bash tools/train_drive_activity.sh
# =============================================================
set -e

CONFIG="configs/recognition/swin/swin_base_drive_activity.py"
WORK_DIR="work_dirs/swin_base_drive_activity"
GPUS=${GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}
GPUS=${GPUS:-1}  # fallback to 1 if nvidia-smi unavailable

# 支持断点续训：若存在 latest.pth 则自动恢复
RESUME_CKPT=""
if [ -f "${WORK_DIR}/latest.pth" ]; then
    RESUME_CKPT="--resume-from ${WORK_DIR}/latest.pth"
    echo "[INFO] 检测到 checkpoint，从 ${WORK_DIR}/latest.pth 继续训练"
fi

mkdir -p "$WORK_DIR"

if [ "$GPUS" -eq 1 ]; then
    echo "[INFO] 单 GPU 训练..."
    python tools/train.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --seed 42 \
        --deterministic \
        --validate \
        $RESUME_CKPT \
        2>&1 | tee "${WORK_DIR}/train.log"
else
    echo "[INFO] ${GPUS} GPU 分布式训练..."
    PORT=${PORT:-29500}
    torchrun \
        --nproc_per_node="$GPUS" \
        --master_port="$PORT" \
        tools/train.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --seed 42 \
        --deterministic \
        --validate \
        --launcher pytorch \
        $RESUME_CKPT \
        2>&1 | tee "${WORK_DIR}/train.log"
fi
