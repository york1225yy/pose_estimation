#!/bin/bash
# ============================================================
#  Video Swin Transformer 行为识别推理脚本
#  适用环境：AutoDL GPU / CUDA 11.8 / Python 3.8 / PyTorch 2.0 / mmcv 2.2.0
#  作者：自动生成
# ============================================================

set -e  # 遇错即停

# ─────────────────────────────────────────────────────────────
#  0. 目录设置（修改 MMACTION2_DIR 至你的实际路径）
# ─────────────────────────────────────────────────────────────
MMACTION2_DIR="$(cd "$(dirname "$0")/mmaction2" 2>/dev/null && pwd || echo "/root/pose_estimation/mmaction2")"
CKPT_DIR="${MMACTION2_DIR}/checkpoints"
OUTPUT_DIR="${MMACTION2_DIR}/outputs"
CONFIG_DIR="${MMACTION2_DIR}/configs/recognition/swin"
LABEL_FILE="${MMACTION2_DIR}/tools/data/kinetics/label_map_k400.txt"

# ─────────────────────────────────────────────────────────────
#  1. 选择推理的 Swin 变体（tiny / small / base）
#     tiny  ：最快，显存最省（~3GB），Top-1 78.9%
#     small ：平衡，（~4GB），Top-1 80.5%
#     base  ：精度高，（~6GB），Top-1 80.6%
# ─────────────────────────────────────────────────────────────
MODEL_VARIANT="tiny"   # 可改为 small / base

case "$MODEL_VARIANT" in
  tiny)
    CONFIG="${CONFIG_DIR}/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"
    CKPT_URL="https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth"
    CKPT_FILE="${CKPT_DIR}/swin_tiny_kinetics400.pth"
    ;;
  small)
    CONFIG="${CONFIG_DIR}/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"
    CKPT_URL="https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-e91ab986.pth"
    CKPT_FILE="${CKPT_DIR}/swin_small_kinetics400.pth"
    ;;
  base)
    CONFIG="${CONFIG_DIR}/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py"
    CKPT_URL="https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-182ec6cc.pth"
    CKPT_FILE="${CKPT_DIR}/swin_base_kinetics400.pth"
    ;;
  *)
    echo "[ERROR] MODEL_VARIANT 只能是 tiny / small / base"
    exit 1
    ;;
esac

# ─────────────────────────────────────────────────────────────
#  2. 指定推理视频（默认使用 repo 自带 demo.mp4）
#     如需推理自己的视频，改为：INPUT_VIDEO="/path/to/your_video.mp4"
# ─────────────────────────────────────────────────────────────
INPUT_VIDEO="${MMACTION2_DIR}/demo/demo.mp4"
OUTPUT_VIDEO="${OUTPUT_DIR}/result_${MODEL_VARIANT}.mp4"

# ─────────────────────────────────────────────────────────────
#  3. 环境检查
# ─────────────────────────────────────────────────────────────
echo "=========================================="
echo "  Video Swin Transformer 行为识别推理"
echo "=========================================="
echo "[INFO] MMAction2 路径 : ${MMACTION2_DIR}"
echo "[INFO] 模型变体       : Swin-${MODEL_VARIANT^}"
echo "[INFO] 输入视频       : ${INPUT_VIDEO}"
echo "[INFO] 输出文件       : ${OUTPUT_VIDEO}"
echo ""

# 检查 Python 和关键依赖
echo "[STEP 1] 检查 Python 环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}')"
python -c "import mmengine; print(f'MMEngine: {mmengine.__version__}')"
python -c "import mmaction; print(f'MMAction2: {mmaction.__version__}')" 2>/dev/null || {
    echo "[WARN] mmaction2 未安装，尝试安装..."
    cd "${MMACTION2_DIR}" && pip install -v -e . -q
}

# ─────────────────────────────────────────────────────────────
#  4. 安装必要的视频依赖
# ─────────────────────────────────────────────────────────────
echo ""
echo "[STEP 2] 确认视频解码依赖..."
python -c "import decord" 2>/dev/null || pip install decord -q && echo "decord OK"
python -c "import moviepy" 2>/dev/null || pip install moviepy -q && echo "moviepy OK"

# ─────────────────────────────────────────────────────────────
#  5. 下载模型权重（已存在则跳过）
# ─────────────────────────────────────────────────────────────
echo ""
echo "[STEP 3] 下载模型权重..."
mkdir -p "${CKPT_DIR}"

if [ -f "${CKPT_FILE}" ]; then
    echo "[INFO] 权重已存在，跳过下载：${CKPT_FILE}"
else
    echo "[INFO] 下载 Swin-${MODEL_VARIANT^} 权重..."
    echo "[INFO] URL: ${CKPT_URL}"
    wget --show-progress -O "${CKPT_FILE}" "${CKPT_URL}"
    echo "[OK] 权重下载完成"
fi

# ─────────────────────────────────────────────────────────────
#  6. 检查输入视频
# ─────────────────────────────────────────────────────────────
echo ""
echo "[STEP 4] 检查输入视频..."
if [ ! -f "${INPUT_VIDEO}" ]; then
    echo "[ERROR] 视频文件不存在: ${INPUT_VIDEO}"
    echo "        请将视频放置到此路径，或修改脚本中 INPUT_VIDEO 变量"
    exit 1
fi
echo "[OK] 视频文件: ${INPUT_VIDEO}"

# ─────────────────────────────────────────────────────────────
#  7. 执行推理
# ─────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "[STEP 5] 开始推理（Swin-${MODEL_VARIANT^}，Kinetics-400）..."
echo "─────────────────────────────────────────────────────"

cd "${MMACTION2_DIR}"

python demo/demo.py \
    "${CONFIG}" \
    "${CKPT_FILE}" \
    "${INPUT_VIDEO}" \
    "${LABEL_FILE}" \
    --device cuda:0 \
    --out-filename "${OUTPUT_VIDEO}" \
    --cfg-options \
        model.backbone.pretrained=None

# ─────────────────────────────────────────────────────────────
#  8. 完成提示
# ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  推理完成！"
echo "  输出视频（含标签叠加）：${OUTPUT_VIDEO}"
echo "=========================================="
