#!/bin/bash
# =============================================================================
# YOWOv3 AutoDL 一键运行脚本
# 适用于：AutoDL GPU 云服务器，UCF24 数据集
# 前提：
#   1. Python 3.8 及 PyTorch 环境已配置好（系统全局）
#   2. 已将 york1225yy/pose_estimation（branch: yowov3）推送至 GitHub
#      （内含 YOWOv3 代码 + 权重文件）
#   3. UCF24 数据集已下载到本机
# =============================================================================

set -e  # 遇到错误立即退出

# ==================== 用户配置区（请根据实际情况修改）====================
# UCF24 数据集路径
UCF24_DATA_ROOT="/root/autodl-tmp/ucf24"

# 项目拉取目录（AutoDL 推荐使用数据盘 /root/autodl-tmp）
REPO_DIR="/root/autodl-tmp/pose_estimation"

# GitHub 仓库地址与分支
GITHUB_REPO="https://github.com/york1225yy/pose_estimation.git"
GITHUB_BRANCH="yowov3"

# YOWOv3 代码目录（仓库内子目录）
PROJECT_DIR="${REPO_DIR}/YOWOv3"

# HuggingFace 镜像（如权重不在 GitHub 中、需从 HF 补充下载时使用，下载慢可取消注释）
export HF_ENDPOINT="https://hf-mirror.com"
# =========================================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() { echo -e "\n${BLUE}========== $1 ==========${NC}"; }

# =============================================================================
# 步骤 0：检查基础环境
# =============================================================================
log_section "步骤 0：检查基础环境"

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU 信息："
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    log_warn "未检测到 nvidia-smi，请确认 GPU 驱动已安装"
fi

# 检查 Python 3.8
if ! command -v python &> /dev/null; then
    log_error "未找到 python，请先配置 Python 3.8 环境后重新运行"
    exit 1
fi
log_info "Python 版本：$(python --version)"

# 检查 PyTorch 与 CUDA
python -c "
import torch
print(f'  PyTorch 版本: {torch.__version__}')
print(f'  CUDA 可用:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU 数量:     {torch.cuda.device_count()}')
    print(f'  GPU 名称:     {torch.cuda.get_device_name(0)}')
else:
    print('  [WARNING] CUDA 不可用，将使用 CPU 运行（速度极慢）')
" || { log_error "PyTorch 未安装或导入失败，请检查环境"; exit 1; }

# =============================================================================
# 步骤 1：从 GitHub 拉取项目（含代码 + 权重）
# =============================================================================
log_section "步骤 1：从 GitHub 拉取项目"

if [ -d "$REPO_DIR/.git" ]; then
    log_warn "仓库已存在，执行 git pull 更新..."
    cd "$REPO_DIR"
    git pull origin "$GITHUB_BRANCH"
else
    log_info "克隆仓库 ${GITHUB_REPO} (branch: ${GITHUB_BRANCH}) ..."
    log_warn "注意：若仓库包含大型权重文件（.pth），克隆时间可能较长"
    git clone -b "$GITHUB_BRANCH" "$GITHUB_REPO" "$REPO_DIR"
fi

# 确认 YOWOv3 子目录存在
if [ ! -f "${PROJECT_DIR}/main.py" ]; then
    log_error "未在 ${PROJECT_DIR} 找到 main.py，请确认 YOWOv3 代码已推送到仓库"
    exit 1
fi

cd "$PROJECT_DIR"
log_info "工作目录：$(pwd)"

# =============================================================================
# 步骤 2：安装 Python 依赖
# =============================================================================
log_section "步骤 2：安装 Python 依赖"

if [ -f "requirements.txt" ]; then
    log_info "安装 requirements.txt 中的依赖（跳过 torch 相关行）..."
    grep -v "^torch" requirements.txt > /tmp/requirements_no_torch.txt
    python -m pip install -r /tmp/requirements_no_torch.txt --quiet
    log_info "依赖安装完成"
else
    log_warn "未找到 requirements.txt，跳过依赖安装"
fi

# =============================================================================
# 步骤 3：验证权重文件（若不在仓库中则从 HuggingFace 补充下载）
# =============================================================================
log_section "步骤 3：验证权重文件"

BASE_HF_URL="https://huggingface.co/manh6054/YOWOv3/tree/main"

download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        local size
        size=$(du -sh "$dest" | cut -f1)
        log_info "  ✓ 已存在：$dest ($size)"
    else
        log_warn "  ✗ 缺失，从 HuggingFace 下载：$(basename "$dest") ..."
        mkdir -p "$(dirname "$dest")"
        wget -q --show-progress -c "$url" -O "$dest" || {
            log_error "wget 下载失败，尝试 curl ..."
            curl -L --progress-bar -o "$dest" "$url" || {
                log_error "下载失败：$url"
                log_warn "请手动下载并放置到 $dest"
            }
        }
    fi
}

log_info "检查骨干预训练权重..."
download_if_missing \
    "$BASE_HF_URL/weights/backbone2D/YOLOv8/v8_n.pth" \
    "weights/backbone2D/YOLOv8/v8_n.pth"
download_if_missing \
    "$BASE_HF_URL/weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth" \
    "weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth"
# download_if_missing \
#     "$BASE_HF_URL/weights/backbone3D/resnet/kinetics_resnet_101_RGB_16_best.pth" \
#     "weights/backbone3D/resnet/kinetics_resnet_101_RGB_16_best.pth"
# download_if_missing \
#     "$BASE_HF_URL/weights/backbone3D/resnext/resnext-101-kinetics.pth" \
#     "weights/backbone3D/resnext/resnext-101-kinetics.pth"
# download_if_missing \
#     "$BASE_HF_URL/weights/backbone3D/I3D/rgb_imagenet.pth" \
#     "weights/backbone3D/I3D/rgb_imagenet.pth"

log_info "检查 C23/C27/C29/C30 模型权重..."
for MODEL in C23; do
    download_if_missing \
        "$BASE_HF_URL/checkpoint/ucf24/$MODEL/config.yaml" \
        "weights/checkpoint/$MODEL/config.yaml"
    download_if_missing \
        "$BASE_HF_URL/checkpoint/ucf24/$MODEL/ema_epoch_7.pth" \
        "weights/checkpoint/$MODEL/ema_epoch_7.pth"
done

# =============================================================================
# 步骤 4：生成 UCF24 评估配置文件
# =============================================================================
log_section "步骤 4：生成各模型评估配置文件"

# 验证数据集路径
if [ ! -d "$UCF24_DATA_ROOT" ]; then
    log_error "UCF24 数据集路径不存在：$UCF24_DATA_ROOT"
    log_warn "请修改脚本顶部的 UCF24_DATA_ROOT 后重新运行"
fi

mkdir -p config

# ——— 生成 C23 配置文件 ———
cat > config/ucf24_C23_eval.yaml << YAML_EOF
BACKBONE2D:
  YOLOv8:
    ver : n

    PRETRAIN:
      n : weights/backbone2D/YOLOv8/v8_n.pth
      s : weights/backbone2D/YOLOv8/v8_s.pth
      m : weights/backbone2D/YOLOv8/v8_m.pth
      l : weights/backbone2D/YOLOv8/v8_l.pth
      x : weights/backbone2D/YOLOv8/v8_x.pth

BACKBONE3D:

  MOBILENET:
    width_mult: 2.0
    PRETRAIN:
      width_mult_0.5x  : weights/backbone3D/mobilenet/kinetics_mobilenet_0.5x_RGB_16_best.pth
      width_mult_1.0x  : weights/backbone3D/mobilenet/kinetics_mobilenet_1.0x_RGB_16_best.pth
      width_mult_1.5x  : weights/backbone3D/mobilenet/kinetics_mobilenet_1.5x_RGB_16_best.pth
      width_mult_2.0x  : weights/backbone3D/mobilenet/kinetics_mobilenet_2.0x_RGB_16_best.pth

  MOBILENETv2:
    width_mult: 1.0
    PRETRAIN:
      width_mult_0.2x  : weights/backbone3D/mobilenetv2/kinetics_mobilenetv2_0.2x_RGB_16_best.pth
      width_mult_0.45x : weights/backbone3D/mobilenetv2/kinetics_mobilenetv2_0.45x_RGB_16_best.pth
      width_mult_0.7x  : weights/backbone3D/mobilenetv2/kinetics_mobilenetv2_0.7x_RGB_16_best.pth
      width_mult_1.0x  : weights/backbone3D/mobilenetv2/kinetics_mobilenetv2_1.0x_RGB_16_best.pth

  SHUFFLENET:
    width_mult: 2.0
    PRETRAIN:
      width_mult_0.5x  : weights/backbone3D/shufflenet/kinetics_shufflenet_0.5x_G3_RGB_16_best.pth
      width_mult_1.0x  : weights/backbone3D/shufflenet/kinetics_shufflenet_1.0x_G3_RGB_16_best.pth
      width_mult_1.5x  : weights/backbone3D/shufflenet/kinetics_shufflenet_1.5x_G3_RGB_16_best.pth
      width_mult_2.0x  : weights/backbone3D/shufflenet/kinetics_shufflenet_2.0x_G3_RGB_16_best.pth

  SHUFFLENETv2:
    width_mult: 2.0
    PRETRAIN:
      width_mult_0.25x : weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_0.25x_RGB_16_best.pth
      width_mult_1.0x  : weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_1.0x_RGB_16_best.pth
      width_mult_1.5x  : weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_1.5x_RGB_16_best.pth
      width_mult_2.0x  : weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth

  I3D:
    PRETRAIN:
      default: weights/backbone3D/I3D/rgb_imagenet.pth

  RESNET:
    ver : 101
    PRETRAIN:
      ver_18 : weights/backbone3D/resnet/kinetics_resnet_18_RGB_16_best.pth
      ver_50 : weights/backbone3D/resnet/kinetics_resnet_50_RGB_16_best.pth
      ver_101: weights/backbone3D/resnet/kinetics_resnet_101_RGB_16_best.pth

  RESNEXT:
    ver : 101
    PRETRAIN:
      ver_101 : weights/backbone3D/resnext/resnext-101-kinetics.pth

LOSS:
  TAL:
    top_k  : 10
    alpha  : 0.5
    beta   : 6.0
    radius : 2.5
    soft_label : False
    scale_cls_loss : 0.5
    scale_box_loss : 7.5
    scale_dfl_loss : 1.5

  SIMOTA:
    top_k  : 10
    gamma  : 0.5
    radius : 2.5
    mode   : balance
    soft_label : True
    dynamic_k     : False
    dynamic_top_k : 40
    scale_cls_loss : 0.5
    scale_box_loss : 7.5
    scale_dfl_loss : 1.5

config_path       : config/ucf24_C23_eval.yaml
dataset           : ucf
loss              : tal
active_checker    : True
num_classes       : 24
backbone2D        : yolov8
backbone3D        : shufflenetv2
fusion_module     : CFAM
mode              : decoupled
interchannels     : [64, 64, 64]
pretrain_path     : weights/checkpoint/C23/ema_epoch_7.pth
data_root         : ${UCF24_DATA_ROOT}
img_size          : 224
clip_length       : 16
batch_size        : 8
num_workers       : 4
acc_grad          : 16
lr                : 0.0001
weight_decay      : 0.0005
max_step_warmup   : 1000
adjustlr_schedule : [1, 2, 3, 4, 5]
max_epoch         : 7
lr_decay          : 0.5
save_folder       : weights/model_checkpoint
sampling_rate     : 1

idx2name:
  0  : Baseketball
  1  : BaseketballDunk
  2  : Biking
  3  : CliffDiving
  4  : CricketBowling
  5  : Diving
  6  : Fencing
  7  : FloorGymnastics
  8  : GolfSwing
  9  : HorseRiding
  10 : IceDancing
  11 : LongJump
  12 : PoleVault
  13 : RopeClimbing
  14 : SalsaSpin
  15 : SkateBoarding
  16 : Skiing
  17 : Skijet
  18 : Soccer Juggling
  19 : Surfing
  20 : TennisSwing
  21 : TrampolineJumping
  22 : VolleyballSpiking
  23 : WalkingWithDog
YAML_EOF
log_info "C23 配置文件生成：config/ucf24_C23_eval.yaml"

# ——— 生成 C27 配置文件 ———
sed 's/backbone3D        : shufflenetv2/backbone3D        : resnet/' \
    config/ucf24_C23_eval.yaml | \
    sed 's/interchannels     : \[64, 64, 64\]/interchannels     : [256, 256, 256]/' | \
    sed 's/weights\/checkpoint\/C23\/ema_epoch_7.pth/weights\/checkpoint\/C27\/ema_epoch_7.pth/' | \
    sed 's/config_path.*ucf24_C23_eval/config_path       : config\/ucf24_C27_eval/' \
    > config/ucf24_C27_eval.yaml
log_info "C27 配置文件生成：config/ucf24_C27_eval.yaml"

# ——— 生成 C29 配置文件 ———
sed 's/backbone3D        : shufflenetv2/backbone3D        : resnext101/' \
    config/ucf24_C23_eval.yaml | \
    sed 's/interchannels     : \[64, 64, 64\]/interchannels     : [256, 256, 256]/' | \
    sed 's/weights\/checkpoint\/C23\/ema_epoch_7.pth/weights\/checkpoint\/C29\/ema_epoch_7.pth/' | \
    sed 's/config_path.*ucf24_C23_eval/config_path       : config\/ucf24_C29_eval/' \
    > config/ucf24_C29_eval.yaml
log_info "C29 配置文件生成：config/ucf24_C29_eval.yaml"

# ——— 生成 C30 配置文件 ———
sed 's/backbone3D        : shufflenetv2/backbone3D        : i3d/' \
    config/ucf24_C23_eval.yaml | \
    sed 's/interchannels     : \[64, 64, 64\]/interchannels     : [256, 256, 256]/' | \
    sed 's/weights\/checkpoint\/C23\/ema_epoch_7.pth/weights\/checkpoint\/C30\/ema_epoch_7.pth/' | \
    sed 's/config_path.*ucf24_C23_eval/config_path       : config\/ucf24_C30_eval/' \
    > config/ucf24_C30_eval.yaml
log_info "C30 配置文件生成：config/ucf24_C30_eval.yaml"

# =============================================================================
# 步骤 5：运行 C23 模型评估
# =============================================================================
log_section "步骤 5：运行 C23 模型评估（UCF24）"

log_info "使用 C23 权重（YOLOv8-n + ShuffleNetv2）在 UCF24 上进行评估..."
log_info "配置文件：config/ucf24_C23_eval.yaml"
log_info "评估权重：weights/checkpoint/C23/ema_epoch_7.pth"

python main.py --mode eval --config config/ucf24_C23_eval.yaml

log_info "C23 评估完成！"

# =============================================================================
# 步骤 6：（可选）运行可视化检测
# =============================================================================
log_section "步骤 6：（可选）运行可视化检测"

read -p "是否运行可视化检测（在数据集图像上绘制检测框）？[y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "运行 C23 检测可视化..."
    python main.py --mode detect --config config/ucf24_C23_eval.yaml
    log_info "检测可视化完成，结果保存在当前目录"
fi

# =============================================================================
# 完成
# =============================================================================
log_section "全部完成！"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}YOWOv3 评估已完成！${NC}"
echo ""
echo -e "项目目录：   ${BLUE}${PROJECT_DIR}${NC}"
echo -e "UCF24 数据： ${BLUE}${UCF24_DATA_ROOT}${NC}"
echo ""
echo -e "快速重新运行评估命令："
echo -e "  ${YELLOW}cd ${PROJECT_DIR}${NC}"
echo -e "  ${YELLOW}# C23 评估（最快，YOLOv8-n + ShuffleNetv2）${NC}"
echo -e "  ${YELLOW}python main.py -m eval -cf config/ucf24_C23_eval.yaml${NC}"
echo -e "  ${YELLOW}# C27 评估（YOLOv8-n + ResNet-101）${NC}"
echo -e "  ${YELLOW}python main.py -m eval -cf config/ucf24_C27_eval.yaml${NC}"
echo -e "  ${YELLOW}# C29 评估（YOLOv8-n + ResNeXt-101）${NC}"
echo -e "  ${YELLOW}python main.py -m eval -cf config/ucf24_C29_eval.yaml${NC}"
echo -e "  ${YELLOW}# C30 评估（最高精度，YOLOv8-n + I3D）${NC}"
echo -e "  ${YELLOW}python main.py -m eval -cf config/ucf24_C30_eval.yaml${NC}"
echo -e "${GREEN}============================================================${NC}"
