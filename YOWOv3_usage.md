# YOWOv3 完整使用文档

> 论文：[YOWOv3: An Efficient and Generalized Framework for Spatiotemporal Action Detection](https://arxiv.org/abs/2408.02623)  
> 代码仓库：https://github.com/hope1337/YOWOv3  
> 权重仓库：https://huggingface.co/manh6054/YOWOv3  
> 适用平台：AutoDL（GPU云服务器），数据集：UCF101-24

---

## 目录

1. [项目概述](#1-项目概述)  
2. [环境准备](#2-环境准备)  
3. [数据集准备](#3-数据集准备)  
4. [权重下载](#4-权重下载)  
5. [配置文件说明](#5-配置文件说明)  
6. [训练](#6-训练)  
7. [评估（Evaluation）](#7-评估evaluation)  
8. [检测可视化（Detect）](#8-检测可视化detect)  
9. [实时摄像头检测（Live）](#9-实时摄像头检测live)  
10. [导出 ONNX](#10-导出-onnx)  
11. [常见问题（FAQ）](#11-常见问题faq)  
12. [权重命名规则速查](#12-权重命名规则速查)

---

## 1. 项目概述

YOWOv3（You Only Watch Once v3）是一种高效的时空动作检测框架，在 UCF101-24 和 AVAv2.2 数据集上达到了先进水平。

**核心架构**：
- **2D 骨干**：YOLOv8（提取空间特征，处理当前帧）
- **3D 骨干**：ShuffleNetv2 / ResNet / ResNeXt / I3D（提取时序特征，处理视频片段）
- **融合模块**：CFAM（跨尺度特征聚合）/ CBAM / SE / Channel 等
- **检测头**：解耦头（Decoupled Head）
- **损失函数**：TAL（Task-Aligned Loss）或 SIMOTA

**主要特点**：
- 支持多动作检测（每个边界框可对应多类动作）
- 输入为 16 帧视频片段（clip），输出动作边界框与类别概率
- 支持 UCF101-24（24类）、AVAv2.2、JHMDB 等多个数据集

---

## 2. 环境准备

### 2.1 前置要求

- 操作系统：Ubuntu 20.04 / 22.04
- GPU：NVIDIA GPU（推荐 >= 12GB 显存）
- CUDA：11.7 或 11.8（AutoDL 默认提供）
- Python：**3.8**（本文档全程使用 Python 3.8 虚拟环境）

### 2.2 创建 Python 3.8 虚拟环境

```bash
# 确认 Python 3.8 可用
python3.8 --version

# 方法一：使用 venv 创建虚拟环境（推荐）
python3.8 -m venv /root/yowov3_env

# 激活虚拟环境
source /root/yowov3_env/bin/activate

# 确认 Python 版本
python --version  # 应输出 Python 3.8.x
```

> **说明**：AutoDL 系统通常预装了 Python 3.8，若未安装请先运行：  
> `apt-get install python3.8 python3.8-venv python3.8-dev -y`

### 2.3 安装 PyTorch（CUDA 版本）

```bash
# 已激活虚拟环境情况下安装 PyTorch（CUDA 11.7）
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 或使用 CUDA 11.8（如果 AutoDL 实例为 CUDA 11.8）：
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
#     --extra-index-url https://download.pytorch.org/whl/cu118

# 验证 CUDA 是否可用
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

### 2.4 克隆仓库并安装其他依赖

```bash
# 切换到项目目录（AutoDL 推荐使用数据盘）
cd /root/autodl-tmp

# 克隆仓库
git clone https://github.com/hope1337/YOWOv3.git
cd YOWOv3

# 激活虚拟环境（若未激活）
source /root/yowov3_env/bin/activate

# 安装依赖（去掉 requirements.txt 中的 torch 相关行以避免覆盖）
pip install certifi charset-normalizer cycler filelock fonttools \
    fvcore h5py huggingface-hub iopath joblib kiwisolver \
    matplotlib numpy==1.21.6 onnx opencv-python==4.9.0.80 \
    packaging pandas Pillow PyYAML requests scikit-learn \
    scipy seaborn setuptools tabulate thop tqdm typer

# 或直接安装（注意 torch 行为空，会跳过）
grep -v "^torch" requirements.txt > requirements_no_torch.txt
pip install -r requirements_no_torch.txt
```

---

## 3. 数据集准备

### 3.1 UCF101-24 数据集

UCF101-24 是一个包含 **24 个动作类别**的视频动作检测数据集。

**数据集目录结构**（下载后放置）：

```
/root/autodl-tmp/ucf24/
├── rgb-images/              # 视频帧图像
│   ├── Basketball/
│   │   ├── v_Basketball_g01_c01/
│   │   │   ├── 00001.jpg
│   │   │   ├── 00002.jpg
│   │   │   └── ...
│   │   └── ...
│   └── ...（共24个动作类别文件夹）
├── labels/                  # 标注文件
│   ├── Basketball/
│   │   ├── v_Basketball_g01_c01/
│   │   │   ├── 00001.txt
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── trainlist.txt            # 训练集列表
└── testlist.txt             # 测试集列表
```

> **已有数据集**：假设已下载至 `/root/autodl-tmp/ucf24`，配置文件中 `data_root` 需指定到该路径。

### 3.2 24 个动作类别

| ID | 类别名称 | ID | 类别名称 |
|----|----------|----|----------|
| 0 | Basketball | 12 | PoleVault |
| 1 | BasketballDunk | 13 | RopeClimbing |
| 2 | Biking | 14 | SalsaSpin |
| 3 | CliffDiving | 15 | SkateBoarding |
| 4 | CricketBowling | 16 | Skiing |
| 5 | Diving | 17 | Skijet |
| 6 | Fencing | 18 | Soccer Juggling |
| 7 | FloorGymnastics | 19 | Surfing |
| 8 | GolfSwing | 20 | TennisSwing |
| 9 | HorseRiding | 21 | TrampolineJumping |
| 10 | IceDancing | 22 | VolleyballSpiking |
| 11 | LongJump | 23 | WalkingWithDog |

---

## 4. 权重下载

### 4.1 需要下载的权重文件

使用 YOWOv3 需要以下几类权重：

| 类型 | 用途 | HuggingFace 路径 |
|------|------|-----------------|
| 2D骨干预训练权重 | 初始化 YOLOv8 骨干 | `weights/backbone2D/YOLOv8/v8_n.pth` |
| 3D骨干预训练权重（C23用） | 初始化 ShuffleNetv2 | `weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth` |
| 3D骨干预训练权重（C27用） | 初始化 ResNet-101 | `weights/backbone3D/resnet/kinetics_resnet_101_RGB_16_best.pth` |
| 3D骨干预训练权重（C29用） | 初始化 ResNeXt-101 | `weights/backbone3D/resnext/resnext-101-kinetics.pth` |
| 3D骨干预训练权重（C30用） | 初始化 I3D | `weights/backbone3D/I3D/rgb_imagenet.pth` |
| 模型权重 C23 | 评估/推理用（最轻量） | `checkpoint/ucf24/C23/ema_epoch_7.pth` |
| 模型权重 C27 | 评估/推理用 | `checkpoint/ucf24/C27/ema_epoch_7.pth` |
| 模型权重 C29 | 评估/推理用 | `checkpoint/ucf24/C29/ema_epoch_7.pth` |
| 模型权重 C30 | 评估/推理用（最强） | `checkpoint/ucf24/C30/ema_epoch_7.pth` |

### 4.2 下载方法（Python 脚本）

项目内置了下载脚本（在 `autodl_setup.sh` 中也包含），手动下载方法：

```bash
# 进入 YOWOv3 项目目录
cd /root/autodl-tmp/YOWOv3

# 使用 wget 逐个下载（推荐）
BASE_URL="https://huggingface.co/manh6054/YOWOv3/resolve/main"

# 下载 backbone 预训练权重
mkdir -p weights/backbone2D/YOLOv8
mkdir -p weights/backbone3D/shufflenetv2
mkdir -p weights/backbone3D/resnet
mkdir -p weights/backbone3D/resnext
mkdir -p weights/backbone3D/I3D

wget -c "$BASE_URL/weights/backbone2D/YOLOv8/v8_n.pth" -O weights/backbone2D/YOLOv8/v8_n.pth
wget -c "$BASE_URL/weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth" \
    -O weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth
wget -c "$BASE_URL/weights/backbone3D/resnet/kinetics_resnet_101_RGB_16_best.pth" \
    -O weights/backbone3D/resnet/kinetics_resnet_101_RGB_16_best.pth
wget -c "$BASE_URL/weights/backbone3D/resnext/resnext-101-kinetics.pth" \
    -O weights/backbone3D/resnext/resnext-101-kinetics.pth
wget -c "$BASE_URL/weights/backbone3D/I3D/rgb_imagenet.pth" \
    -O weights/backbone3D/I3D/rgb_imagenet.pth

# 下载 C 系列模型权重
mkdir -p weights/checkpoint/C23 weights/checkpoint/C27 weights/checkpoint/C29 weights/checkpoint/C30

for model in C23 C27 C29 C30; do
    wget -c "$BASE_URL/checkpoint/ucf24/$model/config.yaml" -O weights/checkpoint/$model/config.yaml
    wget -c "$BASE_URL/checkpoint/ucf24/$model/ema_epoch_7.pth" -O weights/checkpoint/$model/ema_epoch_7.pth
done
```

### 4.3 使用 huggingface_hub 批量下载（可选）

```python
# download_weights.py
from huggingface_hub import hf_hub_download
import os

REPO_ID = "manh6054/YOWOv3"

files_to_download = [
    # backbone 权重
    ("weights/backbone2D/YOLOv8/v8_n.pth",       "weights/backbone2D/YOLOv8/v8_n.pth"),
    ("weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth",
     "weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth"),
    ("weights/backbone3D/resnet/kinetics_resnet_101_RGB_16_best.pth",
     "weights/backbone3D/resnet/kinetics_resnet_101_RGB_16_best.pth"),
    ("weights/backbone3D/resnext/resnext-101-kinetics.pth",
     "weights/backbone3D/resnext/resnext-101-kinetics.pth"),
    ("weights/backbone3D/I3D/rgb_imagenet.pth",
     "weights/backbone3D/I3D/rgb_imagenet.pth"),
    # 模型权重
    ("checkpoint/ucf24/C23/ema_epoch_7.pth", "weights/checkpoint/C23/ema_epoch_7.pth"),
    ("checkpoint/ucf24/C27/ema_epoch_7.pth", "weights/checkpoint/C27/ema_epoch_7.pth"),
    ("checkpoint/ucf24/C29/ema_epoch_7.pth", "weights/checkpoint/C29/ema_epoch_7.pth"),
    ("checkpoint/ucf24/C30/ema_epoch_7.pth", "weights/checkpoint/C30/ema_epoch_7.pth"),
    ("checkpoint/ucf24/C23/config.yaml", "weights/checkpoint/C23/config.yaml"),
    ("checkpoint/ucf24/C27/config.yaml", "weights/checkpoint/C27/config.yaml"),
    ("checkpoint/ucf24/C29/config.yaml", "weights/checkpoint/C29/config.yaml"),
    ("checkpoint/ucf24/C30/config.yaml", "weights/checkpoint/C30/config.yaml"),
]

for hf_path, local_path in files_to_download:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"Downloading {hf_path} ...")
    path = hf_hub_download(repo_id=REPO_ID, filename=hf_path, local_dir=".")
    print(f"  -> saved to {path}")

print("All weights downloaded!")
```

---

## 5. 配置文件说明

### 5.1 配置文件关键参数

以下是 UCF24 + C23 权重对应的配置示例（保存为 `config/ucf24_C23_eval.yaml`）：

```yaml
# ===== 重要：使用 C23 权重进行评估的配置 =====
config_path       : config/ucf24_C23_eval.yaml
dataset           : ucf
loss              : tal
active_checker    : True
num_classes       : 24

# 骨干网络（必须与训练时一致）
backbone2D        : yolov8         # 2D 骨干：YOLOv8
backbone3D        : shufflenetv2   # 3D 骨干：ShuffleNetv2（C23专用）

# 融合模块
fusion_module     : CFAM           # 融合模块：跨尺度特征聚合
mode              : decoupled      # 检测头：解耦头模式
interchannels     : [64, 64, 64]   # 通道数（C23专用：nano小网络对应64）

# 权重路径（与pretrain_path不同，此为已训练好的完整模型）
pretrain_path     : weights/checkpoint/C23/ema_epoch_7.pth

# 数据路径（修改为你的实际路径）
data_root         : /root/autodl-tmp/ucf24

# 输入设置
img_size          : 224            # 图像分辨率（训练时为224，不可更改）
clip_length       : 16             # 输入片段帧数
sampling_rate     : 1              # 帧采样率（每隔1帧取1帧）

# 训练超参数（评估时不使用，但需保留）
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

# 类别映射
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
```

> **警告**：`backbone3D / backbone2D / fusion_module / interchannels / img_size` 
> 等参数必须与训练时的 `config.yaml` 完全一致，否则模型无法正确加载！  
> 每个权重文件夹中已包含对应的 `config.yaml`，强烈建议直接使用。

### 5.2 各 C 系列权重的关键差异配置

| 参数 | C23 | C27 | C29 | C30 |
|------|-----|-----|-----|-----|
| `backbone3D` | shufflenetv2 | resnet | resnext101 | i3d |
| `interchannels` | [64,64,64] | [256,256,256] | [256,256,256] | [256,256,256] |
| `pretrain_path` | `weights/checkpoint/C23/ema_epoch_7.pth` | C27/... | C29/... | C30/... |

---

## 6. 训练

### 6.1 从零开始训练（UCF24）

```bash
# 激活虚拟环境
source /root/yowov3_env/bin/activate
cd /root/autodl-tmp/YOWOv3

# 修改配置文件中的 pretrain_path 为 null（从零开始）
# 并修改 data_root 为实际路径

python main.py --mode train --config config/cf2/ucf_config.yaml
```

### 6.2 基于预训练 backbone 进行训练（推荐）

编辑配置文件，确保骨干预训练权重路径正确：

```yaml
# config 中 backbone 预训练路径会自动从权重文件夹读取
pretrain_path : null   # 不加载完整模型，仅使用骨干预训练
```

然后运行：

```bash
python main.py -m train -cf config/cf2/ucf_config.yaml
```

### 6.3 断点续训（Fine-tune）

```bash
# 修改配置文件中的 pretrain_path 为上次保存的 checkpoint
# pretrain_path : weights/model_checkpoint/epoch_3.pth

python main.py -m train -cf config/cf2/ucf_config.yaml
```

### 6.4 训练输出

训练过程中，每轮结束后会在 `save_folder`（默认 `weights/model_checkpoint/`）保存：
- `epoch_N.pth` - 第 N 轮普通检查点
- `ema_epoch_N.pth` - 第 N 轮 EMA 检查点（**推荐用于评估**）
- `config.yaml` - 训练时的配置文件副本

---

## 7. 评估（Evaluation）

### 7.1 使用 C23 权重评估（推荐起点：最快的配置）

```bash
source /root/yowov3_env/bin/activate
cd /root/autodl-tmp/YOWOv3

# 方法一：直接使用下载的 C23 config.yaml（修改 data_root 和 pretrain_path）
# 先复制 C23 的配置文件
cp weights/checkpoint/C23/config.yaml config/ucf24_C23_eval.yaml

# 修改 config/ucf24_C23_eval.yaml 中的两个路径：
#   pretrain_path: weights/checkpoint/C23/ema_epoch_7.pth
#   data_root:     /root/autodl-tmp/ucf24

# 运行评估
python main.py --mode eval --config config/ucf24_C23_eval.yaml
```

或使用快捷命令（`-m` / `-cf`）：

```bash
python main.py -m eval -cf config/ucf24_C23_eval.yaml
```

### 7.2 评估命令参数说明

```
python main.py -m [mode] -cf [config_file_path]

mode 选项:
  train   - 训练模式
  eval    - 评估模式（计算 mAP@0.5）
  detect  - 可视化检测模式（在数据集上画框）
  live    - 实时摄像头检测
  onnx    - 导出 ONNX 格式
```

### 7.3 评估指标输出

评估完成后会输出：

```
      precision      recall        mAP
       0.XXXXX      0.XXXXX     0.XXXXX
```

- **precision**：精确率（Precision）
- **recall**：召回率（Recall）
- **mAP**：mAP@0.5（IoU=0.5时的平均精度均值）

### 7.4 四个 C 系列模型的评估对比

| 模型 | 3D骨干 | 速度 | 预期精度 |
|------|--------|------|----------|
| **C23** | ShuffleNetv2 | ★★★★★ 最快 | 较低 |
| **C27** | ResNet-101 | ★★★ 中等 | 中等 |
| **C29** | ResNeXt-101 | ★★ 较慢 | 较高 |
| **C30** | I3D | ★ 最慢 | 最高（C系列中） |

---

## 8. 检测可视化（Detect）

在测试集视频上绘制预测框：

```bash
python main.py -m detect -cf config/ucf24_C23_eval.yaml
```

检测输出（可视化图像/视频）通常保存在项目根目录。

---

## 9. 实时摄像头检测（Live）

（AutoDL 服务器通常无摄像头，仅供本地使用参考）

```bash
python main.py -m live -cf config/ucf24_C23_eval.yaml
```

---

## 10. 导出 ONNX

将 PyTorch 模型导出为 ONNX 格式以便部署：

```bash
python main.py -m onnx -cf config/ucf24_C23_eval.yaml
```

---

## 11. 常见问题（FAQ）

### Q1：评估时出现 `CUDA out of memory`

**解决**：减小评估时的 `batch_size`（在 config.yaml 中修改为 4 或 2）：
```yaml
batch_size : 4
```

### Q2：出现 `RuntimeError: Error(s) in loading state_dict`

**原因**：配置文件中的 `backbone3D`、`backbone2D`、`interchannels` 与权重文件不匹配。  
**解决**：使用权重文件夹中自带的 `config.yaml`，不要混用不同权重的配置。

### Q3：`data_root` 路径找不到数据集

**解决**：检查数据集路径，修改 config.yaml 中的 `data_root`：
```yaml
data_root : /root/autodl-tmp/ucf24
```
确保路径下存在 `rgb-images/` 和 `labels/` 文件夹。

### Q4：`pretrain_path` 和 backbone 预训练权重的区别

- **backbone 预训练权重**（如 `v8_n.pth`、`kinetics_shufflenetv2_2.0x.pth`）：  
  存放在 `weights/backbone2D/` 和 `weights/backbone3D/`，  
  在 config.yaml 的 `BACKBONE2D.YOLOv8.PRETRAIN` 等字段中指定路径，  
  用于从预训练骨干开始训练完整 YOWOv3 模型。  

- **`pretrain_path`**：  
  完整的 YOWOv3 模型权重（包含骨干+融合+检测头），  
  如 `weights/checkpoint/C23/ema_epoch_7.pth`，  
  用于加载已训练好的完整模型进行评估或 fine-tune。

### Q5：config 文件中缺少某些选项导致报错

**解决**：参考完整配置文件 `config/cf2/ucf_config.yaml`，将缺少的选项补充进来（如 `img_size`、`sampling_rate` 等可能在旧版本权重 config 中缺失）。

### Q6：下载 HuggingFace 权重速度慢

在 AutoDL 中可使用 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
或先下载到本地再上传至 AutoDL。

---

## 12. 权重命名规则速查

| 字母（列） | 2D骨干 | 损失函数 |
|-----------|--------|---------|
| C | YOLOv8-n（nano） | TAL |
| D | YOLOv8-s（small） | TAL |
| E | YOLOv8-m（medium） | TAL |
| F | YOLOv8-l（large） | TAL |
| G | YOLOv8-x（xlarge） | TAL |
| K | YOLOv8-n（nano） | SIMOTA |
| M | YOLOv8-m（medium） | SIMOTA |

| 数字（行） | 3D骨干 | interchannels |
|-----------|--------|---------------|
| 23 | ShuffleNetv2-2.0x | 自适应（64~256） |
| 27 | ResNet-101 | [256,256,256] |
| 29 | ResNeXt-101 | [256,256,256] |
| 30 | I3D | [256,256,256] |

> 详细说明见 `checkpoint_naming_guide.md`
