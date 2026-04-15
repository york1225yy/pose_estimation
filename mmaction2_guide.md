# MMAction2 完整项目说明文档

> 基于 [open-mmlab/mmaction2](https://github.com/open-mmlab/mmaction2) v1.2.0  
> 本文档涵盖项目结构、文件介绍、安装、训练、推理完整使用指南

---

## 目录

- [项目简介](#项目简介)
- [项目文件结构与说明](#项目文件结构与说明)
- [安装环境](#安装环境)
- [支持的任务与模型](#支持的任务与模型)
- [支持的数据集](#支持的数据集)
- [配置文件系统详解](#配置文件系统详解)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型测试评估](#模型测试评估)
- [推理 Demo](#推理-demo)
- [Video Swin Transformer 使用说明](#video-swin-transformer-使用说明)
- [常见问题](#常见问题)

---

## 项目简介

MMAction2 是 OpenMMLab 基于 PyTorch 的开源视频理解工具箱，支持以下五大任务：

| 任务 | 说明 |
|------|------|
| **行为识别** (Action Recognition) | 对视频片段分类，判断正在发生什么动作 |
| **时序动作检测** (Temporal Localization) | 检测动作在视频时间轴上的起止区间 |
| **时空动作检测** (Spatio-Temporal Detection) | 定位视频中每个人的位置并识别其动作 |
| **骨骼点行为识别** (Skeleton-based Recognition) | 基于人体关键点序列识别行为 |
| **视频检索** (Video Retrieval) | 文本-视频跨模态检索 |

---

## 项目文件结构与说明

```
mmaction2/
├── README.md                        # 英文主说明文档
├── README_zh-CN.md                  # 中文主说明文档
├── CITATION.cff                     # 论文引用信息
├── LICENSE                          # Apache 2.0 开源协议
├── MANIFEST.in                      # 打包清单
├── setup.py                         # pip 安装入口
├── setup.cfg                        # 安装配置（依赖、入口点）
├── requirements.txt                 # 基础依赖列表
├── requirements/                    # 细分依赖
│   ├── optional.txt                 # 可选依赖（moviepy、decord 等）
│   ├── runtime.txt                  # 运行时必须依赖
│   └── tests.txt                    # 测试依赖
│
├── mmaction/                        # 核心源码包
│   ├── __init__.py                  # 版本号定义
│   ├── registry.py                  # 全局注册器（模型、数据集等）
│   │
│   ├── apis/                        # 对外接口层
│   │   ├── inference.py             # init_recognizer / inference_recognizer 等高层 API
│   │   └── inferencers/             # 新式 Inferencer 封装（支持批量、可视化）
│   │
│   ├── models/                      # 模型定义
│   │   ├── recognizers/             # 完整行为识别器（Recognizer2D/3D）
│   │   ├── backbones/               # 骨干网（ResNet、Swin、TimeSformer、VideoMAE …）
│   │   ├── heads/                   # 分类头（TSNHead、I3DHead、SlowFastHead …）
│   │   ├── necks/                   # 颈部网络（TPN 等）
│   │   ├── losses/                  # 损失函数（CrossEntropy、Binary …）
│   │   ├── data_preprocessors/      # 输入归一化预处理
│   │   ├── localizers/              # 时序定位模型（BMN、BSN、TCANet）
│   │   ├── roi_heads/               # RoI 检测头（时空检测用）
│   │   ├── multimodal/              # 多模态模型（ActionCLIP、VindLU）
│   │   ├── similarity/              # 视频检索相关
│   │   ├── task_modules/            # 任务相关通用模块
│   │   └── utils/                   # 模型工具函数
│   │
│   ├── datasets/                    # 数据集与数据处理
│   │   ├── video_dataset.py         # 通用视频数据集（mp4/avi 等）
│   │   ├── rawframe_dataset.py      # 原始帧数据集（图片序列）
│   │   ├── ava_dataset.py           # AVA 时空检测数据集
│   │   ├── pose_dataset.py          # 骨骼点数据集
│   │   ├── audio_dataset.py         # 音频数据集
│   │   └── transforms/              # 数据增强流水线（采样、裁剪、翻转 …）
│   │
│   ├── evaluation/                  # 评估指标
│   │   ├── metrics/                 # AccMetric、AveragePrecision、RetrievalMetric …
│   │   └── functional/              # 指标计算工具函数
│   │
│   ├── engine/                      # 训练引擎扩展
│   │   ├── hooks/                   # 自定义 Hook（OutputHook 等）
│   │   ├── optimizers/              # 优化器（TSAM 等）
│   │   └── runner/                  # Runner 扩展
│   │
│   ├── structures/                  # 数据结构
│   │   └── action_data_sample.py    # 统一的数据样本容器
│   │
│   └── utils/                       # 工具函数（环境收集、日志等）
│
├── configs/                         # 所有模型配置文件（核心）
│   ├── _base_/                      # 基础配置（模型、调度、运行时、数据集模板）
│   │   ├── models/                  # 各骨干网基础模型配置
│   │   ├── schedules/               # 训练调度（cosine/step lr）
│   │   └── default_runtime.py       # 默认运行时（log、checkpoint 频率等）
│   │
│   ├── recognition/                 # 行为识别模型配置
│   │   ├── swin/                    # Video Swin Transformer ← 本次重点
│   │   ├── tsn/                     # TSN（2D CNN）
│   │   ├── tsm/                     # TSM
│   │   ├── i3d/                     # I3D
│   │   ├── slowfast/                # SlowFast
│   │   ├── slowonly/                # SlowOnly
│   │   ├── videomae/                # VideoMAE
│   │   ├── mvit/                    # MViT v2
│   │   ├── uniformer/               # UniFormer v1/v2
│   │   └── ...                      # 其他模型
│   │
│   ├── detection/                   # 时空动作检测配置
│   ├── localization/                # 时序动作定位配置
│   ├── skeleton/                    # 骨骼点识别配置
│   ├── multimodal/                  # 多模态配置
│   └── retrieval/                   # 视频检索配置
│
├── tools/                           # 训练/测试/工具脚本
│   ├── train.py                     # 单卡/多卡训练入口
│   ├── test.py                      # 测试评估入口
│   ├── dist_train.sh                # 多卡分布式训练脚本
│   ├── dist_test.sh                 # 多卡分布式测试脚本
│   ├── slurm_train.sh               # Slurm 集群训练
│   ├── slurm_test.sh                # Slurm 集群测试
│   ├── analysis_tools/              # 分析工具（混淆矩阵、FLOPs、日志分析）
│   ├── convert/                     # 权重转换脚本（从官方 repo 转换）
│   ├── data/                        # 数据集准备脚本
│   │   ├── kinetics/                # Kinetics 数据下载/处理
│   │   ├── ucf101/                  # UCF101 数据处理
│   │   ├── hmdb51/                  # HMDB51 数据处理
│   │   └── ...
│   ├── deployment/                  # 模型部署（ONNX、TensorRT）
│   ├── misc/                        # 其他杂项脚本
│   └── visualizations/             # 可视化工具
│
├── demo/                            # 推理演示脚本
│   ├── demo.py                      # 视频行为识别推理（主要使用）
│   ├── demo_inferencer.py           # 新式 Inferencer API 推理
│   ├── demo_skeleton.py             # 骨骼点行为识别推理
│   ├── demo_spatiotemporal_det.py   # 时空检测推理
│   ├── demo_audio.py                # 音频模态推理
│   ├── long_video_demo.py           # 长视频滑窗推理
│   ├── webcam_demo.py               # 摄像头实时推理
│   ├── demo.mp4                     # 示例视频（篮球 Kinetics-400）
│   ├── demo_skeleton.mp4            # 骨骼点示例视频
│   ├── demo_configs/                # Demo 专用配置文件
│   ├── fuse/                        # 多模态融合推理
│   └── README.md                    # Demo 使用说明
│
├── docs/                            # 文档源文件（ReadTheDocs）
│   ├── en/                          # 英文文档
│   └── zh_cn/                       # 中文文档
│
├── tests/                           # 单元测试
│   ├── test_models/                 # 模型测试
│   ├── test_datasets/               # 数据集测试
│   └── test_apis/                   # API 测试
│
├── projects/                        # 社区贡献项目
│   ├── actionclip/                  # ActionCLIP
│   ├── ctrgcn/                      # CTRGCN 骨骼点识别
│   └── msg3d/                       # MSG3D 骨骼点识别
│
├── docker/                          # Docker 环境
│   └── Dockerfile
│
└── resources/                       # README 图片资源
```

---

## 安装环境

### 依赖要求

| 软件 | 版本要求 |
|------|---------|
| Python | ≥ 3.7 |
| PyTorch | ≥ 1.8 |
| MMCV | ≥ 2.0.0 |
| MMEngine | ≥ 0.6.0 |

### 完整安装步骤

```bash
# 1. 创建 conda 环境（AutoDL上已有 Python 3.8 可跳过）
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# 2. 安装 PyTorch（AutoDL上已装，跳过）
# pip install torch==2.0.0+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 OpenMMLab 工具链
pip install -U openmim
mim install mmengine
mim install "mmcv==2.2.0"   # 与已安装版本对应

# 4. 可选依赖
mim install mmdet   # 时空检测需要
mim install mmpose  # 骨骼点识别需要

# 5. 安装 MMAction2
cd mmaction2
pip install -v -e .

# 6. 安装视频解码依赖
pip install decord moviepy
```

---

## 支持的任务与模型

### 行为识别 (Action Recognition)

| 模型 | 论文 | 特点 |
|------|------|------|
| TSN | ECCV'16 | 经典 2D CNN，稀疏采样 |
| I3D | CVPR'17 | 双流膨胀 3D 卷积 |
| SlowFast | ICCV'19 | 双路径快慢网络 |
| **Video Swin Transformer** | CVPR'22 | 基于窗口注意力的视频 Transformer ← 本文重点 |
| VideoMAE | NeurIPS'22 | 掩码自监督预训练 |
| MViT v2 | CVPR'22 | 多尺度视频 Transformer |
| UniFormer v2 | ArXiv'22 | 统一局部-全局时序建模 |

### 其他任务

| 任务 | 代表模型 |
|------|---------|
| 时序定位 | BSN, BMN, TCANet |
| 时空检测 | SlowOnly+FastRCNN, LFB |
| 骨骼点识别 | ST-GCN, PoseC3D, STGCN++ |
| 视频检索 | CLIP4Clip |

---

## 支持的数据集

| 数据集 | 类别数 | 规模 | 任务 |
|--------|-------|------|------|
| Kinetics-400 | 400 | ~240K 视频 | 行为识别 |
| Kinetics-600/700 | 600/700 | 更大 | 行为识别 |
| UCF101 | 101 | 13,320 | 行为识别 |
| HMDB51 | 51 | 6,766 | 行为识别 |
| Something-Something v1/v2 | 174 | ~220K | 时序行为识别 |
| AVA v2.1/v2.2 | 80 | — | 时空检测 |
| THUMOS14 | 20 | — | 时序定位 |
| NTU RGB+D | 60/120 | — | 骨骼点识别 |

---

## 配置文件系统详解

MMAction2 采用层级继承配置：

```python
# 示例：configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py
_base_ = [
    '../../_base_/models/swin_tiny.py',   # 模型结构
    '../../_base_/default_runtime.py'      # 运行时设置
]

# 可在此覆盖基础配置中的任何字段
model = dict(
    backbone=dict(pretrained='https://…/swin_tiny_patch4_window7_224.pth')
)
```

### 配置文件命名规则

```
{model}-{arch}_{pretrain}_{gpus}x{bsz}-{extra}-{epochs}x{step}x{clips}-{epochs}e_{dataset}-{modality}.py
```

示例：`swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py`
- `swin-tiny`：模型架构
- `p244-w877`：patch size 2×4×4，window size 8×7×7
- `in1k-pre`：ImageNet-1K 预训练
- `8xb8`：8 GPU × batch_size 8
- `amp`：使用混合精度
- `32x2x1`：clip_len=32，frame_interval=2，num_clips=1
- `30e`：训练 30 epoch
- `kinetics400-rgb`：Kinetics-400 数据集，RGB 模态

---

## 数据集准备

以 Kinetics-400 为例（视频形式）：

```bash
# 1. 下载标注文件
cd mmaction2/tools/data/kinetics
bash download_annotations.sh kinetics400

# 2. 下载视频（需要较长时间）
bash download_videos_infer.sh kinetics400

# 3. 生成文件列表
python build_file_list.py kinetics400 data/kinetics400/videos_train/ --level 2 --format videos --num-split 1 --subset train --shuffle
python build_file_list.py kinetics400 data/kinetics400/videos_val/ --level 2 --format videos --num-split 1 --subset val --shuffle
```

数据组织结构：
```
data/kinetics400/
├── videos_train/
│   ├── abseiling/
│   │   ├── 0d91kkzHWvQ.mp4
│   │   └── ...
│   └── ...
├── videos_val/
├── kinetics400_train_list_videos.txt
└── kinetics400_val_list_videos.txt
```

---

## 模型训练

### 单卡训练

```bash
python tools/train.py \
    configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    --work-dir work_dirs/swin_tiny_k400
```

### 多卡分布式训练

```bash
bash tools/dist_train.sh \
    configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    8 \
    --work-dir work_dirs/swin_tiny_k400
```

### 恢复训练

```bash
python tools/train.py \
    configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    --resume work_dirs/swin_tiny_k400/last_checkpoint
```

### 覆盖配置参数

```bash
python tools/train.py config.py \
    --cfg-options train_dataloader.batch_size=4 \
                  optim_wrapper.optimizer.lr=0.0001
```

---

## 模型测试评估

### 单卡测试

```bash
python tools/test.py \
    configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    checkpoints/swin-tiny_kinetics400.pth
```

### 多卡分布式测试

```bash
bash tools/dist_test.sh \
    configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    checkpoints/swin-tiny_kinetics400.pth \
    8
```

### 分析工具

```bash
# 绘制混淆矩阵
python tools/analysis_tools/confusion_matrix.py config.py result.json --show

# 计算模型 FLOPs
python tools/analysis_tools/get_flops.py config.py --shape 1 3 32 224 224

# 分析训练日志
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/*/20*/vis_data/scalars.json --keys loss --legend loss
```

---

## 推理 Demo

### 方式一：demo.py（经典接口）

```bash
python demo/demo.py \
    configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    checkpoints/swin-tiny_kinetics400.pth \
    demo/demo.mp4 \
    tools/data/kinetics/label_map_k400.txt \
    --device cuda:0 \
    --out-filename output.mp4
```

### 方式二：demo_inferencer.py（新式接口，支持批量）

```bash
python demo/demo_inferencer.py \
    --rec swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb \
    --rec-weights checkpoints/swin-tiny_kinetics400.pth \
    --input demo/demo.mp4 \
    --show
```

### 方式三：Python API

```python
from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py'
checkpoint_file = 'checkpoints/swin-tiny_kinetics400.pth'

model = init_recognizer(config_file, checkpoint_file, device='cuda:0')
result = inference_recognizer(model, 'demo/demo.mp4')

# result 是 ActionDataSample，scores 是各类概率
scores = result.pred_scores.item.tolist()
label_map = open('tools/data/kinetics/label_map_k400.txt').readlines()
labels = [l.strip() for l in label_map]

# 打印 Top-5
top5 = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:5]
for idx, score in top5:
    print(f'{labels[idx]}: {score:.4f}')
```

### 长视频推理（滑窗）

```bash
python demo/long_video_demo.py \
    configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
    checkpoints/swin-tiny_kinetics400.pth \
    long_video.mp4 \
    tools/data/kinetics/label_map_k400.txt \
    --threshold 0.4 \
    --stride 16
```

---

## Video Swin Transformer 使用说明

### 模型介绍

Video Swin Transformer（CVPR 2022）将 Swin Transformer 扩展到视频域：
- **局部注意力**：在 3D 时空窗口内计算注意力，降低计算复杂度
- **移位窗口**：通过窗口移位实现跨窗口信息交互
- **时空建模**：同时建模空间外观和时序动态

### 可用模型变体（Kinetics-400）

| 模型 | 预训练 | Top-1 | Top-5 | 参数量 |
|------|-------|-------|-------|-------|
| Swin-T | IN-1K | 78.90% | 93.77% | 28.2M |
| Swin-S | IN-1K | 80.54% | 94.46% | 49.8M |
| Swin-B | IN-1K | 80.57% | 94.49% | 88.0M |
| Swin-L | IN-22K | 83.1%+ | 95.9%+ | 197M |

### 模型权重下载地址

```
Swin-Tiny (推荐，最快):
https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-241016b2.pth

Swin-Small:
https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-small-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-e91ab986.pth

Swin-Base:
https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb/swin-base-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb_20220930-182ec6cc.pth
```

---

## 常见问题

### Q: RuntimeError: CUDA out of memory
减小 batch size 或使用更小的模型变体：
```bash
--cfg-options test_dataloader.batch_size=1
```

### Q: AssertionError: MMCV version 不匹配
确认 mmcv 版本：
```bash
python -c "import mmcv; print(mmcv.__version__)"
mim install "mmcv==2.2.0"
```

### Q: 视频解码失败（decord 报错）
```bash
pip install decord
# 或者使用 opencv 后端
--cfg-options test_pipeline.0.io_backend=cv2
```

### Q: 推理结果乱序（标签对不上）
确保 label 文件与训练数据集对应，Kinetics-400 使用：
```
tools/data/kinetics/label_map_k400.txt
```
