# Video Swin Transformer — 视频行为识别完整指南

> 基于 [Video Swin Transformer](https://arxiv.org/abs/2106.13230)（arXiv:2106.13230）的视频行为识别项目，支持 **CPU / GPU** 两种推理模式。

---

## 目录

1. [项目简介](#1-项目简介)
2. [架构原理](#2-架构原理)
3. [性能指标](#3-性能指标)
4. [环境安装](#4-环境安装)
5. [权重下载](#5-权重下载)
6. [CPU / GPU 推理代码说明](#6-cpu--gpu-推理代码说明)
7. [快速测试 Demo](#7-快速测试-demo)
8. [目录结构](#8-目录结构)
9. [常见问题](#9-常见问题)
10. [引用](#10-引用)

---

## 1. 项目简介

**Video Swin Transformer** 是微软研究院提出的视频理解骨干网络，将图像领域的 Swin Transformer 扩展至时序维度，通过**局部 3D 窗口注意力机制**实现高效的时空建模：

- 在 Kinetics-400 上达到 **84.9% top-1** 精度（预训练于 ImageNet-22K）
- 在 Kinetics-600 上达到 **86.1% top-1** 精度
- 在 Something-Something V2 上达到 **69.6% top-1** 精度
- 相比全局注意力方法，参数量减少 ~3×，预训练数据减少 ~20×

本仓库在原始代码基础上额外提供：
- 独立的 **CPU 推理脚本** `predict_cpu.py`
- 独立的 **GPU 推理脚本** `predict_gpu.py`
- 一键环境安装脚本 `setup_env.sh`
- 一键下载权重和测试脚本 `run_demo.sh`

---

## 2. 架构原理

```
输入视频
    │
    ▼
3D Patch Partition（时空分块嵌入）
    │
    ▼
Video Swin Transformer Block（×N）
    ├── Window Multi-Head Self-Attention (W-MSA)   ← 局部窗口 (P×H×W)
    └── Shifted Window MSA (SW-MSA)                ← 窗口间信息流动
    │
    ▼
Hierarchical Feature Maps（多尺度特征图）
    │
    ▼
Global Average Pooling + 分类头
    │
    ▼
行为类别预测（Kinetics-400 共 400 类）
```

**关键设计**：
- **3D 移位窗口注意力**：在 `(T/2, H/2, W/2)` 的局部窗口内计算自注意力，复杂度从 O(N²) 降至 O(N)
- **时序局部性归纳偏置**：相邻帧在同一窗口内，天然捕捉短程运动
- **预训练迁移**：骨干网络可从 ImageNet 预训练的 Swin Transformer 直接初始化

---

## 3. 性能指标

### Kinetics-400

| 模型 | 预训练 | Top-1 | Top-5 | 参数量 | FLOPs |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Swin-T | ImageNet-1K | 78.8% | 93.6% | 28M | 87.9G |
| Swin-S | ImageNet-1K | 80.6% | 94.5% | 50M | 165.9G |
| Swin-B | ImageNet-1K | 80.6% | 94.6% | 88M | 281.6G |
| Swin-B | ImageNet-22K | **82.7%** | **95.5%** | 88M | 281.6G |

### Kinetics-600

| 模型 | 预训练 | Top-1 | Top-5 |
|:---:|:---:|:---:|:---:|
| Swin-B | ImageNet-22K | **84.0%** | **96.5%** |

### Something-Something V2

| 模型 | 预训练 | Top-1 | Top-5 |
|:---:|:---:|:---:|:---:|
| Swin-B | Kinetics-400 | **69.6%** | **92.7%** |

---

## 4. 环境安装

### 4.1 系统要求

- Python 3.7+
- PyTorch 1.8+（CPU 版本）或 PyTorch 1.8+ + CUDA 10.1/11.0（GPU 版本）
- mmcv-full 1.3.x

### 4.2 一键安装（推荐）

```bash
bash setup_env.sh
```

### 4.3 手动安装

```bash
# 1. 安装 PyTorch（CPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或安装 PyTorch（GPU/CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html

# 3. 安装项目依赖
cd Video-Swin-Transformer
pip install -e .
pip install decord einops timm

cd ..
```

---

## 5. 权重下载

运行以下脚本自动下载 Swin-T（Kinetics-400）权重文件：

```bash
bash run_demo.sh
```

或手动下载：

```bash
mkdir -p checkpoints
# Swin-T，Kinetics-400，ImageNet-1K 预训练（最轻量，推荐 CPU 测试）
wget -O checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
  https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth
```

| 文件名 | 大小 | 数据集 | Top-1 |
|:---|:---:|:---:|:---:|
| `swin_tiny_patch244_window877_kinetics400_1k.pth` | ~110MB | K400 | 78.8% |
| `swin_small_patch244_window877_kinetics400_1k.pth` | ~200MB | K400 | 80.6% |
| `swin_base_patch244_window877_kinetics400_22k.pth` | ~350MB | K400 | 82.7% |

---

## 6. CPU / GPU 推理代码说明

### `predict_cpu.py` — CPU 推理

适用于无显卡环境，自动将模型和数据加载到 CPU。

```bash
python predict_cpu.py \
  --video demo_video.mp4 \
  --config Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
  --checkpoint checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
  --label Video-Swin-Transformer/demo/label_map_k400.txt \
  --top-k 5
```

### `predict_gpu.py` — GPU 推理

需要 CUDA 环境，自动检测并选择可用 GPU（默认 `cuda:0`）。

```bash
python predict_gpu.py \
  --video demo_video.mp4 \
  --config Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
  --checkpoint checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
  --label Video-Swin-Transformer/demo/label_map_k400.txt \
  --device cuda:0 \
  --top-k 5
```

### 两者区别

| 特性 | `predict_cpu.py` | `predict_gpu.py` |
|:---|:---:|:---:|
| 设备 | CPU（自动） | CUDA GPU（可指定） |
| 推理速度 | 较慢（~30-60s/视频） | 快速（~2-5s/视频） |
| 内存要求 | RAM ≥ 8GB | VRAM ≥ 4GB |
| `--device` 参数 | 固定为 `cpu` | 可指定 `cuda:0`, `cuda:1` 等 |
| CUDA 依赖 | 无 | 需要 CUDA 10.1+ |

---

## 7. 快速测试 Demo

```bash
# 完整一键测试（下载权重 + 运行 CPU/GPU 推理）
bash run_demo.sh
```

**脚本流程**：
1. 下载 Swin-T Kinetics-400 权重到 `checkpoints/`
2. 复制 demo 视频到当前目录
3. 分别运行 CPU 和 GPU 推理（GPU 不可用时自动跳过）
4. 打印 Top-5 行为识别结果

**示例输出**：

```
============================================================
  Video Swin Transformer — CPU 推理
  视频: demo_video.mp4
  设备: cpu
============================================================

[步骤 1/3] 加载模型...
  模型参数量: 28,158,856
  设备: cpu

[步骤 2/3] 预处理视频...
  视频信息: 167 帧, 30.0 fps, 分辨率 1280x720

[步骤 3/3] 执行推理...
  推理耗时: 45.32 秒

============================================================
  Top-5 行为识别结果
============================================================
  #1  playing soccer          得分: 12.4521
  #2  kicking soccer ball     得分:  9.8832
  #3  shooting goal (soccer)  得分:  8.1204
  ...
```

---

## 8. 目录结构

```
pose_estimation/
├── README.md                          # 本文档
├── predict_cpu.py                     # CPU 推理脚本
├── predict_gpu.py                     # GPU 推理脚本
├── setup_env.sh                       # 环境安装脚本
├── run_demo.sh                        # 一键测试脚本
├── checkpoints/                       # 权重文件目录（自动创建）
│   └── swin_tiny_patch244_window877_kinetics400_1k.pth
├── demo_video.mp4                     # 测试视频
└── Video-Swin-Transformer/            # 原始项目代码
    ├── configs/recognition/swin/      # Swin 系列配置文件
    ├── demo/                          # 官方 demo 脚本
    ├── mmaction/                      # 核心框架代码
    └── ...
```

---

## 9. 常见问题

**Q: `ModuleNotFoundError: No module named 'mmcv'`**
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0/index.html
```

**Q: `RuntimeError: CUDA error: no kernel image is available for execution`**
> CUDA 版本与 PyTorch 不匹配，重新安装对应 CUDA 版本的 PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Q: CPU 推理速度很慢**
> Swin-T 在 CPU 上处理一段视频约需 30-90 秒（取决于视频长度和机器性能）。如需加速，可减少采样帧数（在配置文件中调整 `num_clips` 和 `clip_len`）。

**Q: `FileNotFoundError: checkpoint not found`**
> 运行 `bash run_demo.sh` 或手动下载权重文件，参见[第 5 节](#5-权重下载)。

---

## 10. 引用

```bibtex
@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}

@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
