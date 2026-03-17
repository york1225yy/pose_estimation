# YOWOv3 权重文件命名含义说明

> 来源仓库：https://github.com/hope1337/YOWOv3  
> 权重仓库：https://huggingface.co/manh6054/YOWOv3

---

## 命名规则总览

YOWOv3 的模型权重文件夹以 **Excel 单元格格式** 命名，即：

```
[列字母][行数字]
  例如：C23、D27、E30、G23 ...
```

- **列字母（Letter）**：决定 **2D骨干网络（YOLOv8 版本）** + **损失函数**
- **行数字（Number）**：决定 **3D骨干网络** + **融合通道数（interchannels）**

---

## 列字母（Letter）含义详解

| 字母 | YOLOv8 2D骨干版本 | 损失函数 | 参数量（2D骨干） |
|------|-------------------|----------|-----------------|
| **C** | YOLOv8-**n**（nano，最小） | TAL（Task-Aligned Loss） | ~3M |
| **D** | YOLOv8-**s**（small） | TAL | ~11M |
| **E** | YOLOv8-**m**（medium） | TAL | ~26M |
| **F** | YOLOv8-**l**（large） | TAL | ~44M |
| **G** | YOLOv8-**x**（extra-large，最大） | TAL | ~68M |
| **K** | YOLOv8-**n**（nano） | SIMOTA（SimOTA Loss） | ~3M |
| **M** | YOLOv8-**m**（medium） | SIMOTA（SimOTA Loss） | ~26M |

> **注意**：列 A、B 未使用；列 H、I、J、L、N、O 等为其他实验配置（含 AVA 数据集或其他变体），在 UCF24 权重中未全部开放。

---

## 行数字（Number）含义详解

| 数字 | 3D骨干网络 | 融合通道数（interchannels）\* | 说明 |
|------|-----------|-------------------------------|------|
| **23** | ShuffleNetv2（width_mult=2.0） | 自适应（见下表） | 轻量级3D骨干，速度最快 |
| **27** | ResNet-101 | [256, 256, 256] | 中等精度与效率平衡 |
| **29** | ResNeXt-101 | [256, 256, 256] | 高精度3D骨干 |
| **30** | I3D（Inflated 3D ConvNet） | [256, 256, 256] | 精度最高，计算量最大 |

### 行23（ShuffleNetv2）的融合通道数随2D骨干自适应变化：

| 组合 | 2D骨干（列） | interchannels |
|------|-------------|---------------|
| C23 | YOLOv8-n | [64, 64, 64] |
| D23 | YOLOv8-s | [128, 128, 128] |
| E23 / M23 | YOLOv8-m | [192, 192, 192] |
| F23、G23、K23 | YOLOv8-l/x 或大型配置 | [256, 256, 256] |

> `interchannels` 是特征融合模块（CFAM）中三个检测尺度的输出通道数列表（对应 small/medium/large 目标尺度）。

---

## C 系列权重详解（用户重点关注）

所有 **C** 系列均满足：  
- 2D骨干：`YOLOv8-n`（nano，最轻量）  
- 损失函数：`TAL`（Task-Aligned Loss）  
- 融合模块：`CFAM`（Cross-scale Feature Aggregation Module）  
- 检测头模式：`decoupled`（解耦头）  
- 图像尺寸：`224×224`  
- 片段长度：`16帧`  
- 采样率：`1`  

| 权重名 | 3D骨干 | interchannels | 数据集 | 特点 |
|--------|--------|---------------|--------|------|
| **C23** | ShuffleNetv2-2.0x | [64, 64, 64] | UCF24 | **速度最快，模型最小** |
| **C27** | ResNet-101 | [256, 256, 256] | UCF24 | 中等速度，精度较好 |
| **C29** | ResNeXt-101 | [256, 256, 256] | UCF24 | 精度高，计算量较大 |
| **C30** | I3D | [256, 256, 256] | UCF24 | **精度最高，计算量最大** |

---

## 所有同一行的对比示例（行30 = I3D）

| 权重名 | 2D骨干 | 3D骨干 | 损失 | interchannels |
|--------|--------|--------|------|---------------|
| C30 | YOLOv8-n | I3D | TAL | [256,256,256] |
| E30 | YOLOv8-m | I3D | TAL | [256,256,256] |

---

## 权重文件结构说明

每个权重文件夹下包含：

```
checkpoint/ucf24/C23/
├── config.yaml          # 训练该模型时使用的完整配置文件（关键！）
├── epoch_1.pth          # 第1轮训练的普通检查点
├── epoch_2.pth
├── ...
├── epoch_7.pth          # 第7轮训练的普通检查点
├── ema_epoch_1.pth      # 第1轮的 EMA（指数移动平均）版本
├── ema_epoch_2.pth
├── ...
├── ema_epoch_7.pth      # 第7轮的 EMA 版本（推荐用于推理/评估）
└── logging.txt          # 训练日志
```

> **推荐使用 `ema_epoch_7.pth`**：EMA 是模型参数的指数移动平均版本，在推理时通常比普通检查点具有更好的精度和泛化性。

---

## 骨干网络预训练权重（backbone）说明

| 文件路径 | 用于哪些权重 |
|---------|------------|
| `weights/backbone2D/YOLOv8/v8_n.pth` | C 系列、K 系列（YOLOv8-n） |
| `weights/backbone2D/YOLOv8/v8_s.pth` | D 系列（YOLOv8-s） |
| `weights/backbone2D/YOLOv8/v8_m.pth` | E 系列、M 系列（YOLOv8-m） |
| `weights/backbone2D/YOLOv8/v8_l.pth` | F 系列（YOLOv8-l） |
| `weights/backbone2D/YOLOv8/v8_x.pth` | G 系列（YOLOv8-x） |
| `weights/backbone3D/shufflenetv2/kinetics_shufflenetv2_2.0x_RGB_16_best.pth` | 行23 |
| `weights/backbone3D/resnet/kinetics_resnet_101_RGB_16_best.pth` | 行27 |
| `weights/backbone3D/resnext/resnext-101-kinetics.pth` | 行29 |
| `weights/backbone3D/I3D/rgb_imagenet.pth` | 行30 |

---

## 快速查找表

```
权重名 = [2D骨干配置][3D骨干配置]

C = n-nano / TAL       D = s-small / TAL      E = m-medium / TAL
F = l-large / TAL      G = x-xlarge / TAL     K = n-nano / SIMOTA
M = m-medium / SIMOTA

23 = ShuffleNetv2      27 = ResNet-101
29 = ResNeXt-101       30 = I3D
```

**所以 C23 = YOLOv8-nano + ShuffleNetv2 + TAL + CFAM，这是速度最快的组合。**  
**C30 = YOLOv8-nano + I3D + TAL + CFAM，这是精度最高（C系列中）的组合。**
