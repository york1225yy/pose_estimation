# Video-Swin-Transformer Demo 使用文档

> 本文档覆盖 `Video-Swin-Transformer/demo/` 目录下所有脚本，以及本项目新增的滑动窗口连续行为识别脚本。
> 所有命令默认在项目根目录 `/workspaces/pose_estimation/` 下执行。

---

## 目录

1. [环境准备](#1-环境准备)
2. [公共说明](#2-公共说明)
3. [demo.py — 单视频行为识别](#3-demopy--单视频行为识别)
4. [long_video_demo.py — 长视频滑动窗口识别（官方版）](#4-long_video_demopy--长视频滑动窗口识别官方版)
5. [webcam_demo.py — 摄像头实时识别](#5-webcam_demopy--摄像头实时识别)
6. [demo_gradcam.py — GradCAM 热力图可视化](#6-demo_gradcampy--gradcam-热力图可视化)
7. [demo_spatiotemporal_det.py — 时空动作检测（视频）](#7-demo_spatiotemporal_detpy--时空动作检测视频)
8. [webcam_demo_spatiotemporal_det.py — 时空动作检测（摄像头）](#8-webcam_demo_spatiotemporal_detpy--时空动作检测摄像头)
9. [sliding_window_cpu.py — 本项目·CPU 滑动窗口连续识别](#9-sliding_window_cpupy--本项目cpu-滑动窗口连续识别)
10. [sliding_window_gpu.py — 本项目·GPU 滑动窗口连续识别](#10-sliding_window_gpupy--本项目gpu-滑动窗口连续识别)
11. [Notebook 文件说明](#11-notebook-文件说明)
12. [常见问题](#12-常见问题)

---

## 1. 环境准备

```bash
# 激活虚拟环境
source /workspaces/pose_estimation/.venv/bin/activate

# 首次使用，安装全部依赖
bash setup_env.sh
```

**本项目常用路径速查：**

| 资源 | 路径 |
|------|------|
| Video Swin Tiny 配置 | `Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py` |
| Video Swin Tiny 权重 | `checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth` |
| Kinetics-400 标签 | `Video-Swin-Transformer/demo/label_map_k400.txt` |
| AVA 标签 | `Video-Swin-Transformer/demo/label_map_ava.txt` |
| 示例视频 | `Video-Swin-Transformer/demo/demo.mp4` |

---

## 2. 公共说明

### `--cfg-options` 参数

所有脚本均支持通过 `--cfg-options` 在命令行直接修改配置，格式为 `key=value`：

```bash
# 示例：修改 backbone 深度
--cfg-options model.backbone.depth=18

# 示例：修改 test 数据 pipeline 的第 0 个步骤类型
--cfg-options data.test.pipeline.0.type=DenseSampleFrames
```

### 设备参数 `--device`

- `cuda:0`（默认）：使用第一块 GPU
- `cuda:1`、`cuda:2`：多卡环境下指定 GPU
- `cpu`：仅 CPU 模式

### 文件路径约定

所有示例命令均以 `$MMACTION2` 代表 `Video-Swin-Transformer/` 目录。

---

## 3. demo.py — 单视频行为识别

**功能：** 对整段视频采样固定帧，一次性推理，生成标注结果视频或 GIF。

**脚本路径：** `Video-Swin-Transformer/demo/demo.py`

### 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `config` | 必填 | — | 模型配置文件路径 |
| `checkpoint` | 必填 | — | 权重文件路径或 URL |
| `video` | 必填 | — | 视频文件路径 / URL / 帧目录 |
| `label` | 必填 | — | 标签文件（每行一个类别名） |
| `--use-frames` | flag | False | 使用帧目录作为输入而非视频文件 |
| `--device` | str | `cuda:0` | 计算设备 |
| `--fps` | int | 30 | 输出视频帧率（帧目录模式有效） |
| `--font-scale` | float | 0.5 | 标签字体大小 |
| `--font-color` | str | `white` | 标签字体颜色（颜色名或 hex） |
| `--target-resolution` | int int | None | 输出分辨率 `W H`，-1 表示保持比例 |
| `--resize-algorithm` | str | `bicubic` | 缩放算法（bilinear/bicubic/area 等） |
| `--out-filename` | str | None | 输出文件路径（.mp4 或 .gif） |

### 使用示例

```bash
# 示例 1：使用 Video Swin Tiny 对 demo.mp4 进行识别，输出标注视频
python Video-Swin-Transformer/demo/demo.py \
    Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    Video-Swin-Transformer/demo/demo.mp4 \
    Video-Swin-Transformer/demo/label_map_k400.txt \
    --device cpu \
    --out-filename result_demo.mp4

# 示例 2：生成 GIF 输出
python Video-Swin-Transformer/demo/demo.py \
    Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    Video-Swin-Transformer/demo/demo.mp4 \
    Video-Swin-Transformer/demo/label_map_k400.txt \
    --out-filename result_demo.gif \
    --target-resolution 640 -1
```

> **注意：** 此脚本依赖 `moviepy`（`pip install moviepy`），且对整段视频仅推理一次，不支持连续帧变化的实时识别。如需连续识别，请使用第 4 节或第 9/10 节的脚本。

---

## 4. long_video_demo.py — 长视频滑动窗口识别（官方版）

**功能：** 对长视频采用滑动窗口方式逐段推理，随着帧的前进动态更新识别结果，输出带标注的视频或 JSON 文件。

**脚本路径：** `Video-Swin-Transformer/demo/long_video_demo.py`

### 核心机制

- 维护一个大小为 `sample_length`（由配置中 `clip_len × num_clips` 决定）的帧队列
- 每 `input_step` 帧从视频中随机采样一帧加入队列
- 每当队列被填满（或达到 `stride` 条件）就执行一次推理
- 推理结果显示在对应帧的左上角

### 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `config` | 必填 | — | 模型配置文件路径 |
| `checkpoint` | 必填 | — | 权重文件路径 |
| `video_path` | 必填 | — | 视频文件路径 |
| `label` | 必填 | — | 标签文件路径 |
| `out_file` | 必填 | — | 输出文件路径（.mp4 或 .json） |
| `--input-step` | int | 1 | 帧采样步长（每 N 帧取一帧） |
| `--device` | str | `cuda:0` | 计算设备 |
| `--threshold` | float | 0.01 | 显示分数阈值 |
| `--stride` | float | 0 | 推理步长比例（0 = 每帧推理一次） |
| `--label-color` | int int int | (255,255,255) | 标签颜色（BGR） |
| `--msg-color` | int int int | (128,128,128) | 等待消息颜色（BGR） |

### 使用示例

```bash
# 示例 1：输出标注视频（每帧推理）
python Video-Swin-Transformer/demo/long_video_demo.py \
    Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    Video-Swin-Transformer/demo/demo.mp4 \
    Video-Swin-Transformer/demo/label_map_k400.txt \
    result_long.mp4 \
    --device cpu \
    --threshold 0.01

# 示例 2：输出 JSON 结果文件，每 4 帧推理一次（stride=0.5 表示滑动半个窗口）
python Video-Swin-Transformer/demo/long_video_demo.py \
    Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    Video-Swin-Transformer/demo/demo.mp4 \
    Video-Swin-Transformer/demo/label_map_k400.txt \
    result_long.json \
    --device cpu \
    --stride 0.5 \
    --input-step 2
```

> **与本项目 sliding_window 脚本的区别：**
> `long_video_demo.py` 使用随机帧采样策略，适合长视频分析；
> 本项目的 `sliding_window_cpu/gpu.py` 使用顺序帧填充，行为更直观，并且界面风格与 `predict_cpu/gpu.py` 统一。

---

## 5. webcam_demo.py — 摄像头实时识别

**功能：** 从摄像头实时读取帧，利用双线程（一个线程推理、一个线程显示）实现实时行为识别。

**脚本路径：** `Video-Swin-Transformer/demo/webcam_demo.py`

### 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `config` | 必填 | — | 模型配置文件路径 |
| `checkpoint` | 必填 | — | 权重文件路径 |
| `label` | 必填 | — | 标签文件路径 |
| `--device` | str | `cuda:0` | 计算设备 |
| `--camera-id` | int | 0 | 摄像头设备 ID（0 为默认摄像头） |
| `--threshold` | float | 0.01 | 显示分数阈值 |
| `--average-size` | int | 1 | 平均最近 N 次推理结果（平滑效果） |
| `--drawing-fps` | int | 20 | 显示帧率上限（0 = 不限制） |
| `--inference-fps` | int | 4 | 推理帧率上限（0 = 不限制） |

### 使用示例

```bash
# 使用摄像头 0，GPU 模式，推理 FPS 限制为 4
python Video-Swin-Transformer/demo/webcam_demo.py \
    Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    Video-Swin-Transformer/demo/label_map_k400.txt \
    --device cuda:0 \
    --camera-id 0 \
    --average-size 3 \
    --drawing-fps 30 \
    --inference-fps 4

# CPU 模式，推理 FPS 适当降低以减少卡顿
python Video-Swin-Transformer/demo/webcam_demo.py \
    Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    Video-Swin-Transformer/demo/label_map_k400.txt \
    --device cpu \
    --inference-fps 1 \
    --average-size 5
```

> **按 `Esc`、`q` 或 `Q` 键退出** 摄像头演示窗口。
> `--average-size` 可对多次推理结果取平均，使识别结果更稳定，建议设为 3~5。

---

## 6. demo_gradcam.py — GradCAM 热力图可视化

**功能：** 对视频/帧目录计算 GradCAM 激活热力图，展示模型关注的时空区域，输出带热力图叠加的视频或 GIF。

**脚本路径：** `Video-Swin-Transformer/demo/demo_gradcam.py`

### 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `config` | 必填 | — | 模型配置文件路径 |
| `checkpoint` | 必填 | — | 权重文件路径 |
| `video` | 必填 | — | 视频文件路径或帧目录 |
| `--use-frames` | flag | False | 输入为帧目录时启用 |
| `--device` | str | `cuda:0` | 计算设备 |
| `--target-layer-name` | str | `backbone/layer4/1/relu` | GradCAM 目标层名称 |
| `--out-filename` | str | None | 输出文件路径（.mp4 或 .gif） |
| `--fps` | int | 5 | 输出视频/GIF 帧率 |
| `--target-resolution` | int int | None | 输出分辨率 `W H` |
| `--resize-algorithm` | str | `bilinear` | 缩放算法 |

### GradCAM 目标层选择

不同模型架构推荐的目标层：

| 模型 | 推荐 `--target-layer-name` |
|------|---------------------------|
| Video Swin | `backbone/layers.3/blocks.1/norm2` |
| I3D | `backbone/layer4/1/relu` |
| SlowFast | `backbone/slow_path/layer4/1/relu` |

### 使用示例

```bash
# 生成 GradCAM 热力图视频
python Video-Swin-Transformer/demo/demo_gradcam.py \
    Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    Video-Swin-Transformer/demo/demo.mp4 \
    --device cpu \
    --target-layer-name "backbone/layers.3/blocks.1/norm2" \
    --out-filename result_gradcam.gif \
    --fps 5 \
    --target-resolution 640 -1
```

---

## 7. demo_spatiotemporal_det.py — 时空动作检测（视频）

**功能：** 结合人体检测（Faster R-CNN）与时空动作识别（SlowOnly/AVA 等）对视频中每个人的动作进行逐帧检测，在边框内显示动作标签。

**脚本路径：** `Video-Swin-Transformer/demo/demo_spatiotemporal_det.py`

### 依赖

- `mmdet`（人体检测）：`pip install mmdet`
- `moviepy`（视频输出）：`pip install moviepy`

### 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--config` | str | AVA SlowOnly 配置 | 时空检测模型配置文件 |
| `--checkpoint` | str | AVA SlowOnly 权重 URL | 时空检测模型权重 |
| `--det-config` | str | `demo/faster_rcnn_r50_fpn_2x_coco.py` | 人体检测配置文件 |
| `--det-checkpoint` | str | Faster R-CNN 权重 URL | 人体检测权重 |
| `--action-score-thr` | float | 0.4 | 动作分数阈值 |
| `--det-score-thr` | float | 0.9 | 人体检测分数阈值 |
| `--input-video` | str | — | 输入视频文件路径 |
| `--label-map` | str | `demo/label_map_ava.txt` | AVA 标签文件 |
| `--device` | str | `cuda:0` | 计算设备 |
| `--output-fps` | int | 15 | 输出视频帧率 |
| `--out-filename` | str | None | 输出视频文件路径 |
| `--show` | flag | False | 是否显示实时窗口 |
| `--display-height` | int | 0 | 显示/处理分辨率高度（0=原始） |
| `--display-width` | int | 0 | 显示/处理分辨率宽度（0=原始） |
| `--predict-stepsize` | int | 8 | 每 N 帧进行一次动作检测 |
| `--clip-vis-length` | int | 8 | 动作标签显示持续帧数 |

### 使用示例

```bash
# 使用 AVA 数据集预训练模型进行时空动作检测
python Video-Swin-Transformer/demo/demo_spatiotemporal_det.py \
    --config Video-Swin-Transformer/configs/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb.py \
    --checkpoint https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --input-video Video-Swin-Transformer/demo/demo.mp4 \
    --label-map Video-Swin-Transformer/demo/label_map_ava.txt \
    --device cuda:0 \
    --out-filename result_det.mp4
```

> **注意：** 此脚本使用 AVA 标签集（`label_map_ava.txt`），识别的是"站立/坐下/行走/交谈"等细粒度动作，而非 Kinetics-400 动作类别。

---

## 8. webcam_demo_spatiotemporal_det.py — 时空动作检测（摄像头）

**功能：** 实时从摄像头或视频流读取画面，对画面中每个人进行实时时空动作检测。使用多线程架构（帧读取线程、人体检测线程、动作识别线程、显示线程）。

**脚本路径：** `Video-Swin-Transformer/demo/webcam_demo_spatiotemporal_det.py`

### 依赖

- `mmdet`：`pip install mmdet`

### 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--config` | str | AVA SlowOnly 配置 | 时空检测模型配置 |
| `--checkpoint` | str | AVA SlowOnly 权重 URL | 时空检测模型权重 |
| `--det-config` | str | Faster R-CNN 配置 | 人体检测配置 |
| `--det-checkpoint` | str | Faster R-CNN 权重 URL | 人体检测权重 |
| `--action-score-thr` | float | 0.4 | 动作分数阈值 |
| `--det-score-thr` | float | 0.9 | 人体检测分数阈值 |
| `--input-video` | str | `0` | 摄像头 ID（0）或视频文件/URL |
| `--label-map` | str | `demo/label_map_ava.txt` | AVA 标签文件 |
| `--device` | str | `cuda:0` | 计算设备 |
| `--output-fps` | int | 15 | 输出帧率 |
| `--out-filename` | str | None | 录制输出视频路径 |
| `--show` | flag | False | 显示窗口 |
| `--display-height` | int | 0 | 显示高度（0=原始） |
| `--display-width` | int | 0 | 显示宽度（0=原始） |
| `--predict-stepsize` | int | 8 | 每 N 帧进行一次动作检测 |
| `--clip-vis-length` | int | 8 | 动作标签显示持续帧数 |

### 使用示例

```bash
# 摄像头实时检测（需 GPU）
python Video-Swin-Transformer/demo/webcam_demo_spatiotemporal_det.py \
    --input-video 0 \
    --label-map Video-Swin-Transformer/demo/label_map_ava.txt \
    --device cuda:0 \
    --show

# 对视频文件进行处理并保存
python Video-Swin-Transformer/demo/webcam_demo_spatiotemporal_det.py \
    --input-video Video-Swin-Transformer/demo/demo.mp4 \
    --label-map Video-Swin-Transformer/demo/label_map_ava.txt \
    --device cuda:0 \
    --out-filename result_webcam_det.mp4
```

---

## 9. sliding_window_cpu.py — 本项目·CPU 滑动窗口连续识别

**功能：** 对视频进行逐帧滑动窗口推理，每收集到 `--stride` 帧新帧就执行一次推理，随帧变化动态更新识别标签。纯 CPU 模式。

**脚本路径：** `sliding_window_cpu.py`（项目根目录）

### 核心原理

```
帧队列（deque，maxlen = sample_length）
 ┌─────────────────────────────┐
 │  frame[t-N] ... frame[t-1] │ ← 每加入 stride 帧就推理一次
 └─────────────────────────────┘
       ↓ 推理结果叠加到每帧上写入输出视频
```

### 参数说明

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--video` | str | 必填 | 输入视频文件路径 |
| `--config` | str | 必填 | 模型配置文件 |
| `--checkpoint` | str | 必填 | 权重文件 |
| `--label` | str | 必填 | 标签文件 |
| `--stride` | int | 8 | 每 N 帧推理一次（越小结果越实时，CPU 负担越大） |
| `--top-k` | int | 5 | 显示 Top-K 个标签 |
| `--threshold` | float | 0.01 | 低于此分数的标签不显示 |
| `--output` | str | `result_sliding_cpu.mp4` | 输出视频路径 |
| `--display` | flag | False | 是否实时显示窗口（需图形界面） |

### 使用示例

```bash
# 基础用法：每 8 帧推理一次
python sliding_window_cpu.py \
    --video Video-Swin-Transformer/demo/demo.mp4 \
    --config Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    --checkpoint checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    --label Video-Swin-Transformer/demo/label_map_k400.txt \
    --stride 8 \
    --top-k 5 \
    --output result_sliding_cpu.mp4

# 更密集识别：每 4 帧推理一次（CPU 耗时更多）
python sliding_window_cpu.py \
    --video my_video.mp4 \
    --config Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    --checkpoint checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    --label Video-Swin-Transformer/demo/label_map_k400.txt \
    --stride 4 \
    --threshold 0.05 \
    --output result_dense_cpu.mp4
```

### stride 参数建议

| stride 值 | 特点 | 适用场景 |
|-----------|------|---------|
| 1 | 每帧推理，CPU 极高负担 | 调试验证 |
| 4~8 | 平衡密度与速度 | 一般使用（推荐） |
| 16~32 | 粗粒度识别，速度快 | 长视频快速分析 |

---

## 10. sliding_window_gpu.py — 本项目·GPU 滑动窗口连续识别

**功能：** 与 `sliding_window_cpu.py` 相同，但优先使用 GPU（CUDA）推理，CUDA 不可用时自动回退 CPU。GPU 模式下推理速度显著提升，可使用更小的 stride 值。

**脚本路径：** `sliding_window_gpu.py`（项目根目录）

### 参数说明

在 CPU 版基础上新增：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--device` | str | `cuda:0` | 计算设备，无 GPU 自动回退 CPU |
| `--stride` | int | **4** | GPU 模式默认步长更小（可更密集） |

其余参数与 CPU 版完全相同。

### 使用示例

```bash
# GPU 模式，每 4 帧推理一次
python sliding_window_gpu.py \
    --video Video-Swin-Transformer/demo/demo.mp4 \
    --config Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    --checkpoint checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    --label Video-Swin-Transformer/demo/label_map_k400.txt \
    --device cuda:0 \
    --stride 4 \
    --top-k 5 \
    --output result_sliding_gpu.mp4

# 多 GPU 环境使用第二块 GPU
python sliding_window_gpu.py \
    --video my_video.mp4 \
    --config Video-Swin-Transformer/configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py \
    --checkpoint checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth \
    --label Video-Swin-Transformer/demo/label_map_k400.txt \
    --device cuda:1 \
    --stride 2 \
    --output result_gpu1.mp4
```

---

## 11. Notebook 文件说明

| 文件 | 语言 | 说明 |
|------|------|------|
| `demo.ipynb` | 中英文 | 交互式演示：加载模型 → 推理 → 可视化，适合入门体验 |
| `mmaction2_tutorial.ipynb` | 英文 | MMAction2 完整教程，覆盖训练/测试/可视化流程 |
| `mmaction2_tutorial_zh-CN.ipynb` | 中文 | 上述教程的中文版 |
| `visualize_heatmap_volume.ipynb` | 英文 | 可视化时序热力图体积（3D heatmap），用于分析模型激活分布 |

在 VS Code 中点击对应 `.ipynb` 文件即可直接打开并运行。

---

## 12. 常见问题

### Q1: `ImportError: No module named 'mmaction'`

```bash
cd Video-Swin-Transformer && pip install -e . && cd ..
```

### Q2: `ImportError: No module named 'mmcv'`

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
# 或 CPU 版本
pip install mmcv-full
```

### Q3: CPU 推理太慢怎么办？

- 增大 `--stride`（如 16 或 32），减少推理频率
- 使用更小的模型配置（如 `swin_tiny` 而非 `swin_base`）
- 如需实时性，优先考虑使用 GPU 版

### Q4: `decord` 读取视频失败

```bash
pip install decord
```
若仍失败（ARM/无 CUDA 环境），可尝试：
```bash
pip install eva-decord
```

### Q5: `moviepy` 相关错误

```bash
pip install moviepy imageio imageio-ffmpeg
```

### Q6: 输出视频无法播放

`mp4v` 编码在部分播放器中兼容性有限，可将输出文件用 ffmpeg 转码：
```bash
ffmpeg -i result_sliding_cpu.mp4 -vcodec libx264 result_h264.mp4
```

### Q7: 滑动窗口开始前没有识别结果？

这是正常现象。脚本需要先积累 `sample_length` 帧（Video Swin Tiny 约为 32 帧）才开始首次推理，在此期间显示 `"Waiting for action ..."`。

---

## 脚本功能对比速查表

| 脚本 | 输入 | 实时连续识别 | 时空检测 | 摄像头 | GPU 支持 |
|------|------|:----------:|:------:|:----:|:------:|
| `predict_cpu.py` | 视频 | ✗ | ✗ | ✗ | ✗ |
| `predict_gpu.py` | 视频 | ✗ | ✗ | ✗ | ✓ |
| `sliding_window_cpu.py` | 视频 | ✓ | ✗ | ✗ | ✗ |
| `sliding_window_gpu.py` | 视频 | ✓ | ✗ | ✗ | ✓ |
| `demo/demo.py` | 视频/帧 | ✗ | ✗ | ✗ | ✓ |
| `demo/long_video_demo.py` | 视频 | ✓ | ✗ | ✗ | ✓ |
| `demo/webcam_demo.py` | 摄像头 | ✓ | ✗ | ✓ | ✓ |
| `demo/demo_gradcam.py` | 视频/帧 | ✗ | ✗ | ✗ | ✓ |
| `demo/demo_spatiotemporal_det.py` | 视频 | ✓ | ✓ | ✗ | ✓ |
| `demo/webcam_demo_spatiotemporal_det.py` | 摄像头/视频 | ✓ | ✓ | ✓ | ✓ |
