# 驾驶舱活动识别 - Video Swin Transformer 微调

基于 [Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) 的驾驶舱行为识别项目，使用 Swin-B 模型在5类活动上进行微调。

## 目标类别

| 标签 ID | 类别名称 |
|---------|----------|
| 0 | sitting_still |
| 1 | eating |
| 2 | fetching_an_object |
| 3 | placing_an_object |
| 4 | reading_magazine |

## 数据集说明

数据来自车内摄像头视频（`.mp4`），标注文件为 `midlevel.chunks_90.csv`，每行代表一个90帧的活动片段。

CSV 字段：`participant_id, file_id, annotation_id, frame_start, frame_end, activity, chunk_id`

## AutoDL 部署步骤

### 1. 克隆代码

```bash
git clone https://github.com/york1225yy/pose_estimation.git
cd pose_estimation/Video-Swin-Transformer
```

### 2. 准备数据目录

```
Video-Swin-Transformer/
└── data/
    ├── video/                  ← 视频根目录
    │   ├── vp1/
    │   │   ├── run1b_2018-05-29-14-02-47.ids_1.mp4
    │   │   └── run2_2018-05-29-14-33-44.ids_1.mp4
    │   ├── vp2/
    │   │   └── ...
    │   └── ...
    └── activity_label/
        └── midlevel.chunks_90.csv
```

将所有参与者的视频复制到对应的 `data/video/vp*/` 子目录，CSV 标签文件放到 `data/activity_label/`。

### 3. 一键安装与准备

```bash
bash setup_autodl.sh
```

该脚本会：
- 安装依赖（mmcv-full, decord, pandas 等）
- 按参与者划分生成 `data/annotations/train.csv` 和 `val.csv`
- 下载 Kinetics-400 预训练权重到 `pretrained/`

### 4. 开始训练

```bash
bash tools/train_drive_activity.sh
```

支持断点续训：若 `work_dirs/swin_base_drive_activity/latest.pth` 存在，自动恢复。

多卡训练（如有多张 GPU）：

```bash
GPUS=2 bash tools/train_drive_activity.sh
```

### 5. 查看训练日志

```bash
tensorboard --logdir work_dirs/swin_base_drive_activity
```

## 配置说明（针对 RTX 4090 24GB 优化）

| 参数 | 值 | 说明 |
|------|----|------|
| `videos_per_gpu` | 4 | 单卡 batch size |
| `cumulative_iters` | 2 | 梯度累积，等效 batch=8 |
| `clip_len` | 32 | 采样帧数 |
| `frame_interval` | 2 | 帧间隔 |
| `lr` | 1e-4 | AdamW 学习率 |
| `backbone lr_mult` | 0.1 | Backbone 使用更小的LR |
| `fp16` | True | 混合精度训练 |
| `total_epochs` | 30 | 训练总轮数 |

## 自定义组件

### DriveActivityDataset

位于 `mmaction/datasets/drive_activity_dataset.py`，继承 `BaseDataset`。

从 CSV 中读取片段级标注，每条记录包含：
- `filename`: 视频文件路径
- `label`: 类别索引（0-4）
- `segment_start`: 片段起始帧
- `segment_end`: 片段结束帧

### SegmentDecordInit

位于 `mmaction/datasets/pipelines/segment_loading.py`，在 Decord 打开视频后，
将 `total_frames` 和 `start_index` 覆盖为片段的实际范围，使 `SampleFrames` 只在该区间内采样。

## 预训练权重

手动下载后放到 `pretrained/` 目录：

```bash
wget "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth" \
     -O pretrained/swin_base_patch244_window877_kinetics400_1k.pth
```
