# 使用自定义数据集训练 Video Swin Transformer 完整指南

> 本文档解释如何将自己的视频数据集接入 MMAction2，使用 Video Swin Transformer (Tiny) 进行训练。

---

## 一、理解数据接入的核心机制

UCF-101 之所以有专门的数据处理脚本（`download_annotations.sh`、`generate_videos_filelist.sh` 等），是因为 MMAction2 的数据加载器依赖两个关键文件：

```
① 标注文件 (ann_file)      —— 告诉模型"有哪些视频、每个视频是什么类别"
② 视频/帧文件              —— 实际的媒体数据
```

UCF-101 的官方标注是 `trainlist01.txt` / `testlist01.txt` 格式，`build_file_list.py` 脚本将其转换为 MMAction2 统一的标注格式。**自定义数据集只需跳过这个转换步骤，直接生成符合格式要求的 txt 文件即可。**

---

## 二、标注文件格式说明

### VideoDataset 格式（推荐，直接读 .mp4/.avi）

```
# 格式: <视频相对路径> <类别标签整数(0-based)>
basketball/clip_001.mp4 0
basketball/clip_002.mp4 0
tennis/clip_001.mp4 1
tennis/clip_002.mp4 1
walking/clip_001.mp4 2
```

> **注意**: 类别标签从 0 开始，`视频相对路径` 是相对于配置文件中 `data_root` 的路径。

### RawframeDataset 格式（预先解帧，I/O 更快）

```
# 格式: <帧目录相对路径> <总帧数> <类别标签整数>
basketball/clip_001 120 0
basketball/clip_002 98 0
tennis/clip_001 145 1
```

---

## 三、推荐的目录结构

```
mmaction2/
└── data/
    └── my_dataset/                         ← 自定义数据集根目录
        ├── videos/                          ← 视频文件（VideoDataset 用）
        │   ├── class_A/
        │   │   ├── video_001.mp4
        │   │   └── video_002.mp4
        │   ├── class_B/
        │   │   └── video_001.mp4
        │   └── class_C/
        │       └── video_001.mp4
        ├── rawframes/                       ← 解帧后的图像（RawframeDataset 用，可选）
        │   ├── class_A/
        │   │   └── video_001/
        │   │       ├── img_00001.jpg
        │   │       └── img_00002.jpg
        │   └── ...
        ├── annotations/                     ← 类别映射文件（可选，便于管理）
        │   └── label_map.txt
        ├── train.txt                        ← 训练集标注文件
        └── val.txt                          ← 验证集标注文件
```

---

## 四、Step-by-Step：准备自定义数据集

### Step 1：整理视频文件

按类别建立子文件夹，将视频放入对应目录：

```
data/my_dataset/videos/
├── class_A/        ← 类别 A 的所有视频
├── class_B/        ← 类别 B 的所有视频
└── class_C/        ← 类别 C 的所有视频
```

建议视频格式：**MP4 (H.264 编码)**，避免使用 `.avi` 等可能有编解码兼容问题的格式。

### Step 2：创建类别映射文件（可选但推荐）

```
# data/my_dataset/annotations/label_map.txt
# 格式: <类别名称>  （行号即为类别 ID，从 0 开始）
class_A
class_B
class_C
```

### Step 3：生成 train.txt 和 val.txt

**方法A: 使用以下 Python 脚本自动生成**

```python
# tools/data/build_custom_filelist.py
import os
import random

# ===== 配置区域 =====
video_root = 'data/my_dataset/videos'   # 视频根目录
output_dir = 'data/my_dataset'          # 输出 txt 目录
val_ratio  = 0.2                        # 验证集比例（20%）
seed       = 42
# ====================

random.seed(seed)

# 扫描所有视频，按类别排序构建标签映射
classes = sorted([d for d in os.listdir(video_root)
                  if os.path.isdir(os.path.join(video_root, d))])
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

print("发现类别:", class_to_idx)

all_samples = []
for cls_name, cls_idx in class_to_idx.items():
    cls_dir = os.path.join(video_root, cls_name)
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            rel_path = os.path.join(cls_name, fname)
            all_samples.append((rel_path, cls_idx))

# 随机划分训练/验证集
random.shuffle(all_samples)
split_idx = int(len(all_samples) * (1 - val_ratio))
train_samples = all_samples[:split_idx]
val_samples   = all_samples[split_idx:]

# 写入 txt 文件
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
    for path, label in train_samples:
        f.write(f'{path} {label}\n')

with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
    for path, label in val_samples:
        f.write(f'{path} {label}\n')

print(f"训练集: {len(train_samples)} 条")
print(f"验证集: {len(val_samples)} 条")
print(f"已保存至 {output_dir}/train.txt 和 val.txt")
```

运行方式：
```bash
cd /workspaces/pose_estimation/mmaction2
python tools/data/build_custom_filelist.py
```

**方法B: 手动编写**

按照第二节的格式，用任何文本编辑器手动创建 `train.txt` 和 `val.txt`。

### Step 4（可选）：提取 RGB 帧

若要使用 `RawframeDataset`（读取更快），可用 OpenCV 提取帧：

```bash
cd /workspaces/pose_estimation/mmaction2
python tools/data/build_rawframes.py \
    data/my_dataset/videos \
    data/my_dataset/rawframes \
    --level 2 \
    --ext mp4 \
    --task rgb
```

然后更新 txt 文件格式为 `<帧目录> <帧数> <标签>`（参考第二节 RawframeDataset 格式）。

---

## 五、创建自定义数据集配置文件

在 `mmaction2/configs/recognition/swin/` 下复制并修改配置文件：

```bash
cd /workspaces/pose_estimation/mmaction2
cp configs/recognition/swin/swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py \
   configs/recognition/swin/swin-tiny_my_dataset.py
```

编辑 `swin-tiny_my_dataset.py`，修改以下字段：

```python
# =============================================================
# 必须修改的字段
# =============================================================

# 1. 修改分类数量（最重要！）
model = dict(
    backbone=dict(
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth'
    ),
    cls_head=dict(
        num_classes=3   # ← 改为你的数据集类别数
    )
)

# 2. 修改数据集路径
dataset_type = 'VideoDataset'   # 或 'RawframeDataset'（如果提前解帧）

data_root     = 'data/my_dataset/videos'   # 训练视频根目录
data_root_val = 'data/my_dataset/videos'   # 验证视频根目录（同一目录时填相同）

ann_file_train = 'data/my_dataset/train.txt'
ann_file_val   = 'data/my_dataset/val.txt'
ann_file_test  = 'data/my_dataset/val.txt'  # 无独立测试集时用验证集代替
```

```python
# =============================================================
# 建议根据数据集规模调整的字段
# =============================================================

# 3. 训练轮数（小数据集 epoch 可以更多）
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,       # 小数据集可适当增加，如 50~100
    val_begin=1,
    val_interval=5
)

# 4. Batch size（根据显存调整）
train_dataloader = dict(
    batch_size=4,        # 显存不足时减小（16GB 显存建议 4~8）
    num_workers=4,       # 根据 CPU 核心数调整
    ...
)

# 5. 学习率（与 batch_size 联动）
# 原始配置: lr=1e-3 对应 batch_size=8×8=64
# 若实际 batch_size=32，则 lr = 1e-3 × (32/64) = 5e-4
optim_wrapper = dict(
    ...
    optimizer=dict(
        lr=5e-4,         # 按比例调整
        ...
    )
)

# 6. 学习率调度（总 epoch 数变化时需同步修改）
param_scheduler = [
    dict(type='LinearLR', ..., end=5),          # warmup 改为 5 epoch
    dict(type='CosineAnnealingLR', T_max=50, end=50)  # 与 max_epochs 保持一致
]

# 7. 自动LR缩放（推荐开启，可免手动计算 lr）
auto_scale_lr = dict(enable=True, base_batch_size=64)
```

---

## 六、完整配置文件示例（3分类自定义数据集）

```python
# configs/recognition/swin/swin-tiny_my_dataset.py

_base_ = [
    '../../_base_/models/swin_tiny.py',
    '../../_base_/default_runtime.py'
]

# ---- 修改1: 分类数 ----
model = dict(
    backbone=dict(
        pretrained='https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth'
    ),
    cls_head=dict(num_classes=3))  # 3 个类别

# ---- 修改2: 数据集路径 ----
dataset_type = 'VideoDataset'
data_root     = 'data/my_dataset/videos'
data_root_val = 'data/my_dataset/videos'
ann_file_train = 'data/my_dataset/train.txt'
ann_file_val   = 'data/my_dataset/val.txt'
ann_file_test  = 'data/my_dataset/val.txt'

file_client_args = dict(io_backend='disk')

# ---- Pipeline 保持不变（或按需调整分辨率）----
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2,
         num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2,
         num_clips=4, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

# ---- DataLoader ----
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# ---- 修改3: 训练轮数 ----
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ---- 优化器（学习率随 batch_size 调整）----
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.02),
    constructor='SwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.),
        relative_position_bias_table=dict(decay_mult=0.),
        norm=dict(decay_mult=0.),
        backbone=dict(lr_mult=0.1)))

# ---- 学习率调度（与 max_epochs 对齐）----
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True,
         begin=0, end=5, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', T_max=50, eta_min=0,
         by_epoch=True, begin=0, end=50)
]

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(interval=50))

auto_scale_lr = dict(enable=True, base_batch_size=64)
```

---

## 七、训练指令

```bash
cd /workspaces/pose_estimation/mmaction2

# 单 GPU 训练
python tools/train.py \
    configs/recognition/swin/swin-tiny_my_dataset.py \
    --work-dir work_dirs/swin_tiny_my_dataset

# 多 GPU 训练（4 GPU）
bash tools/dist_train.sh \
    configs/recognition/swin/swin-tiny_my_dataset.py \
    4 \
    --work-dir work_dirs/swin_tiny_my_dataset

# 从 checkpoint 恢复训练
python tools/train.py \
    configs/recognition/swin/swin-tiny_my_dataset.py \
    --work-dir work_dirs/swin_tiny_my_dataset \
    --resume work_dirs/swin_tiny_my_dataset/epoch_10.pth
```

---

## 八、常见问题排查

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `IndexError: index out of range` | `num_classes` 与实际类别数不匹配 | 检查配置文件中 `cls_head.num_classes` |
| `FileNotFoundError: xxx.mp4` | 视频路径错误 | 检查 `data_root` + `ann_file` 中路径拼接是否正确 |
| `CUDA out of memory` | 显存不足 | 减小 `batch_size`，或减小 `clip_len` |
| `ann_file` 生成的 txt 为空 | 视频目录结构不对 | 确认视频按 `class_name/video.mp4` 二级目录组织 |
| `KeyError: xxx class` | 标签值超出范围 | 确认标签从 0 开始连续编号 |
| 训练精度异常低 | 预训练权重未加载 | 检查 `pretrained` URL 是否可访问，或改为本地路径 |

---

## 九、与 UCF-101 流程的对比总结

| 步骤 | UCF-101（有专用脚本） | 自定义数据集 |
|------|----------------------|-------------|
| 下载标注 | `bash download_annotations.sh` | ❌ 不需要（自己定义类别） |
| 下载视频 | `bash download_videos.sh` | 自行准备视频并分类整理 |
| 生成 filelist | `bash generate_videos_filelist.sh`（调用 `build_file_list.py`，需官方标注格式） | **直接编写或用脚本生成 `train.txt`/`val.txt`** |
| 修改配置 | 修改路径 + 类别数 | 修改路径 + 类别数（完全相同） |

**核心结论：UCF-101 的脚本工具链只是帮你将官方标注格式转换为 MMAction2 的 txt 格式。自定义数据集跳过这个转换，直接生成 txt 文件即可，后续训练流程完全相同。**
