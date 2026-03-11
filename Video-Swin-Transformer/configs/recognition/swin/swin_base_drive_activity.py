# ============================================================
# Swin-B 用于驾驶舱活动识别（5 类）
# 基于 Kinetics-400 预训练权重进行微调
# 硬件目标：NVIDIA RTX 4090 (24 GB)
# ============================================================

_base_ = [
    '../../_base_/models/swin/swin_base.py',
    '../../_base_/default_runtime.py',
]

# ---------- 模型设置 ----------
model = dict(
    backbone=dict(
        patch_size=(2, 4, 4),
        drop_path_rate=0.3,
    ),
    cls_head=dict(
        num_classes=5,        # 5 类活动
        in_channels=1024,
    ),
    test_cfg=dict(max_testing_views=4),
)

# ---------- 数据集设置 ----------
dataset_type = 'DriveActivityDataset'
# AutoDL 上视频根目录（vp1/, vp2/ 等子目录均在此目录下）
data_root = 'data/video'
# CSV 标注文件（train/val 由准备脚本生成）
ann_file_train = 'data/annotations/train.csv'
ann_file_val   = 'data/annotations/val.csv'
ann_file_test  = 'data/annotations/val.csv'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_bgr=False,
)

# 每个训练 clip 采样 32 帧，帧间隔 2
train_pipeline = [
    dict(type='SegmentDecordInit'),          # 自定义：读取视频并限定片段范围
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label']),
]

val_pipeline = [
    dict(type='SegmentDecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2,
         num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs']),
]

test_pipeline = [
    dict(type='SegmentDecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2,
         num_clips=4, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs']),
]

# ---------- DataLoader 设置（4090 24GB 优化） ----------
# videos_per_gpu=4：Swin-B 32帧时约 18-20 GB，保留余量
data = dict(
    videos_per_gpu=2,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=2,
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=2,
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root,
        pipeline=val_pipeline,
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

# ---------- 评估指标 ----------
evaluation = dict(
    interval=1,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    metric_options=dict(top_k_accuracy=dict(topk=(1, 3))),
    save_best='top1_acc',
)

# ---------- 优化器（AdamW + 余弦退火） ----------
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.02,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
        }
    ),
)

# GradientCumulativeFp16OptimizerHook：同时支持梯度累积 + fp16 混合精度
# 等效 batch_size = videos_per_gpu × cumulative_iters = 4 × 2 = 8
optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook',
    grad_clip=dict(max_norm=5, norm_type=2),
    cumulative_iters=2,
    loss_scale='dynamic',
)

# 总训练 epoch
total_epochs = 30

# 余弦退火学习率
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2,          # 前 2 个 epoch warm-up
)

# ---------- 运行时设置 ----------
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
)

work_dir = 'work_dirs/swin_base_drive_activity'

# ---------- 预训练权重 ----------
# 从 Kinetics-400 预训练模型微调，下载后放到 pretrained/ 目录
# 权重下载：
#   https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth
load_from = 'pretrained/swin_base_patch244_window877_kinetics400_1k.pth'
