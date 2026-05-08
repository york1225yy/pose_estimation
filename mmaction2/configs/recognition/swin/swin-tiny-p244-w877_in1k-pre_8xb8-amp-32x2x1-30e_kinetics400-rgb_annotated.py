# =============================================================================
# Video Swin Transformer (Tiny) 完整配置文件 - 逐行注释版
# 原始配置: swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py
#
# 文件名含义拆解:
#   swin-tiny        → 模型大小: tiny 版本
#   p244             → patch_size: (2, 4, 4) 即时间维度2帧, 空间维度4x4像素
#   w877             → window_size: (8, 7, 7) 即时间窗口8帧, 空间7x7 patch
#   in1k-pre         → 预训练来源: ImageNet-1K
#   8xb8             → 8块GPU × 每GPU batch_size=8, 共64样本/步
#   amp              → 开启自动混合精度 (Automatic Mixed Precision) 训练
#   32x2x1           → 采样策略: 每个clip取32帧, 相邻帧间隔2帧, 每个视频取1个clip
#   30e              → 训练30个epoch
#   kinetics400-rgb  → 数据集: Kinetics-400, 使用RGB模态
# =============================================================================


# -----------------------------------------------------------------------------
# 1. 继承基础配置 (_base_)
# -----------------------------------------------------------------------------
# MMAction2 支持配置继承，以下两个基础文件提供了模型结构和运行时默认参数。
# 当前配置文件中的同名字段会覆盖基础配置中的对应值。
_base_ = [
    '../../_base_/models/swin_tiny.py',   # 定义了 SwinTransformer3D backbone +
                                           # ActionDataPreprocessor + I3DHead
                                           # 其中 num_classes=400, in_channels=768
    '../../_base_/default_runtime.py'     # 定义了日志、checkpoint 保存等运行时默认行为
]


# -----------------------------------------------------------------------------
# 2. 模型配置 (model)
# -----------------------------------------------------------------------------
# 这里只覆盖基础配置中需要修改的字段，其余字段从 _base_ 继承。
model = dict(
    backbone=dict(
        # pretrained: 加载预训练的 2D ImageNet 权重路径（URL 或本地路径）。
        # 由于 pretrained2d=True（在 _base_ 中设置），框架会自动将 2D 权重
        # inflate（膨胀）为 3D 权重，用于初始化 3D 时序模型。
        # 设为 None 则从头随机初始化。
        pretrained=  # noqa: E251
        'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_tiny_patch4_window7_224.pth'  # noqa: E501
    ))
# 注意：_base_/models/swin_tiny.py 中完整的模型定义如下（供参考，不在此重复）:
# model = dict(
#     type='Recognizer3D',               # 3D 视频识别器
#     backbone=dict(
#         type='SwinTransformer3D',       # 3D Video Swin Transformer
#         arch='tiny',                   # 模型规模: tiny / small / base / large
#         pretrained=None,               # 此处被上方覆盖
#         pretrained2d=True,             # True: 从2D ImageNet权重inflate; False: 直接加载3D权重
#         patch_size=(2, 4, 4),          # 时空 patch 大小: (T, H, W) = (2帧, 4px, 4px)
#         window_size=(8, 7, 7),         # 注意力窗口: (T, H, W) = (8帧, 7patch, 7patch)
#         mlp_ratio=4.,                  # FFN 中间层维度放大倍数
#         qkv_bias=True,                 # QKV 线性层是否使用偏置
#         drop_rate=0.,                  # Embedding dropout 率
#         attn_drop_rate=0.,             # Attention dropout 率
#         drop_path_rate=0.1),           # Stochastic depth 随机深度丢弃率
#     data_preprocessor=dict(
#         type='ActionDataPreprocessor',
#         mean=[123.675, 116.28, 103.53], # ImageNet RGB 均值 (归一化用)
#         std=[58.395, 57.12, 57.375],    # ImageNet RGB 标准差
#         format_shape='NCTHW'),          # 输入张量格式: N(batch)×C(通道)×T(时间)×H×W
#     cls_head=dict(
#         type='I3DHead',
#         in_channels=768,               # Tiny 版 backbone 输出维度
#         num_classes=400,               # 分类数量，使用自定义数据集时必须修改此项！
#         spatial_type='avg',            # 空间池化方式: avg (平均池化)
#         dropout_ratio=0.5,             # 分类头 dropout 率
#         average_clips='prob'))         # 多 clip 融合方式: 'prob'(概率平均) 或 'score'


# -----------------------------------------------------------------------------
# 3. 数据集设置
# -----------------------------------------------------------------------------

# dataset_type: 数据集类型
#   - 'VideoDataset': 直接读取视频文件（.mp4/.avi 等），推荐用于存储空间有限的场景
#   - 'RawframeDataset': 读取预先解码的图像帧（jpg/png），I/O 更快但占用空间大
#   - 'VideoDatasetV1': 旧版视频数据集（兼容性用）
dataset_type = 'VideoDataset'

# data_root: 训练集视频文件根目录
# 文件列表 (ann_file_train) 中的视频路径是相对于该目录的相对路径
data_root = 'data/kinetics400/videos_train'

# data_root_val: 验证集/测试集视频文件根目录
data_root_val = 'data/kinetics400/videos_val'

# ann_file_train: 训练集标注文件路径
# 文件格式（VideoDataset）: 每行 "<视频相对路径> <类别标签整数>"
# 示例: "basketball/v_basketball_g01_c01.mp4 0"
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'

# ann_file_val: 验证集标注文件路径（格式同上）
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'

# ann_file_test: 测试集标注文件路径（通常与验证集相同）
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'

# file_client_args: 文件读取后端参数
# io_backend='disk': 从本地磁盘读取
# 也支持 'memcached'、'petrel'(云存储) 等分布式存储后端
file_client_args = dict(io_backend='disk')


# -----------------------------------------------------------------------------
# 4. 数据预处理 Pipeline（训练集）
# -----------------------------------------------------------------------------
train_pipeline = [
    # Step1: 初始化视频解码器（Decord 库），解析视频元信息但不立即解码
    # 比 OpenCV 更快，支持随机帧访问
    dict(type='DecordInit', **file_client_args),

    # Step2: 帧采样策略
    #   clip_len=32:       每个 clip 采样 32 帧作为模型输入
    #   frame_interval=2:  相邻采样帧之间跳过1帧（即每隔1帧取1帧），时序跨度更大
    #   num_clips=1:       每个视频随机采样 1 个 clip（训练时随机位置）
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),

    # Step3: 使用 Decord 解码采样到的帧为 numpy 图像数组
    dict(type='DecordDecode'),

    # Step4: 等比缩放，将短边缩放到 256 像素（-1 表示长边按比例自动计算）
    dict(type='Resize', scale=(-1, 256)),

    # Step5: 随机裁剪到 224×224（RandomResizedCrop 先随机选区域再 Resize）
    # 相比直接 RandomCrop，该操作同时引入了尺度和长宽比的随机扰动，增强数据多样性
    dict(type='RandomResizedCrop'),

    # Step6: 将随机裁剪结果 Resize 到精确的 224×224
    # keep_ratio=False: 不保持长宽比，强制拉伸到目标尺寸
    dict(type='Resize', scale=(224, 224), keep_ratio=False),

    # Step7: 随机水平翻转，flip_ratio=0.5 表示 50% 概率翻转（数据增强）
    dict(type='Flip', flip_ratio=0.5),

    # Step8: 将帧列表整理成模型所需的张量格式
    # input_format='NCTHW': N(batch) × C(通道) × T(时间帧) × H(高) × W(宽)
    dict(type='FormatShape', input_format='NCTHW'),

    # Step9: 打包数据，生成 DataSample 对象，供模型 forward() 接收
    dict(type='PackActionInputs')
]


# -----------------------------------------------------------------------------
# 5. 数据预处理 Pipeline（验证集）
# -----------------------------------------------------------------------------
# 验证集不做随机增强，使用确定性的中心裁剪
val_pipeline = [
    dict(type='DecordInit', **file_client_args),

    # test_mode=True: 使用确定性采样（从视频中心位置均匀采样），保证可复现
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,          # 验证时每视频取 1 个 clip（速度快，精度略低）
        test_mode=True),

    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),

    # CenterCrop: 从中心裁剪固定大小区域，替代训练时的随机裁剪
    dict(type='CenterCrop', crop_size=224),

    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]


# -----------------------------------------------------------------------------
# 6. 数据预处理 Pipeline（测试集）
# -----------------------------------------------------------------------------
# 测试时使用多 clip + 多 crop 策略，提升最终精度
test_pipeline = [
    dict(type='DecordInit', **file_client_args),

    # num_clips=4: 从视频中均匀取 4 个时间片段（时序 TTA - 测试时增强）
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        test_mode=True),

    dict(type='DecordDecode'),

    # 缩放到 224（注意测试 Resize 目标比验证时的 256 小）
    dict(type='Resize', scale=(-1, 224)),

    # ThreeCrop: 从左、中、右（或上、中、下）裁剪 3 个 224×224 区域（空间 TTA）
    # 4 clips × 3 crops = 12 个片段，取概率平均作为最终预测
    dict(type='ThreeCrop', crop_size=224),

    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]


# -----------------------------------------------------------------------------
# 7. DataLoader 配置
# -----------------------------------------------------------------------------

# 训练 DataLoader
train_dataloader = dict(
    batch_size=8,              # 每块 GPU 每步处理 8 个视频样本
    num_workers=8,             # 数据预处理的并行进程数（建议设为 CPU 核心数的一半）
    persistent_workers=True,   # workers 在 epoch 结束后保持存活，避免重启开销
    sampler=dict(
        type='DefaultSampler',
        shuffle=True           # 训练时打乱数据顺序
    ),
    dataset=dict(
        type=dataset_type,           # 'VideoDataset'
        ann_file=ann_file_train,     # 训练集标注文件
        data_prefix=dict(
            video=data_root          # 视频文件根目录（与 ann_file 中路径拼接）
        ),
        pipeline=train_pipeline      # 使用训练 pipeline
    ))

# 验证 DataLoader
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False              # 验证/测试时不打乱，保证结果可复现
    ),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True             # 启用测试模式（禁用部分随机增强）
    ))

# 测试 DataLoader
test_dataloader = dict(
    batch_size=1,                  # 测试时 batch_size=1（因为每个视频对应多个片段）
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True
    ))


# -----------------------------------------------------------------------------
# 8. 评估指标
# -----------------------------------------------------------------------------

# val_evaluator: 验证集评估指标
# AccMetric: 计算 Top-1 和 Top-5 分类准确率
val_evaluator = dict(type='AccMetric')

# test_evaluator: 测试集使用相同的评估指标
test_evaluator = val_evaluator


# -----------------------------------------------------------------------------
# 9. 训练/验证/测试循环配置
# -----------------------------------------------------------------------------

train_cfg = dict(
    type='EpochBasedTrainLoop',  # 按 epoch 训练（另有 IterBasedTrainLoop 按步数训练）
    max_epochs=30,               # 总训练 30 个 epoch
    val_begin=1,                 # 从第 1 个 epoch 结束后开始进行验证
    val_interval=3               # 每隔 3 个 epoch 进行一次验证（减少验证开销）
)
val_cfg = dict(type='ValLoop')   # 标准验证循环
test_cfg = dict(type='TestLoop') # 标准测试循环


# -----------------------------------------------------------------------------
# 10. 优化器配置
# -----------------------------------------------------------------------------
optim_wrapper = dict(
    # AmpOptimWrapper: 启用自动混合精度 (AMP/FP16) 训练
    # 在保持精度的同时减少显存占用、加快训练速度
    type='AmpOptimWrapper',

    optimizer=dict(
        type='AdamW',          # AdamW 优化器（Adam + 权重衰减解耦）
        lr=1e-3,               # 全局基础学习率（但 backbone 会乘以 lr_mult=0.1）
        betas=(0.9, 0.999),    # Adam 的一阶/二阶动量系数
        weight_decay=0.02      # L2 权重衰减系数，防止过拟合
    ),

    # SwinOptimWrapperConstructor: Swin 专用优化器构造器
    # 支持对不同层设置不同的学习率和权重衰减
    constructor='SwinOptimWrapperConstructor',

    paramwise_cfg=dict(
        # 以下特殊层不施加权重衰减（decay_mult=0.），保持其数值稳定
        absolute_pos_embed=dict(decay_mult=0.),            # 绝对位置编码
        relative_position_bias_table=dict(decay_mult=0.),  # 相对位置偏置表
        norm=dict(decay_mult=0.),                          # LayerNorm/BatchNorm 层

        # backbone 整体学习率为全局 lr 的 0.1 倍（精调时 backbone 学习率应更小）
        backbone=dict(lr_mult=0.1)
    ))


# -----------------------------------------------------------------------------
# 11. 学习率调度策略
# -----------------------------------------------------------------------------
param_scheduler = [
    # 阶段1: 线性 Warmup（从 lr×0.1 线性增长到 lr）
    # 避免训练初期梯度不稳定导致模型发散
    dict(
        type='LinearLR',
        start_factor=0.1,              # 起始学习率 = lr × 0.1
        by_epoch=True,                 # 以 epoch 为单位
        begin=0,
        end=2.5,                       # 前 2.5 个 epoch 线性增长
        convert_to_iter_based=True     # 转换为按迭代步数执行，使增长更平滑
    ),

    # 阶段2: Cosine Annealing（从 lr 余弦衰减到 0）
    # 在整个训练过程中平滑地降低学习率，有助于收敛到更好的极值
    dict(
        type='CosineAnnealingLR',
        T_max=30,                      # 余弦半周期 = 总 epoch 数
        eta_min=0,                     # 最终学习率下降到 0
        by_epoch=True,
        begin=0,
        end=30
    )
]


# -----------------------------------------------------------------------------
# 12. 默认钩子配置（覆盖 _base_/default_runtime.py 中的默认值）
# -----------------------------------------------------------------------------
default_hooks = dict(
    # checkpoint: 保存策略
    #   interval=3: 每 3 个 epoch 保存一次 checkpoint
    #   max_keep_ckpts=5: 最多保留最近 5 个 checkpoint，自动删除旧的
    checkpoint=dict(interval=3, max_keep_ckpts=5),

    # logger: 日志打印策略
    #   interval=100: 每 100 个迭代步打印一次训练日志
    logger=dict(interval=100)
)


# -----------------------------------------------------------------------------
# 13. 自动缩放学习率
# -----------------------------------------------------------------------------
auto_scale_lr = dict(
    # enable=False: 默认关闭自动学习率缩放
    # 若开启，框架会根据实际 batch_size 与 base_batch_size 的比值自动调整 lr
    # 公式: actual_lr = lr × (actual_batch_size / base_batch_size)
    enable=False,

    # base_batch_size: 配置文件中 lr 对应的参考 batch_size
    # 本配置: 8 GPUs × 8 samples/GPU = 64
    base_batch_size=64
)
