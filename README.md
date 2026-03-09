# Skeleton-based Action Recognition with ST-GCN

基于骨架的行为识别系统，使用 **ST-GCN（Spatial-Temporal Graph Convolutional Network）** 对 OpenPose 3D 骨架序列进行多类别动作分类。

---

## 目录

- [项目结构](#项目结构)
- [数据导入说明](#数据导入说明)
- [快速开始](#快速开始)
- [评估指标说明](#评估指标说明)
- [模型架构](#模型架构)
- [训练配置](#训练配置)

---

## 项目结构

```
pose_estimation/
│
├── activity_label/                        # 活动标签文件
│   ├── tasklevel.chunks_90.csv            # 全量标签（所有参与者，8479条）
│   ├── tasklevel.chunks_90.split_0.train.csv
│   ├── tasklevel.chunks_90.split_0.val.csv
│   ├── tasklevel.chunks_90.split_0.test.csv
│   ├── tasklevel.chunks_90.split_1.*.csv  # 交叉验证 Split 1
│   └── tasklevel.chunks_90.split_2.*.csv  # 交叉验证 Split 2
│
├── pose_all/                              # OpenPose 3D 骨架数据（按需添加）
│   ├── run1b_2018-05-29-14-02-47.ids_1.openpose.3d.csv   # vp1 run1
│   ├── run2_2018-05-29-14-33-44.ids_1.openpose.3d.csv    # vp1 run2
│   ├── run1_2018-05-03-14-08-31.ids_1.openpose.3d.csv    # vp2 run1
│   └── run2_2018-05-24-17-22-26.ids_1.openpose.3d.csv    # vp2 run2
│
├── stgcn/                                 # 模型核心代码包
│   ├── __init__.py                        # 公开导出接口
│   ├── graph.py                           # OpenPose 25关节骨架图定义
│   ├── model.py                           # ST-GCN 官方9层模型
│   └── dataset.py                         # 数据集加载与预处理
│
├── train.py                               # 训练、验证、测试主脚本
├── requirements.txt                       # Python 依赖
└── output/                                # 训练输出（自动生成，不提交）
    ├── best_model.pth                     # 验证集最优模型权重
    ├── last_model.pth                     # 最后一个 epoch 权重
    ├── training_history.json              # 每 epoch 训练曲线
    └── test_results.json                  # 测试集最终评估结果
```

---

## 数据导入说明

### 1. 标签文件格式（activity_label/）

每个 CSV 文件包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `participant_id` | int | 参与者编号（1=vp1, 2=vp2, ...） |
| `file_id` | str | 对应骨架文件的逻辑ID，格式为 `vpX/run_name.ids_1` |
| `annotation_id` | int | 动作标注序号 |
| `frame_start` | int | 片段起始帧（含） |
| `frame_end` | int | 片段结束帧（不含） |
| `activity` | str | 动作类别名称 |
| `chunk_id` | int | 90帧分块的序号 |

**示例：**
```
participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id
1,vp1/run1b_2018-05-29-14-02-47.ids_1,0,0,90,fasten_seat_belt,0
1,vp1/run1b_2018-05-29-14-02-47.ids_1,0,90,180,fasten_seat_belt,1
```

### 2. 骨架文件格式（pose_all/）

每个 CSV 对应一次采集 session 的完整骨架时序，行为帧，列为关节坐标：

| 列名 | 说明 |
|------|------|
| `frame_id` | 帧编号（从0开始） |
| `timestamp` | Unix 时间戳（秒） |
| `{joint}_x` | 关节 X 坐标（米，世界坐标系） |
| `{joint}_y` | 关节 Y 坐标 |
| `{joint}_z` | 关节 Z 坐标（深度） |
| `{joint}_p` | 关节检测置信度（0~1） |

共 25 个关节（OpenPose Body-25，排除 background），每个关节 4 列，加 `frame_id` 和 `timestamp` 共 **102 列**。

25 个关节名称：
```
nose, neck, rShoulder, rElbow, rWrist,
lShoulder, lElbow, lWrist, rHip, rKnee,
rAnkle, lHip, lKnee, lAnkle, rEye,
lEye, rEar, lEar, lBigToe, lSmallToe,
lHeel, rBigToe, rSmallToe, rHeel, midHip
```

### 3. 如何添加新参与者数据

代码会**自动发现** `pose_all/` 目录下所有 `*.openpose.3d.csv` 文件，并通过文件名与标签 `file_id` 自动匹配，无需修改任何代码。

**匹配规则：**

| 标签中的 file_id | 对应骨架文件名 |
|-----------------|--------------|
| `vp3/run1_2018-xx-xx.ids_1` | `run1_2018-xx-xx.ids_1.openpose.3d.csv` |

规则：去掉 `vpX/` 前缀后，加 `.openpose.3d.csv` 后缀即为文件名。

**添加步骤：**

```bash
# 1. 将新的骨架 CSV 放入 pose_all/
cp run1_xxx.ids_1.openpose.3d.csv /workspaces/pose_estimation/pose_all/

# 2. 确认 activity_label/tasklevel.chunks_90.csv 中有对应的标签行
#    （文件中已包含 vp1~vp15 全部标签，只需有对应 pose CSV 即可）

# 3. 直接运行训练，程序自动加载所有匹配数据
python train.py
```

### 4. 当前数据统计（pose_all/ 中 4 个 CSV）

| 参与者 | 动作片段数 |
|--------|-----------|
| vp1（run1b + run2） | ~582 |
| vp2（run1 + run2） | ~567 |
| **合计** | **1149** |

**各类别样本分布：**

| 动作类别 | 样本数 | 占比 |
|----------|--------|------|
| fasten_seat_belt | 30 | 2.6% |
| hand_over | 16 | 1.4% |
| work | 128 | 11.1% |
| eat_drink | 133 | 11.6% |
| read_write_newspaper | 136 | 11.8% |
| read_write_magazine | 154 | 13.4% |
| **watch_video** | **367** | **31.9%** |
| put_on_jacket | 60 | 5.2% |
| take_off_jacket | 37 | 3.2% |
| put_on_sunglasses | 32 | 2.8% |
| take_off_sunglasses | 26 | 2.3% |
| final_task | 30 | 2.6% |

> ⚠️ 数据存在明显类别不平衡（watch_video 367 vs hand_over 16），建议关注少数类的 F1-macro 值。

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行训练

```bash
# 默认参数（推荐 CPU 快速验证）
python train.py

# GPU 服务器推荐配置
python train.py --epochs 80 --lr 0.01 --batch_size 32 --graph_strategy spatial --num_workers 4
```

### 所有命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 80 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 0.01 | 初始学习率 |
| `--weight_decay` | 1e-4 | L2 正则化系数 |
| `--dropout` | 0.0 | Dropout 概率 |
| `--max_frames` | 90 | 时序窗口长度（帧数）|
| `--graph_strategy` | spatial | 邻接矩阵策略：`uniform` / `distance` / `spatial` |
| `--num_workers` | 4 | DataLoader 工作进程数（Windows 建议设为 0）|
| `--seed` | 42 | 随机种子 |

---

## 评估指标说明

训练结束后，程序在测试集上输出以下指标：

### 1. Accuracy（准确率）

$$\text{Accuracy} = \frac{\text{预测正确的样本数}}{\text{总样本数}}$$

- **用途**：整体表现概览
- **局限**：类别不平衡时会高估性能（大类主导分母）

### 2. Precision（精确率，macro 宏平均）

$$\text{Precision}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c}$$

- 每类先单独计算精确率，再取算术平均
- **含义**：预测为某类时，真正属于该类的比例（误报率低则精确率高）

### 3. Recall（召回率，macro 宏平均）

$$\text{Recall}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}$$

- **含义**：某类的真实样本中，被正确预测到的比例（漏报率低则召回率高）

### 4. F1-macro（宏平均 F1，主要指标）

$$\text{F1}_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

- **推荐主指标**：所有类别权重相同，对少数类更敏感，适合不平衡数据集
- 模型保存策略：**验证集 F1-macro 最高**时保存 `best_model.pth`

### 5. F1-weighted（加权 F1，辅助指标）

$$\text{F1}_{\text{weighted}} = \sum_{c=1}^{C} \frac{n_c}{N} \cdot \text{F1}_c$$

- 按真实样本数加权，大类贡献更多
- 反映整体性能，但受多数类（watch_video）影响较大

### 6. Confusion Matrix（混淆矩阵）

行为**真实类别**，列为**预测类别**，对角线为正确预测数：

```
            fasten_  hand_ov     work  eat_dri  ...
fasten_          5        0        0        0
hand_ov          0        2        0        1
work             0        0       20        3
...
```

非对角线元素揭示哪些类别容易被混淆。

### 7. Classification Report（每类详细报告）

```
                      precision  recall  f1-score  support
 fasten_seat_belt         0.83    0.71      0.77        7
        hand_over         0.67    0.67      0.67        3
             work         0.82    0.87      0.84       30
             ...
        macro avg         0.74    0.72      0.73      173
     weighted avg         0.81    0.80      0.80      173
```

### 指标选用建议

| 场景 | 推荐指标 |
|------|----------|
| 类别均衡 | Accuracy 或 F1-macro |
| 类别不平衡（本项目）| **F1-macro**（主），F1-weighted（辅） |
| 关注最差类表现 | Classification Report 中最低 F1 的类别 |
| 模型选择标准 | 验证集 F1-macro 最高 → `best_model.pth` |

---

## 模型架构

**ST-GCN（Spatial-Temporal Graph Convolutional Network）** — 官方 9 层架构，约 3.13M 参数。

```
输入张量: (N, 3, 90, 25)
          batch × (x,y,z) × 帧数 × 关节数
  ↓
BatchNorm1d(75)
  ↓
Layer 1: STGCNBlock(3   → 64,  stride=1, no residual)
Layer 2: STGCNBlock(64  → 64,  stride=1)
Layer 3: STGCNBlock(64  → 64,  stride=1)
Layer 4: STGCNBlock(64  → 128, stride=2)   ← 时序下采样 ×0.5
Layer 5: STGCNBlock(128 → 128, stride=1)
Layer 6: STGCNBlock(128 → 128, stride=1)
Layer 7: STGCNBlock(128 → 256, stride=2)   ← 时序下采样 ×0.5
Layer 8: STGCNBlock(256 → 256, stride=1)
Layer 9: STGCNBlock(256 → 256, stride=1)
  ↓
Global Average Pooling
  ↓
Linear(256 → 12)
  ↓
输出: (N, 12) logits
```

每个 STGCNBlock = 空间图卷积（GCN）+ 时序卷积（TCN），包含可学习边重要性权重 **M**。

**graph_strategy 选项：**

| 策略 | 分区数 K | 说明 |
|------|---------|------|
| `uniform` | 1 | 所有邻居权重相同 |
| `distance` | 2 | 自身节点 vs 1-hop 邻居 |
| `spatial` | 3 | 自身 / 向心（靠近质心）/ 离心（远离质心）|

---

## 训练配置

| 项目 | 值 |
|------|-----|
| 优化器 | SGD + Nesterov（momentum=0.9）|
| 初始学习率 | 0.01 |
| 学习率衰减 | MultiStepLR，epoch [30, 50, 70] 各乘以 0.1 |
| 权重衰减 | 1e-4 |
| 数据划分 | 70% train / 15% val / 15% test（seed=42）|
| 训练增强 | 随机高斯噪声(σ=0.01) + 随机缩放([0.9,1.1]) + 随机时序偏移 |
| 模型保存 | 验证集 F1-macro 最高时保存 `output/best_model.pth` |
| 推荐 GPU | NVIDIA RTX 3090（24GB），约 80 epoch 数分钟完成 |