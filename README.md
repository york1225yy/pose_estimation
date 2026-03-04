# 驾驶员坐姿估计 - Pose Estimation

基于 **YOLOv8-Pose** 和 **YOLO11-Pose** 的驾驶员坐姿识别系统，通过骨骼关键点检测 + 规则引擎实现 10 类坐姿分类。

---

## 项目结构

```
pose_estimation/
├── README.md
├── requirements.txt
├── docs/
│   └── 算法调研报告.md          # 算法调研文档
├── utils/                       # 共享工具模块
│   ├── __init__.py
│   ├── posture_rules.py         # 坐姿分类规则引擎（10 类）
│   └── visualization.py         # 骨骼可视化 & 标注
├── yolov8_pose/                 # YOLOv8-Pose 方案
│   ├── infer.py                 # 推理脚本（图片/文件夹/视频）
│   └── export_onnx.py           # ONNX 导出脚本
└── yolo11_pose/                 # YOLO11-Pose 方案
    ├── infer.py                 # 推理脚本（图片/文件夹/视频）
    └── export_onnx.py           # ONNX 导出脚本
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 1. YOLOv8-Pose 推理

```bash
# 单张图片
cd yolov8_pose
python infer.py --source ../test.jpg

# 文件夹（批量）
python infer.py --source ../images/

# 视频
python infer.py --source ../video.mp4

# 指定模型 & 设备
python infer.py --source ../video.mp4 --model yolov8m-pose.pt --device cuda:0

# 无界面模式（服务器/无显示器环境）
python infer.py --source ../video.mp4 --no-show
```

### 2. YOLO11-Pose 推理

```bash
cd yolo11_pose
python infer.py --source ../test.jpg
python infer.py --source ../video.mp4 --model yolo11s-pose.pt
```

### 3. 导出 ONNX（用于 TDA4 部署）

```bash
# YOLOv8
cd yolov8_pose
python export_onnx.py --model yolov8s-pose.pt --imgsz 640 --opset 11 --simplify

# YOLO11
cd yolo11_pose
python export_onnx.py --model yolo11s-pose.pt --imgsz 640 --opset 11 --simplify
```

---

## 输出说明

| 输出 | 说明 |
|------|------|
| `output/result_*.jpg` | 带骨骼 + 坐姿标注的结果图片 |
| `output/result_*.mp4` | 带骨骼 + 坐姿标注的结果视频 |
| `output/posture_log.csv` | 逐帧参数日志（CSV） |
| 控制台 | 每帧实时打印所有几何参数 |

### CSV 日志字段

| 字段 | 含义 |
|------|------|
| 帧序号 | 视频帧编号 |
| 人序号 | 多人时的检测编号 |
| 躯干前倾角(度) | 肩-髋连线与垂直方向夹角 |
| 头前倾角(度) | 耳-肩连线与躯干轴夹角 |
| 头侧倾角(度) | 双耳连线偏离水平的角度 |
| 躯干旋转角(度) | 双肩水平投影比 arccos |
| 肩膀抬起量(归一化) | 双肩Y坐标差 / 肩距 |
| 判定坐姿 | 10 类坐姿之一 |

---

## 10 类坐姿说明

| # | 坐姿 | 触发条件 |
|---|------|----------|
| 1 | 标准驾驶坐姿 | 默认（无异常触发） |
| 2 | 轻微前倾 | 躯干前倾 8°~20° |
| 3 | 严重前倾 | 躯干前倾 > 20° |
| 4 | 后仰放松 | 头后仰 10°~25°，躯干 < 8° |
| 5 | 侧身取物 | 躯干旋转 > 15° + 肩膀抬起 > 20% |
| 6 | 半躺休息 | 靠背 ≥ 120° 或 后仰 > 35° |
| 7 | 低头玩手机 | 头前倾 > 25° |
| 8 | 睡姿 | 头侧倾 > 35° |
| 9 | 斜躺副驾 | 半躺 + 侧向倾斜 > 0.15 |
| 10 | 直立办公 | 躯干 < 5° + 手臂抬起 |

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source` | 必填 | 输入源(图片/文件夹/视频) |
| `--model` | `yolov8s-pose.pt` / `yolo11s-pose.pt` | 模型权重 |
| `--device` | 自动 | 推理设备 |
| `--conf` | 0.5 | 检测置信度阈值 |
| `--kp-conf` | 0.4 | 关键点置信度阈值 |
| `--imgsz` | 640 | 推理尺寸 |
| `--output` | `output` | 输出目录 |
| `--no-show` | False | 无界面模式 |
| `--no-params` | False | 不叠加参数面板 |
| `--save-csv` | True | 保存 CSV 日志 |

---

## TDA4 部署流程

1. 导出 ONNX（opset 11, 静态 batch）
2. 使用 TI TIDL 工具链转换为 TDA4 可执行格式
3. 在 TDA4 上用 TIDL Runtime / ONNX Runtime 推理
4. 后处理（坐姿规则引擎）在 ARM A72 上运行

详见 [docs/算法调研报告.md](docs/算法调研报告.md) 第 7 节。