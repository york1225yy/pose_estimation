# UCF24 数据格式说明 & 自定义数据集训练指南

本文档面向希望了解 UCF24 数据格式、或使用自采数据训练 YOWOv3 的用户。

---

## 一、UCF24 数据集目录结构

```
ucf24/
├── rgb-images/                  # 视频帧图像（解帧后的 JPEG）
│   ├── Basketball/
│   │   ├── v_Basketball_g01_c01/
│   │   │   ├── 00001.jpg
│   │   │   ├── 00002.jpg
│   │   │   └── ...
│   │   └── v_Basketball_g01_c02/
│   │       └── ...
│   ├── Biking/
│   └── ... (共 24 个动作类别文件夹)
│
├── labels/                      # 标注文件（与 rgb-images 目录结构完全对应）
│   ├── Basketball/
│   │   ├── v_Basketball_g01_c01/
│   │   │   ├── 00001.txt        # 每帧一个 .txt 文件（仅标注有人的帧）
│   │   │   ├── 00005.txt
│   │   │   └── ...
│   │   └── ...
│   └── ...
│
├── trainlist.txt                # 训练集分割文件
└── testlist.txt                 # 测试集分割文件
```

---

## 二、标注文件格式（labels/.../*.txt）

每个 `.txt` 文件对应视频中**一帧**的标注，文件中每行代表该帧中**一个人**的动作标注：

```
<class_id> <x1> <y1> <x2> <y2>
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `class_id` | 整数 | 动作类别 ID，**从 1 开始**（代码读取时会自动 `-1` 变成从 0 开始） |
| `x1` | 浮点数 | 人体框左上角 x 坐标（**像素值，非归一化**） |
| `y1` | 浮点数 | 人体框左上角 y 坐标（像素值） |
| `x2` | 浮点数 | 人体框右下角 x 坐标（像素值） |
| `y2` | 浮点数 | 人体框右下角 y 坐标（像素值） |

> **注意**：原始 UCF24 中坐标为**像素绝对值**，代码在 `transforms.py` 的 `UCF_transform.__call__` 中统一除以宽高完成归一化：
> ```python
> targets[:, :4] /= np.array([W, H, W, H])
> ```

**示例** — `labels/Basketball/v_Basketball_g08_c01/00070.txt`：
```
1 105.0 82.0 298.0 365.0
1 312.0 90.0 480.0 360.0
```
表示第 70 帧中有两个人，均在做 Basketball（类别 1）动作。

---

## 三、分割文件格式（trainlist.txt / testlist.txt）

每行是一个**关键帧**的标注路径（相对于 `ucf24/` 根目录），格式为：

```
labels/<类别名>/<视频名>/<帧编号>.txt
```

**示例**（`trainlist.txt` 前几行）：
```
labels/Basketball/v_Basketball_g08_c01/00070.txt
labels/Basketball/v_Basketball_g08_c01/00071.txt
labels/Biking/v_Biking_g01_c01/00016.txt
...
```

代码读取分割文件后，会：
1. 解析出 `class_name`、`video_name`、`key_frame_idx`
2. 从 `rgb-images/<class_name>/<video_name>/` 读取以关键帧为结尾的 **16 帧**图像组成 clip
3. 读取对应 `labels/` 下的 `.txt` 标注

---

## 四、UCF24 的 24 个动作类别

| ID（从 1 起） | 类别名 | ID（从 1 起） | 类别名 |
|:---:|---|:---:|---|
| 1 | Basketball | 13 | PoleVault |
| 2 | BasketballDunk | 14 | RopeClimbing |
| 3 | Biking | 15 | SalsaSpin |
| 4 | CliffDiving | 16 | SkateBoarding |
| 5 | CricketBowling | 17 | Skiing |
| 6 | Diving | 18 | Skijet |
| 7 | Fencing | 19 | SoccerJuggling |
| 8 | FloorGymnastics | 20 | Surfing |
| 9 | GolfSwing | 21 | TennisSwing |
| 10 | HorseRiding | 22 | TrampolineJumping |
| 11 | IceDancing | 23 | VolleyballSpiking |
| 12 | LongJump | 24 | WalkingWithDog |

---

## 五、自定义数据集所需格式

### 5.1 原始采集要求

| 项目 | 要求 |
|------|------|
| 视频格式 | MP4、AVI 等主流格式均可，需能被 `ffmpeg` / `cv2` 读取 |
| 帧率 | 建议 ≥ 15 fps（模型默认读取 16 帧 clip，`sampling_rate=1` 即连续帧） |
| 分辨率 | 无严格要求，最终会 resize 到 `img_size`（默认 224×224） |
| 标注内容 | 每帧中每个人的**人体边界框**（x1,y1,x2,y2 像素值）+ **动作类别 ID** |

### 5.2 目标目录结构（需整理成以下格式）

```
my_dataset/
├── rgb-images/
│   └── <action_class>/
│       └── <video_name>/
│           ├── 00001.jpg
│           ├── 00002.jpg
│           └── ...
├── labels/
│   └── <action_class>/
│       └── <video_name>/
│           ├── 00001.txt   ← 仅需为"有标注的帧"创建
│           └── ...
├── trainlist.txt
└── testlist.txt
```

---

## 六、数据转换步骤（视频 → 训练所需格式）

### 步骤 1：视频解帧

将原始视频解帧为 JPEG 图片，保存到 `rgb-images/<class>/<video_name>/` 下：

```bash
# 以 ffmpeg 批量解帧为例
# 假设原始视频为 videos/Basketball/v_Basketball_my01.mp4
mkdir -p my_dataset/rgb-images/Basketball/v_Basketball_my01
ffmpeg -i videos/Basketball/v_Basketball_my01.mp4 \
       -q:v 2 \
       my_dataset/rgb-images/Basketball/v_Basketball_my01/%05d.jpg
```

> 帧编号需为**5位零填充**（`00001.jpg`），与代码中 `'{:05d}.jpg'.format(idx)` 一致。

### 步骤 2：标注帧

对需要参与训练的帧，在对应路径创建 `.txt` 标注文件，格式参照第二节：

```
<class_id> <x1_pixel> <y1_pixel> <x2_pixel> <y2_pixel>
```

**工具推荐**：
- [LabelImg](https://github.com/HumanSignal/labelImg)（支持导出 YOLO 格式，需转换坐标格式）
- [CVAT](https://cvat.org/)（支持视频标注，可导出多种格式）

> **坐标格式转换**（YOLO 格式 → UCF24 格式）：
> YOLO 格式为归一化中心点 `(cx, cy, w, h)`，需转换为像素左上右下坐标：
> ```python
> x1 = (cx - w/2) * img_W
> y1 = (cy - h/2) * img_H
> x2 = (cx + w/2) * img_W
> y2 = (cy + h/2) * img_H
> ```

### 步骤 3：生成 trainlist.txt / testlist.txt

按以下格式写入分割文件（路径相对于数据集根目录）：

```python
import os

root = "my_dataset"
output_train = open(os.path.join(root, "trainlist.txt"), "w")
output_test  = open(os.path.join(root, "testlist.txt"),  "w")

labels_dir = os.path.join(root, "labels")
for class_name in sorted(os.listdir(labels_dir)):
    class_dir = os.path.join(labels_dir, class_name)
    all_videos = sorted(os.listdir(class_dir))
    
    # 例：按视频名字母序，前 80% 训练，后 20% 测试
    split_idx = int(len(all_videos) * 0.8)
    train_videos = all_videos[:split_idx]
    test_videos  = all_videos[split_idx:]
    
    for video_name in train_videos:
        video_dir = os.path.join(class_dir, video_name)
        for txt_file in sorted(os.listdir(video_dir)):
            line = f"labels/{class_name}/{video_name}/{txt_file}\n"
            output_train.write(line)
    
    for video_name in test_videos:
        video_dir = os.path.join(class_dir, video_name)
        for txt_file in sorted(os.listdir(video_dir)):
            line = f"labels/{class_name}/{video_name}/{txt_file}\n"
            output_test.write(line)

output_train.close()
output_test.close()
print("Done!")
```

### 步骤 4：修改 config 文件

将原有 `ucf_config.yaml` 复制后修改以下字段：

```yaml
dataset       : ucf          # 保持不变，复用 UCF 数据加载器
num_classes   : <你的类别数>  # 修改为实际类别数（例如 5）
data_root     : /path/to/my_dataset
pretrain_path : null          # 从头训练；若迁移学习则填写 checkpoint 路径

idx2name:
  0 : class_A
  1 : class_B
  ...
```

> **注意**：`num_classes` 必须与 `idx2name` 的条目数一致。

### 步骤 5：开始训练

```bash
cd YOWOv3
python main.py --mode train --config config/my_dataset_config.yaml
```

---

## 七、常见问题

**Q：标注时类别 ID 从 0 还是 1 开始？**  
A：`.txt` 标注文件中从 **1 开始**，代码读取时会自动 `-1`：
```python
label = int(line[0]) - 1  # load_data.py line 73
```

**Q：不是每一帧都需要标注吗？**  
A：不需要。只需在 `trainlist.txt` / `testlist.txt` 中列出**有标注的帧**对应的 `.txt` 路径即可。没有出现在分割文件中的帧不会被加载。

**Q：一帧中有多个人怎么办？**  
A：在同一个 `.txt` 文件中写多行，每行一个人：
```
1 100 50 300 400
2 350 60 520 390
```

**Q：clip 的 16 帧如何确定？**  
A：以 `trainlist.txt` 中每行的帧为**关键帧（最后一帧）**，向前取 16 帧（`sampling_rate=1` 为连续帧）。若前向帧不足，则重复使用第 1 帧填充。

**Q：能否用于多标签（一人同时做多个动作）？**  
A：UCF24 数据加载器支持多标签训练（训练时 label 为 one-hot 向量），但评估时按单标签处理。每行只需一个类别 ID 即可满足单标签场景。
