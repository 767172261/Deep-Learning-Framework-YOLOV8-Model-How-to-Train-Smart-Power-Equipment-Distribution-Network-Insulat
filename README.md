电力设备 **电力设备配网绝缘导线烧伤检测图像数据集** 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5b1f84b6b2c64826b72b96213dafeaf4.png)

| 属性 | 详情 |
|------|------|
| **数据集名称** | 配网绝缘导线烧伤检测数据集（Distribution Insulated Conductor Burn Detection Dataset） |
| **总图像数量** | 1,500+ 张 |
| **图像内容** | 户外配电线路中绝缘导线的局部特写，包含不同程度的烧伤、熔融、碳化、鼓包等缺陷 |
| **拍摄环境** | 多光照条件（晴天/阴天/逆光）、多角度（地面手持、无人机航拍）、多季节背景 |
| **标注格式** | PASCAL VOC（`.xml`），可一键转换为 YOLO（`.txt`）格式 |
| **任务类型** | 目标检测（定位烧伤区域）或图像分类（按损伤等级） |
| **类别数量** | 1 类（默认）或可扩展为多级损伤分类 |
| **典型类别标签** | `burn_damage`（烧伤缺陷）<br>（部分数据集可能细分为：`mild_burn`, `severe_burn`, `melted`, `carbonized` 等） |
| **标注质量** | 边界框精准覆盖烧伤区域，由电力运维专家审核 |
| **适用模型** | YOLOv5/v8/v10、Faster R-CNN、SSD 等目标检测框架 |
| **应用场景** | 电网智能巡检、无人机自动诊断、配网设备健康评估、预防性维护预警 |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5932baa4f20f43f49ee2838c31d5b0de.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d03081e41f6446c6a5c305d97912ae17.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/712ade229d9b4dae9e09c10b6a8a34f4.png)
：

✅ VOC → YOLO 格式转换脚本  
✅ YOLOv8 训练代码（支持单类/多类）  
✅ 模型验证与推理示例  
✅ 部署建议  

---

## 📁 一、数据目录结构（原始 VOC）

```
insulation_burn_dataset/
├── JPEGImages/          # 所有图像 (.jpg)
└── Annotations/         # 对应标注 (.xml)
```

> 假设类别为单类：`burn_damage`（若为多类，请确保 XML 中 `<name>` 字段一致）

---

## 🔁 二、VOC 转 YOLO 格式脚本（`voc2yolo.py`）

```python
# voc2yolo.py
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 配置
CLASS_NAMES = ['burn_damage']  # 单类；若多类，请按顺序列出，如 ['mild', 'severe']
DATASET_PATH = 'insulation_burn_dataset'
OUTPUT_PATH = 'insulation_burn_yolo'

os.makedirs(f'{OUTPUT_PATH}/images/train', exist_ok=True)
os.makedirs(f'{OUTPUT_PATH}/labels/train', exist_ok=True)

# 获取所有 XML 文件
xml_files = [f for f in os.listdir(f'{DATASET_PATH}/Annotations') if f.endswith('.xml')]

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

for xml_file in tqdm(xml_files, desc="转换 VOC → YOLO"):
    tree = ET.parse(f'{DATASET_PATH}/Annotations/{xml_file}')
    root = tree.getroot()
    
    image_name = root.find('filename').text
    img_path = f'{DATASET_PATH}/JPEGImages/{image_name}'
    
    # 复制图像到新目录
    import shutil
    shutil.copy(img_path, f'{OUTPUT_PATH}/images/train/{image_name}')
    
    # 获取图像尺寸
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    # 写入 YOLO 标签文件
    label_file = open(f'{OUTPUT_PATH}/labels/train/{xml_file.replace(".xml", ".txt")}', 'w')
    
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASS_NAMES:
            continue
        cls_id = CLASS_NAMES.index(cls_name)
        
        xmlbox = obj.find('bndbox')
        b = (
            float(xmlbox.find('xmin').text),
            float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymin').text),
            float(xmlbox.find('ymax').text)
        )
        bb = convert_bbox((w, h), b)
        label_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")
    
    label_file.close()

print("✅ VOC → YOLO 转换完成！")
```

> ✅ 运行后生成：
> ```
> insulation_burn_yolo/
> ├── images/train/
> └── labels/train/
> ```

---

## 📄 三、划分训练/验证集（可选）

由于数据量较小（1500+），建议手动划分或使用以下脚本：

```python
# split_train_val.py
import os
import random
import shutil

IMG_DIR = 'insulation_burn_yolo/images/train'
LBL_DIR = 'insulation_burn_yolo/labels/train'
OUT_DIR = 'insulation_burn_yolo'

os.makedirs(f'{OUT_DIR}/images/val', exist_ok=True)
os.makedirs(f'{OUT_DIR}/labels/val', exist_ok=True)

files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
random.seed(42)
random.shuffle(files)
val_files = files[:300]  # 约 20% 作为验证集

for f in val_files:
    # 移动图像
    shutil.move(f'{IMG_DIR}/{f}', f'{OUT_DIR}/images/val/{f}')
    # 移动标签
    lbl = f.replace('.jpg', '.txt')
    shutil.move(f'{LBL_DIR}/{lbl}', f'{OUT_DIR}/labels/val/{lbl}')

print(f"✅ 划分完成：训练集 {len(files)-300} 张，验证集 300 张")
```

---

## 📄 四、配置文件 `burn_defect.yaml`

```yaml
path: ./insulation_burn_yolo
train: images/train
val: images/val

nc: 1
names: ['burn_damage']
```

> 若为多类（如 3 类），则：
> ```yaml
> nc: 3
> names: ['mild_burn', 'severe_burn', 'melted']
> ```

---

## 🚀 五、YOLOv8 训练代码（`train_burn.py`）

```python
# train_burn.py
from ultralytics import YOLO

# 1. 加载预训练模型（推荐 yolov8s）
model = YOLO('yolov8s.pt')

# 2. 开始训练
results = model.train(
    data='burn_defect.yaml',
    epochs=150,
    imgsz=640,               # 导线烧伤区域通常较小，640 足够
    batch=16,                # 根据 GPU 显存调整（16GB 可设 32）
    name='burn_defect_yolov8s',
    device=0,
    workers=8,
    patience=30,
    save=True,
    exist_ok=True,

    # 数据增强（提升小缺陷识别）
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.6,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.001,
    fliplr=0.5,
    mosaic=1.0,              # 合成多图提升小目标召回
    mixup=0.1,
    copy_paste=0.3           # 对稀有烧伤样本增强
)

print("✅ 训练完成！最佳模型路径：", results.save_dir / 'weights' / 'best.pt')
```

---

## 🔍 六、模型验证与推理

### 1. 验证性能
```python
from ultralytics import YOLO
model = YOLO('runs/detect/burn_defect_yolov8s/weights/best.pt')
metrics = model.val(data='burn_defect.yaml')
print(f"mAP@0.5: {metrics.box.map50:.4f}")
```

### 2. 单图推理
```python
results = model('test_wire.jpg', conf=0.25)
annotated = results[0].plot()
cv2.imwrite('burn_result.jpg', annotated)
```

### 3. 视频流检测（无人机巡检）
```python
cap = cv2.VideoCapture('drone_inspection.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    results = model(frame, conf=0.2)
    annotated = results[0].plot()
    cv2.imshow('Burn Detection', annotated)
    if cv2.waitKey(1) == ord('q'): break
cap.release()
cv2.destroyAllWindows()
```

---


