import json
import os
import glob

# 1. 严格对应你的标注类别，顺序不能错
# 索引 0: trash_can, 索引 1: fire_hydrant
classes = ["trash_can", "fire_hydrant","manhole_cover","air_conditioner"]


def convert_coordinates(size, box):
    """将绝对像素坐标转换为YOLO要求的相对中心点归一化坐标"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh


# 2. 设置路径（根据你的项目实际目录调整）
json_dir = './datasets/images/train'  # 你放10个json的文件夹
label_dir = './datasets/labels/train'  # 转换后txt存放的文件夹

if not os.path.exists(label_dir):
    os.makedirs(label_dir)

# 3. 遍历转换
json_files = glob.glob(os.path.join(json_dir, "*.json"))
print(f"检测到 {len(json_files)} 个标注文件，开始转换...")

for json_path in json_files:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        img_w = data['imageWidth']
        img_h = data['imageHeight']

        txt_name = os.path.basename(json_path).replace(".json", ".txt")
        txt_path = os.path.join(label_dir, txt_name)

        with open(txt_path, 'w') as out_f:
            for shape in data['shapes']:
                label = shape['label']
                if label not in classes:
                    continue

                cls_id = classes.index(label)
                points = shape['points']
                # Labelme矩形框通常保存为两个点: [[x1, y1], [x2, y2]]
                x1, y1 = points[0]
                x2, y2 = points[1]

                # 转换坐标
                bb = convert_coordinates((img_w, img_h), (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                out_f.write(f"{cls_id} {' '.join([f'{a:.6f}' for a in bb])}\n")

print(f"--- 转换完成！ ---")
print(f"生成的标注文件已存放在: {label_dir}")
