import os
import cv2
import yaml

image_dir = "ImageSet/images/val"
label_dir = "ImageSet/smallLabels/val"
big_label_dir = "ImageSet/bigLabels/val"
output_dir = "val"
yaml_path = "ImageSet/smallLabel_data.yaml"
yaml_big_path = "ImageSet/bigLabel_data.yaml"

os.makedirs(output_dir, exist_ok=True)

with open(yaml_path, 'r') as f:
    class_names = yaml.safe_load(f)['names']
with open(yaml_big_path, 'r') as f:
    big_class_names = yaml.safe_load(f)['names']

for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
    big_label_path = os.path.join(big_label_dir, os.path.splitext(img_name)[0] + ".txt")
    if not os.path.exists(label_path):
        print(f"Not Exist: {label_path}")
        continue

    image = cv2.imread(img_path)
    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()
    with open(big_label_path, "r") as f:
        big_lines = f.readlines()
    for i, (line,big_line) in enumerate(zip(lines,big_lines)):
        parts = line.strip().split()
        cls_id = int(parts[0])
        parts2 = big_line.strip().split()
        big_cls_id = int(parts2[0])
        x_center, y_center, box_w, box_h = map(float, parts[1:])

        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)

        crop = image[y1:y2, x1:x2]
        class_name = class_names[cls_id]
        big_class_name = big_class_names[big_cls_id]
        save_name = f"{os.path.splitext(img_name)[0]}_obj{i}_{big_class_name}_{class_name}.jpg"
        save_path = os.path.join(output_dir, str(big_cls_id))
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, str(cls_id))
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, save_name)
        cv2.imwrite(save_path, crop)

        print(f"Saved: {save_path}")
