import os
import cv2
import yaml

image_dir = "ImageSet/images/val"
label_dir = "ImageSet/smallLabels/val"
big_label_dir = "ImageSet/bigLabels/val"
output_dir = "val"
yaml_path = "ImageSet/smallLabel_data.yaml"
yaml_big_path = "ImageSet/bigLabel_data.yaml"

model = YOLO("runs/detect/train14/weights/best.pt")

os.makedirs(output_dir, exist_ok=True)

with open(yaml_path, 'r') as f:
    class_names = yaml.safe_load(f)['names']
with open(yaml_big_path, 'r') as f:
    big_class_names = yaml.safe_load(f)['names']

for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    results = model.predict(source=img_path, save=False)
    image = cv2.imread(img_path)
    for i, result in enumerate(results):
        boxes = result.boxes
        for j, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            
            x1, y1, x2, y2 = xyxy
            cropped = image[y1:y2, x1:x2]

            '''
            filename = f"img{i}_obj{j}_{class_name}_{conf:.2f}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, cropped)
            print(f"Saved: {save_path}")
            '''

