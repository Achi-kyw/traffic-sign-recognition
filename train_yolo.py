from ultralytics import YOLO
import cv2
import os
import yaml

model = YOLO("runs/detect/train18/weights/best.pt")
model = YOLO("yolo11n.pt")


model.train(
    data="ImageSet/BigLabel_data.yaml",
    epochs=50,
    imgsz=1920,
    batch=8
)

metrics = model.val()
print(metrics)

img_path = "ImageSet/images/train/00000_000000005586.jpg"
output_dir = "object"
model_path = "runs/detect/train8/weights/best.pt"
yaml_path = "ImageSet/smallLabel_data.yaml"
os.makedirs(output_dir, exist_ok=True)

with open(yaml_path, 'r') as f:
    class_names = yaml.safe_load(f)['names']


model = YOLO(model_path)
results = model.predict(source=img_path, save=False)

image = cv2.imread(img_path)

for i, result in enumerate(results):
    boxes = result.boxes
    for j, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
        
        x1, y1, x2, y2 = xyxy
        cropped = image[y1:y2, x1:x2]

        filename = f"img{i}_obj{j}_{class_name}_{conf:.2f}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cropped)
        print(f"Saved: {save_path}")

        
'''

# results = model.predict(source="your_image.jpg", save=True)

results = model.predict(source="ImageSet/images/val", save=True)

# results = model.predict(source="test_image", save=True)


img_path = "ImageSet/images/train/00000_000000005586.jpg"
output_dir = "object"
model_path = "runs/detect/train8/weights/best.pt"
yaml_path = "ImageSet/smallLabel_data.yaml"
os.makedirs(output_dir, exist_ok=True)

with open(yaml_path, 'r') as f:
    small_class_names = yaml.safe_load(f)['names']

model = YOLO(model_path)
results = model.predict(source=img_path, save=False)

image = cv2.imread(img_path)

for i, result in enumerate(results):
    boxes = result.boxes
    for j, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Image {i}, Object {j}: Class ID: {cls_id}, Confidence: {conf:.2f}")
        class_name = small_class_names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
        print(xyxy)
        x1, y1, x2, y2 = xyxy
        cropped = image[y1:y2, x1:x2]

        filename = f"img{i}_obj{j}_{class_name}_{conf:.2f}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cropped)
        print(f"Saved: {save_path}")
'''