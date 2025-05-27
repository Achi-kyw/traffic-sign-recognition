from ultralytics import YOLO
import cv2
import os
import yaml

# model = YOLO("runs/detect/train30/weights/best.pt")

model = YOLO("yolo11n.pt")

if os.path.exists("ImageSet/bigLabels"):
    os.rename("ImageSet/bigLabels", "ImageSet/labels")
cwd = os.getcwd()
data_path = os.path.join(cwd, "ImageSet/bigLabel_data.yaml")
model.train(
    data=data_path,
    epochs=50,
    imgsz=1920,
    batch=8,
    hsv_h=0.01,
    hsv_v=0.6,
    device=2
)


metrics = model.val()

if os.path.exists("ImageSet/labels"):
    os.rename("ImageSet/labels", "ImageSet/bigLabels")

results = model.predict(source="ImageSet/images/val", save=True)
results = model.predict(source="test_image", save=True)
