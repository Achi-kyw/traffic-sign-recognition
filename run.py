from ultralytics import YOLO
import cv2
import os
import yaml

# model = YOLO("runs/detect/train14/weights/best.pt")
model = YOLO("yolo11n.pt")


model.train(
    data="ImageSet/BigLabel_data.yaml",
    epochs=50,
    imgsz=1920,
    batch=8
)

metrics = model.val()