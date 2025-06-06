from ultralytics import YOLO
import cv2
import os
import yaml
from argparse import ArgumentParser

def train(model,data_path,yaml_path,train_output_dir):
    model = YOLO(model)
    if os.path.exists(data_path):
        os.rename(data_path, "ImageSet/labels")
    cwd = os.getcwd()
    new_data_path = os.path.join(cwd, yaml_path)
    model.train(
        data=new_data_path,
        epochs=50,
        imgsz=1920,
        batch=8,
        hsv_h=0.020,
        hsv_v=0.6,
        device=0,
        name=train_output_dir,
        exist_ok=True
    )
    metrics = model.val()
    if os.path.exists("ImageSet/labels"):
        os.rename("ImageSet/labels", data_path)
    return model


def test(testdata_path, testdata_output, model):
    results = model.predict(source=testdata_path, save=True, name=testdata_output)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--data_path", type=str, default="ImageSet/bigLabels")
    parser.add_argument("--yaml_path", type=str, default="ImageSet/bigLabel_data.yaml")
    parser.add_argument("--train_output_dir", type=str, default="yolo")
    parser.add_argument("--testdata_path", type=str, default=None)
    parser.add_argument("--testdata_output", type=str, default="yolo_test")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model = train(args.model,args.data_path,args.yaml_path,args.train_output_dir)
    # model = YOLO("runs/detect/yolo_640/weights/best.pt")
    if args.testdata_path is not None:
        test(args.testdata_path, args.testdata_output, model)