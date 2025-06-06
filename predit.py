import os
import cv2
import yaml
from utils.predict_mobile import do_mobile_predict
from ultralytics import YOLO
from tqdm import tqdm
from utils.easy_ocr import do_easyocr
import logging
from groq import Groq
from argparse import ArgumentParser

def main(yolo_model_path, mobilenet_model_path, image_dir, yaml_small_path, yaml_big_path, output_dir):
    with open('api.txt','r') as f:
        s = f.read()
        client = Groq(api_key=s)
    logging.getLogger('absl').setLevel(logging.ERROR)

    model = YOLO(yolo_model_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(yaml_small_path, 'r') as f:
        class_names = yaml.safe_load(f)['names']
    with open(yaml_big_path, 'r') as f:
        big_class_names = yaml.safe_load(f)['names']


    for img_name in tqdm(os.listdir(image_dir)):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(image_dir,img_name)
        results = model.predict(source=img_path, save=False)
        image = cv2.imread(img_path)
        txt_path = os.path.join(output_dir,img_name.split('.')[0]+'.txt')
        data=[]
        with open(txt_path,"w") as f:
            print("",end="",file=f)
        for i, result in enumerate(results):
            boxes = result.boxes
            for j, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                x1, y1, x2, y2 = xyxy
                cropped = image[y1:y2, x1:x2]
                if cls_id !=6:
                    predict_cls_id, confidence = do_mobile_predict(cls_id, cropped, mobilenet_model_path)
                    class_name = class_names[int(predict_cls_id)]
                    if predict_cls_id==74 or predict_cls_id==75:
                        words = do_easyocr(cropped)
                        combined = "".join([text for (_, text, __) in words])
                        with open(txt_path,"a") as f:
                            output = f"偵測到號誌：{class_name}:限速為「{combined}」 位於方框 x1={x1}, x2={x2}, y1={y1}, y2={y2}, 信心程度 {confidence}"
                            print(output,file=f)
                            data.append(output)
                    else:
                        with open(txt_path,"a") as f:
                            output = f"偵測到號誌：{class_name} 位於方框 x1={x1}, x2={x2}, y1={y1}, y2={y2}, 信心程度 {confidence}"
                            print(output,file=f)
                            data.append(output)
                else:
                    words = do_easyocr(cropped)
                    combined = "".join([text for (_, text, __) in words])
                    with open(txt_path,"a") as f:
                        output = f"偵測到路牌，文字內容「{combined}」 位於方框 x1={x1}, x2={x2}, y1={y1}, y2={y2}"
                        print(output,file=f)
                        data.append(output)
        if len(data)==0:
            with open(txt_path,"a") as f:
                print("未偵測到任何號誌",file=f)
            continue
        text_message = f'我拿到一張街景圖片，並偵測出以下路牌資訊以及路牌在圖片上的座標。請將路牌在圖片上的位置納入考量，並依據這些資訊總結出一些【給駕駛的指示】。路牌資訊與座標：{data}。重要指令：1. 你的最終回答【必須且只能】包含【一組】`【` 和 `】` 符號。2. 這組`【` 和 `】` 符號內【必須只包含】最終總結出的駕駛指示，不要有其他文字。3. 【不要】輸出任何額外的 `【】` 符號。請直接以繁體中文輸出這項指示。'
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_message},
                    ],
                }
            ],
            model="llama3-70b-8192",
        )
        print(response.choices[0].message.content)
        with open(txt_path,"a") as f:
            print(response.choices[0].message.content,file=f)
    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--yolo_model_path", type=str, default="runs/detect/yolo/weights/best.pt")
    parser.add_argument("--mobilenet_model_path", type=str, default="mobile_net")
    parser.add_argument("--testdata_path", type=str, default="test_image")
    parser.add_argument("--yaml_small_path", type=str, default="ImageSet/smallLabel_data.yaml")
    parser.add_argument("--yaml_big_path", type=str, default="ImageSet/bigLabel_data.yaml")
    parser.add_argument("--output_dir", type=str, default="output")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args.yolo_model_path, args.mobilenet_model_path, args.testdata_path, args.yaml_small_path, args.yaml_big_path, args.output_dir)
    