import os
import cv2
import yaml
from utils.predict_mobile import do_mobile_predict
from ultralytics import YOLO
from tqdm import tqdm
from utils.easy_ocr import do_easyocr
import logging
from groq import Groq

with open('api.txt','r') as f:
    s = f.read()
    client = Groq(api_key=s)
logging.getLogger('absl').setLevel(logging.ERROR)
image_dir = "test_image"
label_dir = "ImageSet/smallLabels/val"
big_label_dir = "ImageSet/bigLabels/val"
yaml_path = "ImageSet/smallLabel_data.yaml"
yaml_big_path = "ImageSet/bigLabel_data.yaml"
result_output_dir = "output"

model = YOLO("runs/detect/train44/weights/best.pt")
os.makedirs(result_output_dir, exist_ok=True)

with open(yaml_path, 'r') as f:
    class_names = yaml.safe_load(f)['names']
with open(yaml_big_path, 'r') as f:
    big_class_names = yaml.safe_load(f)['names']


for img_name in tqdm(os.listdir(image_dir)):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    img_path = os.path.join(image_dir,img_name)
    results = model.predict(source=img_path, save=False)
    image = cv2.imread(img_path)
    txt_path = os.path.join(result_output_dir,img_name.split('.')[0]+'.txt')
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
                predict_cls_id, confidence = do_mobile_predict(cls_id, cropped)
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
    text_message = f'我拿到一張街景圖片，並偵測出以下路牌資訊以及路牌在圖片上的座標。請將路牌在圖片上的位置納入考量，並依據這些資訊總結出一項【給駕駛的指示】。路牌資訊與座標：{data}。重要指令：1. 你的最終回答【必須且只能】包含【一組】`【` 和 `】` 符號。2. 這組`【` 和 `】` 符號內【必須只包含】最終總結出的那一項駕駛指示，不要有其他文字。3. 【不要】輸出任何額外的 `【】` 符號。\請直接以繁體中文輸出這項指示。例如，你的最終輸出應該是像這樣：【前方路口請右轉】'
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
    

