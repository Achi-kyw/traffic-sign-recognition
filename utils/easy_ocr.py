import cv2
import easyocr
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return clahe_img


def draw_chinese_text(image_cv, text, position, font_path='/System/Library/Fonts/STHeiti Light.ttc', font_size=20):
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=(255, 0, 0))  # RGB 顏色

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def run_ocr(image_path, output_path):
    preprocessed = preprocess_image(image_path)
    reader = easyocr.Reader(['ch_tra'])
    results = reader.readtext(preprocessed)

    original_img = cv2.imread(image_path)

    for (bbox, text, confidence) in results:
        print(f"辨識文字：{text}（信心度：{confidence:.2f}）")

        pts = [tuple(map(int, point)) for point in bbox]
        cv2.polylines(original_img, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

        x, y = pts[0]
        original_img = draw_chinese_text(original_img, text, (x, max(y-10,0)))

    cv2.imwrite(output_path, original_img)

def do_easyocr(img):
    preprocessed = preprocess_image(img)
    reader = easyocr.Reader(['ch_tra'],gpu=False)
    results = reader.readtext(preprocessed)
    
    return results
