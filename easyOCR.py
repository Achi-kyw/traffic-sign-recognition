import cv2
import easyocr
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# def preprocess_image(image_path):         # 灰階 + 直方圖均衡 + Otsu 二值化
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     image = cv2.equalizeHist(image)
#     _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return thresh

# def preprocess_image(image_path):         # 灰階 + 高斯模糊
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)  # 去雜訊但不破壞結構
#     return blur

def preprocess_image(image_path):           # 灰階 + CLAHE 局部直方增強
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return clahe_img


def draw_chinese_text(image_cv, text, position, font_path='/System/Library/Fonts/STHeiti Light.ttc', font_size=20):
    # 將 OpenCV 圖片轉成 PIL 格式
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=(255, 0, 0))  # RGB 顏色

    # 轉回 OpenCV 格式
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def run_ocr(image_path, output_path):
    preprocessed = preprocess_image(image_path)
    reader = easyocr.Reader(['ch_tra'])
    results = reader.readtext(preprocessed)

    original_img = cv2.imread(image_path)

    for (bbox, text, confidence) in results:
        print(f"辨識文字：{text}（信心度：{confidence:.2f}）")

        # bbox: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        pts = [tuple(map(int, point)) for point in bbox]
        cv2.polylines(original_img, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

        x, y = pts[0]
        original_img = draw_chinese_text(original_img, text, (x, max(y-10,0)))

    # 存檔
    cv2.imwrite(output_path, original_img)

for i in range(1, 11):
    image_path = f'TestSign/image{i}.png'
    output_path = f'TestSign/Result/result{i}.png'
    print(f"\n處理圖片：{image_path}")
    run_ocr(image_path, output_path)
    print(f"結果儲存：{output_path}")
