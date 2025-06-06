import cv2
import easyocr
import os
import glob
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from argparse import ArgumentParser
from rouge_score import rouge_scorer
import json
from text_evaluate import character_accuracy, character_error_rate, levenshtein_similarity

# def preprocess_image(img):         # 灰階 + 直方圖均衡 + Otsu 二值化Add commentMore actions
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     image = cv2.equalizeHist(gray)
#     _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return thresh

# def preprocess_image(img):         # 灰階 + 高斯模糊
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)  # 去雜訊但不破壞結構
#     return blur

# def preprocess_image(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     clahe_img = clahe.apply(gray)
#     return clahe_img

def preprocess_image(img):
    return img

def draw_chinese_text(image_cv, text, position, font_path='/System/Library/Fonts/STHeiti Light.ttc', font_size=20):
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=(255, 0, 0))  # RGB 顏色

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def evaluate_ocr_results(ocr_results: dict, ground_truth_file: str):
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        gt_dict = json.load(f)

    total_acc, total_cer, total_lev = 0, 0, 0
    count = 0

    for filename, gt_text in gt_dict.items():
        if filename not in ocr_results:
            print(f"Warning: {filename} not found in OCR results.")
            continue

        pred_text = ocr_results[filename]
        acc = character_accuracy(gt_text, pred_text)
        cer = character_error_rate(gt_text, pred_text)
        lev = levenshtein_similarity(gt_text, pred_text)

        print(f"{filename}")
        print(f"gt_text: {gt_text}")
        print(f"pred_text: {pred_text}")
        print(f"accuracy: {acc:.4f}, CER: {cer:.4f}, Levenshtein similarity: {lev:.4f}")

        total_acc += acc
        total_cer += cer
        total_lev += lev
        count += 1

    if count > 0:
        print(f"\n=== Average Scores ===")
        print(f"Accuracy: {total_acc / count:.4f}")
        print(f"CER: {total_cer / count:.4f}")
        print(f"Levenshtein Similarity: {total_lev / count:.4f}")


def run_ocr(img, output_path):
    preprocessed = preprocess_image(img)
    reader = easyocr.Reader(['ch_tra'])
    results = reader.readtext(preprocessed)

    original_img = preprocessed.copy()
    texts = []

    for (bbox, text, confidence) in results:
        texts.append(text)
        print(f"辨識文字：{text}（信心度：{confidence:.2f}）")

        pts = [tuple(map(int, point)) for point in bbox]
        cv2.polylines(original_img, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

        x, y = pts[0]
        original_img = draw_chinese_text(original_img, text, (x, max(y-10,0)))

    cv2.imwrite(output_path, original_img)
    combined_text = ''.join(texts).replace(' ', '')
    print(f"整合結果：{combined_text}")

    return combined_text

def do_easyocr(img):
    preprocessed = preprocess_image(img)
    reader = easyocr.Reader(['ch_tra'],gpu=False)
    results = reader.readtext(preprocessed)
    
    return results

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image_dir', type=str, help='Path to the input image dir')
    parser.add_argument('--output_dir', type=str, default="easy_ocr")
    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = []
    image_paths.extend(glob.glob(os.path.join(args.image_dir, '*.png')))

    if not image_paths:
        print(f"no files found in {args.image_dir}")
        exit(1)
    
    ocr_results = {}
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        output_path = os.path.join(args.output_dir, f"ocr_{filename}")

        print(f"processing {filename}")
        img = cv2.imread(image_path)
        ocr_text = run_ocr(img, output_path)
        ocr_results[filename] = ocr_text
        
    evaluate_ocr_results(ocr_results, f"{args.image_dir}/ground_truth.json")
    print("finished processing all images")
