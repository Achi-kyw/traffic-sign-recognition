from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='chinese_cht')
for i in range(1, 2):
    image_path = f'TestSign/image{i}.png'
    print(f"\n處理圖片：{image_path}")
    result = ocr.ocr(image_path)

    print(f"辨識結果：{result}")