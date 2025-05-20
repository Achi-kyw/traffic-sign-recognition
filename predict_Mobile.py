import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

def preprocess_img(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model_index, model_path, img_path):
    # 載入模型＋預處理圖片
    model = tf.keras.models.load_model(model_path)
    img_array = preprocess_img(img_path)

    # 預測
    offsets = [0, 2, 5, 19, 46, 73]
    preds = model.predict(img_array)
    pred_class_idx = np.argmax(preds, axis=1)[0]
    pred_class = offsets[model_index] + pred_class_idx
    confidence = preds[0][pred_class_idx]

    print(f"Prediction: {pred_class} (confidence: {confidence:.4f})")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_index> <image_path>")
        sys.exit(1)

    model_path = "Model/model_" + sys.argv[1] + ".h5"
    image_path = sys.argv[2]

    predict_image(int(sys.argv[1]), model_path, image_path)
