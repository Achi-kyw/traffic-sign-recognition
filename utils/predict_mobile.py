import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import sys
import cv2

def preprocess_img(img, target_size=(160, 160)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = img.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model_index, model_path, img):
    model = tf.keras.models.load_model(model_path)
    img_array = preprocess_img(img)

    offsets = [0, 2, 5, 19, 46, 73]
    preds = model.predict(img_array)
    pred_class_idx = np.argmax(preds, axis=1)[0]
    pred_class = offsets[model_index] + pred_class_idx
    confidence = preds[0][pred_class_idx]
    return (pred_class, confidence)
    # print(f"Prediction: {pred_class} (confidence: {confidence:.4f})")

def do_mobile_predict(model_index, img, model_path):
    model_path = f"{model_path}/model_" + str(model_index) + ".h5"
    return predict_image(model_index, model_path, img)
