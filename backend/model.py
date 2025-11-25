# model.py
import os
import numpy as np
import cv2
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_weights.h5")
MODEL = tf.keras.models.load_model(MODEL_PATH)
MODEL.trainable = False

def preprocess_bgr(img_bgr: np.ndarray):
    import cv2
    target_size = (128, 128)  # 모델이 기대하는 입력 크기
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    tensor = img_resized.astype("float32") / 255.0
    return np.expand_dims(tensor, axis=0)  # (1,H,W,C)


def predict_prob(img_bgr: np.ndarray):
    x = preprocess_bgr(img_bgr)
    logits = MODEL.predict(x, verbose=0)
    if logits.ndim == 2 and logits.shape[1] == 1:
        v = logits[0,0]
        if v < 0.0 or v > 1.0:
            p1 = 1.0 / (1.0 + np.exp(-v))
        else:
            p1 = float(v)
        p0 = 1.0 - p1
        probs = [p0, p1]
        pred = int(p1 >= 0.5)
    else:
        probs_arr = tf.nn.softmax(logits, axis=1).numpy()[0]
        probs = probs_arr.tolist()
        pred = int(np.argmax(probs_arr))
    return probs, pred
