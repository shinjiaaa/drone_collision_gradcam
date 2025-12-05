import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

class CollisionModel:
    def __init__(self, model_path, input_size=(128, 128)):
        self.input_size = input_size
        self.model = load_model(model_path)
        self.model.trainable = False

        # --- Sequential 호출 강제(build) ---
        dummy = np.zeros((1, input_size[0], input_size[1], 3), dtype=np.float32)
        try:
            self.model(dummy)
        except Exception as e:
            print("⚠️ 모델 호출 시 오류:", e)

        # 마지막 conv layer 찾기
        self.last_conv_layer = self._find_last_conv_layer()
        if self.last_conv_layer is None:
            print("⚠️ Warning: Conv layer not found in model.")

    # 마지막 Conv 레이어 자동 탐색
    def _find_last_conv_layer(self):
        for layer in reversed(self.model.layers):
            if "conv" in layer.name.lower():
                return layer.name
        return None

    # 입력 전처리
    def preprocess(self, bgr):
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    # Grad-CAM 계산
    def compute_gradcam(self, x):
        if self.last_conv_layer is None:
            return np.zeros((x.shape[1], x.shape[2]), dtype=np.float32)

        last_conv = self.model.get_layer(self.last_conv_layer)

        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[last_conv.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            loss = predictions[:, 0]  # binary classification

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        heatmap = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
        for i in range(pooled_grads.shape[0]):
            heatmap += pooled_grads[i] * conv_outputs[:, :, i]

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-6)
        return heatmap

    # 예측 + Grad-CAM
    def predict_and_gradcam(self, bgr):
        x = self.preprocess(bgr)

        preds = self.model.predict(x)[0]
        prob = float(preds[0])
        pred_label = "collision" if prob > 0.5 else "normal"

        heatmap = self.compute_gradcam(x)
        heatmap_resized = cv2.resize(heatmap, (bgr.shape[1], bgr.shape[0]))
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            cv2.COLORMAP_JET
        )

        overlay = cv2.addWeighted(bgr, 0.6, heatmap_color, 0.4, 0)

        # ROI 컨투어
        thresh = (heatmap_resized * 255).astype(np.uint8)
        _, th = cv2.threshold(thresh, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bbox = None
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            bbox = (x, y, w, h)

        return overlay, pred_label, {"bbox": bbox, "prob": prob}
