import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Conv2D


class CollisionModel:
    def __init__(self, model_path, input_size=(128, 128)):
        self.input_size = input_size

        # 모델 로드
        loaded_model = load_model(model_path)

        try:
            self.model = clone_model(loaded_model)
            self.model.set_weights(loaded_model.get_weights())
            self.model.trainable = False
        except Exception as e:
            print(f"⚠️ clone 실패, 원본 모델 사용: {e}")
            self.model = loaded_model

        # 모델 빌드 보장
        dummy = np.zeros((1, input_size[0], input_size[1], 3), dtype=np.float32)
        _ = self.model(dummy)

        # 마지막 Conv layer 탐색
        self.last_conv_layer_name = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, Conv2D):
                self.last_conv_layer_name = layer.name
                break

        if self.last_conv_layer_name is None:
            print("⚠️ Conv layer 없음 → Grad-CAM 불가")

    # -------------------------------
    # Preprocess
    # -------------------------------
    def preprocess(self, bgr):
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        img = img.astype(np.float32) / 255.0
        return tf.convert_to_tensor(img[None, ...])

    # -------------------------------
    # Grad-CAM
    # -------------------------------
    def compute_gradcam(self, x_tensor):
        if self.last_conv_layer_name is None:
            return np.zeros(self.input_size, dtype=np.float32)

        with tf.GradientTape() as tape:
            current = x_tensor
            conv_out = None

            for layer in self.model.layers:
                current = layer(current)
                if layer.name == self.last_conv_layer_name:
                    conv_out = current

            loss = current[:, 0]

        grads = tape.gradient(loss, conv_out)
        if grads is None:
            return np.zeros(self.input_size, dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0].numpy()

        heatmap = np.zeros(conv_out.shape[:2], dtype=np.float32)
        for i in range(pooled_grads.shape[0]):
            heatmap += pooled_grads[i] * conv_out[:, :, i]

        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    # -------------------------------
    # BBox 중심 수축
    # -------------------------------
    @staticmethod
    def shrink_bbox(bbox, ratio=0.85):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        nw, nh = int(w * ratio), int(h * ratio)
        nx, ny = int(cx - nw / 2), int(cy - nh / 2)
        return nx, ny, nw, nh

    # -------------------------------
    # Predict + Grad-CAM + ROI
    # -------------------------------
    def predict_and_gradcam(self, bgr):
        # 예측
        x_tensor = self.preprocess(bgr)
        prob = float(self.model(x_tensor).numpy()[0][0])
        label = "collision" if prob > 0.5 else "normal"

        # Grad-CAM
        heatmap = self.compute_gradcam(x_tensor)
        heatmap = cv2.resize(heatmap, (bgr.shape[1], bgr.shape[0]))

        # 시각화
        heat_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr, 0.6, heat_color, 0.4, 0)

        # -------------------------------
        # ROI 보정 (핵심)
        # -------------------------------
        bbox = None
        heat_uint8 = np.uint8(255 * heatmap)
        thresh_val = int(heat_uint8.max() * 0.6)

        _, th = cv2.threshold(heat_uint8, thresh_val, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contours = [
                c
                for c in contours
                if cv2.contourArea(c) > 0.01 * bgr.shape[0] * bgr.shape[1]
            ]
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                bbox = self.shrink_bbox((x, y, w, h))

        return overlay, label, {"bbox": bbox, "prob": prob}
