import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Input, Conv2D, Layer  # Layer 임포트 추가
import os


class CollisionModel:
    def __init__(self, model_path, input_size=(128, 128)):
        self.input_size = input_size

        # 1. 모델 로드
        loaded_model = load_model(model_path)

        try:
            # 2. 모델 구조를 복제하고 가중치 전송
            self.model = clone_model(loaded_model)
            self.model.set_weights(loaded_model.get_weights())
            self.model.trainable = False

        except Exception as e:
            print(f"⚠️ clone_model 또는 가중치 전송 실패: {e}")
            self.model = loaded_model  # 실패 시 기존 모델을 사용하도록 폴백

        self.last_conv_layer_name = None

        dummy = np.zeros((1, input_size[0], input_size[1], 3), dtype=np.float32)
        try:
            _ = self.model(dummy)
        except Exception as e:
            print(f"⚠️ 모델 빌드 검증 실패: {e}")

        # 마지막 Conv layer 찾기
        conv_layers = [
            layer for layer in self.model.layers if isinstance(layer, Conv2D)
        ]

        if conv_layers:
            self.last_conv_layer_name = conv_layers[-1].name

        if self.last_conv_layer_name is None:
            print("⚠️ Conv 레이어 없음 → Grad-CAM을 계산할 수 없습니다.")

    # 입력 전처리
    def preprocess(self, bgr):
        # BGR을 RGB로 변환
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # 입력 크기로 리사이즈
        img = cv2.resize(img, self.input_size)
        # 정규화 (0.0 ~ 1.0)
        img = img.astype(np.float32) / 255.0
        # 배치 차원 추가
        return tf.convert_to_tensor(np.expand_dims(img, axis=0))

    # Grad-CAM 계산
    def compute_gradcam(self, x_tensor):
        if self.last_conv_layer_name is None:
            return np.zeros((x_tensor.shape[1], x_tensor.shape[2]), dtype=np.float32)
        with tf.GradientTape(persistent=True) as tape:
            # 입력 텐서를 추적 대상으로 설정
            tape.watch(x_tensor)

            current_tensor = x_tensor
            conv_outputs = None

            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue

                try:
                    current_tensor = layer(current_tensor)
                except Exception as e:
                    continue

                if layer.name == self.last_conv_layer_name:
                    conv_outputs = current_tensor

            predictions = current_tensor

            if conv_outputs is None:
                print("⚠️ Grad-CAM: Conv Layer 출력을 찾지 못했습니다.")
                return np.zeros(
                    (x_tensor.shape[1], x_tensor.shape[2]), dtype=np.float32
                )
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)

        del tape

        if grads is None:
            print("⚠️ Gradient Tape에서 기울기 계산에 실패했습니다. (grads is None)")
            return np.zeros((x_tensor.shape[1], x_tensor.shape[2]), dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()

        heatmap = np.zeros(conv_outputs_np.shape[:2], dtype=np.float32)
        for i in range(pooled_grads_np.shape[0]):
            heatmap += pooled_grads_np[i] * conv_outputs_np[:, :, i]

        heatmap = np.maximum(heatmap, 0)
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max

        return heatmap

    def predict_and_gradcam(self, bgr):
        # 1. 전처리
        x_tensor = self.preprocess(bgr)

        # 2. 예측
        preds = self.model(x_tensor).numpy()[0]
        prob = float(preds[0])
        label = "collision" if prob > 0.5 else "normal"

        # 3. Grad-CAM 계산
        heatmap = self.compute_gradcam(x_tensor)

        # 4. 시각화 및 ROI
        heatmap_resized = cv2.resize(heatmap, (bgr.shape[1], bgr.shape[0]))

        # 컬러맵 적용
        heat_color = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )

        # 원본 이미지와 히트맵 오버레이
        overlay = cv2.addWeighted(bgr, 0.6, heat_color, 0.4, 0)

        # ROI Bounding Box 추출
        bbox = None
        if self.last_conv_layer_name is not None:
            # 8비트 이미지로 변환
            thresh = (heatmap_resized * 255).astype(np.uint8)
            # 임계값 (threshold)을 설정하여 중요한 영역만 이진화 (180 이상)
            _, th = cv2.threshold(thresh, 180, 255, cv2.THRESH_BINARY)
            # 윤곽선 찾기
            contours, _ = cv2.findContours(
                th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # 가장 큰 윤곽선에 대한 Bounding Box 계산
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                bbox = (x, y, w, h)

        return overlay, label, {"bbox": bbox, "prob": prob}
