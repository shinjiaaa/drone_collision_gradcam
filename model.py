import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, clone_model
from tensorflow.keras.layers import Input, Conv2D, Layer  # Layer 임포트 추가
import os


class CollisionModel:
    """
    충돌 예측 모델을 로드하고 Grad-CAM을 사용하여 시각화하는 클래스.
    """

    def __init__(self, model_path, input_size=(128, 128)):
        self.input_size = input_size

        # 1. 모델 로드
        loaded_model = load_model(model_path)

        # --- 해결책: clone_model을 사용하여 깨끗한 Functional 모델 확보 ---
        try:
            # 2. 모델 구조를 복제하고 가중치 전송
            self.model = clone_model(loaded_model)
            self.model.set_weights(loaded_model.get_weights())
            self.model.trainable = False

            print("✅ 모델 구조 복제 및 가중치 전송 완료.")

        except Exception as e:
            print(f"⚠️ clone_model 또는 가중치 전송 실패: {e}")
            self.model = loaded_model  # 실패 시 기존 모델을 사용하도록 폴백

        # Grad-CAM 관련 변수 초기화
        self.last_conv_layer_name = None

        # 모델 빌드 강제 (복제 후 안정성 확보)
        dummy = np.zeros((1, input_size[0], input_size[1], 3), dtype=np.float32)
        try:
            _ = self.model(dummy)
            print("✅ 모델 빌드 검증 완료.")
        except Exception as e:
            print(f"⚠️ 모델 빌드 검증 실패: {e}")

        # 마지막 Conv layer 찾기
        conv_layers = [
            layer for layer in self.model.layers if isinstance(layer, Conv2D)
        ]

        if conv_layers:
            self.last_conv_layer_name = conv_layers[-1].name
            print(f"✅ 마지막 Conv 레이어 발견: {self.last_conv_layer_name}")

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
        # Grad-CAM 계산을 위해 입력은 tf.Tensor여야 합니다.
        return tf.convert_to_tensor(np.expand_dims(img, axis=0))

    # Grad-CAM 계산
    def compute_gradcam(self, x_tensor):
        if self.last_conv_layer_name is None:
            return np.zeros((x_tensor.shape[1], x_tensor.shape[2]), dtype=np.float32)

        # Grad-CAM 계산: Sub-Model 없이 GradientTape 내에서 직접 Conv 출력을 얻습니다.
        # persistent=True를 사용하여 두 단계의 기울기를 계산할 수 있게 합니다.
        with tf.GradientTape(persistent=True) as tape:
            # 입력 텐서를 추적 대상으로 설정
            tape.watch(x_tensor)

            # 모델의 모든 레이어를 순회하며 Conv Layer를 포함한 전체 계산 경로를 재구성합니다.
            current_tensor = x_tensor
            conv_outputs = None

            # self.model.layers를 순회하며 텐서 계산을 재실행합니다.
            for layer in self.model.layers:
                # Input Layer는 건너뜁니다.
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue

                # 이전 텐서를 현재 레이어에 통과시킵니다.
                try:
                    current_tensor = layer(current_tensor)
                except Exception as e:
                    # Functional 모델에서 레이어 재호출 시 발생하는 예외를 무시하고 진행 (최후의 수단)
                    continue

                # 찾던 마지막 Conv Layer에 도달하면 출력을 저장합니다.
                if layer.name == self.last_conv_layer_name:
                    conv_outputs = current_tensor

            # 최종 예측은 루프의 마지막 출력입니다. (Conv output을 포함한 계산 경로)
            predictions = current_tensor

            # conv_outputs가 아직 None이거나, predictions가 None이면 오류를 반환합니다.
            if conv_outputs is None:
                print("⚠️ Grad-CAM: Conv Layer 출력을 찾지 못했습니다.")
                return np.zeros(
                    (x_tensor.shape[1], x_tensor.shape[2]), dtype=np.float32
                )

            # 예측값 중 충돌(collision) 확률 (첫 번째 클래스)에 대한 손실을 정의합니다.
            loss = predictions[:, 0]

        # 1. 최종 예측(loss)에 대한 Conv 출력(conv_outputs)의 기울기 계산
        # 이제 loss는 conv_outputs를 포함하는 명확한 계산 경로를 가집니다.
        grads = tape.gradient(loss, conv_outputs)

        # 메모리 해제를 위해 persistent tape 삭제
        del tape

        if grads is None:
            # 이 코드가 실행되는 것은 이제 Layer 순회 중 문제가 발생했거나,
            # Conv output이 GradientTape의 경로에 올바르게 포함되지 않았음을 의미합니다.
            print("⚠️ Gradient Tape에서 기울기 계산에 실패했습니다. (grads is None)")
            return np.zeros((x_tensor.shape[1], x_tensor.shape[2]), dtype=np.float32)

        # Global Average Pooling (가중치) 계산
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # NumPy 배열로 변환
        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()

        # 히트맵 계산: 가중치(pooled_grads)와 활성화 맵(conv_outputs)의 곱의 합
        heatmap = np.zeros(conv_outputs_np.shape[:2], dtype=np.float32)
        for i in range(pooled_grads_np.shape[0]):
            heatmap += pooled_grads_np[i] * conv_outputs_np[:, :, i]

        # ReLU 적용 (음수 값 제거)
        heatmap = np.maximum(heatmap, 0)
        # 정규화 (0 ~ 1)
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max

        return heatmap

    # 예측 + Grad-CAM 시각화
    def predict_and_gradcam(self, bgr):
        # 1. 전처리 (결과는 tf.Tensor)
        x_tensor = self.preprocess(bgr)

        # 2. 예측
        # self.model(x_tensor)는 tf.Tensor를 반환합니다.
        preds = self.model(x_tensor).numpy()[0]
        prob = float(preds[0])
        label = "collision" if prob > 0.5 else "normal"

        # 3. Grad-CAM 계산
        heatmap = self.compute_gradcam(x_tensor)

        # 4. 시각화 및 ROI (Region of Interest)
        # 원본 이미지 크기로 히트맵 리사이즈
        heatmap_resized = cv2.resize(heatmap, (bgr.shape[1], bgr.shape[0]))

        # 컬러맵 적용 (JET 컬러맵을 사용하여 시각화)
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
