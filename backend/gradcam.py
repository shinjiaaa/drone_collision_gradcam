# gradcam.py - Keras(.h5) 모델 전용 Grad-CAM 구현
import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple

from model import MODEL, preprocess_bgr, predict_prob

def get_target_conv_layer(model: tf.keras.Model):
    """
    네 모델의 마지막 conv 레이어는 이름 'conv2d_2'로 보였음.
    안전하게 마지막으로 'Conv2D' 타입인 레이어를 찾아 리턴.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise RuntimeError("No Conv2D layer found in model.")

TARGET_LAYER = get_target_conv_layer(MODEL)

def make_gradcam_heatmap(img_bgr: np.ndarray, class_index: int, upsample_to: Tuple[int,int]=(224,224)) -> np.ndarray:
    """
    img_bgr: original BGR ndarray (HxWx3)
    class_index: 0 or 1 (사용: 1 -> collision)
    returns: heatmap float32 (upsampled to upsample_to) in range 0..1
    """
    # 1) 전처리 (RGB normalized)
    x = preprocess_bgr(img_bgr, target_size=upsample_to)  # (1,H,W,3) RGB [0..1]
    img_tensor = tf.convert_to_tensor(x)

    # 2) 모델에서 target layer output과 predictions을 함께 뽑는 서브모델
    grad_model = tf.keras.models.Model(
        [MODEL.inputs],
        [TARGET_LAYER.output, MODEL.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        # predictions shape could be (1,1) or (1,num_classes)
        if predictions.shape[-1] == 1:
            # single-output -> use probability for collision class (index 1)
            # if model outputs raw logits, apply sigmoid
            pred_val = predictions[:, 0]
            # ensure scalar for gradient (use sigmoid to get prob)
            prob = tf.math.sigmoid(pred_val)
            score = prob[:, 0]
        else:
            # multi-class
            score = predictions[:, class_index]

    # gradients of the target score w.r.t. conv outputs
    grads = tape.gradient(score, conv_outputs)  # shape (1, H, W, C) or (1, H, W, C) depending on ordering
    # pooled grads across spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # shape (1, C)

    conv_outputs = conv_outputs[0]  # (H, W, C)
    pooled_grads = pooled_grads[0]  # (C,)

    # weight the channels by corresponding gradients
    cam = tf.zeros(conv_outputs.shape[:2], dtype=tf.float32)  # (H, W)
    for i in range(pooled_grads.shape[0]):
        cam += pooled_grads[i] * conv_outputs[:, :, i]

    cam = tf.nn.relu(cam)
    cam = cam - tf.reduce_min(cam)
    maxv = tf.reduce_max(cam)
    if maxv > 0:
        cam = cam / maxv
    cam_np = cam.numpy().astype(np.float32)  # (H_cam, W_cam) values 0..1

    # resize heatmap to upsample_to (e.g., model input) or original image later
    heatmap = cv2.resize(cam_np, (upsample_to[1], upsample_to[0]))
    return heatmap

def overlay_heatmap_on_bgr(orig_bgr: np.ndarray, heatmap: np.ndarray, alpha: float=0.5) -> np.ndarray:
    """
    heatmap: HxW float 0..1, orig_bgr: HxWx3
    returns BGR overlay uint8
    """
    h, w = orig_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, orig_bgr, 1 - alpha, 0)
    return overlay

def predict_and_gradcam(img_bgr: np.ndarray):
    """
    통합 함수: 예측(probs,pred) + gradcam overlay image 반환
    """
    probs, pred = predict_prob(img_bgr)
    # class index: collision class is 1 (we use probs[1])
    heatmap = make_gradcam_heatmap(img_bgr, class_index=1, upsample_to=(224,224))
    overlay = overlay_heatmap_on_bgr(img_bgr, heatmap, alpha=0.5)
    return {
        "probs": probs,
        "pred": int(pred),
        "overlay": overlay
    }
