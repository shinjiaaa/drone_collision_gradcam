import sys
import os
import traceback
import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "model_weights.h5"  # 필요하면 경로 수정

def safe_imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imread 실패: {path}")
    return img

def print_model_info(model):
    print("=== Model summary ===")
    model.summary()
    print("=== model.input_shape ===", getattr(model, "input_shape", None))
    try:
        print("model.inputs:", model.inputs)
    except Exception:
        pass
    print("Layer names and output shapes:")
    for layer in model.layers:
        try:
            print(f"  {layer.name} -> {getattr(layer, 'output_shape', layer.output.shape if hasattr(layer, 'output') else 'unknown')}")
        except Exception:
            print(f"  {layer.name} -> (cannot read shape)")

def guess_required_size(model):
    # try to infer target HW from first layer or model.input_shape
    ishape = getattr(model, "input_shape", None)
    if ishape and len(ishape) >= 3:
        return (ishape[1], ishape[2])
    # fallback: search first Conv2D layer output receptive field estimate is hard — skip
    return None

def preprocess_for_model(img_bgr, target_size):
    print(f"[preprocess] original shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print("[preprocess] after BGR->RGB:", img.shape)
    img_resized = cv2.resize(img, target_size)
    print("[preprocess] after resize:", img_resized.shape)
    x = img_resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    print("[preprocess] final input shape:", x.shape, "dtype:", x.dtype)
    return x

def main(img_path):
    try:
        if not os.path.exists(MODEL_PATH):
            print("ERROR: model file not found at", MODEL_PATH)
            return

        print("Loading model:", MODEL_PATH)
        model = load_model(MODEL_PATH)
        print("Model loaded OK")

        print_model_info(model)

        # infer target size
        tgt = guess_required_size(model)
        if tgt is None:
            print("Could not infer target size from model.input_shape. Defaulting to (224,224).")
            tgt = (224, 224)
        else:
            print("Inferred target size from model:", tgt)

        # read image
        img = safe_imread(img_path) 

        # try multiple sizes: original, inferred, common sizes
        sizes_to_try = [tgt, (224,224), (256,256), (416,416)]
        tried = set()
        for s in sizes_to_try:
            if s in tried: continue
            tried.add(s)
            try:
                print("\n--- Attempting preprocess with size:", s)
                x = preprocess_for_model(img, s)
                print("Calling model.predict with input shape:", x.shape)
                preds = model.predict(x)
                print("predict OK; preds shape:", np.shape(preds), "value:", preds)
                # if success, exit loop
                break
            except Exception as e:
                print("predict failed for size", s)
                traceback.print_exc()
                continue

        print("\nDone.")

    except Exception as e:
        print("Unhandled exception:")
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_inference.py path/to/image.jpg")
    else:
        main(sys.argv[1])
