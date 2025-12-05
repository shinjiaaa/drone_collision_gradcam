import cv2
import threading
import time
import numpy as np
from model import CollisionModel

class Camera:
    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.pred_result = None
        self.model = None

    def set_model(self, collision_model):
        self.model = collision_model

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            print("❌ Camera open failed")
            return
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._predictor, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame

    def _predictor(self):
        while self.running:
            frame = None
            with self.lock:
                if self.frame is not None:
                    frame = self.frame.copy()
            if frame is None or self.model is None:
                time.sleep(0.01)
                continue
            overlay, label, info = self.model.predict_and_gradcam(frame)
            self.pred_result = (overlay, label, info)
            time.sleep(0.05)

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def read_pred(self):
        return self.pred_result

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()


# -----------------------------
# CollisionModel 로드
# -----------------------------
model = CollisionModel("rebuilt_model.h5", input_size=(128,128))

# -----------------------------
# Camera 생성 및 모델 주입
# -----------------------------
camera = Camera(source=0, width=640, height=480)
camera.set_model(model)
