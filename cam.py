import sys
import os

# [강력 처방] 현재 폴더(.)가 라이브러리 경로보다 앞서는 것을 방지
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)
sys.path.append(current_dir) # 현재 폴더를 맨 뒤로 보냄

import cv2
import threading
import time
import numpy as np

# ZED SDK 임포트 시도
try:
    import pyzed.sl as sl
    print("✅ [DEBUG] ZED SDK 로드 성공!")
except (ImportError, AttributeError) as e:
    print(f"❌ [DEBUG] ZED SDK 로드 실패: {e}")
    sl = None




class RiskRefiner:
    def __init__(self, area_gain=2.0, min_bbox_area_ratio=0.01, max_bbox_area_ratio=0.6):
        self.area_gain = area_gain
        self.min_ratio = min_bbox_area_ratio
        self.max_ratio = max_bbox_area_ratio

    def refine_bbox(self, heatmap, bbox, frame_shape):
        if bbox is None:
            return None

        h, w = frame_shape[:2]
        x, y, bw, bh = bbox
        area_ratio = (bw * bh) / (w * h)

        if area_ratio < self.min_ratio or area_ratio > self.max_ratio:
            return None

        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        edges = cv2.Canny(heatmap_uint8, 50, 150)
        
        roi_edges = edges[y : y + bh, x : x + bw]
        if roi_edges.size == 0:
            return None
        
        edge_density = np.mean(roi_edges > 0)
        if edge_density < 0.01:
            return None

        return bbox

    def refine_risk(self, prob, bbox, frame_shape):
        if bbox is None:
            return prob

        h, w = frame_shape[:2]
        _, _, bw, bh = bbox
        area_ratio = (bw * bh) / (w * h)

        boosted = prob * (1.0 + self.area_gain * area_ratio)
        return float(np.clip(boosted, 0.0, 1.0))


class UniversalCamera:
    def __init__(self, mode="zed", source=0, width=640, height=480):
        self.mode = mode.lower()
        self.source = source
        self.width = width
        self.height = height
        
        self.cap = None    
        self.zed = None    
        self.runtime_params = None
        
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.model = None
        self.refiner = RiskRefiner() 
        self.pred_result = None

    def set_model(self, collision_model):
        self.model = collision_model

    def start(self):
        if self.mode == "zed":
            if sl is None:
                print("❌ pyzed 라이브러리가 설치되지 않았습니다.")
                return
            
            self.zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            init_params.coordinate_units = sl.UNIT.METER

            if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
                print("❌ ZED 2 카메라 열기 실패")
                return
            
            self.runtime_params = sl.RuntimeParameters()
            print("✅ ZED 2 카메라 연결 성공")
        
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                print(f"❌ 웹캠(Source:{self.source}) 연결 실패")
                return
            print("✅ 웹캠 연결 성공")

        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._predictor, daemon=True).start()

    # --- 추가된 메서드 (FastAPI 연동용) ---
    def read(self):
        """가장 최신의 원본 프레임을 읽어옴"""
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def read_pred(self):
        """_predictor 스레드에서 생성한 최신 예측 결과(오버레이 등)를 읽어옴"""
        with self.lock:
            return self.pred_result
    # ----------------------------------

    def _reader(self):
        if self.mode == "zed":
            image_zed = sl.Mat()
            while self.running:
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                    frame = image_zed.get_data()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    frame = cv2.resize(frame, (self.width, self.height))
                    with self.lock:
                        self.frame = frame
        else:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (self.width, self.height))
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.01)

    def _predictor(self):
        while self.running:
            frame = None
            with self.lock:
                if self.frame is not None:
                    frame = self.frame.copy()
            
            if frame is None or self.model is None:
                time.sleep(0.01)
                continue

            # 모델 추론 및 Grad-CAM 데이터 획득
            overlay, label, info = self.model.predict_and_gradcam(frame)
            
            # RiskRefiner 정밀 보정
            if 'heatmap' in info and 'bbox' in info:
                refined_bbox = self.refiner.refine_bbox(info['heatmap'], info['bbox'], frame.shape)
                refined_prob = self.refiner.refine_risk(info['prob'], refined_bbox, frame.shape)
                
                info['refined_prob'] = refined_prob
                info['is_valid'] = refined_bbox is not None
            
            with self.lock:
                self.pred_result = (overlay, label, info)
            
            time.sleep(0.05)

    def stop(self):
        self.running = False
        if self.mode == "zed" and self.zed:
            self.zed.close()
        elif self.cap:
            self.cap.release()
        print(f"🛑 {self.mode} 시스템 종료")