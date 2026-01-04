import sys
import os
import cv2
import threading
import time
import numpy as np
from collections import deque

# ZED SDK 임포트
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
        if bbox is None: return None
        h, w = frame_shape[:2]
        x, y, bw, bh = bbox
        area_ratio = (bw * bh) / (w * h)
        if area_ratio < self.min_ratio or area_ratio > self.max_ratio: return None
        
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        edges = cv2.Canny(heatmap_uint8, 50, 150)
        roi_edges = edges[y : y + bh, x : x + bw]
        if roi_edges.size == 0: return None
        if np.mean(roi_edges > 0) < 0.01: return None
        return bbox

    def refine_risk(self, prob, bbox, frame_shape):
        if bbox is None: return prob
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
        self.depth_map = None # Depth 데이터 저장용
        self.running = False
        self.lock = threading.Lock()
        self.model = None
        self.refiner = RiskRefiner() 
        self.pred_result = None

        # --- [추가] 시계열 및 TTC 변수 ---
        self.risk_history = deque(maxlen=10)
        self.prev_distance = None
        self.prev_time = None

    def set_model(self, collision_model):
        self.model = collision_model

    def start(self):
        if self.mode == "zed":
            if sl is None: return
            self.zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            init_params.coordinate_units = sl.UNIT.METER
            if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS: return
            self.runtime_params = sl.RuntimeParameters()
        else:
            self.cap = cv2.VideoCapture(self.source)
        
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._predictor, daemon=True).start()

    def _reader(self):
        if self.mode == "zed":
            image_zed = sl.Mat()
            depth_zed = sl.Mat()
            while self.running:
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                    self.zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                    frame = cv2.cvtColor(image_zed.get_data(), cv2.COLOR_BGRA2BGR)
                    frame = cv2.resize(frame, (self.width, self.height))
                    with self.lock:
                        self.frame = frame
                        self.depth_map = depth_zed
        else:
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.resize(frame, (self.width, self.height))
                    with self.lock: self.frame = frame
                time.sleep(0.01)

    def _calculate_latency(self, t_start, t1_end, t2_end, t3_end):
        latency_t1 = t1_end - t_start
        latency_t2 = t2_end - t1_end
        latency_t3 = t3_end - t2_end
        total_loop = t3_end - t_start
        fps = 1.0 / total_loop if total_loop > 0 else 0
        
        # T1: 스테레오 카메라에서 영상 수집
        # T2: CNN 모델이 충돌 분류 & Grad-CAM 생성
        # T3: 히트맵, 위험도, LLM을 UI에 제공
        print(f"\n[실시간성 검증] FPS: {fps:.1f} Hz")
        print(f" - T1 (Sensing): {latency_t1:.4f}s")
        print(f" - T2 (Inference): {latency_t2:.4f}s")
        print(f" - T3 (Interpretation): {latency_t3:.4f}s")
        print(f" - Total Pipeline: {total_loop:.4f}s")
        
        return fps

    def _predictor(self):
        while self.running:
            t_start = time.perf_counter() # [T1 시작]

            frame = self.read()
            if frame is None or self.model is None:
                time.sleep(0.01)
                continue
            t1_end = time.perf_counter() # [T1 종료 / T2 시작]

            # CNN 추론 및 Grad-CAM 생성 (T2)
            overlay, label, info = self.model.predict_and_gradcam(frame)
            t2_end = time.perf_counter() # [T2 종료 / T3 시작]
            
            # 1. 위험률 정수화
            prob_percent = int(info.get('prob', 0.0) * 100)
            self.risk_history.append(prob_percent)
            
            # 2. 추세 요약
            trend_str = "유지"
            if len(self.risk_history) >= 3:
                diff = self.risk_history[-1] - self.risk_history[-3]
                if diff > 15: trend_str = "급상승"
                elif diff < -15: trend_str = "하락"

            # 3. TTC 계산
            ttc = 99.0
            bbox = info.get('bbox')
            if bbox and self.mode == "zed":
                with self.lock:
                    local_depth = self.depth_map
                if local_depth:
                    x, y, bw, bh = bbox
                    cx, cy = int(x + bw/2), int(y + bh/2)
                    err, d_val = local_depth.get_value(cx, cy)
                    if err == sl.ERROR_CODE.SUCCESS and np.isfinite(d_val):
                        now = time.perf_counter()
                        if self.prev_distance and self.prev_time:
                            v = (self.prev_distance - d_val) / (now - self.prev_time)
                            if v > 0.05: ttc = d_val / v
                        self.prev_distance, self.prev_time = d_val, now

                        if d_val < 1.5:
                            ttc = min(ttc, 1.0) 

            info.update({
                'prob_percent': prob_percent,
                'trend_str': trend_str,
                'ttc': ttc
            })
            
            with self.lock:
                self.pred_result = (overlay, label, info)
            
            t3_end = time.perf_counter() # [T3 종료]

            # 신규 추가: 성능 지표 계산 함수 호출
            self._calculate_latency(t_start, t1_end, t2_end, t3_end)

            time.sleep(0.05)

    def read(self):
        with self.lock: return self.frame.copy() if self.frame is not None else None

    def read_pred(self):
        with self.lock: return self.pred_result

    def stop(self):
        self.running = False
        if self.mode == "zed" and self.zed: self.zed.close()
        elif self.cap: self.cap.release()