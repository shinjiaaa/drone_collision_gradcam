import numpy as np
import cv2


class RiskRefiner:
    def __init__(
        self,
        area_gain=2.0,
        min_bbox_area_ratio=0.01,
        max_bbox_area_ratio=0.6,
    ):
        self.area_gain = area_gain
        self.min_ratio = min_bbox_area_ratio
        self.max_ratio = max_bbox_area_ratio

    def refine_bbox(self, heatmap, bbox, frame_shape):
        if bbox is None:
            return None

        h, w = frame_shape[:2]
        x, y, bw, bh = bbox

        area_ratio = (bw * bh) / (w * h)

        # 너무 작거나 너무 크면 배경/노이즈로 간주
        if area_ratio < self.min_ratio or area_ratio > self.max_ratio:
            return None

        # Edge 기반 필터 (객체 아닌 부드러운 영역 제거)
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        edges = cv2.Canny(heatmap_uint8, 50, 150)

        edge_density = np.mean(edges[y : y + bh, x : x + bw] > 0)

        # 에지가 거의 없으면 배경
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
