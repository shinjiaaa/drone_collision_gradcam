# utils.py - 이미지 I/O, base64 헬퍼
import io
import base64
import numpy as np
from PIL import Image
import cv2
import os

def read_imagefile_bytes(file_bytes: bytes) -> np.ndarray:
    """업로드된 바이트를 읽어 OpenCV BGR ndarray로 반환."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(img)  # RGB
    arr_bgr = arr[..., ::-1].copy()  # RGB -> BGR
    return arr_bgr

def ndarray_to_jpeg_bytes(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    """BGR ndarray -> JPEG bytes"""
    ret, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ret else b""

def bytes_to_datauri_jpeg(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")

def ensure_working_dir_is_project_root():
    """(선택) 현재 작업 디렉토리가 backend 파일 위치와 다른 경우 대비 helper"""
    # 사용 시 필요하면 구현/호출
    pass
