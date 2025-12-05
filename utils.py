import base64
import cv2
import numpy as np

def bgr_to_jpeg_bytes(bgr, quality=90):
    ret, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        return None
    return buf.tobytes()

def jpeg_bytes_to_base64(jpeg_bytes):
    return base64.b64encode(jpeg_bytes).decode('ascii')
