from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from cam import Camera
from model import CollisionModel
from llm_integration import describe_heatmap
import numpy as np

import asyncio
import cv2
import io
import time


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------
# 전역 인스턴스
# --------------------------------------------------
camera = Camera(source=0, width=640, height=480)  # 드론캠이면 URL/인덱스 변경
model = CollisionModel("rebuilt_model.h5", input_size=(128,128))

# LLM 최신 설명 저장
latest_description = {"text": "", "ts": 0}


# --------------------------------------------------
# index 페이지
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# --------------------------------------------------
# MJPEG 스트리밍 제너레이터
# --------------------------------------------------
def gen_frames():
    while True:
        frame = camera.read()
        if frame is None:
            continue

        # 예측 및 Grad-CAM
        overlay_bgr, pred_label, heatmap_roi = model.predict_and_gradcam(frame)

        # JPEG 인코딩
        ret, jpeg = cv2.imencode(".jpg", overlay_bgr)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )


# --------------------------------------------------
# 스트리밍 엔드포인트
# --------------------------------------------------
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# 최신 LLM Heatmap 설명 엔드포인트
# --------------------------------------------------
@app.get("/latest_description")
async def get_description():
    return JSONResponse(latest_description)


# --------------------------------------------------
# LLM 백그라운드 worker
# --------------------------------------------------
async def llm_worker():
    """1초마다 Grad-CAM ROI를 LLM에 보내 최신 설명 업데이트"""
    global latest_description

    while True:
        frame = camera.read()
        if frame is None:
            await asyncio.sleep(0.2)
            continue

        overlay_bgr, pred_label, heatmap_roi = model.predict_and_gradcam(frame)

        try:
            text = await describe_heatmap(pred_label, heatmap_roi)
            latest_description = {"text": text, "ts": time.time()}
        except Exception:
            pass

        await asyncio.sleep(1.0)  # 1초마다 업데이트

from fastapi import UploadFile, File
from utils import bgr_to_jpeg_bytes, jpeg_bytes_to_base64


@app.post("/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    # 파일 읽기
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        return {"error": "이미지 디코딩 실패"}

    # 모델 분석
    overlay_bgr, pred_label, heatmap_meta = model.predict_and_gradcam(bgr)

    # LLM 설명 생성
    try:
        text = await describe_heatmap(pred_label, heatmap_meta)
    except:
        text = f"{pred_label} 영역 감지됨"

    # 이미지 → base64 변환
    jpeg_bytes = bgr_to_jpeg_bytes(overlay_bgr)
    b64 = jpeg_bytes_to_base64(jpeg_bytes)

    return {
        "image": b64,
        "text": text
    }



# --------------------------------------------------
# 이벤트: 서버 시작
# --------------------------------------------------
@app.on_event("startup")
async def startup_event():
    camera.start()
    asyncio.create_task(llm_worker())


# --------------------------------------------------
# 이벤트: 서버 종료
# --------------------------------------------------
@app.on_event("shutdown")
def shutdown_event():
    camera.stop()


# --------------------------------------------------
# 실행
# --------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
