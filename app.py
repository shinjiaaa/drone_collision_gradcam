from fastapi import FastAPI, Response, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import cv2
import time
import numpy as np

from cam import Camera
from model import CollisionModel
from llm_integration import describe_heatmap
from utils import bgr_to_jpeg_bytes, jpeg_bytes_to_base64


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

FRAME_WIDTH = 640

camera = Camera(source=0, width=FRAME_WIDTH, height=480)
model = CollisionModel("rebuilt_model.h5", input_size=(128, 128))

latest_description = {"text": "", "prob_percent": None, "ts": 0}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


def gen_frames():
    while True:
        frame = camera.read()
        if frame is None:
            continue

        overlay_bgr, _, _ = model.predict_and_gradcam(frame)
        ret, jpeg = cv2.imencode(".jpg", overlay_bgr)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/latest_description")
async def get_description():
    return JSONResponse(latest_description)


async def llm_worker():
    global latest_description

    while True:
        frame = camera.read()
        if frame is None:
            await asyncio.sleep(0.2)
            continue

        _, label, heatmap_meta = model.predict_and_gradcam(frame)

        try:
            result = await describe_heatmap(label, heatmap_meta, FRAME_WIDTH)
            latest_description = {
                "text": result["text"],
                "prob_percent": result["prob_percent"],
                "ts": time.time(),
            }
        except Exception as e:
            print("LLM 오류:", e)

        await asyncio.sleep(1.0)


@app.post("/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        return {"error": "이미지 디코딩 실패"}

    overlay_bgr, label, heatmap_meta = model.predict_and_gradcam(bgr)

    try:
        result = await describe_heatmap(label, heatmap_meta, FRAME_WIDTH)
    except Exception:
        result = {"text": f"{label} 감지됨", "prob_percent": None}

    jpeg_bytes = bgr_to_jpeg_bytes(overlay_bgr)
    b64 = jpeg_bytes_to_base64(jpeg_bytes)

    return {
        "image": b64,
        "text": result["text"],
        "prob_percent": result["prob_percent"],
    }


@app.on_event("startup")
async def startup_event():
    camera.start()
    asyncio.create_task(llm_worker())


@app.on_event("shutdown")
def shutdown_event():
    camera.stop()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
