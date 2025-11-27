from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import numpy as np
import cv2
from backend.model import predict_prob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    import base64, io
    from PIL import Image

    # 업로드 파일 그대로 읽기
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # 결과 이미지(PIL) → base64
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "prediction": 1,
        "probabilities": [0.1, 0.9],
        "image_base64": img_b64,
    }
