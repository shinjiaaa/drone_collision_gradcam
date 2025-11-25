# routes.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from backend.model import predict_prob

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Collision Grad-CAM API running"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    probs, pred = predict_prob(img_bgr)
    return JSONResponse({"probabilities": probs, "prediction": pred})

# 나중에 gradcam 라우트도 여기에 추가 가능
