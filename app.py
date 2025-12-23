from fastapi import FastAPI, Response, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import cv2
import time
import numpy as np
import os

# 모듈 임포트
from cam import UniversalCamera as Camera 
from model import CollisionModel
from llm_integration import describe_heatmap
from utils import bgr_to_jpeg_bytes, jpeg_bytes_to_base64

try:
    from risk_refiner import RiskRefiner
    risk_refiner = RiskRefiner()
except ImportError:
    risk_refiner = None

app = FastAPI()

# 전역 변수 설정
current_realtime_prob = 0.0  # 실시간 모델 확률 저장용
latest_description = {"text": "시스템 초기화 중...", "prob_percent": 0, "ts": 0}

# Static 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

FRAME_WIDTH = 640

# 카메라 및 모델 초기화
camera = Camera(mode="zed", source=0, width=FRAME_WIDTH, height=480)
model = CollisionModel("rebuilt_model.h5", input_size=(128, 128))
camera.set_model(model)

@app.get("/", response_class=HTMLResponse)
async def index():
    template_path = os.path.join(current_dir, "templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h2>templates/index.html 파일을 찾을 수 없습니다.</h2>")

def gen_frames():
    """실시간 영상 스트리밍 및 확률 전역 변수 업데이트"""
    global current_realtime_prob
    while True:
        res = camera.read_pred()
        if res is None:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            display_frame = frame
        else:
            display_frame, label, info = res
            current_realtime_prob = info.get('prob', 0)

        ret, jpeg = cv2.imencode(".jpg", display_frame)
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
# app.py 내의 analyze_upload 함수 수정
@app.post("/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        return {"error": "이미지 디코딩 실패"}

    # 1. 모델 분석 실행
    overlay_bgr, label, heatmap_meta = model.predict_and_gradcam(bgr)

    # 2. [핵심] 수치 증폭/보정 로직을 LLM 호출 전에 먼저 실행
    raw = heatmap_meta.get('prob', 0)
    if raw <= 0.001: final_val = 0
    elif 0.001 < raw <= 0.02: final_val = 30 + (raw * 1000) 
    elif 0.02 < raw <= 0.1: final_val = 60 + (raw * 100)
    else: final_val = 85 + (raw * 10)
    
    prob_percent = int(min(final_val, 99)) # 화면에 표시될 94% 같은 수치

    try:
        # 3. [중요] 보정된 prob_percent를 LLM에 전달!!
        result = await describe_heatmap(
            label, 
            heatmap_meta, 
            FRAME_WIDTH, 
            calibrated_prob=prob_percent  # <--- 이 부분을 반드시 넣어야 함!
        )
        llm_text = result["text"]
    except Exception as e:
        print(f"이미지 분석 중 LLM 오류: {e}")
        llm_text = f"{label} 감지됨 (분석 오류)"

    jpeg_bytes = bgr_to_jpeg_bytes(overlay_bgr)
    b64 = jpeg_bytes_to_base64(jpeg_bytes)

    return {
        "image": b64,
        "text": llm_text,           # 이제 LLM이 94%를 인지한 답변을 보냄
        "prob_percent": prob_percent, # UI 하단에 표시될 수치
    }

async def llm_worker():
    global latest_description, current_realtime_prob
    print("🚀 LLM Worker 시작됨 (초강력 모드)")
    
    while True:
        res = camera.read_pred()
        if res is None:
            await asyncio.sleep(0.5)
            continue

        _, label, info = res 

        # app.py 의 llm_worker 내부
        try:
            # 1. 먼저 수치를 증폭시킵니다.
            raw = current_realtime_prob 
            if raw <= 0.001: final_val = 0
            elif 0.001 < raw <= 0.02: final_val = 30 + (raw * 1000) 
            elif 0.02 < raw <= 0.1: final_val = 60 + (raw * 100)
            else: final_val = 85 + (raw * 10)
            prob_percent = int(min(final_val, 99))

            # 2. [수정] 증폭된 수치를 LLM 함수에 인자로 같이 보냅니다.
            result = await describe_heatmap(label, info, FRAME_WIDTH, calibrated_prob=prob_percent)
            
            latest_description = {
                "text": result["text"],
                "prob_percent": prob_percent, # 화면에 표시될 수치
                "ts": time.time(),
            }
            print(f"🔥 [SUPER BOOST] 원본: {raw:.4f} -> 결과: {prob_percent}%")
            
        except Exception as e:
            print(f"❌ LLM Worker 오류: {e}")

        await asyncio.sleep(0.8)

@app.on_event("startup")
async def startup_event():
    print(f"📂 Static 경로 확인: {static_dir}")
    camera.start()
    asyncio.create_task(llm_worker())

@app.on_event("shutdown")
def shutdown_event():
    camera.stop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)