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

app = FastAPI()

# static 폴더가 없을 경우 자동 생성 (에러 방지)
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

FRAME_WIDTH = 640

# 카메라 및 모델 초기화 (mode="zed" 또는 "webcam")
camera = Camera(mode="zed", source=0, width=FRAME_WIDTH, height=480)
model = CollisionModel("rebuilt_model.h5", input_size=(128, 128))
camera.set_model(model)

# 전역 변수: 최신 LLM 분석 결과 저장
latest_description = {"text": "시스템 초기화 중...", "prob_percent": 0, "ts": 0}

@app.get("/", response_class=HTMLResponse)
async def index():
    # templates/index.html 파일이 있는지 확인 필요
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h2>index.html 파일을 찾을 수 없습니다.</h2>")

def gen_frames():
    """실시간 비디오 스트리밍을 위한 제너레이터"""
    while True:
        # cam.py에서 추가한 read_pred() 사용
        res = camera.read_pred()
        
        if res is None:
            # 아직 예측 결과가 없으면 원본 프레임이라도 시도
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            display_frame = frame
        else:
            # 오버레이 이미지(Grad-CAM 합본) 추출
            display_frame, label, info = res

        # JPEG 인코딩
        ret, jpeg = cv2.imencode(".jpg", display_frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )

@app.get("/video_feed")
def video_feed():
    """웹 페이지에 비디오 스트림 전달"""
    return StreamingResponse(
        gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/latest_description")
async def get_description():
    """프론트엔드에서 최신 LLM 텍스트를 가져가는 엔드포인트"""
    return JSONResponse(latest_description)

async def llm_worker():
    """백그라운드에서 주기적으로 LLM 분석 수행"""
    global latest_description
    print("🚀 LLM Worker 시작됨")
    
    while True:
        res = camera.read_pred()
        if res is None:
            await asyncio.sleep(0.5)
            continue

        _, label, info = res # info에는 refined_prob, bbox 등이 포함됨

        try:
            # LLM 분석 호출
            result = await describe_heatmap(label, info, FRAME_WIDTH)
            
            # 전역 변수 업데이트
            latest_description = {
                "text": result["text"],
                "prob_percent": result.get("prob_percent", 0),
                "ts": time.time(),
            }
        except Exception as e:
            print(f"❌ LLM Worker 오류: {e}")

        # LLM API 비용 및 부하를 고려해 1~2초 간격으로 수행
        await asyncio.sleep(1.5)

@app.post("/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    """이미지 업로드 분석 (실시간 외 수동 분석용)"""
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        return {"error": "이미지 디코딩 실패"}

    overlay_bgr, label, info = model.predict_and_gradcam(bgr)
    
    # 수동 업로드 시에도 LLM 설명 생성
    try:
        result = await describe_heatmap(label, info, FRAME_WIDTH)
    except Exception:
        result = {"text": f"{label} 감지됨", "prob_percent": round(info.get('prob', 0)*100, 1)}

    jpeg_bytes = bgr_to_jpeg_bytes(overlay_bgr)
    b64 = jpeg_bytes_to_base64(jpeg_bytes)

    return {
        "image": b64,
        "text": result["text"],
        "prob_percent": result["prob_percent"],
    }

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 카메라 가동 및 LLM 태스크 할당"""
    camera.start()
    asyncio.create_task(llm_worker())

@app.on_event("shutdown")
def shutdown_event():
    """서버 종료 시 카메라 자원 해제"""
    camera.stop()

if __name__ == "__main__":
    # 포트 8000에서 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)