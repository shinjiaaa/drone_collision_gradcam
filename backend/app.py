# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import router as api_router

def create_app():
    app = FastAPI(title="Collision Grad-CAM API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 개발용
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True
    )
    # ⚡ 핵심: 라우터 등록
    app.include_router(api_router, prefix="/api")
    return app

app = create_app()
