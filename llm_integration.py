import os
import json
import time
import random
import asyncio

# --- OpenAI API 설정 및 클라이언트 초기화 ---
# 이 파일에서만 OpenAI 관련 의존성을 관리합니다.
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# .env 파일에서 API 키를 로드하여 클라이언트 초기화
try:
    # Canvas 환경에서는 API 키가 자동으로 제공되므로, 실제 배포 환경을 위해 dotenv 로직을 유지합니다.
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None
    print("⚠️ 경고: OpenAI API 클라이언트 초기화 실패. API 키를 확인해주세요.")

# 사용할 GPT 모델 정의
GPT_MODEL = "gpt-3.5-turbo"


# --------------------------------------------------
# LLM API 통신 함수 (OpenAI GPT)
# --------------------------------------------------
async def call_openai_api(system_prompt, user_query):
    """지수 백오프를 사용하여 OpenAI GPT API를 호출합니다."""

    if client is None:
        return "모델 해석 생성 중 오류가 발생했습니다. (OpenAI 클라이언트 미초기화)"

    default_response = "모델 해석 생성 중 오류가 발생했습니다."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    max_retries = 3
    base_delay = 1.0  # 초

    for attempt in range(max_retries):
        try:
            # OpenAI API 호출은 동기식이지만, async 함수 내에서 비동기로 다루어집니다.
            response = client.chat.completions.create(
                model=GPT_MODEL, messages=messages, temperature=0.0
            )

            text_part = response.choices[0].message.content.strip()

            if text_part:
                return text_part
            else:
                raise Exception("GPT response missing text part.")

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(delay)
            else:
                return default_response

    return default_response


# --------------------------------------------------
# Grad-CAM 설명 함수 (Textual Explanation Generator)
# --------------------------------------------------
async def describe_heatmap(label, heatmap_meta):
    """
    Grad-CAM 메타데이터를 기반으로 LLM에 자연어 설명을 요청합니다.
    """
    bbox = heatmap_meta.get("bbox")
    prob_percent = heatmap_meta.get("prob_percent")
    prob = heatmap_meta.get("prob")

    # Bounding Box 정보를 간결하게 포맷팅
    if bbox is not None:
        x, y, w, h = bbox
        if w > 0 and h > 0:
            bbox_desc = (
                f"Area detected at position ({x}, {y}) with size {w}x{h} pixels."
            )
        else:
            bbox_desc = "Diffuse focus without a clear bounding box."
    else:
        bbox_desc = "No distinct highlighted area (bbox) found, focus is diffuse."

    # LLM에게 전달할 시스템 역할 정의
    system_prompt = (
        "You are an AI-powered safety assistant for an autonomous system. "
        "Your task is to interpret model predictions and Grad-CAM data (heatmap location) "
        "and provide a single, actionable, and very short textual alert. "
        "Use simple, direct language. Start with the alert level (e.g., 'HIGH ALERT', 'MONITOR')."
    )

    # LLM에게 전달할 사용자 쿼리 (확률을 백분율로 표시)
    user_query = (
        f"The model predicted '{label}' with a probability of {prob_percent}. "
        f"The Grad-CAM heatmap shows a significant focus on an area. {bbox_desc}. "
        f"Based on this, provide a concise, single-sentence alert that combines "
        f"the risk level and the potential location of the threat. If probability is > 0.7, use HIGH ALERT. "
        f"If probability is between 0.5 and 0.7, use WARNING. If < 0.5, use MONITOR."
    )

    llm_explanation = await call_openai_api(system_prompt, user_query)

    return llm_explanation
