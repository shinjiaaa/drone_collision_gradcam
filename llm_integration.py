import os
import json
import time
import random
import asyncio

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None
    print("⚠️ 경고: OpenAI API 클라이언트 초기화 실패. API 키를 확인해주세요.")

GPT_MODEL = "gpt-3.5-turbo"


async def call_openai_api(system_prompt, user_query):
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


async def describe_heatmap(label, heatmap_meta):
    bbox = heatmap_meta.get("bbox")
    prob_percent = heatmap_meta.get("prob_percent")
    prob = heatmap_meta.get("prob")

    # if bbox is not None:
    #     x, y, w, h = bbox
    #     if w > 0 and h > 0:
    #         bbox_desc = (
    #             f"Area detected at position ({x}, {y}) with size {w}x{h} pixels."
    #         )
    #     else:
    #         bbox_desc = "Diffuse focus without a clear bounding box."
    # else:
    #     bbox_desc = "No distinct highlighted area (bbox) found, focus is diffuse."

    # system_prompt = (
    #     "You are an AI-powered safety assistant for an autonomous system. "
    #     "Your task is to interpret model predictions and Grad-CAM data (heatmap location) "
    #     "and provide a single, actionable, and very short textual alert. "
    #     "Use simple, direct language. Start with the alert level (e.g., 'HIGH ALERT', 'MONITOR')."
    # )

    # user_query = (
    #     f"The model predicted '{label}' with a probability of {prob_percent}. "
    #     f"The Grad-CAM heatmap shows a significant focus on an area. {bbox_desc}. "
    #     f"Based on this, provide a concise, single-sentence alert that combines "
    #     f"the risk level and the potential location of the threat. If probability is > 0.7, use HIGH ALERT. "
    #     f"If probability is between 0.5 and 0.7, use WARNING. If < 0.5, use MONITOR."
    # )

    if bbox is not None:
        x, y, w, h = bbox
        if w > 0 and h > 0:
            bbox_desc = f"Grad-CAM이 ({x}, {y}) 위치에서 {w}x{h} 픽셀 크기의 영역에 집중하고 있습니다."
        else:
            bbox_desc = "명확한 경계 상자가 없어, 관심 영역이 퍼져 있는 형태입니다."
    else:
        bbox_desc = "경계 상자(bbox)가 감지되지 않아, 관심 영역이 뚜렷하지 않습니다."

    system_prompt = (
        "당신은 자율 시스템을 위한 AI 기반 안전 보조장치입니다. "
        "모델 예측 결과와 Grad-CAM 데이터(열지도 집중 영역)를 해석하여 "
        "단 하나의 행동 지향적이고 매우 짧은 경고 문장을 생성하는 것이 역할입니다. "
        "문장은 단순하고 직접적으로 표현하며, 경고 수준(예: '고위험', '주의 필요')으로 시작해야 합니다."
    )

    user_query = (
        f"모델이 '{label}'을(를) {prob_percent} 확률로 예측했습니다. "
        f"Grad-CAM은 특정 영역에 높은 집중을 보였습니다. {bbox_desc} "
        f"이를 바탕으로 위험 수준과 잠재적 위험 위치를 결합한 "
        f"간결한 한 문장 경고를 생성하세요. "
        f"확률이 0.7 이상이면 '고위험', 0.5~0.7 사이는 '주의', 0.5 미만은 '안전'을 사용하세요."
    )

    llm_explanation = await call_openai_api(system_prompt, user_query)

    return llm_explanation
