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
    print("OpenAI API 클라이언트 초기화 실패. API 키를 확인하세요.")

GPT_MODEL = "gpt-3.5-turbo"

LABEL_MAP = {
    "collision": "장애물",
    "person": "사람",
    "car": "차량",
    "normal": "정상 주행 경로"
}

async def call_openai_api(system_prompt, user_query):
    if client is None:
        return "모델 해석 생성 중 오류가 발생했습니다. (OpenAI 클라이언트 미초기화)"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL, messages=messages, temperature=0.7 # 약간의 창의성을 위해 0.7 권장
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
                return "상황 분석 중 오류가 발생했습니다. 수동 조종에 주의하세요."
    return "응답 지연으로 인해 해석을 생성할 수 없습니다."

def get_direction_from_bbox(bbox, frame_width):
    x, y, w, h = bbox
    center_x = x + w / 2
    if center_x < frame_width * 0.33:
        return "좌측"
    elif center_x > frame_width * 0.66:
        return "우측"
    else:
        return "중앙 전방"

async def describe_heatmap(label, info, frame_width):
    """
    label: "collision" 또는 "normal"
    info: {'bbox': (x, y, w, h), 'prob': 원본확률, 'refined_prob': 보정확률, 'is_valid': 보정결과}
    """
    refined_prob = info.get('refined_prob', 0.0)
    prob_percent = round(refined_prob * 100, 1)
    bbox = info.get('bbox')
    object_name = LABEL_MAP.get(label, label)
    
    # 1. 장애물 위치 분석
    position_text = "전방"
    if bbox:
        position_text = get_direction_from_bbox(bbox, frame_width)

    # 2. 위험 수준 정의
    if refined_prob > 0.8:
        level = "매우 높음 (즉시 회피 필요)"
    elif refined_prob > 0.5:
        level = "주의 (장애물 감지)"
    else:
        level = "안전 (정상 비행)"

    # 3. 프롬프트 구성
    system_prompt = (
        "당신은 드론의 지능형 안전 보조 시스템입니다. 조종사에게 상황을 설명하고 안전 지침을 제공하세요.\n"
        "Grad-CAM 시각화와 위험률 보정 알고리즘을 통해 도출된 결과를 바탕으로 말하세요."
    )

    user_query = f"""
    [현재 분석 데이터]
    - 감지된 대상: {object_name}
    - 장애물 추정 위치: {position_text}
    - 보정된 충돌 위험도: {prob_percent}%
    - 현재 위험 수준: {level}

    [미션]
    1. 현재 상황을 조종사에게 한국어로 명확히 설명하세요.
    2. '시각적 근거' 또는 '위험률 보정'이 적용되었음을 은연중에 언급하여 신뢰성을 확보하세요.
    3. 장애물 위치를 피할 수 있는 구체적인 회피 방향을 포함해 2-3문장으로 답하세요.
    """

    # 실제 API 호출
    text = await call_openai_api(system_prompt, user_query)

    return {
        "text": text,
        "prob_percent": prob_percent
    }