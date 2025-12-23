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
                model=GPT_MODEL, messages=messages, temperature=0.7 
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
    """
    히트맵의 빨간색 영역(BBox)의 중심점을 계산하여 위치 판정
    """
    if not bbox:
        return "중앙 전방"
    
    x, y, w, h = bbox
    # 박스의 중심점 계산 (중앙을 기준으로 판정 정확도 향상)
    center_x = x + (w / 2)
    
    # 프레임을 3등분 (35% / 30% / 35% 비중)
    left_threshold = frame_width * 0.35
    right_threshold = frame_width * 0.65
    
    if center_x < left_threshold:
        return "좌측"
    elif center_x > right_threshold:
        return "우측"
    else:
        return "중앙 전방"

async def describe_heatmap(label, info, frame_width, calibrated_prob=None):
    """
    히트맵 시각화 결과와 위험 수치를 결합하여 자연스러운 안내문 생성
    """
    # 1. 수치 보정 확인 (전달받은 증폭 수치가 없으면 백업 계산)
    if calibrated_prob is None:
        refined_prob = info.get('refined_prob', 0.0)
        calibrated_prob = int(refined_prob * 100)

    bbox = info.get('bbox')
    object_name = LABEL_MAP.get(label, label)
    
    # 2. 위치 판정 (히트맵 중심점 기준)
    position_text = get_direction_from_bbox(bbox, frame_width)

    # 3. 위험 등급 설정
    if calibrated_prob >= 85:
        level = "매우 높음 (즉시 회피 기동 필요)"
    elif calibrated_prob >= 75:
        level = "주의 (장애물 감지)"
    else:
        level = "안전 (정상 비행)"

    # 4. 프롬프트 구성 (히트맵 시각적 일치성 강조 및 기술 용어 제거)
    system_prompt = (
        "당신은 드론의 지능형 안전 보조 AI입니다. Grad-CAM 히트맵 분석 결과를 바탕으로 조종사에게 브리핑하세요.\n"
        "특히 히트맵에서 '빨간색으로 강조된 영역'은 현재 가장 위험한 지점입니다.\n"
        "조종사가 보고 있는 빨간색 영역과 당신의 설명이 정확히 일치해야 신뢰를 얻을 수 있습니다."
    )

    user_query = f"""
    [시각적 분석 데이터]
    - 감지 대상: {object_name}
    - 히트맵 주요 강조 지점(빨간 영역 위치): {position_text}
    - 현재 충돌 위험도: {calibrated_prob}%
    - 현재 위험 등급: {level}

    [작성 지침 - 중요]
    1. '보정된', '알고리즘', '수치상' 같은 딱딱한 기술 용어는 절대 쓰지 마세요.
    2. "현재 충돌 위험도는 {calibrated_prob}%입니다."라고 문장을 시작하세요.
    3. 히트맵의 빨간색 영역인 '{position_text}'에 위험 요소가 있음을 알리세요.
    4. 반드시 해당 빨간색 영역({position_text})을 피해서 반대 방향으로 이동하라는 구체적인 지시를 포함해 2문장으로 답하세요.
    """

    # OpenAI API 호출
    text = await call_openai_api(system_prompt, user_query)

    return {
        "text": text,
        "prob_percent": calibrated_prob
    }