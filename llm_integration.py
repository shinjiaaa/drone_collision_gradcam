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

async def describe_heatmap(label, info, frame_width, **kwargs):
    """
    label: 감지된 클래스명
    info: 모델 분석 정보 (prob_percent, ttc, trend_str, bbox 포함)
    frame_width: 화면 너비
    **kwargs: calibrated_prob 등 예상치 못한 인자 대응
    """
    # 1. 데이터 추출 (다양한 인자 이름 대응)
    prob_percent = kwargs.get('calibrated_prob') 
    if prob_percent is None:
        prob_percent = info.get('prob_percent', 0)
    
    # 소수점일 경우를 대비해 한 번 더 정수화
    prob_percent = int(prob_percent if prob_percent >= 1 else prob_percent * 100)
        
    ttc = info.get('ttc', 99.0)
    trend = info.get('trend_str', "유지")
    bbox = info.get('bbox')
    
    # 2. 장애물 위치 및 회피 방향 매핑
    pos = get_direction_from_bbox(bbox, frame_width)
    
    evade_map = {
        "중앙 전방": "좌측 혹은 우측으로 회피",
        "좌측": "우측으로 회피",
        "우측": "좌측으로 회피",
        "전방": "후방으로 기동"
    }
    evade_dir = evade_map.get(pos, "경로 재탐색")

    # [지연 방지 필터링] 위험도가 낮고 안정적이면 LLM 호출 생략
    if prob_percent < 40 and trend != "급상승":
        return {"text": "현재 경로 안전 확보 중.", "prob_percent": prob_percent}

    # 3. LLM 페르소나 및 지침
    system_prompt = (
        "당신은 드론 지능형 부조종사입니다. 조종사에게 수치적 데이터와 명확한 행동 지침만 전달하세요. "
        "0.97 같은 소수점은 절대 쓰지 마세요. 안정적이라는 안심 멘트도 하지 마세요."
    )

    # 4. 프롬프트 구성
    ttc_text = f"{ttc:.1f}초 후 충돌! " if ttc < 10 else ""
    warning_suffix = " 경고!" if trend == "급상승" else ""

    # 1. TTC 문구 사전 준비 (10초 미만일 때만 구체적인 초 명시)
    if ttc < 10:
        ttc_sentence = f"{ttc:.1f}초 후 충돌 예정입니다. "
    else:
        ttc_sentence = ""

    # 2. 프롬프트 구성
    user_query = f"""
    [실시간 분석 데이터]
    - 위험률: {prob_percent}%
    - 예상시간: {f'{ttc:.1f}초' if ttc < 10 else '여유 있음'}
    - 장애물 위치: {pos}
    - 추세: {trend}
    - 회피 방향: {evade_dir}

    [작성 지침]
    1. 다음 형식을 반드시 지켜서 한 문장으로 출력하세요: {pos}에 장애물이 감지되었습니다. {ttc_sentence}{evade_dir}하세요.{' 경고!' if trend == '급상승' else ''}
    2. 소수점 수치(0.97 등)나 다른 설명은 일절 배제하세요.
    3. 반드시 위 형식에 맞춰 조종사가 즉각 행동할 수 있도록 짧고 단호하게 답하세요.
    """

    # 5. API 호출
    text = await call_openai_api(system_prompt, user_query)

    return {
        "text": text,
        "prob_percent": prob_percent
    }