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
}


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


def get_avoid_direction(direction):
    if direction == "좌측":
        return "우측"
    elif direction == "우측":
        return "좌측"
    else:
        return "상승"


def get_direction_from_bbox(bbox, frame_width):
    """
    bbox 중심 x 좌표 기준으로 좌/전/우 판단
    """
    x, y, w, h = bbox
    center_x = x + w / 2

    if center_x < frame_width / 3:
        return "좌측"
    elif center_x > frame_width * 2 / 3:
        return "우측"
    else:
        return "전방"


async def describe_heatmap(label, heatmap_meta, frame_width):
    bbox = heatmap_meta.get("bbox")
    prob = heatmap_meta.get("prob")

    object_name = LABEL_MAP.get(label, label)

    if prob is None:
        prob_percent = "알 수 없음"
    else:
        prob_percent = f"{int(prob * 100)}%"

    if bbox and bbox[2] > 0 and bbox[3] > 0:
        direction = get_direction_from_bbox(bbox, frame_width)
    else:
        direction = "전방"

    avoid_direction = get_avoid_direction(direction)

    system_prompt = (
        "당신은 자율 이동체용 충돌 회피 경고 문장 생성기입니다.\n"
        "아래 정보를 사용하여 반드시 한 문장의 경고를 생성하세요.\n"
        "다른 설명은 절대 추가하지 마세요."
    )

    user_query = (
        f"{direction}에 있는 {object_name}과 충돌할 확률이 {prob_percent}입니다. "
        f"{avoid_direction}으로 회피하세요."
    )

    text = await call_openai_api(system_prompt, user_query)

    return {"text": text, "prob_percent": prob_percent}
