import os
import asyncio

# 선택적으로 openai 라이브러리를 사용 (없으면 fallback 사용)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_AVAILABLE and OPENAI_KEY:
    openai.api_key = OPENAI_KEY


# --------------------------------------------------
# Grad-CAM 설명 함수
# --------------------------------------------------
async def describe_heatmap(label, heatmap_meta):
    """
    LLM에 heatmap 요약(bbox, 확률 등)을 보내 자연어 설명을 받아온다.
    실패하면 간단한 rule-based 설명을 반환함.
    """

    bbox = heatmap_meta.get("bbox") if isinstance(heatmap_meta, dict) else None
    prob = heatmap_meta.get("prob") if isinstance(heatmap_meta, dict) else None

    # --------------------------------------------------
    # OpenAI API 사용 가능할 때
    # --------------------------------------------------
    if OPENAI_AVAILABLE and OPENAI_KEY:
        system_prompt = (
            "You describe Grad-CAM highlighted regions for a drone collision "
            "detection system. Keep outputs short and practical."
        )

        user_prompt = (
            f"Prediction: {label}. "
            f"Probability: {prob:.3f}. "
            f"Heatmap bbox: {bbox}. "
            "Explain what the highlighted area likely represents "
            "(e.g., object ahead, ground reflection, drone propeller), "
            "and give one short suggestion to reduce false positives."
        )

        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=80,
                temperature=0.3,
            )
            text = resp["choices"][0]["message"]["content"].strip()
            return text

        except Exception:
            pass  # 실패 시 fallback 아래에서 처리

    # --------------------------------------------------
    # Fallback (LLM 사용 불가)
    # --------------------------------------------------
    if prob is None:
        return f"모델 예측: {label}. Heatmap 정보가 부족합니다."

    if bbox is None:
        return f"모델 예측: {label} (확률 {prob:.2f}). 강조된 영역이 없습니다."

    x, y, w, h = bbox
    desc = (
        f"모델 예측: {label} (확률 {prob:.2f}). "
        f"강조된 영역은 화면 내 {w}x{h} 크기의 직사각형입니다."
    )
    suggestion = "카메라 각도 조정 또는 배경 노이즈 필터링을 고려하세요."

    return desc + " " + suggestion
