// ----------------------------------------
// 위험률 표시 업데이트
// ----------------------------------------
function updateRiskIndicator(probPercentStr) {
    const riskElement = document.getElementById('risk-percent');

    // 클래스 초기화
    riskElement.className = 'risk-level-label';

    // "87.32%" → 87.32
    const prob = parseFloat(probPercentStr.replace('%', ''));

    let className;
    let statusText;

    if (prob >= 85) {
        className = 'danger';
        statusText = '위험';
    } else if (prob >= 75) {
        className = 'warning';
        statusText = '주의';
    } else {
        className = 'safe';
        statusText = '안전';
    }

    riskElement.classList.add(className);
    riskElement.textContent = `${probPercentStr} (${statusText})`;
}

// ----------------------------------------
// 주기적 LLM 설명 + 위험률 갱신
// ----------------------------------------
async function fetchDesc() {
    try {
        const res = await fetch('/latest_description');
        const j = await res.json();

        // LLM 설명
        document.getElementById('desc').textContent = j.text || '설명 없음';

        // 위험률
        if (j.prob_percent !== undefined && j.prob_percent !== null) {
            updateRiskIndicator(j.prob_percent);
        } else {
            document.getElementById('risk-percent').textContent = '--% (로딩 중)';
        }

    } catch (e) {
        console.error('LLM 설명 및 위험률 로딩 실패:', e);
        document.getElementById('desc').textContent = '⚠️ 설명 로딩 중 오류 발생';
        document.getElementById('risk-percent').textContent = '--% (오류)';
    }
}

// 최초 1회 + 주기적 호출
document.addEventListener('DOMContentLoaded', fetchDesc);
setInterval(fetchDesc, 1200);

// ----------------------------------------
// 이미지 업로드 및 분석
// ----------------------------------------
async function uploadImage() {
    const fileInput = document.getElementById('imgInput');
    const file = fileInput.files[0];
    const riskElement = document.getElementById('risk-percent');
    const descElement = document.getElementById('desc');

    if (!file) {
        alert('이미지를 선택하세요!');
        return;
    }

    // UI 초기 상태
    riskElement.textContent = '... 분석 중 ...';
    riskElement.className = 'risk-level-label';
    descElement.textContent = '이미지를 서버로 전송하여 분석을 시작합니다...';

    // 업로드 이미지 미리보기
    const reader = new FileReader();
    reader.onload = () => {
        document.getElementById('uploadedPreview').src = reader.result;
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/analyze_upload', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            throw new Error(`HTTP 오류! 상태 코드: ${res.status}`);
        }

        const j = await res.json();

        // 결과 이미지 표시
        document.getElementById('uploadedPreview').src =
            `data:image/jpeg;base64,${j.image}`;

        // LLM 설명
        descElement.textContent = j.text || 'LLM 설명 없음';

        // 위험률 (prob_percent만 사용)
        if (j.prob_percent !== undefined && j.prob_percent !== null) {
            updateRiskIndicator(j.prob_percent);
        } else {
            riskElement.textContent = '분석 완료 (확률 정보 없음)';
            riskElement.classList.add('safe');
        }

    } catch (err) {
        console.error('이미지 분석 중 오류 발생:', err);
        alert(`이미지 분석 중 오류 발생: ${err.message}`);
        descElement.textContent = '❌ 이미지 분석 실패';
        riskElement.textContent = '--% (실패)';
        riskElement.className = 'risk-level-label danger';
    }
}
