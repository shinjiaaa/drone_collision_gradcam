
// -------------------------------
// 충돌 위험률 시각화 함수
// -------------------------------
function updateRiskIndicator(probPercentStr) {
    const riskElement = document.getElementById('risk-percent');
    
    // 기본 상태 초기화
    riskElement.className = 'risk-level-label';

    // 문자열에서 숫자 추출 (예: "78.50%" -> 78.5)
    const prob = parseFloat(probPercentStr.replace('%', ''));

    let className = 'safe';
    let statusText = '안전';

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

// -------------------------------
// LLM 설명 및 위험률 주기적 업데이트
// -------------------------------
async function fetchDesc() {
    try {
        const res = await fetch('/latest_description');
        const j = await res.json();
        
        // 1. LLM 텍스트 업데이트
        document.getElementById('desc').textContent = j.text || '설명 없음';
        
        // 2. 위험률 업데이트 (prob_percent 필드가 존재한다고 가정)
        if (j.prob_percent) {
            updateRiskIndicator(j.prob_percent);
        } else {
            // 초기 로딩 또는 데이터 부재 시
            document.getElementById('risk-percent').textContent = '--% (로딩 중)';
        }

    } catch (e) {
        console.error("LLM 설명 및 위험률 로딩 실패:", e);
        // 오류 발생 시 UI에 표시
        document.getElementById('desc').textContent = '⚠️ 설명 로딩 중 오류 발생';
        document.getElementById('risk-percent').textContent = '--% (오류)';
    }
}

// 1.2초마다 업데이트
setInterval(fetchDesc, 1200);
// 페이지 로드 시 즉시 호출
document.addEventListener('DOMContentLoaded', fetchDesc);


// -------------------------------
// 이미지 업로드 분석
// -------------------------------
async function uploadImage() {
    const fileInput = document.getElementById("imgInput");
    const file = fileInput.files[0];
    const riskElement = document.getElementById('risk-percent');
    const descElement = document.getElementById("desc");

    if (!file) {
        alert("이미지를 선택하세요!");
        return;
    }

    // 분석 중임을 표시
    riskElement.textContent = '... 분석 중 ...';
    riskElement.className = 'risk-level-label';
    descElement.textContent = '이미지를 서버로 전송하여 분석을 시작합니다...';

    // 미리보기 표시
    const reader = new FileReader();
    reader.onload = () => {
        document.getElementById("uploadedPreview").src = reader.result;
    };
    reader.readAsDataURL(file);

    // 서버로 전송
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/analyze_upload", {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            throw new Error(`HTTP 오류! 상태 코드: ${res.status}`);
        }

        const j = await res.json();
        
        // 업로드 분석 결과 이미지 표시
        document.getElementById("uploadedPreview").src = `data:image/jpeg;base64,${j.image}`;
        
        // LLM 설명 및 위험률 업데이트
        descElement.textContent = j.text || "LLM 설명 없음";
        
        // 'analyze_upload' 결과는 확률 정보가 포함되어 있다고 가정합니다.
        // LLM 설명 텍스트 자체에서 확률을 파싱해야 하므로, 이 부분은 LLM 응답 포맷에 따라 조정이 필요합니다.
        
        // 임시로 LLM 텍스트에서 확률을 추출하는 로직 (만약 LLM 응답 포맷이 고정된다면)
        // 예: "HIGH ALERT: Collision probability 92.50%. The model... "
        const probMatch = j.text.match(/(\d+\.\d{2})%/);
        if (probMatch) {
            updateRiskIndicator(probMatch[1] + '%');
        } else {
            // 확률 정보가 없으면 기본값 표시
            riskElement.textContent = '분석 완료 (확률 정보 없음)';
            riskElement.classList.add('safe');
        }

    } catch (err) {
        console.error("이미지 분석 중 오류 발생:", err);
        alert(`이미지 분석 중 오류 발생: ${err.message}`);
        descElement.textContent = '❌ 이미지 분석 실패';
        riskElement.textContent = '--% (실패)';
        riskElement.className = 'risk-level-label danger';
    }
}