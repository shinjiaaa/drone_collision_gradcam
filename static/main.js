
function updateRiskIndicator(probPercentStr) {
    const riskElement = document.getElementById('risk-percent');
    
    riskElement.className = 'risk-level-label';

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

async function fetchDesc() {
    try {
        const res = await fetch('/latest_description');
        const j = await res.json();
        
        // 1. LLM 텍스트 업데이트
        document.getElementById('desc').textContent = j.text || '설명 없음';
        
        // 2. 위험률 업데이트
        if (j.prob_percent) {
            updateRiskIndicator(j.prob_percent);
        } else {
            document.getElementById('risk-percent').textContent = '--% (로딩 중)';
        }

    } catch (e) {
        console.error("LLM 설명 및 위험률 로딩 실패:", e);
        document.getElementById('desc').textContent = '⚠️ 설명 로딩 중 오류 발생';
        document.getElementById('risk-percent').textContent = '--% (오류)';
    }
}

setInterval(fetchDesc, 1200);
document.addEventListener('DOMContentLoaded', fetchDesc);

async function uploadImage() {
    const fileInput = document.getElementById("imgInput");
    const file = fileInput.files[0];
    const riskElement = document.getElementById('risk-percent');
    const descElement = document.getElementById("desc");

    if (!file) {
        alert("이미지를 선택하세요!");
        return;
    }
    riskElement.textContent = '... 분석 중 ...';
    riskElement.className = 'risk-level-label';
    descElement.textContent = '이미지를 서버로 전송하여 분석을 시작합니다...';

    const reader = new FileReader();
    reader.onload = () => {
        document.getElementById("uploadedPreview").src = reader.result;
    };
    reader.readAsDataURL(file);

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
        
        document.getElementById("uploadedPreview").src = `data:image/jpeg;base64,${j.image}`;
        
        descElement.textContent = j.text || "LLM 설명 없음";
        
        const probMatch = j.text.match(/(\d+\.\d{2})%/);
        if (probMatch) {
            updateRiskIndicator(probMatch[1] + '%');
        } else {
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