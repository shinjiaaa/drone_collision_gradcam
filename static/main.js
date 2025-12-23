/**
 * 위험 수치 및 레이블 업데이트 (타겟 지정 가능)
 * @param {number|string} probValue - 서버에서 넘어온 확률값
 * @param {string} targetId - 업데이트할 HTML 요소의 ID (기본값: 실시간용 risk-percent)
 */
function updateRiskIndicator(probValue, targetId = 'risk-percent') {
    const riskElement = document.getElementById(targetId);
    if (!riskElement) return;

    // 초기화: 기존 클래스 제거 후 기본 클래스 추가
    riskElement.className = 'risk-level-label';

    // 1. 값 처리 (숫자든 문자열이든 안전하게 숫자로 변환)
    let prob = 0;
    if (typeof probValue === 'string') {
        prob = parseFloat(probValue.replace('%', ''));
    } else {
        prob = parseFloat(probValue);
    }
    if (isNaN(prob)) prob = 0;

    // 2. 위험도 등급 판별
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

    // 3. UI 업데이트 (정수로 출력)
    riskElement.classList.add(className);
    riskElement.textContent = `${Math.floor(prob)}% (${statusText})`;
}

/**
 * [실시간] LLM 설명 및 확률 정보 가져오기
 */
async function fetchDesc() {
    try {
        const res = await fetch('/latest_description'); 
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        
        const j = await res.json();

        // 실시간용 텍스트 영역 (desc) 업데이트
        const descElement = document.getElementById('desc');
        if (descElement) {
            descElement.textContent = j.text || '설명 없음';
        }

        // 실시간용 위험도 영역 (risk-percent) 업데이트
        if (j.prob_percent !== undefined && j.prob_percent !== null) {
            updateRiskIndicator(j.prob_percent, 'risk-percent');
        }

    } catch (e) {
        console.error('실시간 로딩 실패:', e);
    }
}

// 1.2초마다 실시간 갱신 실행
document.addEventListener('DOMContentLoaded', fetchDesc);
setInterval(fetchDesc, 1200);

/**
 * [업로드] 이미지 파일 분석 및 결과 표시
 */
async function uploadImage() {
    const fileInput = document.getElementById('imgInput');
    const file = fileInput.files[0];
    
    // 업로드 전용 타겟 요소들
    const uploadRiskElement = document.getElementById('upload-risk-percent');
    const uploadDescElement = document.getElementById('upload-desc');
    const preview = document.getElementById('uploadedPreview');

    if (!file) {
        alert('이미지를 선택하세요!');
        return;
    }

    // 업로드 전용 영역 로딩 상태 표시
    if (uploadRiskElement) {
        uploadRiskElement.textContent = '... 분석 중 ...';
        uploadRiskElement.className = 'risk-level-label';
    }
    if (uploadDescElement) {
        uploadDescElement.textContent = 'AI가 업로드된 이미지를 정밀 분석하고 있습니다...';
    }

    // 선택한 이미지 미리보기 표시
    const reader = new FileReader();
    reader.onload = () => { if (preview) preview.src = reader.result; };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const res = await fetch('/analyze_upload', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            if (res.status === 404) {
                throw new Error("서버에서 업로드 경로(/analyze_upload)를 찾을 수 없습니다.");
            }
            throw new Error(`서버 오류 (상태 코드: ${res.status})`);
        }

        const j = await res.json();

        // 1. 결과 이미지 (Grad-CAM) 표시
        if (preview && j.image) {
            preview.src = `data:image/jpeg;base64,${j.image}`;
        }

        // 2. 업로드 전용 설명글 (upload-desc) 업데이트
        if (uploadDescElement) {
            uploadDescElement.textContent = j.text || '분석 결과 설명이 없습니다.';
        }

        // 3. 업로드 전용 위험도 (upload-risk-percent) 업데이트
        if (j.prob_percent !== undefined && j.prob_percent !== null) {
            updateRiskIndicator(j.prob_percent, 'upload-risk-percent');
        } else {
            if (uploadRiskElement) {
                uploadRiskElement.textContent = '분석 완료';
                uploadRiskElement.className = 'risk-level-label safe';
            }
        }

    } catch (err) {
        console.error('이미지 분석 중 오류 발생:', err);
        alert(err.message);
        if (uploadDescElement) uploadDescElement.textContent = '이미지 분석 실패';
        if (uploadRiskElement) {
            uploadRiskElement.textContent = '--%';
            uploadRiskElement.className = 'risk-level-label danger';
        }
    }
}