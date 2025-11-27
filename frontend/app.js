const fileInput = document.getElementById("fileinput");
const sendBtn = document.getElementById("send");
const resultDiv = document.getElementById("result");
const predSpan = document.getElementById("pred");
const probsSpan = document.getElementById("probs");
const outImg = document.getElementById("outimg");
const loading = document.getElementById("loading");

let selectedFile = null;
const API_BASE = "http://localhost:8000/api";

fileInput.addEventListener("change", (e) => {
  selectedFile = e.target.files[0];
  sendBtn.disabled = !selectedFile;
});

sendBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  loading.classList.remove("hidden");
  resultDiv.classList.add("hidden");

  const fd = new FormData();
  fd.append("file", selectedFile);

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: fd
    });

    if (!res.ok) {
      const err = await res.json();
      alert("서버 오류: " + (err.detail || JSON.stringify(err)));
      return;
    }

    const data = await res.json();     // ⬅ 먼저 선언
    console.log("서버 응답:", data);   // ⬅ 그 다음 사용

    predSpan.textContent = data.prediction;

    if (data.probabilities && Array.isArray(data.probabilities)) {
      probsSpan.textContent = data.probabilities
        .map(v => v.toFixed(3))
        .join(", ");
    } else {
      probsSpan.textContent = "확률 없음";
    }

    outImg.src = "data:image/jpeg;base64," + data.image_base64;



    resultDiv.classList.remove("hidden");

  } catch (e) {
    alert("네트워크 오류: " + e.message);
  } finally {
    loading.classList.add("hidden");
  }
});
