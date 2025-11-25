const fileInput = document.getElementById("fileinput");
const sendBtn = document.getElementById("send");
const resultDiv = document.getElementById("result");
const predSpan = document.getElementById("pred");
const probsSpan = document.getElementById("probs");
const outImg = document.getElementById("outimg");
const loading = document.getElementById("loading");

let selectedFile = null;
const API_BASE = "http://localhost:8000/api"; // backend에서 /api/predict 사용

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
    const data = await res.json();
    predSpan.textContent = data.pred;
    probsSpan.textContent = data.probs.map(v => v.toFixed(3)).join(", ");
    outImg.src = data.image;
    resultDiv.classList.remove("hidden");
  } catch (e) {
    alert("네트워크 오류: " + e.message);
  } finally {
    loading.classList.add("hidden");
  }
});
