async function uploadImage() {
    const fileInput = document.getElementById("upload_input");
    const file = fileInput.files[0];

    if (!file) {
        alert("이미지를 선택하세요.");
        return;
    }

    const form = new FormData();
    form.append("file", file);

    try {
        const res = await fetch("/analyze_upload", {
            method: "POST",
            body: form
        });

        const data = await res.json();

        // 결과 표시
        const resultImg = document.getElementById("uploaded_img");
        const resultDesc = document.getElementById("uploaded_desc");

        resultImg.src = "data:image/jpeg;base64," + data.image;
        resultDesc.textContent = data.text;

    } catch (e) {
        console.error(e);
        alert("분석 실패");
    }
}
