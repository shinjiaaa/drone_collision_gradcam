import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ===============================
# 1. 설정
# ===============================
MODEL_PATH = "rebuilt_model.h5"
VAL_ROOT = "collision_dataset/validation"
IMG_SIZE = (128, 128)
THRESHOLD = 0.5

# ===============================
# 2. 모델 로드
# ===============================
model = load_model(MODEL_PATH)

all_images = []
all_labels = []
sequence_ranges = []

start_idx = 0

# ===============================
# 3. 시퀀스별 데이터 로드
# ===============================
for seq_name in sorted(os.listdir(VAL_ROOT)):
    seq_path = os.path.join(VAL_ROOT, seq_name)

    if not os.path.isdir(seq_path):
        continue

    img_dir = os.path.join(seq_path, "images")
    label_file = os.path.join(seq_path, "labels.txt")

    if not os.path.exists(img_dir) or not os.path.exists(label_file):
        continue

    frame_files = sorted(
        os.listdir(img_dir),
        key=lambda x: int("".join(filter(str.isdigit, x)))
    )

    with open(label_file, "r") as f:
        labels = [int(line.strip()) for line in f if line.strip() != ""]

    assert len(frame_files) == len(labels), \
        f"[ERROR] {seq_name}: frame 수와 label 수가 다름"

    images = []
    for fname in frame_files:
        img_path = os.path.join(img_dir, fname)
        img = cv2.imread(img_path)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        images.append(img)

    images = np.array(images)
    labels = np.array(labels)

    all_images.append(images)
    all_labels.append(labels)

    end_idx = start_idx + len(labels)
    sequence_ranges.append((seq_name, start_idx, end_idx))
    start_idx = end_idx

# ===============================
# 4. 전체 병합
# ===============================
X = np.concatenate(all_images, axis=0)
y_true = np.concatenate(all_labels, axis=0)

# ===============================
# 5. 예측
# ===============================
y_prob = model.predict(X).ravel()
y_pred = (y_prob >= THRESHOLD).astype(int)

# ===============================
# 6. Confusion Matrix
# ===============================
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Collision"]
)
disp.plot()
plt.title("Confusion Matrix (Sequential Validation)")
plt.show()

# ===============================
# 7. ROC Curve
# ===============================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="red")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# ===============================
# 8. Precision–Recall Curve
# ===============================
precision, recall, _ = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

plt.figure()
plt.plot(recall, precision, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.show()

# ===============================
# 9. 시퀀스별 위험률 변화
# ===============================
plt.figure(figsize=(12, 4))

for seq_name, s, e in sequence_ranges:
    plt.plot(range(s, e), y_prob[s:e], label=seq_name)

plt.xlabel("Frame Index (Temporal Order)")
plt.ylabel("Collision Probability")
plt.title("Temporal Collision Risk per Sequence")
plt.legend()
plt.grid()
plt.show()

# ===============================
# 10. 정량 지표 계산
# ===============================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# ===============================
# 11. 성능 지표 Bar Chart (추가됨)
# ===============================
metrics = {
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-score": f1
}

plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values())
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Collision Classification Performance Metrics")

for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha="center")

plt.grid(axis="y")
plt.show()

# ===============================
# 12. 정량 지표 출력
# ===============================
print("===== Collision Classification Performance (Sequential Validation) =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")
print(f"PR AUC    : {ap:.4f}")
