
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Xác định thư mục gốc dự án
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL = ROOT / "model"

RESULTS.mkdir(exist_ok=True)
MODEL.mkdir(exist_ok=True)

# 2. Khai báo đúng 5 đặc trưng cảm biến
FEATURES = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio"
]

TARGET = "Occupancy"

# 3. Đọc dữ liệu
train = pd.read_csv(DATA / "datatraining.txt")
test1 = pd.read_csv(DATA / "datatest.txt")
test2 = pd.read_csv(DATA / "datatest2.txt")

# 4. Tách X (đặc trưng) và y (nhãn)
X_train = train[FEATURES]
y_train = train[TARGET]

# 5. Tạo pipeline: chuẩn hóa + Logistic Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# 6. Huấn luyện chỉ trên datatraining
pipeline.fit(X_train, y_train)

print("Đã huấn luyện Logistic Regression!")

# 7. Lưu mô hình
joblib.dump(
    pipeline,
    MODEL / "baseline.joblib"
)

# 8. Đánh giá từng tập kiểm thử
all_metrics = {}

for name, df in [
    ("datatest", test1),
    ("datatest2", test2)
]:
    X_test = df[FEATURES]
    y_test = df[TARGET]

    # Dự đoán trạng thái phòng
    y_pred = pipeline.predict(X_test)

    # Xác suất có người (lớp 1)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Ma trận nhầm lẫn: thứ tự nhãn [0, 1]
    cm = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # Tính chỉ số
    macro_f1 = f1_score(
        y_test, y_pred,
        average="macro",
        zero_division=0
    )

    false_empty_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0 else None
    )

    accuracy = accuracy_score(y_test, y_pred)

    all_metrics[name] = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "false_empty_rate": (
            float(false_empty_rate)
            if false_empty_rate is not None else None
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    print(f"\n--- {name} ---")
    print("Accuracy:", round(accuracy, 4))
    print("Macro-F1:", round(macro_f1, 4))
    print("False Empty Rate:", false_empty_rate)
    print("TN, FP, FN, TP:", tn, fp, fn, tp)

    # 9. Vẽ Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Empty", "Occupied"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Baseline - {name}")
    plt.tight_layout()

    plt.savefig(
        RESULTS / f"confusion_matrix_baseline_{name}.png",
        dpi=200
    )
    plt.close()

# 10. Xuất kết quả thành JSON
with open(
    RESULTS / "metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(all_metrics, f, indent=4)

print("\nĐã lưu metrics.json và các biểu đồ.")
