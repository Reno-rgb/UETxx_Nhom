
import joblib
import pandas as pd
from pathlib import Path

# Đường dẫn dự án
ROOT = Path(__file__).resolve().parent.parent

# Nạp mô hình baseline đã huấn luyện
model = joblib.load(ROOT / "model" / "baseline.joblib")

FEATURES = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio"
]

# Ngưỡng cảnh báo baseline
THRESHOLD = 0.5
REQUIRED_SAMPLES = 3


def check_room(sensor_sequence):
    """
    Nhận một chuỗi mẫu cảm biến theo thời gian.
    Trả về kết quả dự đoán và cảnh báo từng mẫu.
    """

    df = pd.DataFrame(
        sensor_sequence,
        columns=FEATURES
    )

    probabilities = model.predict_proba(df)[:, 1]

    consecutive_empty = 0
    results = []

    for i, prob in enumerate(probabilities):

        # Xác suất có người thấp
        if prob < THRESHOLD:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        # Chỉ cảnh báo sau 3 mẫu liên tiếp
        alert = consecutive_empty >= REQUIRED_SAMPLES

        if alert:
            action = "De xuat kiem tra phong de tiet kiem dien"
        else:
            action = "Chua de xuat tat dien"

        results.append({
            "sample": i + 1,
            "probability_occupied": round(float(prob), 4),
            "predicted_occupancy": int(prob >= THRESHOLD),
            "consecutive_empty": consecutive_empty,
            "alert": alert,
            "action": action
        })

    return results


if __name__ == "__main__":

    # Chuỗi dữ liệu cảm biến giả lập
    samples = [
        [23.5, 28.0, 0.0, 450.0, 0.005],
        [23.4, 28.1, 0.0, 452.0, 0.005],
        [23.6, 28.2, 0.0, 455.0, 0.005],
        [24.0, 30.0, 400.0, 600.0, 0.006]
    ]

    output = check_room(samples)

    for result in output:
        print(result)

import joblib
import pandas as pd
from pathlib import Path

# Đường dẫn dự án
ROOT = Path(__file__).resolve().parent.parent

# Nạp mô hình baseline đã huấn luyện
model = joblib.load(ROOT / "model" / "baseline.joblib")

FEATURES = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio"
]

# Ngưỡng cảnh báo baseline
THRESHOLD = 0.5
REQUIRED_SAMPLES = 3


def check_room(sensor_sequence):
    """
    Nhận một chuỗi mẫu cảm biến theo thời gian.
    Trả về kết quả dự đoán và cảnh báo từng mẫu.
    """

    df = pd.DataFrame(
        sensor_sequence,
        columns=FEATURES
    )

    probabilities = model.predict_proba(df)[:, 1]

    consecutive_empty = 0
    results = []

    for i, prob in enumerate(probabilities):

        # Xác suất có người thấp
        if prob < THRESHOLD:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        # Chỉ cảnh báo sau 3 mẫu liên tiếp
        alert = consecutive_empty >= REQUIRED_SAMPLES

        if alert:
            action = "De xuat kiem tra phong de tiet kiem dien"
        else:
            action = "Chua de xuat tat dien"

        results.append({
            "sample": i + 1,
            "probability_occupied": round(float(prob), 4),
            "predicted_occupancy": int(prob >= THRESHOLD),
            "consecutive_empty": consecutive_empty,
            "alert": alert,
            "action": action
        })

    return results


if __name__ == "__main__":

    # Chuỗi dữ liệu cảm biến giả lập
    samples = [
        [23.5, 28.0, 0.0, 450.0, 0.005],
        [23.4, 28.1, 0.0, 452.0, 0.005],
        [23.6, 28.2, 0.0, 455.0, 0.005],
        [24.0, 30.0, 400.0, 600.0, 0.006]
    ]

    output = check_room(samples)

    for result in output:
        print(result)



def alerts_from_probabilities(probabilities, threshold=0.5):
    count = 0
    alerts = []

    for prob in probabilities:
        count = count + 1 if prob < threshold else 0
        alerts.append(count >= 3)

    return alerts


# Kiểm tra độc lập logic cảnh báo
test_probs = [0.12, 0.18, 0.08, 0.75, 0.10, 0.09, 0.07]

expected = [False, False, True, False, False, False, True]

actual = alerts_from_probabilities(test_probs)

assert actual == expected, "Logic cảnh báo chưa đúng!"
print("Logic cảnh báo hoạt động đúng.")
