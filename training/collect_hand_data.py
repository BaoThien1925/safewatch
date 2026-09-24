"""Thu data landmark tay cho 1 nhãn cụ thể, lưu thẳng ra CSV (append được,
không lo ghi đè/mất data như file Excel).

Khác với Make_Dataset_WithWebcam.py gốc:
- Chọn nhãn TRƯỚC khi quay (không phải sửa tay cột Label sau khi export).
- Nhắc đổi điều kiện (tay trái/phải, khoảng cách, góc) sau mỗi lượt ngắn,
  để dữ liệu đa dạng hơn là 1 lượt quay dài cùng 1 tư thế.
- Lưu landmark THÔ (chưa chuẩn hoá) — giống cách dataset gốc lưu, để linh
  hoạt đổi công thức chuẩn hoá sau này mà không cần quay lại.

Cách dùng:
    python training/collect_hand_data.py "Fist"
    python training/collect_hand_data.py "Fist" --frames 300 --out data/new_gestures.csv

Nhãn phải có trong hand_labels.json (thêm dòng mới vào đó trước nếu là nhãn
hoàn toàn mới). Chạy nhiều lượt ngắn (mặc định 300 frame/lượt, ~10 giây ở
30fps) cho MỖI nhãn, đổi tay/góc/khoảng cách giữa các lượt, rồi chạy lại
script với cùng nhãn để nối thêm data (append) — không ghi đè lượt trước.

Nhấn 'q' để dừng giữa lượt, 's' để bỏ frame hiện tại (không lưu) nếu tư thế
bị lỗi.
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import mediapipe as mp

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(SCRIPT_DIR, "hand_labels.json")
DEFAULT_OUT_PATH = os.path.join(SCRIPT_DIR, "data", "collected_hand_landmarks.csv")

COLUMNS = ["Label"] + [f"hand_mark_{i:02d}{axis}" for i in range(21) for axis in ["x", "y", "z"]]

VARIATION_HINTS = [
    "Tay phải, khoảng cách gần (~30cm)",
    "Tay phải, khoảng cách xa (~1m)",
    "Tay trái, khoảng cách gần",
    "Tay trái, khoảng cách xa",
    "Tay phải, hơi nghiêng/xoay",
    "Tay trái, hơi nghiêng/xoay",
    "Tay phải, ánh sáng/góc camera khác (di chuyển ra chỗ khác nếu có thể)",
    "Tay trái, ánh sáng/góc camera khác",
]


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_label_name(labels, label_arg):
    for idx, name in labels.items():
        if name.lower() == label_arg.lower():
            return name
    raise SystemExit(
        f"Nhãn '{label_arg}' chưa có trong {LABELS_PATH}. "
        f"Thêm nhãn mới vào file đó trước (dạng \"<số>\": \"<tên>\"), rồi chạy lại."
    )


def count_existing_rows(out_path, label_name):
    if not os.path.exists(out_path):
        return 0
    count = 0
    with open(out_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0] == label_name:
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Thu data landmark tay cho 1 nhãn.")
    parser.add_argument("label", help="Tên nhãn, phải khớp 1 dòng trong hand_labels.json")
    parser.add_argument("--frames", type=int, default=300, help="Số frame thu trong lượt này (mặc định 300)")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="File CSV để append data vào")
    args = parser.parse_args()

    labels = load_labels()
    label_name = resolve_label_name(labels, args.label)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    file_exists = os.path.exists(args.out)
    already_collected = count_existing_rows(args.out, label_name)

    hint_idx = (already_collected // args.frames) % len(VARIATION_HINTS)
    hint = VARIATION_HINTS[hint_idx]

    print(f"Nhãn: {label_name}")
    print(f"Đã có sẵn {already_collected} mẫu cho nhãn này trong {args.out}")
    print(f"Gợi ý tư thế lượt này: {hint}")
    print("Nhấn 'q' để dừng sớm, 's' để tạm dừng/tiếp tục thu.")
    print("Chuẩn bị trong 3 giây...")
    time.sleep(3)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        raise SystemExit("Không mở được webcam.")

    rows = []
    paused = False

    while len(rows) < args.frames:
        success, frame = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            if not paused:
                feature_vector = []
                for lm in landmarks.landmark:
                    feature_vector.extend([lm.x, lm.y, lm.z])
                rows.append([label_name] + feature_vector)

        status = "PAUSED (nhan 's' de tiep tuc)" if paused else "RECORDING"
        cv2.putText(frame, f"{label_name} | {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {len(rows)}/{args.frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, hint, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("Thu data cu chi tay", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

    if not rows:
        print("Không thu được frame nào (không có tay trong khung hình?). Không lưu gì cả.")
        return

    with open(args.out, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(COLUMNS)
        writer.writerows(rows)

    print(f"\nĐã lưu thêm {len(rows)} mẫu cho nhãn '{label_name}' vào {args.out}")
    print(f"Tổng cộng nhãn này giờ có {already_collected + len(rows)} mẫu.")
    print("Chạy lại script (cùng nhãn) để thu thêm lượt khác với tư thế/điều kiện khác nhé.")


if __name__ == "__main__":
    main()
