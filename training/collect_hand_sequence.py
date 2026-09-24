"""Thu data DẠNG CHUỖI (nhiều frame liên tiếp = 1 mẫu) cho pipeline nhận diện
Signal for Help theo chuyển động — khác với collect_hand_data.py (mỗi frame
tĩnh = 1 mẫu, dùng cho model 14 nhãn tĩnh).

2 nhãn duy nhất (mô hình nhị phân, tách biệt hoàn toàn khỏi model 14 nhãn):
    signal    - đúng chuỗi động tác Signal for Help (xoè tay -> gập ngón cái
                vào lòng bàn tay -> khép 4 ngón còn lại che lên trên)
    negative  - bất kỳ cử động/tư thế tay khác (đi lại tay tự nhiên, các cử
                chỉ tĩnh khác, vẫy tay bình thường...) để model học phân biệt
                "đây là tín hiệu có chủ đích" khỏi cử động ngẫu nhiên.

Cách dùng:
    python training/collect_hand_sequence.py signal --count 30
    python training/collect_hand_sequence.py negative --count 30

Mỗi "mẫu" là 1 cửa sổ SIGNAL_SEQUENCE_LENGTH frame liên tiếp có tay. Với nhãn
"signal", cứ LẶP LẠI động tác 3 bước liên tục trong lúc quay (làm chậm, rõ
từng bước) — script tự cắt thành nhiều mẫu chồng lấp nhau khi bạn lặp lại.
Với nhãn "negative", cứ cử động tay tự nhiên/ngẫu nhiên, hoặc giữ các cử chỉ
tĩnh khác (nắm tay, xoè tay, OK...) trong lúc quay.

Nhấn 'q' để dừng sớm.
"""

import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_features import SIGNAL_SEQUENCE_LENGTH

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT_PATH = os.path.join(SCRIPT_DIR, "data", "signal_sequences.csv")
VALID_LABELS = {"signal", "negative"}

# Cách nhau STRIDE frame giữa 2 mẫu liên tiếp (mẫu chồng lấp nếu STRIDE <
# SIGNAL_SEQUENCE_LENGTH) — chồng lấp giúp thu nhiều mẫu hơn từ 1 lượt quay
# lặp lại động tác, không cần dừng giữa mỗi lần.
STRIDE = 10

COLUMNS = ["Label"] + [
    f"f{frame:02d}_mark{i:02d}{axis}" for frame in range(SIGNAL_SEQUENCE_LENGTH) for i in range(21) for axis in ["x", "y", "z"]
]


def main():
    parser = argparse.ArgumentParser(description="Thu data dạng chuỗi cho Signal for Help.")
    parser.add_argument("label", choices=sorted(VALID_LABELS), help="'signal' hoặc 'negative'")
    parser.add_argument("--count", type=int, default=30, help="Số mẫu (chuỗi) muốn thu trong lượt này")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="File CSV để append data vào")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    file_exists = os.path.exists(args.out)

    if args.label == "signal":
        instruction = "Lap lai CHAM: xoe tay -> gap ngon cai vao long ban tay -> khep 4 ngon che len"
    else:
        instruction = "Cu dong tay TU NHIEN / cac tu the khac (KHONG lam dong tac Signal for Help)"

    print(f"Nhan: {args.label}")
    print(f"Huong dan: {instruction}")
    print(f"Muc tieu: {args.count} mau")
    print("Nhan 'q' de dung som.")
    print("Chuan bi trong 3 giay...")
    time.sleep(3)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        raise SystemExit("Không mở được webcam.")

    frame_buffer = deque(maxlen=SIGNAL_SEQUENCE_LENGTH)
    samples = []
    frames_since_last_sample = STRIDE  # cho phép lưu mẫu ngay khi buffer đầy lần đầu

    while len(samples) < args.count:
        success, frame = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            vec = []
            for lm in landmarks.landmark:
                vec.extend([lm.x, lm.y, lm.z])
            frame_buffer.append(vec)
            frames_since_last_sample += 1

            if len(frame_buffer) == SIGNAL_SEQUENCE_LENGTH and frames_since_last_sample >= STRIDE:
                flat = [v for frame_vec in frame_buffer for v in frame_vec]
                samples.append([args.label] + flat)
                frames_since_last_sample = 0
        else:
            # Mất tay giữa chừng -> reset buffer, tránh nối 2 đoạn không liên tục thành 1 mẫu.
            frame_buffer.clear()
            frames_since_last_sample = STRIDE

        cv2.putText(frame, f"{args.label} | Mau: {len(samples)}/{args.count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, instruction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.imshow("Thu data chuoi dong tac", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not samples:
        print("Không thu được mẫu nào. Không lưu gì cả.")
        return

    with open(args.out, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(COLUMNS)
        writer.writerows(samples)

    print(f"\nĐã lưu thêm {len(samples)} mẫu nhãn '{args.label}' vào {args.out}")
    print("Chạy lại script (đổi --count nếu cần) để thu thêm.")


if __name__ == "__main__":
    main()
