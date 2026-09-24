"""Chuẩn hoá landmark bàn tay — dùng chung giữa training và app.py để đảm bảo
tiền xử lý lúc train và lúc predict luôn khớp nhau.

Input: vector 63 số (21 landmark x,y,z thô từ MediaPipe Hands).
Output: vector 63 số đã chuẩn hoá — bất biến với vị trí tay trong khung hình
(dời gốc về cổ tay) và bất biến với khoảng cách tay tới camera (chia theo
khoảng cách cổ tay -> gốc ngón giữa).
"""

import json
import os

import numpy as np

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

HAND_LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training", "hand_labels.json")


SIGNAL_SEQUENCE_LENGTH = 30  # ~1 giay o 30fps, du cho chuoi 3 buoc cua Signal for Help


def normalize_hand_sequence(frames):
    """Chuẩn hoá từng frame trong 1 chuỗi (list các vector 63 số thô) —
    dùng chung giữa training/train_signal_sequence_model.py và app.py cho
    pipeline nhận diện Signal for Help theo chuyển động."""
    return [normalize_hand_landmarks(frame) for frame in frames]


def load_hand_labels(labels_path=HAND_LABELS_PATH):
    """Đọc training/hand_labels.json — nguồn duy nhất cho danh sách nhãn tay,
    dùng chung giữa app.py và các script train để không bao giờ bị lệch thứ
    tự/tên nhãn. Trả về list tên nhãn theo đúng thứ tự index 0, 1, 2, ...
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return [labels[str(i)] for i in range(len(labels))]


def normalize_hand_landmarks(vector):
    points = np.array(vector, dtype=np.float64).reshape(21, 3)

    wrist = points[WRIST_IDX].copy()
    points -= wrist

    scale = np.linalg.norm(points[MIDDLE_MCP_IDX])
    if scale < 1e-6:
        scale = 1e-6
    points /= scale

    return points.reshape(-1).tolist()
