"""Chuẩn hoá landmark bàn tay — dùng chung giữa training và app.py để đảm bảo
tiền xử lý lúc train và lúc predict luôn khớp nhau.

Input: vector 63 số (21 landmark x,y,z thô từ MediaPipe Hands).
Output: vector 63 số đã chuẩn hoá — bất biến với vị trí tay trong khung hình
(dời gốc về cổ tay) và bất biến với khoảng cách tay tới camera (chia theo
khoảng cách cổ tay -> gốc ngón giữa).
"""

import numpy as np

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def normalize_hand_landmarks(vector):
    points = np.array(vector, dtype=np.float64).reshape(21, 3)

    wrist = points[WRIST_IDX].copy()
    points -= wrist

    scale = np.linalg.norm(points[MIDDLE_MCP_IDX])
    if scale < 1e-6:
        scale = 1e-6
    points /= scale

    return points.reshape(-1).tolist()
