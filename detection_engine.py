"""Engine nhận diện dùng chung — tách từ app.py để các trang trong views/
(live_view, settings, dashboard) đều import cùng 1 nguồn, không lặp code.

app.py (bản gốc, 1 trang) vẫn giữ nguyên hoạt động độc lập như 1 fallback
đơn giản; views/live_view.py là bản giao diện mới, dùng module này.
"""

import os
from collections import Counter

import mediapipe as mp

# Bản mediapipe mới (0.10.35, xem lịch sử git — bump từ 0.10.21 để né bug
# PermissionError trên Streamlit Cloud) không tự gắn `mp.solutions` vào
# namespace gốc nữa (package __init__.py giờ chỉ eager-import Tasks API
# mới). API cũ (Pose/Hands) vẫn còn, chỉ cần import submodule trực tiếp
# thay vì trông cậy vào thuộc tính `mp.solutions`.
from mediapipe import solutions as mp_solutions
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from hand_features import load_hand_labels, normalize_hand_landmarks, normalize_hand_sequence

# Mặc định TensorFlow tự spawn nhiều thread nội bộ (intra/inter-op) CHO MỖI
# model. Ở đây có 3 model (Ngã, Tay, Signal) + 2 MediaPipe Solution (cũng tự
# có thread pool riêng) + thread mình tự tạo để chạy song song Pose/Hands
# (xem views/live_view.py) — tổng số thread thực tế dễ vượt xa số nhân CPU
# thật, gây tranh chấp CPU (oversubscription) khiến chạy CHẬM HƠN thay vì
# nhanh hơn. Giới hạn TF về ít thread nội bộ (nhường chỗ cho thread mình tự
# quản lý) sửa đúng vấn đề này. Phải set TRƯỚC khi TF chạy op đầu tiên.
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass  # TF đã init rồi (module bị import lại) -- không set lại được, bỏ qua.

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FALL_MODEL_PATH = os.path.join(MODELS_DIR, "LSTM_model.h5")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "HandLandMarks_Model_Extended.keras")
SIGNAL_MODEL_PATH = os.path.join(MODELS_DIR, "SignalForHelpSequenceModel.keras")

FALL_TIMESTEPS = 10
WARMUP_FRAMES = 40

HAND_LABELS = load_hand_labels()
SOS_HAND_LABELS = {"Need Ambulance", "Need Help", "Signal For Help"}
UNCERTAIN_LABEL = "Uncertain"

FALL_VOTE_WINDOW = 5
HAND_VOTE_WINDOW = 5
ALERT_HOLD_SECONDS = 2.0
SIGNAL_CONSECUTIVE_REQUIRED = 5

DEFAULT_FALL_GRACE_SECONDS = 120
DEFAULT_HAND_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_ENABLE_FALL = True
DEFAULT_ENABLE_HAND = True
DEFAULT_ENABLE_SIGNAL = False

# Nhãn kỹ thuật -> văn bản thân thiện hiển thị cho người dùng cuối (yêu cầu
# UX: không hiện "NotFall"/"class_id" trực tiếp trên giao diện chính).
FALL_LABEL_DISPLAY = {
    "Warmup...": "Đang khởi động...",
    "NotFall": "Không phát hiện ngã",
    "Fall": "Đã phát hiện ngã",
}
HAND_LABEL_DISPLAY = {
    "Warmup...": "Đang khởi động...",
    "No hands": "Không thấy tay",
    "Uncertain": "Không rõ cử chỉ",
}


def friendly_fall_label(raw_label):
    return FALL_LABEL_DISPLAY.get(raw_label, raw_label)


def friendly_hand_label(raw_label):
    return HAND_LABEL_DISPLAY.get(raw_label, raw_label)


CAMERA_RESOLUTIONS = {
    "320x240": (320, 240),
    "640x480": (640, 480),
    "1280x720": (1280, 720),
}

SETTINGS_DEFAULTS = {
    "enable_fall": DEFAULT_ENABLE_FALL,
    "enable_hand": DEFAULT_ENABLE_HAND,
    "enable_signal": DEFAULT_ENABLE_SIGNAL,
    "fall_grace_seconds": DEFAULT_FALL_GRACE_SECONDS,
    "hand_confidence_threshold": DEFAULT_HAND_CONFIDENCE_THRESHOLD,
    "signal_confidence_threshold": DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD,
    "alert_hold_seconds": ALERT_HOLD_SECONDS,
    "show_overlay": True,
    "mirror_camera": False,
    "camera_resolution": "640x480",
    "sound_alert": True,
    "silent_monitoring": True,
    "notify_email": False,
    "notify_sms": False,
    "notify_push": False,
    "retention_days": 30,
    "save_snapshot": False,
    "save_video": False,
}


def ensure_settings_defaults():
    """Gọi ở đầu mỗi trang cần đọc cấu hình — đảm bảo session_state luôn có
    đủ key với giá trị mặc định, để Live View/Dashboard chạy được cả khi
    người dùng chưa từng mở trang Cấu hình."""
    for key, value in SETTINGS_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def majority_label(votes, default):
    if not votes:
        return default
    return Counter(votes).most_common(1)[0][0]


def build_fall_model():
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(FALL_TIMESTEPS, 99)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="sigmoid"))
    model.load_weights(FALL_MODEL_PATH)
    return model


def build_hand_model():
    return tf.keras.models.load_model(HAND_MODEL_PATH)


def build_signal_model():
    if not os.path.exists(SIGNAL_MODEL_PATH):
        return None
    return tf.keras.models.load_model(SIGNAL_MODEL_PATH)


@st.cache_resource
def load_models():
    return build_fall_model(), build_hand_model(), build_signal_model()


def warm_up_models(fall_model, hand_model, signal_model):
    """Gọi predict 1 lần với input giả (toàn số 0) để TF trace/biên dịch
    graph ngay bây giờ, không phải lúc frame thật đầu tiên tới (mới thấy
    chậm rõ ở vài giây đầu, kể cả khi model đã cache qua @st.cache_resource
    -- cache chỉ tránh load lại file, không tránh được chi phí lần predict
    đầu tiên)."""
    dummy_fall = [[0.0] * 99 for _ in range(FALL_TIMESTEPS)]
    predict_fall(fall_model, dummy_fall)

    dummy_hand = [0.0] * 63
    predict_hand(hand_model, dummy_hand, confidence_threshold=1.1)  # threshold>1 -> luôn "Uncertain", không quan trọng

    if signal_model is not None:
        from hand_features import SIGNAL_SEQUENCE_LENGTH

        dummy_signal = [[0.0] * 63 for _ in range(SIGNAL_SEQUENCE_LENGTH)]
        predict_signal_sequence(signal_model, dummy_signal, confidence_threshold=1.1)


@st.cache_resource
def get_mediapipe():
    # model_complexity=0 (lite) thay vì mặc định 1 (full) — nhanh hơn rõ rệt
    # trên CPU. LƯU Ý: model Ngã được train trên landmark sinh ra từ Pose
    # complexity=1 (mặc định lúc train) — đổi complexity đồng nghĩa đổi luôn
    # phân phối landmark đầu vào, CHƯA được kiểm chứng model Ngã còn chính
    # xác y như cũ hay không. Nên test kỹ (tự ngã thử) trước khi tin tưởng.
    mp_pose = mp_solutions.pose.Pose(model_complexity=0)
    mp_hands = mp_solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    return mp_pose, mp_hands


def pose_landmarks_to_vector(results):
    lm = []
    for landmark in results.pose_landmarks.landmark:
        lm.extend([landmark.x, landmark.y, landmark.z])
    return lm


def hand_landmarks_to_vector(results):
    lm = []
    for landmark in results.multi_hand_landmarks[0].landmark:
        lm.extend([landmark.x, landmark.y, landmark.z])
    return lm


def predict_fall(model, lm_list):
    data = np.expand_dims(np.array(lm_list, dtype=np.float32), axis=0)
    result = model(data, training=False).numpy()
    return "Fall" if result[0][0] > 0.5 else "NotFall"


def predict_hand(model, lm_vector, confidence_threshold):
    normalized = normalize_hand_landmarks(lm_vector)
    lstm_input = np.array(normalized, dtype=np.float32).reshape(1, 21, 3)
    dnn_input = np.array(normalized, dtype=np.float32).reshape(1, 63)
    result = model([lstm_input, dnn_input], training=False).numpy()[0]
    confidence = float(np.max(result))
    if confidence < confidence_threshold:
        return UNCERTAIN_LABEL, confidence
    return HAND_LABELS[int(np.argmax(result))], confidence


def predict_signal_sequence(model, frame_sequence, confidence_threshold):
    normalized = normalize_hand_sequence(frame_sequence)
    data = np.expand_dims(np.array(normalized, dtype=np.float32), axis=0)
    probability = float(model(data, training=False).numpy()[0][0])
    return probability >= confidence_threshold, probability


def draw_pose(img, results):
    mp_solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_solutions.pose.POSE_CONNECTIONS)
    return img


def draw_hand(img, results):
    for hand_lms in results.multi_hand_landmarks:
        mp_solutions.drawing_utils.draw_landmarks(img, hand_lms, mp_solutions.hands.HAND_CONNECTIONS)
    return img
