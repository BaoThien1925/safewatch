import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from hand_features import normalize_hand_landmarks

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FALL_MODEL_PATH = os.path.join(MODELS_DIR, "LSTM_model.h5")
# Model đã train lại với landmark chuẩn hoá (gốc = cổ tay, scale theo cổ tay
# -> gốc ngón giữa) — accuracy 99.67% so với 81.21% của model cũ trên cùng
# test set, đặc biệt sửa lỗi nhận sai Need Ambulance/Need Help (xem
# training/compare_models.py). Bắt buộc phải chuẩn hoá landmark bằng đúng
# hand_features.normalize_hand_landmarks trước khi đưa vào model này.
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "HandLandMarks_Model_Normalized.keras")

FALL_TIMESTEPS = 10
WARMUP_FRAMES = 40

HAND_LABELS = ["Like", "Dislike", "OK", "Neutral", "Need Ambulance", "Need Help", "Signal For Help"]
SOS_HAND_LABELS = {"Need Ambulance", "Need Help", "Signal For Help"}

# Debounce: chỉ đổi nhãn hiển thị/cảnh báo khi 1 nhãn chiếm đa số trong
# VOTE_WINDOW lần predict gần nhất, để 1 frame nhiễu không làm nhãn nhảy loạn.
FALL_VOTE_WINDOW = 5
HAND_VOTE_WINDOW = 5

# Giữ cảnh báo SOS tối thiểu chừng này giây sau khi trigger, tránh nhấp nháy
# khi tín hiệu dao động ngay ở ngưỡng.
ALERT_HOLD_SECONDS = 2.0


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
    # File .h5 gốc được lưu bằng Keras 2 (có kwarg "time_major" trên LSTM đã bị
    # loại bỏ ở Keras 3), nên load_model() thẳng sẽ lỗi khi giải mã config.
    # Dựng lại đúng kiến trúc rồi chỉ load trọng số là cách né lỗi này.
    model.load_weights(FALL_MODEL_PATH)
    return model


def build_hand_model():
    # Model này được train và lưu lại bằng model.save() chuẩn của bản Keras
    # đang dùng (training/retrain_hand_model.py), nên load_model() đọc thẳng
    # được đầy đủ kiến trúc + trọng số — không cần vá lỗi lệch tên layer như
    # file .keras gốc (được lưu từ 1 phiên Keras khác, xem lịch sử git).
    return tf.keras.models.load_model(HAND_MODEL_PATH)


@st.cache_resource
def load_models():
    return build_fall_model(), build_hand_model()


@st.cache_resource
def get_mediapipe():
    mp_pose = mp.solutions.pose.Pose()
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
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
    data = np.expand_dims(np.array(lm_list), axis=0)
    result = model.predict(data, verbose=0)
    return "Fall" if result[0][0] > 0.5 else "NotFall"


def predict_hand(model, lm_vector):
    # Model chỉ nhận 1 frame (21 điểm x,y,z) mỗi lần predict — kiến trúc gốc
    # không có input theo thời gian thật, nên gom 10 frame rồi chỉ dùng frame
    # cuối (bản cũ) chỉ tốn ~10 frame chờ vô ích. Predict ngay mỗi frame có
    # tay, và dùng vote ở tầng ứng dụng (majority_label) để làm mượt kết quả.
    #
    # normalize_hand_landmarks bắt buộc phải khớp với tiền xử lý lúc train
    # (training/retrain_hand_model.py), nếu không model sẽ dự đoán sai hết.
    normalized = normalize_hand_landmarks(lm_vector)
    lstm_input = np.array(normalized).reshape(1, 21, 3)
    dnn_input = np.array(normalized).reshape(1, 63)
    result = model.predict([lstm_input, dnn_input], verbose=0)
    return HAND_LABELS[int(np.argmax(result))]


def draw_pose(img, results):
    mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    return img


def draw_hand(img, results):
    for hand_lms in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(img, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)
    return img


def main():
    st.set_page_config(page_title="Fall + SOS Hand Demo", layout="wide")
    st.title("Demo hợp nhất: Phát hiện Ngã quỵ + Tín hiệu tay SOS")

    with st.sidebar:
        st.header("Cấu hình")
        enable_fall = st.checkbox("Bật phát hiện Ngã", value=True)
        enable_hand = st.checkbox("Bật phát hiện tay SOS", value=True)
        run = st.checkbox("Chạy webcam", value=False, key="run_webcam")

    fall_model, hand_model = load_models()
    pose_detector, hand_detector = get_mediapipe()

    video_col, status_col = st.columns([2, 1])
    frame_placeholder = video_col.empty()
    fall_placeholder = status_col.empty()
    hand_placeholder = status_col.empty()
    alert_placeholder = status_col.empty()
    fps_placeholder = status_col.empty()

    if not run:
        frame_placeholder.info("Tích chọn 'Chạy webcam' ở thanh bên để bắt đầu.")
        return

    # Backend mặc định (Media Foundation) của OpenCV trên Windows hay bị treo
    # nhiều giây khi mở webcam; CAP_DSHOW mở nhanh và ổn định hơn hẳn.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        st.error("Không mở được webcam. Kiểm tra webcam có đang bị ứng dụng khác chiếm dụng, hoặc chưa cấp quyền camera cho Python trong Windows Settings > Privacy > Camera.")
        return

    # Sliding window: giữ đúng FALL_TIMESTEPS frame gần nhất, predict lại mỗi
    # khi có frame mới (không chờ gom hẳn 1 lô 10 frame rồi xoá) — giảm độ
    # trễ cảm nhận so với bản chia khối rời rạc cũ.
    fall_lm_window = deque(maxlen=FALL_TIMESTEPS)
    fall_votes = deque(maxlen=FALL_VOTE_WINDOW)
    hand_votes = deque(maxlen=HAND_VOTE_WINDOW)

    fall_label = "Warmup..."
    hand_label = "Warmup..."
    alert_until = 0.0
    frame_count = 0
    prev_time = time.time()

    while run:
        success, frame = cap.read()
        if not success:
            st.error("Không đọc được webcam.")
            break

        frame_count += 1
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_count > WARMUP_FRAMES:
            if enable_fall:
                pose_results = pose_detector.process(img_rgb)
                if pose_results.pose_landmarks:
                    fall_lm_window.append(pose_landmarks_to_vector(pose_results))
                    frame = draw_pose(frame, pose_results)
                    if len(fall_lm_window) == FALL_TIMESTEPS:
                        fall_votes.append(predict_fall(fall_model, list(fall_lm_window)))
                        fall_label = majority_label(fall_votes, fall_label)

            if enable_hand:
                hand_results = hand_detector.process(img_rgb)
                if hand_results.multi_hand_landmarks:
                    frame = draw_hand(frame, hand_results)
                    hand_votes.append(predict_hand(hand_model, hand_landmarks_to_vector(hand_results)))
                    hand_label = majority_label(hand_votes, hand_label)
                else:
                    hand_votes.clear()
                    hand_label = "No hands"

        raw_sos = (enable_fall and fall_label == "Fall") or (enable_hand and hand_label in SOS_HAND_LABELS)
        now = time.time()
        if raw_sos:
            alert_until = now + ALERT_HOLD_SECONDS
        sos_alert = now < alert_until

        fps = 1 / (now - prev_time) if now > prev_time else 0
        prev_time = now

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        fall_placeholder.metric("Trạng thái ngã", fall_label if enable_fall else "Đã tắt")
        hand_placeholder.metric("Cử chỉ tay", hand_label if enable_hand else "Đã tắt")
        fps_placeholder.caption(f"FPS: {fps:.1f}")

        if sos_alert:
            alert_placeholder.error("🚨 CẢNH BÁO SOS 🚨")
        else:
            alert_placeholder.success("Bình thường")

        run = st.session_state["run_webcam"]

    cap.release()


if __name__ == "__main__":
    main()
