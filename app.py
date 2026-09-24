import os
import threading
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from hand_features import (
    SIGNAL_SEQUENCE_LENGTH,
    load_hand_labels,
    normalize_hand_landmarks,
    normalize_hand_sequence,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FALL_MODEL_PATH = os.path.join(MODELS_DIR, "LSTM_model.h5")
# Model 14 nhãn (7 gốc + 7 tự thu: Fist, Peace Sign, Rock On, One Finger,
# Three Fingers, Call Me, Gun Sign), train trên landmark đã chuẩn hoá —
# accuracy 99.54% (xem training/retrain_extended_hand_model.py). Bắt buộc
# phải chuẩn hoá landmark bằng đúng hand_features.normalize_hand_landmarks
# trước khi đưa vào model này. Danh sách nhãn khớp với training/hand_labels.json.
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "HandLandMarks_Model_Extended.keras")
# Pipeline độc lập thứ 3 (không đụng gì tới model 14 nhãn tĩnh ở trên): model
# nhị phân nhận diện CHUỖI động tác Signal for Help theo thời gian (xoè tay
# -> gập ngón cái -> khép 4 ngón), vì cử chỉ này là 1 chuyển động, không phải
# 1 tư thế tĩnh — xem training/train_signal_sequence_model.py. Optional: chỉ
# bật khi model đã được train (chạy training/collect_hand_sequence.py rồi
# training/train_signal_sequence_model.py trước).
SIGNAL_MODEL_PATH = os.path.join(MODELS_DIR, "SignalForHelpSequenceModel.keras")
SIGNAL_CONFIDENCE_THRESHOLD = 0.7
# Model chuỗi train trên rất ít data (60 mẫu, 1 buổi quay) nên dễ báo dương
# giả (false positive) với cử động tay ngẫu nhiên. Yêu cầu N cửa sổ liên
# tiếp đều dự đoán "signal" mới xác nhận thật — giảm false positive tức thời
# mà không cần thu thêm data (dù data đa dạng hơn vẫn là hướng sửa gốc).
SIGNAL_CONSECUTIVE_REQUIRED = 5

FALL_TIMESTEPS = 10
WARMUP_FRAMES = 40

HAND_LABELS = load_hand_labels()
SOS_HAND_LABELS = {"Need Ambulance", "Need Help", "Signal For Help"}

# Debounce: chỉ đổi nhãn hiển thị/cảnh báo khi 1 nhãn chiếm đa số trong
# VOTE_WINDOW lần predict gần nhất, để 1 frame nhiễu không làm nhãn nhảy loạn.
FALL_VOTE_WINDOW = 5
HAND_VOTE_WINDOW = 5

# Giữ cảnh báo SOS tối thiểu chừng này giây sau khi trigger, tránh nhấp nháy
# khi tín hiệu dao động ngay ở ngưỡng.
ALERT_HOLD_SECONDS = 2.0

# Model tay là bộ phân loại closed-set: argmax luôn ép output thành 1 trong 7
# nhãn dù cử chỉ thật không nằm trong tập train. Chỉ nhận nhãn khi xác suất
# softmax cao nhất vượt ngưỡng này, còn lại coi là "Uncertain" (không tính
# vào SOS) để tay làm động tác lạ không bị ép nhận nhầm thành cử chỉ SOS.
DEFAULT_HAND_CONFIDENCE_THRESHOLD = 0.8
UNCERTAIN_LABEL = "Uncertain"


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


def build_signal_model():
    if not os.path.exists(SIGNAL_MODEL_PATH):
        return None
    return tf.keras.models.load_model(SIGNAL_MODEL_PATH)


@st.cache_resource
def load_models():
    return build_fall_model(), build_hand_model(), build_signal_model()


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
    # Gọi model trực tiếp (model(x)) thay vì model.predict(x) — predict() có
    # overhead quản lý dataset/callback không cần thiết khi gọi liên tục mỗi
    # frame với 1 mẫu duy nhất, đo thực tế chậm hơn rõ so với gọi trực tiếp.
    data = np.expand_dims(np.array(lm_list, dtype=np.float32), axis=0)
    result = model(data, training=False).numpy()
    return "Fall" if result[0][0] > 0.5 else "NotFall"


def predict_hand(model, lm_vector, confidence_threshold):
    # Model chỉ nhận 1 frame (21 điểm x,y,z) mỗi lần predict — kiến trúc gốc
    # không có input theo thời gian thật, nên gom 10 frame rồi chỉ dùng frame
    # cuối (bản cũ) chỉ tốn ~10 frame chờ vô ích. Predict ngay mỗi frame có
    # tay, và dùng vote ở tầng ứng dụng (majority_label) để làm mượt kết quả.
    #
    # normalize_hand_landmarks bắt buộc phải khớp với tiền xử lý lúc train
    # (training/retrain_hand_model.py), nếu không model sẽ dự đoán sai hết.
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
        hand_confidence_threshold = st.slider(
            "Ngưỡng tin cậy tay (%)",
            min_value=0,
            max_value=100,
            value=int(DEFAULT_HAND_CONFIDENCE_THRESHOLD * 100),
            help="Cử chỉ tay có xác suất dự đoán thấp hơn ngưỡng này sẽ hiện 'Uncertain' thay vì bị ép nhận thành 1 trong 7 cử chỉ đã train.",
        ) / 100
        run = st.checkbox("Chạy webcam", value=False, key="run_webcam")

    fall_model, hand_model, signal_model = load_models()
    pose_detector, hand_detector = get_mediapipe()

    enable_signal = False
    with st.sidebar:
        if signal_model is not None:
            enable_signal = st.checkbox(
                "Bật nhận diện chuỗi Signal for Help (thử nghiệm, còn hay báo sai)",
                value=False,
                help="Model chỉ train trên 60 mẫu 1 buổi quay, dễ báo dương giả (false positive). "
                "Tạm để mặc định tắt, tự bật lên nếu muốn thử tiếp — cần thu thêm data đa dạng để dùng thật.",
            )
        else:
            st.caption(
                "Chưa có model chuỗi Signal for Help — chạy "
                "training/collect_hand_sequence.py rồi training/train_signal_sequence_model.py "
                "để bật tính năng này."
            )

    video_col, status_col = st.columns([2, 1])
    frame_placeholder = video_col.empty()
    fall_placeholder = status_col.empty()
    hand_placeholder = status_col.empty()
    hand_confidence_placeholder = status_col.empty()
    signal_placeholder = status_col.empty()
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
    signal_window = deque(maxlen=SIGNAL_SEQUENCE_LENGTH)

    fall_label = "Warmup..."
    hand_label = "Warmup..."
    hand_confidence = 0.0
    signal_detected = False
    signal_probability = 0.0
    signal_consecutive_count = 0
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
            # Pose (ngã) và Hands (tay) là 2 MediaPipe Solution độc lập, không
            # chia sẻ state, nên chạy song song bằng thread thay vì tuần tự
            # giúp giảm độ trễ mỗi frame gần một nửa (2 lệnh .process() nặng
            # nhất trong loop từng chạy nối tiếp nhau).
            detection_results = {}

            def run_pose():
                detection_results["pose"] = pose_detector.process(img_rgb)

            def run_hands():
                detection_results["hands"] = hand_detector.process(img_rgb)

            threads = []
            if enable_fall:
                t = threading.Thread(target=run_pose)
                t.start()
                threads.append(t)
            if enable_hand:
                t = threading.Thread(target=run_hands)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            if enable_fall:
                pose_results = detection_results["pose"]
                if pose_results.pose_landmarks:
                    fall_lm_window.append(pose_landmarks_to_vector(pose_results))
                    frame = draw_pose(frame, pose_results)
                    if len(fall_lm_window) == FALL_TIMESTEPS:
                        fall_votes.append(predict_fall(fall_model, list(fall_lm_window)))
                        fall_label = majority_label(fall_votes, fall_label)

            if enable_hand:
                hand_results = detection_results["hands"]
                if hand_results.multi_hand_landmarks:
                    frame = draw_hand(frame, hand_results)
                    raw_vector = hand_landmarks_to_vector(hand_results)
                    raw_label, hand_confidence = predict_hand(hand_model, raw_vector, hand_confidence_threshold)
                    hand_votes.append(raw_label)
                    hand_label = majority_label(hand_votes, hand_label)

                    if enable_signal:
                        signal_window.append(raw_vector)
                        if len(signal_window) == SIGNAL_SEQUENCE_LENGTH:
                            raw_signal, signal_probability = predict_signal_sequence(
                                signal_model, list(signal_window), SIGNAL_CONFIDENCE_THRESHOLD
                            )
                            signal_consecutive_count = signal_consecutive_count + 1 if raw_signal else 0
                            signal_detected = signal_consecutive_count >= SIGNAL_CONSECUTIVE_REQUIRED
                else:
                    hand_votes.clear()
                    hand_label = "No hands"
                    signal_window.clear()  # mất tay giữa chừng -> chuỗi không còn liên tục, reset
                    signal_consecutive_count = 0
                    signal_detected = False

        raw_sos = (
            (enable_fall and fall_label == "Fall")
            or (enable_hand and hand_label in SOS_HAND_LABELS)
            or (enable_hand and enable_signal and signal_detected)
        )
        now = time.time()
        if raw_sos:
            alert_until = now + ALERT_HOLD_SECONDS
        sos_alert = now < alert_until

        fps = 1 / (now - prev_time) if now > prev_time else 0
        prev_time = now

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        fall_placeholder.metric("Trạng thái ngã", fall_label if enable_fall else "Đã tắt")
        hand_placeholder.metric("Cử chỉ tay", hand_label if enable_hand else "Đã tắt")
        if enable_hand:
            hand_confidence_placeholder.caption(f"Độ tin cậy: {hand_confidence * 100:.0f}%")
        if enable_hand and enable_signal:
            signal_placeholder.caption(f"Chuỗi Signal for Help: {'Có' if signal_detected else 'Không'} ({signal_probability * 100:.0f}%)")
        fps_placeholder.caption(f"FPS: {fps:.1f}")

        if sos_alert:
            alert_placeholder.error("🚨 CẢNH BÁO SOS 🚨")
        else:
            alert_placeholder.success("Bình thường")

        run = st.session_state["run_webcam"]

    cap.release()


if __name__ == "__main__":
    main()
