import io
import time
import zipfile

import cv2
import h5py
import mediapipe as mp
import numpy as np
import streamlit as st
import tensorflow as tf
from keras.layers import LSTM, Concatenate, Dense, Dropout, Input
from keras.models import Model, Sequential

FALL_MODEL_PATH = r"D:\DeTaiAI\NgaQuy\Code\LSTM_model.h5"
HAND_MODEL_PATH = r"D:\DeTaiAI\NCKH SV 2024\Code\Hand_Detection\Model_HandDetection\HandLandMarks_Model_300Epochs_new.keras"

FALL_TIMESTEPS = 10
HAND_TIMESTEPS = 10
WARMUP_FRAMES = 40

HAND_LABELS = ["Like", "Dislike", "OK", "Neutral", "Need Ambulance", "Need Help", "Signal For Help"]
SOS_HAND_LABELS = {"Need Ambulance", "Need Help", "Signal For Help"}


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
    lstm_input = Input(shape=(21, 3))
    lstm_branch = LSTM(64, return_sequences=True, name="lstm")(lstm_input)
    lstm_branch = LSTM(64, name="lstm_1")(lstm_branch)
    lstm_branch = Dropout(0.5)(lstm_branch)

    dnn_input = Input(shape=(63,))
    dnn_branch = Dense(128, activation="relu", name="dense")(dnn_input)
    dnn_branch = Dropout(0.5)(dnn_branch)
    dnn_branch = Dense(64, activation="relu", name="dense_1")(dnn_branch)

    combined = Concatenate()([lstm_branch, dnn_branch])
    output = Dense(7, activation="softmax", name="dense_2")(combined)
    model = Model(inputs=[lstm_input, dnn_input], outputs=output)

    # File .keras gốc lưu tên layer nội bộ lệch so với model dựng lại tươi
    # (dense/dense_2/dense_4, lstm/lstm_2 thay vì dense/dense_1/dense_2,
    # lstm/lstm_1) do được lưu từ 1 phiên Keras có bộ đếm tên layer khác,
    # nên model.load_weights() theo tên sẽ không khớp và báo thiếu biến.
    # Đọc thẳng mảng trọng số theo đúng vị trí trong model.weights.h5 rồi
    # gán bằng set_weights() để né việc khớp tên.
    with zipfile.ZipFile(HAND_MODEL_PATH) as z:
        weights_h5 = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")

    def get(name, idx):
        return np.array(weights_h5["_layer_checkpoint_dependencies\\" + name]["vars"][str(idx)])

    model.get_layer("lstm").set_weights([get("lstm\\cell", 0), get("lstm\\cell", 1), get("lstm\\cell", 2)])
    model.get_layer("lstm_1").set_weights([get("lstm_2\\cell", 0), get("lstm_2\\cell", 1), get("lstm_2\\cell", 2)])
    model.get_layer("dense").set_weights([get("dense", 0), get("dense", 1)])
    model.get_layer("dense_1").set_weights([get("dense_2", 0), get("dense_2", 1)])
    model.get_layer("dense_2").set_weights([get("dense_4", 0), get("dense_4", 1)])
    weights_h5.close()
    return model


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


def predict_hand(model, lm_list):
    data = np.array(lm_list).reshape(1, HAND_TIMESTEPS, 21, 3)
    dnn_data = np.array(lm_list).reshape(1, HAND_TIMESTEPS, 63)
    lstm_last = data[:, -1, :, :]
    dnn_last = dnn_data[:, -1, :]
    result = model.predict([lstm_last, dnn_last], verbose=0)
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

    fall_lm_list = []
    hand_lm_list = []
    fall_label = "Warmup..."
    hand_label = "Warmup..."
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
                    fall_lm_list.append(pose_landmarks_to_vector(pose_results))
                    frame = draw_pose(frame, pose_results)
                    if len(fall_lm_list) == FALL_TIMESTEPS:
                        fall_label = predict_fall(fall_model, fall_lm_list)
                        fall_lm_list = []

            if enable_hand:
                hand_results = hand_detector.process(img_rgb)
                if hand_results.multi_hand_landmarks:
                    hand_lm_list.append(hand_landmarks_to_vector(hand_results))
                    frame = draw_hand(frame, hand_results)
                    if len(hand_lm_list) == HAND_TIMESTEPS:
                        hand_label = predict_hand(hand_model, hand_lm_list)
                        hand_lm_list = []
                else:
                    hand_label = "No hands"

        sos_alert = (enable_fall and fall_label == "Fall") or (enable_hand and hand_label in SOS_HAND_LABELS)

        now = time.time()
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
