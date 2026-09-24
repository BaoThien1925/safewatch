"""Trang Xem trực tiếp — webcam + 3 pipeline nhận diện (Ngã, Tay tĩnh,
Signal for Help theo chuỗi). Logic nhận diện y hệt app.py gốc (xem
detection_engine.py); phần khác biệt ở đây chỉ là lớp trình bày: 70/30
layout, banner 3 trạng thái Safe/Monitoring/Emergency, ẩn thuật ngữ kỹ
thuật khỏi giao diện chính (đưa vào "Thông tin nâng cao").

Bật/tắt pipeline + ngưỡng được cấu hình ở trang Cấu hình, đọc qua
st.session_state — trang này chỉ hiển thị + có nút Bắt đầu/Dừng giám sát.
"""

import threading
import time
from collections import deque

import cv2
import streamlit as st

import detection_engine as engine
import event_log
from hand_features import SIGNAL_SEQUENCE_LENGTH
from theme import badge_html, status_banner_html

engine.ensure_settings_defaults()

st.title("Xem trực tiếp")

s = st.session_state
enable_fall = s["enable_fall"]
enable_hand = s["enable_hand"]
enable_signal = s["enable_signal"]
fall_grace_seconds = s["fall_grace_seconds"]
hand_confidence_threshold = s["hand_confidence_threshold"]
signal_confidence_threshold = s["signal_confidence_threshold"]
show_overlay = s["show_overlay"]
mirror_camera = s["mirror_camera"]
alert_hold_seconds = s["alert_hold_seconds"]
cam_width, cam_height = engine.CAMERA_RESOLUTIONS.get(s["camera_resolution"], (640, 480))

top_col1, top_col2, top_col3 = st.columns([3, 2, 1])
with top_col1:
    st.selectbox("Camera", ["Webcam chính (mặc định)"], disabled=True)
with top_col2:
    st.caption("Bật/tắt pipeline và ngưỡng: xem trang **Cấu hình**")
with top_col3:
    run = st.checkbox("▶️ Bắt đầu giám sát", value=False, key="run_webcam")

video_col, status_col = st.columns([7, 3])

live_badge_placeholder = video_col.empty()
frame_placeholder = video_col.empty()

status_banner_placeholder = status_col.empty()
fall_line_placeholder = status_col.empty()
hand_line_placeholder = status_col.empty()
advanced_placeholder = status_col.container()

if not run:
    live_badge_placeholder.empty()
    frame_placeholder.info("Bấm '▶️ Bắt đầu giám sát' ở trên để xem camera trực tiếp.")
    status_banner_placeholder.markdown(
        status_banner_html("safe", "CHƯA GIÁM SÁT", "Bấm Bắt đầu giám sát để hệ thống theo dõi."),
        unsafe_allow_html=True,
    )
    st.stop()

live_badge_placeholder.markdown(badge_html("live", "● LIVE"), unsafe_allow_html=True)

fall_model, hand_model, signal_model = engine.load_models()
if enable_signal and signal_model is None:
    enable_signal = False
pose_detector, hand_detector = engine.get_mediapipe()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, cam_width)
cap.set(4, cam_height)

if not cap.isOpened():
    st.error(
        "Không mở được webcam. Kiểm tra webcam có đang bị ứng dụng khác chiếm dụng, "
        "hoặc chưa cấp quyền camera cho Python trong Windows Settings > Privacy > Camera."
    )
    st.stop()

event_log.init_db()
was_alerting = False
was_fall_monitoring = False
episode_escalated = False
recovered_message_until = 0.0

fall_lm_window = deque(maxlen=engine.FALL_TIMESTEPS)
fall_votes = deque(maxlen=engine.FALL_VOTE_WINDOW)
hand_votes = deque(maxlen=engine.HAND_VOTE_WINDOW)
signal_window = deque(maxlen=SIGNAL_SEQUENCE_LENGTH)

fall_label = "Warmup..."
fall_down_since = None
fall_escalated = False
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

    if mirror_camera:
        frame = cv2.flip(frame, 1)

    frame_count += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count > engine.WARMUP_FRAMES:
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
                fall_lm_window.append(engine.pose_landmarks_to_vector(pose_results))
                if show_overlay:
                    frame = engine.draw_pose(frame, pose_results)
                if len(fall_lm_window) == engine.FALL_TIMESTEPS:
                    fall_votes.append(engine.predict_fall(fall_model, list(fall_lm_window)))
                    fall_label = engine.majority_label(fall_votes, fall_label)

        if enable_hand:
            hand_results = detection_results["hands"]
            if hand_results.multi_hand_landmarks:
                if show_overlay:
                    frame = engine.draw_hand(frame, hand_results)
                raw_vector = engine.hand_landmarks_to_vector(hand_results)
                raw_label, hand_confidence = engine.predict_hand(hand_model, raw_vector, hand_confidence_threshold)
                hand_votes.append(raw_label)
                hand_label = engine.majority_label(hand_votes, hand_label)

                if enable_signal:
                    signal_window.append(raw_vector)
                    if len(signal_window) == SIGNAL_SEQUENCE_LENGTH:
                        raw_signal, signal_probability = engine.predict_signal_sequence(
                            signal_model, list(signal_window), signal_confidence_threshold
                        )
                        signal_consecutive_count = signal_consecutive_count + 1 if raw_signal else 0
                        signal_detected = signal_consecutive_count >= engine.SIGNAL_CONSECUTIVE_REQUIRED
            else:
                hand_votes.clear()
                hand_label = "No hands"
                signal_window.clear()
                signal_consecutive_count = 0
                signal_detected = False

    now = time.time()

    if enable_fall:
        if fall_label == "Fall":
            if fall_down_since is None:
                fall_down_since = now
                was_fall_monitoring = True
                episode_escalated = False
        else:
            if was_fall_monitoring and not episode_escalated:
                event_log.log_event("Ngã - đã phục hồi", "Đứng lên lại trong lúc theo dõi, không báo động")
                recovered_message_until = now + 4.0
            was_fall_monitoring = False
            fall_down_since = None
        fall_escalated = fall_down_since is not None and (now - fall_down_since) >= fall_grace_seconds
        if fall_escalated:
            episode_escalated = True
    else:
        fall_down_since = None
        fall_escalated = False

    raw_sos = (
        (enable_fall and fall_escalated)
        or (enable_hand and hand_label in engine.SOS_HAND_LABELS)
        or (enable_hand and enable_signal and signal_detected)
    )
    if raw_sos:
        alert_until = now + alert_hold_seconds
    sos_alert = now < alert_until

    if sos_alert and not was_alerting:
        if enable_fall and fall_escalated:
            event_log.log_event("Ngã (không đứng lên lại)", f"Sau {fall_grace_seconds:.0f}s theo dõi")
        elif enable_hand and hand_label in engine.SOS_HAND_LABELS:
            event_log.log_event(hand_label, f"Độ tin cậy {hand_confidence * 100:.0f}%")
        elif enable_hand and enable_signal and signal_detected:
            event_log.log_event("Signal for Help (chuỗi động tác)", f"Xác suất {signal_probability * 100:.0f}%")
    was_alerting = sos_alert

    fps = 1 / (now - prev_time) if now > prev_time else 0
    prev_time = now

    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    # Ghi lại trạng thái mới nhất vào session_state để trang Dashboard đọc
    # được — Streamlit chỉ chạy 1 trang/lần nên Dashboard không "live" thật,
    # chỉ hiện đúng trạng thái ghi nhận lần cuối trang này còn chạy.
    if sos_alert:
        st.session_state["last_status"] = "emergency"
    elif enable_fall and fall_down_since is not None and not fall_escalated:
        st.session_state["last_status"] = "monitoring"
    else:
        st.session_state["last_status"] = "safe"
    st.session_state["last_status_time"] = now
    st.session_state["camera_online"] = True

    # ----- Banner trạng thái tổng thể (Safe / Monitoring / Emergency) -----
    if sos_alert:
        status_banner_placeholder.markdown(
            status_banner_html("emergency", "🚨 TÌNH HUỐNG KHẨN CẤP", "SOS đã được kích hoạt"),
            unsafe_allow_html=True,
        )
    elif enable_fall and fall_down_since is not None and not fall_escalated:
        remaining = max(0, fall_grace_seconds - (now - fall_down_since))
        progress = min(1.0, (now - fall_down_since) / fall_grace_seconds)
        with status_banner_placeholder.container():
            st.markdown(
                status_banner_html(
                    "monitoring",
                    "⏳ ĐANG THEO DÕI SAU NGÃ",
                    "Phát hiện dấu hiệu ngã. Hệ thống đang xác minh tình trạng.",
                ),
                unsafe_allow_html=True,
            )
            st.progress(progress, text=f"Còn {remaining:.0f}s trước khi báo động")
    elif now < recovered_message_until:
        status_banner_placeholder.markdown(
            status_banner_html("safe", "✅ ĐÃ PHỤC HỒI", "Người dùng đã đứng lên lại — cảnh báo đã được huỷ."),
            unsafe_allow_html=True,
        )
    else:
        status_banner_placeholder.markdown(
            status_banner_html("safe", "AN TOÀN", "Không phát hiện ngã hoặc tín hiệu cầu cứu."),
            unsafe_allow_html=True,
        )

    fall_line_placeholder.markdown(
        f"**Phát hiện ngã:** {engine.friendly_fall_label(fall_label) if enable_fall else 'Đã tắt'}"
    )
    hand_line_placeholder.markdown(
        f"**Cử chỉ tay:** {engine.friendly_hand_label(hand_label) if enable_hand else 'Đã tắt'}"
    )

    with advanced_placeholder:
        with st.expander("Thông tin nâng cao", expanded=False):
            st.caption(f"Nhãn thô (ngã): `{fall_label}`")
            st.caption(f"Nhãn thô (tay): `{hand_label}`")
            st.caption(f"Độ tin cậy tay: {hand_confidence * 100:.0f}%")
            if enable_signal:
                st.caption(f"Signal chuỗi: {'Có' if signal_detected else 'Không'} ({signal_probability * 100:.0f}%)")
            st.caption(f"FPS: {fps:.1f}")
            st.caption(f"Độ phân giải: {cam_width}x{cam_height}")
            st.caption("Model: LSTM (ngã) · DNN+LSTM 14 nhãn (tay) · LSTM nhị phân (signal, thử nghiệm)")

    run = st.session_state["run_webcam"]

cap.release()
st.session_state["camera_online"] = False
