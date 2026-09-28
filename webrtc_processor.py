"""VideoProcessor cho streamlit-webrtc — thay cho cv2.VideoCapture(0) đọc
webcam CỦA MÁY CHẠY SERVER (chỉ work khi chạy local). Với streamlit-webrtc,
trình duyệt của MỖI NGƯỜI XEM tự capture webcam của HỌ qua WebRTC, gửi từng
frame lên hàm recv() dưới đây để xử lý, rồi phát video đã xử lý ngược lại —
đây là điều kiện bắt buộc để app hoạt động đúng khi deploy lên 1 địa chỉ
web thật cho nhiều người dùng riêng biệt.

Khác biệt lớn nhất so với vòng lặp while cũ trong live_view.py: recv() được
gọi bởi streamlit-webrtc trên 1 THREAD RIÊNG, liên tục, độc lập với vòng đời
script Streamlit chính (script chính chạy xong gần như ngay sau khi gọi
webrtc_streamer(), nhưng recv() vẫn tiếp tục chạy). Mọi state cần giữ giữa
các frame (buffer landmark, vote, timer...) phải nằm trong instance của
class này (self.*), không phải biến cục bộ trong script chính. Để script
chính (chạy trên thread khác) đọc được kết quả mới nhất, dùng self.lock +
self.public_state (dict) — ghi trong recv(), đọc ở script chính qua
get_state().
"""

import threading
import time
from collections import deque

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

import detection_engine as engine
import event_log
from hand_features import SIGNAL_SEQUENCE_LENGTH


class DetectionVideoProcessor(VideoProcessorBase):
    def __init__(self, fall_model, hand_model, signal_model, pose_detector, hand_detector, config):
        self.fall_model = fall_model
        self.hand_model = hand_model
        self.signal_model = signal_model
        self.pose_detector = pose_detector
        self.hand_detector = hand_detector

        # Snapshot cấu hình tại thời điểm bắt đầu stream — đổi ở trang Cấu
        # hình trong lúc đang stream sẽ KHÔNG áp dụng ngay, cần dừng/bắt đầu
        # lại stream để nhận cấu hình mới (đơn giản hoá, chấp nhận được).
        self.config = config

        self.lock = threading.Lock()
        self.public_state = {
            "fall_label": "Warmup...",
            "hand_label": "Warmup...",
            "hand_confidence": 0.0,
            "signal_detected": False,
            "signal_probability": 0.0,
            "fall_down_since": None,
            "fall_escalated": False,
            "sos_alert": False,
            "fps": 0.0,
            "monitor_start_time": time.time(),
        }

        self.fall_lm_window = deque(maxlen=engine.FALL_TIMESTEPS)
        self.fall_votes = deque(maxlen=engine.FALL_VOTE_WINDOW)
        self.hand_votes = deque(maxlen=engine.HAND_VOTE_WINDOW)
        self.signal_window = deque(maxlen=SIGNAL_SEQUENCE_LENGTH)

        self.fall_label = "Warmup..."
        self.fall_down_since = None
        self.fall_escalated = False
        self.was_fall_monitoring = False
        self.episode_escalated = False
        self.recovered_message_until = 0.0
        self.hand_label = "Warmup..."
        self.hand_confidence = 0.0
        self.signal_detected = False
        self.signal_probability = 0.0
        self.signal_consecutive_count = 0
        self.alert_until = 0.0
        self.was_alerting = False
        self.frame_count = 0
        self.prev_time = time.time()

        event_log.init_db()

    def get_state(self):
        with self.lock:
            return dict(self.public_state)

    # Webcam thường gửi ~30 frame/giây, nhưng recv() phải xử lý XONG mỗi
    # frame mới được trả về hiển thị (khác hẳn bản cv2 cũ, video giờ hiện
    # trực tiếp qua <video> của trình duyệt, không qua Streamlit nữa) — 1
    # frame chạy đủ Pose+Hands+2-3 model TF dễ mất >100ms, tức <10fps thật,
    # gây giật rõ dù máy mạnh hay yếu. Chỉ chạy AI mỗi PROCESS_EVERY_N_FRAMES
    # frame, các frame còn lại trả về ngay (không chờ AI) — video mượt hơn
    # nhiều, đổi lại tốc độ CẬP NHẬT kết quả nhận diện giảm (vẫn đủ nhanh
    # cho mục đích an toàn — không cần kiểm tra 30 lần/giây).
    PROCESS_EVERY_N_FRAMES = 3

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cfg = self.config

        if cfg["mirror_camera"]:
            img = cv2.flip(img, 1)

        self.frame_count += 1
        should_process = (
            self.frame_count > engine.WARMUP_FRAMES
            and self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0
        )

        if should_process:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_detector.process(img_rgb) if cfg["enable_fall"] else None
            hand_results = self.hand_detector.process(img_rgb) if cfg["enable_hand"] else None

            if cfg["enable_fall"] and pose_results.pose_landmarks:
                self.fall_lm_window.append(engine.pose_landmarks_to_vector(pose_results))
                if cfg["show_overlay"]:
                    img = engine.draw_pose(img, pose_results)

            raw_vector = None
            if cfg["enable_hand"]:
                if hand_results.multi_hand_landmarks:
                    if cfg["show_overlay"]:
                        img = engine.draw_hand(img, hand_results)
                    raw_vector = engine.hand_landmarks_to_vector(hand_results)
                    self.signal_window.append(raw_vector)
                else:
                    self.hand_votes.clear()
                    self.hand_label = "No hands"
                    self.signal_window.clear()
                    self.signal_consecutive_count = 0
                    self.signal_detected = False

            if cfg["enable_fall"] and len(self.fall_lm_window) == engine.FALL_TIMESTEPS:
                self.fall_votes.append(engine.predict_fall(self.fall_model, list(self.fall_lm_window)))
                self.fall_label = engine.majority_label(self.fall_votes, self.fall_label)

            if cfg["enable_hand"] and raw_vector is not None:
                raw_label, self.hand_confidence = engine.predict_hand(
                    self.hand_model, raw_vector, cfg["hand_confidence_threshold"]
                )
                self.hand_votes.append(raw_label)
                self.hand_label = engine.majority_label(self.hand_votes, self.hand_label)

            if (
                cfg["enable_hand"]
                and cfg["enable_signal"]
                and self.signal_model is not None
                and raw_vector is not None
                and len(self.signal_window) == SIGNAL_SEQUENCE_LENGTH
            ):
                raw_signal, self.signal_probability = engine.predict_signal_sequence(
                    self.signal_model, list(self.signal_window), cfg["signal_confidence_threshold"]
                )
                self.signal_consecutive_count = self.signal_consecutive_count + 1 if raw_signal else 0
                self.signal_detected = self.signal_consecutive_count >= engine.SIGNAL_CONSECUTIVE_REQUIRED

        now = time.time()

        if cfg["enable_fall"]:
            if self.fall_label == "Fall":
                if self.fall_down_since is None:
                    self.fall_down_since = now
                    self.was_fall_monitoring = True
                    self.episode_escalated = False
            else:
                if self.was_fall_monitoring and not self.episode_escalated:
                    event_log.log_event("Ngã - đã phục hồi", "Đứng lên lại trong lúc theo dõi, không báo động")
                    self.recovered_message_until = now + 4.0
                self.was_fall_monitoring = False
                self.fall_down_since = None
            self.fall_escalated = self.fall_down_since is not None and (now - self.fall_down_since) >= cfg[
                "fall_grace_seconds"
            ]
            if self.fall_escalated:
                self.episode_escalated = True
        else:
            self.fall_down_since = None
            self.fall_escalated = False

        raw_sos = (
            (cfg["enable_fall"] and self.fall_escalated)
            or (cfg["enable_hand"] and self.hand_label in engine.SOS_HAND_LABELS)
            or (cfg["enable_hand"] and cfg["enable_signal"] and self.signal_detected)
        )
        if raw_sos:
            self.alert_until = now + cfg["alert_hold_seconds"]
        sos_alert = now < self.alert_until

        if sos_alert and not self.was_alerting:
            if cfg["enable_fall"] and self.fall_escalated:
                event_log.log_event("Ngã (không đứng lên lại)", f"Sau {cfg['fall_grace_seconds']:.0f}s theo dõi")
            elif cfg["enable_hand"] and self.hand_label in engine.SOS_HAND_LABELS:
                event_log.log_event(self.hand_label, f"Độ tin cậy {self.hand_confidence * 100:.0f}%")
            elif cfg["enable_hand"] and cfg["enable_signal"] and self.signal_detected:
                event_log.log_event(
                    "Signal for Help (chuỗi động tác)", f"Xác suất {self.signal_probability * 100:.0f}%"
                )
        self.was_alerting = sos_alert

        fps = 1 / (now - self.prev_time) if now > self.prev_time else 0
        self.prev_time = now

        with self.lock:
            self.public_state.update(
                {
                    "fall_label": self.fall_label,
                    "hand_label": self.hand_label,
                    "hand_confidence": self.hand_confidence,
                    "signal_detected": self.signal_detected,
                    "signal_probability": self.signal_probability,
                    "fall_down_since": self.fall_down_since,
                    "fall_escalated": self.fall_escalated,
                    "sos_alert": sos_alert,
                    "recovered_message_until": self.recovered_message_until,
                    "fps": fps,
                }
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")
