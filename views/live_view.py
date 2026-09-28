"""Trang chủ (Xem trực tiếp) — dùng streamlit-webrtc thay vì
cv2.VideoCapture(0) đọc webcam CỦA MÁY CHẠY SERVER. Với streamlit-webrtc,
trình duyệt của MỖI NGƯỜI XEM tự capture webcam CỦA HỌ qua WebRTC và gửi
từng frame lên DetectionVideoProcessor.recv() (webrtc_processor.py) để xử
lý — đây là điều kiện bắt buộc để deploy app lên 1 địa chỉ web thật cho
nhiều người dùng riêng biệt (bản cv2 cũ chỉ chạy đúng khi mở local trên
đúng máy có webcam, xem app.py và lịch sử git để tham khảo bản đó).

Xử lý AI thật sự chạy trong DetectionVideoProcessor, trên 1 thread riêng
do streamlit-webrtc quản lý — liên tục, độc lập với vòng đời script này.
Vòng lặp while ở cuối file CHỈ để polling + cập nhật hiển thị (banner,
card, FPS), không xử lý frame nào cả.
"""

import time

import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    get_cloudflare_ice_servers,
    get_hf_ice_servers,
    webrtc_streamer,
)

import detection_engine as engine
import event_log
from theme import sidebar_status_html, status_banner_html
from webrtc_processor import DetectionVideoProcessor

engine.ensure_settings_defaults()

# STUN công khai của Google luôn có sẵn — đủ cho mạng "dễ" (nhà, hầu hết
# trường học). Mạng chặt hơn (NAT đối xứng, firewall công ty/1 số mobile
# network) cần thêm TURN server (máy trung chuyển thật) mới kết nối được —
# STUN không đủ trong trường hợp đó, biểu hiện là "Connection is taking
# longer than expected". Ưu tiên Cloudflare Realtime TURN (ổn định hơn,
# cần CF_TURN_KEY_ID + CF_TURN_API_TOKEN trong Secrets), rơi về TURN miễn
# phí Hugging Face (HF_TOKEN) nếu không có Cloudflare, cuối cùng rơi về
# STUN-only nếu không có cả 2 (Settings > Secrets trên Streamlit Cloud).
_ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
try:
    # st.secrets ném StreamlitSecretNotFoundError ngay khi truy cập (không
    # phải chỉ khi thiếu key) nếu máy không có file secrets.toml — bình
    # thường lúc chạy local (chỉ Streamlit Cloud mới tự có qua Settings >
    # Secrets), nên phải bọc try/except quanh CẢ việc đọc, không chỉ .get().
    _cf_key_id = st.secrets.get("CF_TURN_KEY_ID")
    _cf_api_token = st.secrets.get("CF_TURN_API_TOKEN")
    _hf_token = st.secrets.get("HF_TOKEN")
except Exception:
    _cf_key_id = _cf_api_token = _hf_token = None

if _cf_key_id and _cf_api_token:
    try:
        _ice_servers = get_cloudflare_ice_servers(_cf_key_id, _cf_api_token)
    except Exception:
        pass  # key sai/hết hạn/mạng lỗi lúc xin TURN -> thử HF bên dưới thay vì crash.
if _ice_servers == [{"urls": ["stun:stun.l.google.com:19302"]}] and _hf_token:
    try:
        _ice_servers = get_hf_ice_servers(_hf_token)
    except Exception:
        pass  # rơi về STUN, còn hơn crash cả app.

RTC_CONFIGURATION = RTCConfiguration({"iceServers": _ice_servers})

st.title("Trang chủ")
st.caption("Giám sát an toàn cá nhân bằng AI")

s = st.session_state
config = {
    "enable_fall": s["enable_fall"],
    "enable_hand": s["enable_hand"],
    "enable_signal": s["enable_signal"],
    "fall_grace_seconds": s["fall_grace_seconds"],
    "hand_confidence_threshold": s["hand_confidence_threshold"],
    "signal_confidence_threshold": s["signal_confidence_threshold"],
    "mirror_camera": s["mirror_camera"],
    "alert_hold_seconds": s["alert_hold_seconds"],
    "show_overlay": s["show_overlay"],
}

control_col1, control_col2 = st.columns([2, 3])
with control_col1:
    st.selectbox("Camera", ["Webcam của bạn (qua trình duyệt)"], disabled=True)
with control_col2:
    config["show_overlay"] = st.checkbox("Hiện điểm landmark", value=s["show_overlay"], key="show_overlay")
    st.caption(
        "Bật/tắt pipeline và ngưỡng: xem trang **Cấu hình**. Đổi cấu hình cần Stop rồi Start lại "
        "stream bên dưới để áp dụng (cấu hình được chốt lúc bắt đầu stream)."
    )

video_col, action_col = st.columns([7, 3])

fall_model, hand_model, signal_model = engine.load_models()
if config["enable_signal"] and signal_model is None:
    config["enable_signal"] = False
pose_detector, hand_detector = engine.get_mediapipe()

# Làm nóng model 1 lần (xem detection_engine.warm_up_models) trước khi
# stream thật bắt đầu, để chi phí trace/biên dịch graph lần đầu không lộ ra
# thành giật ở vài frame đầu tiên người dùng xem.
engine.warm_up_models(fall_model, hand_model, signal_model)


def processor_factory():
    return DetectionVideoProcessor(fall_model, hand_model, signal_model, pose_detector, hand_detector, config)


cam_width, cam_height = engine.CAMERA_RESOLUTIONS.get(s["camera_resolution"], (640, 480))

with video_col:
    webrtc_ctx = webrtc_streamer(
        key="safewatch-live",
        video_processor_factory=processor_factory,
        rtc_configuration=RTC_CONFIGURATION,
        # Không giới hạn độ phân giải ở đây thì trình duyệt tự capture ở độ
        # phân giải MẶC ĐỊNH của webcam (thường 720p+) — nặng hơn hẳn cho
        # mọi bước sau (mã hoá gửi lên, decode ở server, chạy AI), nhất là
        # trên CPU giới hạn của free tier cloud. "ideal" là gợi ý, không ép
        # buộc — webcam yếu hơn vẫn hoạt động, chỉ không vượt quá mức này.
        media_stream_constraints={
            "video": {"width": {"ideal": cam_width}, "height": {"ideal": cam_height}},
            "audio": False,
        },
        # Hàng đợi nội bộ mặc định giữ tới 4 frame trước khi xử lý — cộng
        # dồn thành độ trễ vài trăm ms tới vài giây nếu xử lý không theo
        # kịp. Giảm về 1 (giống CAP_PROP_BUFFERSIZE=1 ở bản cv2 cũ): luôn
        # lấy đúng frame mới nhất, chấp nhận rớt frame cũ thay vì trễ dồn.
        video_receiver_size=1,
    )

with action_col:
    status_banner_placeholder = st.empty()
    fall_line_placeholder = st.empty()
    hand_line_placeholder = st.empty()
    advanced_placeholder = st.empty()

st.markdown("")
card_col1, card_col2, card_col3 = st.columns(3)
with card_col1:
    camera_card_placeholder = st.empty()
with card_col2:
    duration_card_placeholder = st.empty()
with card_col3:
    incidents_card_placeholder = st.empty()


def render_kpi_card(placeholder, label, value, sub):
    with placeholder.container(border=True):
        st.markdown(f'<div class="sw-kpi-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sw-kpi-value">{value}</div>', unsafe_allow_html=True)
        st.caption(sub)


def count_today_incidents():
    return sum(1 for ts, _, _ in event_log.get_events(limit=1000) if ts.startswith(time.strftime("%Y-%m-%d")))


event_log.init_db()


def render_idle():
    status_banner_placeholder.markdown(
        status_banner_html("safe", "CHƯA GIÁM SÁT", 'Bấm "START" ở khung camera bên trái để hệ thống theo dõi.'),
        unsafe_allow_html=True,
    )
    fall_line_placeholder.markdown("**Phát hiện ngã:** Đã tắt")
    hand_line_placeholder.markdown("**Cử chỉ tay:** Đã tắt")
    advanced_placeholder.empty()
    render_kpi_card(camera_card_placeholder, "📷 Trạng thái camera", "Chưa hoạt động", "Webcam của bạn")
    render_kpi_card(duration_card_placeholder, "⏱️ Thời gian giám sát", "00:00:00", "Chưa bắt đầu")
    render_kpi_card(incidents_card_placeholder, "🚨 Sự cố phát hiện", str(count_today_incidents()), "Hôm nay")
    st.session_state["camera_online"] = False
    st.session_state["last_status"] = None
    if st.session_state.get("sidebar_status_placeholder") is not None:
        st.session_state["sidebar_status_placeholder"].markdown(sidebar_status_html(False), unsafe_allow_html=True)


if not webrtc_ctx.state.playing:
    st.session_state["_was_playing"] = False
    render_idle()
    st.stop()

if not st.session_state.get("_was_playing"):
    st.session_state["monitor_start_time"] = time.time()
st.session_state["_was_playing"] = True
st.session_state["camera_online"] = True
if st.session_state.get("sidebar_status_placeholder") is not None:
    st.session_state["sidebar_status_placeholder"].markdown(sidebar_status_html(True), unsafe_allow_html=True)

while webrtc_ctx.state.playing:
    processor = webrtc_ctx.video_processor
    if processor is None:
        time.sleep(0.2)
        continue

    state = processor.get_state()
    now = time.time()

    fall_label = state["fall_label"]
    hand_label = state["hand_label"]
    sos_alert = state["sos_alert"]
    fall_down_since = state["fall_down_since"]
    fall_escalated = state["fall_escalated"]
    recovered_message_until = state.get("recovered_message_until", 0.0)

    if sos_alert:
        st.session_state["last_status"] = "emergency"
    elif config["enable_fall"] and fall_down_since is not None and not fall_escalated:
        st.session_state["last_status"] = "monitoring"
    else:
        st.session_state["last_status"] = "safe"

    if sos_alert:
        status_banner_placeholder.markdown(
            status_banner_html("emergency", "🚨 TÌNH HUỐNG KHẨN CẤP", "SOS đã được kích hoạt"),
            unsafe_allow_html=True,
        )
    elif config["enable_fall"] and fall_down_since is not None and not fall_escalated:
        remaining = max(0, config["fall_grace_seconds"] - (now - fall_down_since))
        progress = min(1.0, (now - fall_down_since) / config["fall_grace_seconds"])
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
        f"**Phát hiện ngã:** {engine.friendly_fall_label(fall_label) if config['enable_fall'] else 'Đã tắt'}"
    )
    hand_line_placeholder.markdown(
        f"**Cử chỉ tay:** {engine.friendly_hand_label(hand_label) if config['enable_hand'] else 'Đã tắt'}"
    )

    with advanced_placeholder.container():
        with st.expander("Thông tin nâng cao", expanded=False):
            st.caption(f"Nhãn thô (ngã): `{fall_label}`")
            st.caption(f"Nhãn thô (tay): `{hand_label}`")
            st.caption(f"Độ tin cậy tay: {state['hand_confidence'] * 100:.0f}%")
            if config["enable_signal"]:
                st.caption(
                    f"Signal chuỗi: {'Có' if state['signal_detected'] else 'Không'} "
                    f"({state['signal_probability'] * 100:.0f}%)"
                )
            st.caption(f"FPS xử lý: {state['fps']:.1f}")
            st.caption("Model: LSTM (ngã) · DNN+LSTM 14 nhãn (tay) · LSTM nhị phân (signal, thử nghiệm)")

    elapsed = now - st.session_state["monitor_start_time"]
    hh, rem = divmod(int(elapsed), 3600)
    mm, ss = divmod(rem, 60)
    render_kpi_card(camera_card_placeholder, "📷 Trạng thái camera", "Đang hoạt động", "Webcam của bạn")
    render_kpi_card(
        duration_card_placeholder, "⏱️ Thời gian giám sát", f"{hh:02d}:{mm:02d}:{ss:02d}", "Đang giám sát"
    )
    render_kpi_card(incidents_card_placeholder, "🚨 Sự cố phát hiện", str(count_today_incidents()), "Hôm nay")

    time.sleep(0.3)

st.session_state["_was_playing"] = False
st.session_state["camera_online"] = False
st.session_state["last_status"] = None
if st.session_state.get("sidebar_status_placeholder") is not None:
    st.session_state["sidebar_status_placeholder"].markdown(sidebar_status_html(False), unsafe_allow_html=True)
