"""Trang Cấu hình — 4 tab: Nhận diện, Camera, Cảnh báo, Lưu trữ. Các giá trị
đọc/ghi qua st.session_state (key trùng SETTINGS_DEFAULTS trong
detection_engine.py) để trang Xem trực tiếp đọc lại được ngay.

Lưu ý thật: mục Email/SMS/Push/Lưu snapshot/video hiện chỉ là toggle giao
diện — demo này chưa có backend gửi thông báo hay ghi video thật.
"""

import streamlit as st

import detection_engine as engine
import event_log

engine.ensure_settings_defaults()
event_log.init_db()

st.title("Cấu hình")

tab_detect, tab_camera, tab_alert, tab_storage = st.tabs(["Nhận diện", "Camera", "Cảnh báo", "Lưu trữ"])

with tab_detect:
    st.session_state["enable_fall"] = st.checkbox("Bật phát hiện Ngã", value=st.session_state["enable_fall"])
    st.session_state["fall_grace_seconds"] = st.slider(
        "Thời gian xác minh sau khi Ngã (giây)",
        min_value=5,
        max_value=300,
        value=st.session_state["fall_grace_seconds"],
        step=5,
        help="Ngã xong tự đứng lên lại trong thời gian này sẽ KHÔNG báo động.",
    )
    st.session_state["enable_hand"] = st.checkbox("Bật phát hiện cử chỉ tay", value=st.session_state["enable_hand"])
    st.session_state["hand_confidence_threshold"] = st.slider(
        "Ngưỡng tin cậy cử chỉ tay (%)",
        min_value=0,
        max_value=100,
        value=int(st.session_state["hand_confidence_threshold"] * 100),
        help="Cử chỉ có độ tin cậy thấp hơn ngưỡng này sẽ hiện 'Không rõ cử chỉ'.",
    ) / 100

    signal_col1, signal_col2 = st.columns([3, 1])
    with signal_col1:
        st.session_state["enable_signal"] = st.checkbox(
            "Bật Signal for Help theo chuyển động", value=st.session_state["enable_signal"]
        )
    with signal_col2:
        st.markdown(
            '<span style="background:#F1F5F9;color:#64748B;padding:0.15rem 0.5rem;'
            'border-radius:6px;font-size:0.72rem;font-weight:600;">Experimental</span>',
            unsafe_allow_html=True,
        )
    st.session_state["signal_confidence_threshold"] = st.slider(
        "Ngưỡng tin cậy chuỗi Signal for Help (%)",
        min_value=0,
        max_value=100,
        value=int(st.session_state["signal_confidence_threshold"] * 100),
    ) / 100
    st.caption("Model này train trên rất ít data, còn hay báo sai — xem chi tiết ở README/training/.")

    st.session_state["alert_hold_seconds"] = st.slider(
        "Thời gian giữ cảnh báo tối thiểu (giây)",
        min_value=0.5,
        max_value=10.0,
        value=float(st.session_state["alert_hold_seconds"]),
        step=0.5,
        help="Tránh cảnh báo nhấp nháy khi tín hiệu dao động ngay ở ngưỡng.",
    )

with tab_camera:
    st.session_state["camera_resolution"] = st.selectbox(
        "Độ phân giải camera",
        list(engine.CAMERA_RESOLUTIONS.keys()),
        index=list(engine.CAMERA_RESOLUTIONS.keys()).index(st.session_state["camera_resolution"]),
    )
    st.session_state["show_overlay"] = st.checkbox(
        "Hiển thị landmark đè lên video", value=st.session_state["show_overlay"]
    )
    st.session_state["mirror_camera"] = st.checkbox(
        "Lật ngược camera (mirror)", value=st.session_state["mirror_camera"]
    )
    st.caption("Nguồn camera: hiện chỉ hỗ trợ 1 webcam local (index 0). Nhiều camera cần bản mở rộng sau.")

with tab_alert:
    st.markdown("**Kênh gửi cảnh báo** *(giao diện minh hoạ — chưa có backend gửi thật trong demo này)*")
    st.session_state["notify_email"] = st.checkbox("Gửi email khi có cảnh báo", value=st.session_state["notify_email"])
    st.session_state["notify_sms"] = st.checkbox("Gửi SMS khi có cảnh báo", value=st.session_state["notify_sms"])
    st.session_state["notify_push"] = st.checkbox("Push notification", value=st.session_state["notify_push"])
    st.page_link("views/emergency_contacts.py", label="Quản lý người liên hệ khẩn cấp →", icon="📞")
    st.markdown("---")
    st.session_state["sound_alert"] = st.checkbox(
        "Phát âm thanh khi báo động khẩn cấp", value=st.session_state["sound_alert"]
    )
    st.session_state["silent_monitoring"] = st.checkbox(
        "Theo dõi im lặng sau Ngã (không phát âm thanh trong lúc xác minh)",
        value=st.session_state["silent_monitoring"],
        help="Đúng ý tưởng gốc: tránh gây chú ý không cần thiết trong tình huống nhạy cảm (VD bị khống chế).",
    )

with tab_storage:
    st.session_state["retention_days"] = st.slider(
        "Lưu lịch sử bao nhiêu ngày", min_value=7, max_value=365, value=st.session_state["retention_days"], step=7
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Áp dụng — xoá dữ liệu cũ hơn giới hạn"):
            removed = event_log.prune_older_than(st.session_state["retention_days"])
            st.success(f"Đã xoá {removed} bản ghi cũ hơn {st.session_state['retention_days']} ngày.")
    with col2:
        st.caption("Áp dụng ngay cho lịch sử hiện có trong database.")

    st.markdown("---")
    st.session_state["save_snapshot"] = st.checkbox(
        "Lưu ảnh chụp lúc xảy ra sự cố", value=st.session_state["save_snapshot"], disabled=True
    )
    st.session_state["save_video"] = st.checkbox(
        "Lưu video lúc xảy ra sự cố", value=st.session_state["save_video"], disabled=True
    )
    st.caption("Chưa hỗ trợ trong bản demo — cần thêm dung lượng lưu trữ và luồng ghi video riêng.")
