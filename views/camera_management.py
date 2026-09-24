"""Trang Quản lý camera — demo chỉ hỗ trợ 1 webcam local thật, nên "Thêm
camera" bị disable kèm giải thích, thay vì giả vờ hỗ trợ nhiều camera."""

import streamlit as st

import detection_engine as engine
from theme import badge_html

engine.ensure_settings_defaults()

st.title("Quản lý camera")

st.button("➕ Thêm camera", disabled=True, help="Demo hiện chỉ hỗ trợ 1 webcam local (index 0).")
st.caption("Muốn dùng nhiều camera/IP camera cần bản mở rộng phần capture — hiện chưa hỗ trợ.")

st.markdown("---")

camera_online = st.session_state.get("camera_online", False)

with st.container(border=True):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Camera 1 — Webcam chính")
        st.caption("Nguồn: webcam local (index 0)")
    with c2:
        st.markdown(badge_html("safe", "Online") if camera_online else badge_html("neutral", "Offline"), unsafe_allow_html=True)

    st.markdown("")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Độ phân giải", st.session_state["camera_resolution"])
    with m2:
        st.metric("FPS mục tiêu", "~15-25")
    with m3:
        pipelines_on = sum(
            [st.session_state["enable_fall"], st.session_state["enable_hand"], st.session_state["enable_signal"]]
        )
        st.metric("AI pipeline đang bật", f"{pipelines_on}/3")

    st.markdown(
        f"- Phát hiện Ngã: {'✅ Bật' if st.session_state['enable_fall'] else '⛔ Tắt'}\n"
        f"- Cử chỉ tay: {'✅ Bật' if st.session_state['enable_hand'] else '⛔ Tắt'}\n"
        f"- Signal for Help (chuỗi, thử nghiệm): {'✅ Bật' if st.session_state['enable_signal'] else '⛔ Tắt'}"
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        st.page_link("views/live_view.py", label="Xem trực tiếp", icon="🎥")
    with b2:
        st.page_link("views/settings.py", label="Sửa cấu hình", icon="⚙️")
    with b3:
        st.button("Xoá camera", disabled=True, help="Camera duy nhất của hệ thống, không thể xoá trong demo.")
