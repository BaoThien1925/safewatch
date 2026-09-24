"""Trang Dashboard — tổng quan hệ thống. Lưu ý: Streamlit chỉ chạy 1 trang
tại 1 thời điểm trong 1 session, nên khối trạng thái lớn ở đây hiện đúng
trạng thái GHI NHẬN LẦN CUỐI từ trang Xem trực tiếp (session_state), không
phải trạng thái "sống" song song thật — vì demo chỉ chạy trên 1 máy/1 luồng.
"""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import detection_engine as engine
import event_log
from theme import badge_html, status_banner_html

engine.ensure_settings_defaults()
event_log.init_db()

st.title("Dashboard")

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown(f"### Xin chào, {st.session_state.get('user_name', 'Người dùng')} 👋")
    st.caption(datetime.now().strftime("%A, %d/%m/%Y — %H:%M"))
with header_col2:
    today_count = sum(
        1
        for ts, _, _ in event_log.get_events(limit=1000)
        if ts.startswith(datetime.now().strftime("%Y-%m-%d"))
    )
    st.markdown(f"🔔 **{today_count}** cảnh báo hôm nay")

st.markdown("---")

# ----- KPI -----
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
camera_online = st.session_state.get("camera_online", False)
last_status = st.session_state.get("last_status")

with kpi1:
    with st.container(border=True):
        st.markdown('<div class="sw-kpi-label">Tổng camera</div>', unsafe_allow_html=True)
        st.markdown('<div class="sw-kpi-value">1</div>', unsafe_allow_html=True)
with kpi2:
    with st.container(border=True):
        st.markdown('<div class="sw-kpi-label">Camera đang hoạt động</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sw-kpi-value">{1 if camera_online else 0}</div>', unsafe_allow_html=True)
with kpi3:
    with st.container(border=True):
        st.markdown('<div class="sw-kpi-label">Camera đang cảnh báo</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sw-kpi-value">{1 if last_status == "emergency" else 0}</div>', unsafe_allow_html=True)
with kpi4:
    with st.container(border=True):
        st.markdown('<div class="sw-kpi-label">Sự cố hôm nay</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sw-kpi-value">{today_count}</div>', unsafe_allow_html=True)

st.markdown("")

# ----- Trạng thái tổng thể -----
if last_status == "emergency":
    st.markdown(status_banner_html("emergency", "🚨 TÌNH HUỐNG KHẨN CẤP", "SOS đã được kích hoạt"), unsafe_allow_html=True)
elif last_status == "monitoring":
    st.markdown(status_banner_html("monitoring", "⏳ ĐANG THEO DÕI SAU NGÃ", "Đang xác minh tình trạng người dùng"), unsafe_allow_html=True)
elif last_status == "safe":
    st.markdown(status_banner_html("safe", "AN TOÀN", "Không có cảnh báo nào đang hoạt động"), unsafe_allow_html=True)
else:
    st.markdown(status_banner_html("safe", "CHƯA GIÁM SÁT", "Vào trang Xem trực tiếp để bắt đầu"), unsafe_allow_html=True)

st.caption("Trạng thái hiện theo lần ghi nhận cuối từ trang Xem trực tiếp (demo chạy trên 1 luồng, không đa nhiệm thật).")

st.markdown("---")

# ----- Danh sách camera (mock: chỉ có 1 webcam thật) -----
st.subheader("Camera")
cam_col, _ = st.columns([1, 3])
with cam_col:
    with st.container(border=True):
        st.markdown("**Camera 1 – Webcam chính**")
        badge = badge_html("safe", "Online") if camera_online else badge_html("neutral", "Offline")
        st.markdown(badge, unsafe_allow_html=True)
        st.caption("640x480 · MediaPipe Pose + Hands")
        st.page_link("views/live_view.py", label="Xem trực tiếp →", icon="🎥")

st.markdown("---")

# ----- Biểu đồ 7 ngày -----
st.subheader("Sự cố trong 7 ngày gần nhất")

events = event_log.get_events(limit=2000)


def bucket_type(event_type):
    if event_type.startswith("Ngã - đã phục hồi"):
        return "Recovered / Cancelled"
    if event_type.startswith("Ngã"):
        return "Fall"
    return "SOS Gesture"


rows = []
for ts, event_type, _ in events:
    try:
        date = datetime.fromisoformat(ts).date()
    except ValueError:
        continue
    rows.append({"date": date, "bucket": bucket_type(event_type)})

today = datetime.now().date()
date_range = [today - timedelta(days=i) for i in range(6, -1, -1)]

if rows:
    df = pd.DataFrame(rows)
    pivot = df.groupby(["date", "bucket"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(date_range, fill_value=0)
    for col in ["Fall", "SOS Gesture", "Recovered / Cancelled"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot.index = [d.strftime("%d/%m") for d in pivot.index]
    st.bar_chart(pivot[["Fall", "SOS Gesture", "Recovered / Cancelled"]])
else:
    st.info("Chưa có sự cố nào được ghi lại trong 7 ngày gần đây.")
