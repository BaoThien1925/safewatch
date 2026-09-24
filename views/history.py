"""Trang Lịch sử sự cố — bản restyle của pages/1_Lich_su_canh_bao.py cũ, có
thêm bộ lọc theo thời gian/loại sự cố."""

from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

import event_log
from theme import badge_html

st.title("Lịch sử sự cố")
event_log.init_db()


def bucket_and_status(event_type):
    if event_type.startswith("Ngã - đã phục hồi"):
        return "Ngã", "Đã phục hồi"
    if event_type.startswith("Ngã"):
        return "Ngã", "SOS đã kích hoạt"
    return "SOS tay", "SOS đã kích hoạt"


events = event_log.get_events(limit=1000)
rows = []
for ts, event_type, details in events:
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        continue
    bucket, status = bucket_and_status(event_type)
    rows.append({"dt": dt, "Thời gian": ts, "Loại sự cố": event_type, "Nhóm": bucket, "Chi tiết": details, "Trạng thái": status, "Camera": "Webcam chính"})

df = pd.DataFrame(rows)

# ----- KPI -----
total = len(df)
fall_count = int((df["Nhóm"] == "Ngã").sum()) if total else 0
sos_count = int((df["Nhóm"] == "SOS tay").sum()) if total else 0
recovered_count = int((df["Trạng thái"] == "Đã phục hồi").sum()) if total else 0

k1, k2, k3, k4 = st.columns(4)
for col, label, value in [
    (k1, "Tổng sự cố", total),
    (k2, "Phát hiện ngã", fall_count),
    (k3, "SOS tay", sos_count),
    (k4, "Đã phục hồi / huỷ", recovered_count),
]:
    with col:
        with st.container(border=True):
            st.markdown(f'<div class="sw-kpi-label">{label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sw-kpi-value">{value}</div>', unsafe_allow_html=True)

st.markdown("---")

# ----- Bộ lọc -----
f1, f2, f3 = st.columns(3)
with f1:
    days = st.selectbox("Khoảng thời gian", ["7 ngày", "30 ngày", "Tất cả"], index=0)
with f2:
    loai = st.selectbox("Loại sự cố", ["Tất cả", "Ngã", "SOS tay"], index=0)
with f3:
    trang_thai = st.selectbox("Trạng thái", ["Tất cả", "SOS đã kích hoạt", "Đã phục hồi"], index=0)

filtered = df.copy()
if not filtered.empty:
    if days == "7 ngày":
        filtered = filtered[filtered["dt"] >= datetime.now() - timedelta(days=7)]
    elif days == "30 ngày":
        filtered = filtered[filtered["dt"] >= datetime.now() - timedelta(days=30)]
    if loai != "Tất cả":
        filtered = filtered[filtered["Nhóm"] == loai]
    if trang_thai != "Tất cả":
        filtered = filtered[filtered["Trạng thái"] == trang_thai]

st.subheader(f"Chi tiết ({len(filtered)} kết quả)")
if filtered.empty:
    st.info("Không có sự cố nào khớp bộ lọc.")
else:
    display_df = filtered[["Thời gian", "Loại sự cố", "Camera", "Chi tiết", "Trạng thái"]].reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with st.expander("Xoá toàn bộ lịch sử"):
    confirm = st.checkbox("Tôi chắc chắn muốn xoá toàn bộ lịch sử (không thể hoàn tác)")
    if st.button("Xoá lịch sử", disabled=not confirm):
        event_log.clear_events()
        st.success("Đã xoá toàn bộ lịch sử.")
        st.rerun()
