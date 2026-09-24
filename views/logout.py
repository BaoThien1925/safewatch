"""Trang Đăng xuất — demo không có hệ thống đăng nhập thật nên chỉ là màn
hình minh hoạ, không có tác dụng thật (không xoá session)."""

import streamlit as st

st.title("Đăng xuất")
st.info(
    "Demo này chưa có hệ thống đăng nhập/tài khoản thật, nên đây chỉ là giao diện minh hoạ. "
    "Đóng tab trình duyệt để kết thúc phiên làm việc."
)
st.page_link("views/dashboard.py", label="← Quay lại Dashboard", icon="📊")
