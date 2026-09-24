"""Entry point mới của SafeWatch — multi-page qua st.navigation, thay cho
app.py (vẫn giữ nguyên, chạy độc lập như bản demo 1 trang đơn giản hơn).

Chạy: streamlit run main.py
"""

import streamlit as st

import profile_store
from theme import APP_NAME, inject_global_theme, render_sidebar_brand, render_sidebar_profile

st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide")
inject_global_theme()

if "user_name" not in st.session_state:
    st.session_state["user_name"] = profile_store.load_profile()["name"]

with st.sidebar:
    render_sidebar_brand()

# Dùng emoji thay vì cú pháp :material/...: — font icon Material Symbols
# không load được trên máy này (kể cả icon nội bộ của Streamlit cũng bị lỗi
# hiện chữ thô), emoji thì luôn hiện đúng vì không phụ thuộc font ngoài.
pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon="📊", default=True),
    st.Page("views/live_view.py", title="Xem trực tiếp", icon="🎥"),
    st.Page("views/history.py", title="Lịch sử sự cố", icon="📋"),
    st.Page("views/camera_management.py", title="Quản lý camera", icon="📷"),
    st.Page("views/emergency_contacts.py", title="Người liên hệ khẩn cấp", icon="📞"),
    st.Page("views/settings.py", title="Cấu hình", icon="⚙️"),
    st.Page("views/account.py", title="Tài khoản của tôi", icon="👤"),
    st.Page("views/logout.py", title="Đăng xuất", icon="🚪"),
]

nav = st.navigation(pages)

with st.sidebar:
    render_sidebar_profile(
        name=st.session_state.get("user_name", "Người dùng"),
        role=profile_store.load_profile()["role"],
        system_online=st.session_state.get("camera_online", False),
    )

nav.run()
