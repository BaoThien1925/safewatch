"""Trang Tài khoản của tôi — demo không có auth thật (không có đăng nhập/
đăng ký/nhiều user), nên đây là 1 hồ sơ tĩnh duy nhất lưu vào file JSON."""

import streamlit as st

import detection_engine as engine
import profile_store

engine.ensure_settings_defaults()

st.title("Tài khoản của tôi")

profile = profile_store.load_profile()

avatar_col, info_col = st.columns([1, 4])
with avatar_col:
    st.markdown(
        f"""
        <div style="width:72px;height:72px;border-radius:50%;background:#2563EB;color:white;
                    display:flex;align-items:center;justify-content:center;font-size:1.8rem;font-weight:700;">
            {profile['name'].strip()[0].upper() if profile['name'].strip() else '?'}
        </div>
        """,
        unsafe_allow_html=True,
    )
with info_col:
    st.markdown(f"### {profile['name']}")
    st.caption(profile["role"])

st.markdown("---")

with st.form("profile_form"):
    st.subheader("Thông tin cá nhân")
    c1, c2 = st.columns(2)
    with c1:
        name = st.text_input("Họ và tên", value=profile["name"])
        email = st.text_input("Email", value=profile["email"])
        phone = st.text_input("Số điện thoại", value=profile["phone"])
    with c2:
        birth_date = st.text_input("Ngày sinh", value=profile["birth_date"])
        address = st.text_input("Địa chỉ", value=profile["address"])
        role = st.selectbox(
            "Vai trò tài khoản",
            ["Người dùng cá nhân", "Người thân / Người giám hộ", "Quản trị viên"],
            index=["Người dùng cá nhân", "Người thân / Người giám hộ", "Quản trị viên"].index(profile["role"])
            if profile["role"] in ["Người dùng cá nhân", "Người thân / Người giám hộ", "Quản trị viên"]
            else 0,
        )

    st.subheader("Thông tin an toàn cá nhân")
    health_notes = st.text_area("Ghi chú sức khoẻ (bệnh nền, thuốc đang dùng...)", value=profile["health_notes"])
    watch_area = st.text_input("Khu vực giám sát chính", value=profile["watch_area"])

    if st.form_submit_button("💾 Lưu thay đổi"):
        profile_store.save_profile(
            {
                "name": name,
                "email": email,
                "phone": phone,
                "birth_date": birth_date,
                "address": address,
                "role": role,
                "health_notes": health_notes,
                "watch_area": watch_area,
            }
        )
        st.session_state["user_name"] = name
        st.success("Đã lưu hồ sơ.")
        st.rerun()

st.markdown("---")
st.subheader("Bảo mật")
st.caption("Demo này chưa có hệ thống đăng nhập/tài khoản thật, nên các mục dưới đây chỉ minh hoạ giao diện.")
st.button("Đổi mật khẩu", disabled=True)
st.button("Bật xác thực 2 bước", disabled=True)
st.button("Đăng xuất khỏi tất cả thiết bị", disabled=True)

st.markdown("---")
st.subheader("Cá nhân hoá")
st.selectbox("Giao diện", ["Light", "Dark (chưa hỗ trợ)"], disabled=True)
st.selectbox("Ngôn ngữ", ["Tiếng Việt"], disabled=True)
