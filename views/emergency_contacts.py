"""Trang Người liên hệ khẩn cấp — CRUD thật (lưu SQLite qua contacts.py).
Không gửi thông báo thật (chưa có backend SMS/email trong demo)."""

import streamlit as st

import contacts

contacts.init_db()

st.title("Người liên hệ khẩn cấp")

with st.expander("➕ Thêm người liên hệ mới"):
    with st.form("add_contact_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Họ và tên")
            relationship = st.selectbox("Quan hệ", ["Người thân chính", "Người giám hộ", "Cơ sở y tế", "Khác"])
        with c2:
            phone = st.text_input("Số điện thoại")
            email = st.text_input("Email")
        priority = st.slider("Mức ưu tiên (1 = ưu tiên cao nhất)", 1, 5, 1)
        submitted = st.form_submit_button("Lưu người liên hệ")
        if submitted:
            if not name or not phone:
                st.error("Cần ít nhất Họ tên và Số điện thoại.")
            else:
                contacts.add_contact(name, relationship, phone, email, priority)
                st.success(f"Đã thêm {name}.")
                st.rerun()

st.markdown("---")

rows = contacts.get_contacts()
if not rows:
    st.info("Chưa có người liên hệ khẩn cấp nào. Thêm ít nhất 1 người ở trên.")
else:
    for contact_id, name, relationship, phone, email, priority, notify_enabled in rows:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{priority}. {name}**")
                st.caption(relationship or "—")
            with col2:
                st.caption(f"📞 {phone or '—'}")
                st.caption(f"✉️ {email or '—'}")
            with col3:
                new_enabled = st.checkbox("Nhận cảnh báo", value=bool(notify_enabled), key=f"notify_{contact_id}")
                if new_enabled != bool(notify_enabled):
                    contacts.set_notify_enabled(contact_id, new_enabled)
                    st.rerun()
                if st.button("Xoá", key=f"delete_{contact_id}"):
                    contacts.delete_contact(contact_id)
                    st.rerun()
