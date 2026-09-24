"""Lưu hồ sơ người dùng — demo không có hệ thống auth/nhiều tài khoản thật,
nên chỉ lưu 1 hồ sơ duy nhất vào file JSON (không phải database nhiều user)."""

import json
import os

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "profile.json")

DEFAULT_PROFILE = {
    "name": "Nguyễn Văn A",
    "email": "vana@example.com",
    "phone": "090xxxxxxx",
    "birth_date": "01/01/1990",
    "address": "",
    "role": "Người dùng cá nhân",
    "health_notes": "",
    "watch_area": "Phòng khách",
}


def load_profile():
    if not os.path.exists(PROFILE_PATH):
        return dict(DEFAULT_PROFILE)
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = dict(DEFAULT_PROFILE)
    merged.update(data)
    return merged


def save_profile(profile):
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
