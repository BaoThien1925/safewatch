"""Lưu danh sách người liên hệ khẩn cấp — SQLite, CRUD thật (khác với mục
Camera/Tài khoản là mock tĩnh), vì đây là dữ liệu người dùng thực sự nhập
và cần giữ lại giữa các lần chạy app.
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "events.db")


def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                relationship TEXT,
                phone TEXT,
                email TEXT,
                priority INTEGER DEFAULT 3,
                notify_enabled INTEGER DEFAULT 1
            )
            """
        )


def add_contact(name, relationship, phone, email, priority):
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO contacts (name, relationship, phone, email, priority, notify_enabled) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (name, relationship, phone, email, priority),
        )


def get_contacts():
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT id, name, relationship, phone, email, priority, notify_enabled "
            "FROM contacts ORDER BY priority ASC, id ASC"
        )
        return cursor.fetchall()


def delete_contact(contact_id):
    with _get_connection() as conn:
        conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))


def set_notify_enabled(contact_id, enabled):
    with _get_connection() as conn:
        conn.execute("UPDATE contacts SET notify_enabled = ? WHERE id = ?", (1 if enabled else 0, contact_id))
