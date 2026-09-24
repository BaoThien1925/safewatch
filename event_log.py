"""Lưu lịch sử các lần cảnh báo SOS (Ngã đã leo thang, cử chỉ tay SOS, chuỗi
Signal for Help) vào SQLite — dùng chung giữa app.py (trang theo dõi trực
tiếp, ghi log) và pages/ (trang lịch sử, đọc log).

SQLite là lựa chọn nhẹ nhất cho demo/đồ án: 1 file, không cần cài server
riêng, có sẵn trong Python (module sqlite3).
"""

import os
import sqlite3
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "events.db")


def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT
            )
            """
        )


def log_event(event_type, details=""):
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO events (timestamp, event_type, details) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), event_type, details),
        )


def get_events(limit=200):
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT timestamp, event_type, details FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()


def count_events_by_type():
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT event_type, COUNT(*) FROM events GROUP BY event_type ORDER BY COUNT(*) DESC"
        )
        return cursor.fetchall()


def clear_events():
    with _get_connection() as conn:
        conn.execute("DELETE FROM events")


def prune_older_than(days):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with _get_connection() as conn:
        cursor = conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
        return cursor.rowcount
