"""Design tokens + CSS dùng chung cho toàn bộ app SafeWatch — 1 nơi duy nhất
định nghĩa màu/font/style để các trang không bị lệch nhau.
"""

import streamlit as st

COLOR_SAFE = "#16A34A"
COLOR_MONITORING = "#F97316"
COLOR_EMERGENCY = "#DC2626"
COLOR_NAVY = "#0F172A"
COLOR_BLUE = "#2563EB"
COLOR_BG = "#F8FAFC"
COLOR_CARD = "#FFFFFF"
COLOR_BORDER = "#E2E8F0"
COLOR_TEXT_MUTED = "#64748B"

APP_NAME = "SafeWatch"
APP_TAGLINE = "Giám sát an toàn cá nhân bằng AI"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Chỉ áp font ở body/root, KHÔNG dùng selector rộng kiểu [class*="css"] —
nó đè luôn cả font ligature riêng mà Streamlit dùng để hiện icon Material
(:material/...:), làm icon hiện ra thành chữ thô đè lên label. Font ở đây
chỉ ảnh hưởng qua inheritance, phần tử nào tự khai font riêng (icon) vẫn
giữ đúng font của nó. */
html, body, .stApp {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    /* Cỡ chữ gốc toàn app — Streamlit dùng rem cho hầu hết label/caption/
    checkbox nên tăng cỡ gốc này sẽ kéo to theo tỉ lệ toàn bộ giao diện,
    không chỉ riêng sidebar. Mặc định trình duyệt/Streamlit ~14-16px, nhiều
    người thấy hơi nhỏ (đặc biệt lúc demo/chiếu), nên nâng lên rõ rệt. */
    font-size: 16px;
}}

.stApp {{
    background-color: {COLOR_BG};
}}

/* Chữ sidebar (tên nav + brand + profile) — to hơn cỡ chữ gốc 1 chút, vì
đây là phần điều hướng chính, nhưng không quá to để tránh vỡ dòng/tràn chữ
trong sidebar (đã bị ở mức 1.3rem trước đó). */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] span,
[data-testid="stSidebarNavItems"] a,
[data-testid="stSidebarNavItems"] span {{
    font-size: 1.05rem !important;
}}
.sw-brand-name {{ font-size: 1.1rem !important; }}
.sw-brand-tagline {{ font-size: 0.78rem !important; }}

/* Sidebar branding */
[data-testid="stSidebarHeader"] {{
    padding-bottom: 0;
}}

.sw-brand {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0 1rem 0;
}}
.sw-brand-mark {{
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: {COLOR_NAVY};
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 15px;
}}
.sw-brand-name {{
    font-weight: 800;
    font-size: 1.05rem;
    color: {COLOR_NAVY};
    line-height: 1.1;
}}
.sw-brand-tagline {{
    font-size: 0.7rem;
    color: {COLOR_TEXT_MUTED};
}}

/* Generic card */
.sw-card {{
    background: {COLOR_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}}

/* Big status banner */
.sw-status-banner {{
    border-radius: 16px;
    padding: 1.75rem 2rem;
    text-align: center;
    color: white;
}}
.sw-status-banner .sw-status-label {{
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: 0.02em;
}}
.sw-status-banner .sw-status-sub {{
    font-size: 0.9rem;
    opacity: 0.92;
    margin-top: 0.35rem;
}}
.sw-status-safe {{ background: linear-gradient(135deg, #16A34A, #15803D); }}
.sw-status-monitoring {{ background: linear-gradient(135deg, #F97316, #C2410C); }}
.sw-status-emergency {{ background: linear-gradient(135deg, #DC2626, #991B1B); }}

/* Small status badge (pill) */
.sw-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}}
.sw-badge-safe {{ background: #DCFCE7; color: #166534; }}
.sw-badge-monitoring {{ background: #FFEDD5; color: #9A3412; }}
.sw-badge-emergency {{ background: #FEE2E2; color: #991B1B; }}
.sw-badge-neutral {{ background: #F1F5F9; color: {COLOR_TEXT_MUTED}; }}
.sw-badge-live {{ background: {COLOR_EMERGENCY}; color: white; }}

/* KPI card */
.sw-kpi-value {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {COLOR_NAVY};
}}
.sw-kpi-label {{
    font-size: 0.8rem;
    color: {COLOR_TEXT_MUTED};
    font-weight: 500;
}}

hr {{ margin: 0.5rem 0; }}
</style>
"""


def inject_global_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar_brand():
    st.markdown(
        f"""
        <div class="sw-brand">
            <div class="sw-brand-mark">SW</div>
            <div>
                <div class="sw-brand-name">{APP_NAME}</div>
                <div class="sw-brand-tagline">{APP_TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_profile(name="Nguyễn Văn A", role="Người dùng cá nhân"):
    """Chỉ render avatar/tên/role — KHÔNG kèm trạng thái hệ thống nữa.

    Trạng thái online/offline cần cập nhật LIÊN TỤC trong lúc trang Xem
    trực tiếp đang chạy vòng lặp webcam, nhưng main.py chỉ chạy phần này
    ĐÚNG 1 LẦN trước khi vào nav.run() — không có cơ hội cập nhật lại
    trong suốt lúc vòng lặp còn sống, nên trước đây hiện sai (báo "Hệ
    thống ngưng" dù camera đang live). Trạng thái đúng giờ render bằng
    sidebar_status_placeholder(), gọi và cập nhật trực tiếp TỪ BÊN TRONG
    views/live_view.py (cùng 1 script đang chạy liên tục, giống cách
    frame_placeholder/status_banner_placeholder đã cập nhật đúng).
    """
    st.markdown("---")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:0.6rem;">
            <div style="width:40px;height:40px;border-radius:50%;background:{COLOR_BLUE};
                        color:white;display:flex;align-items:center;justify-content:center;
                        font-weight:700;font-size:1rem;">
                {name.strip()[0].upper() if name.strip() else "?"}
            </div>
            <div>
                <div style="font-weight:600;font-size:1rem;color:{COLOR_NAVY};">{name}</div>
                <div style="font-size:0.85rem;color:{COLOR_TEXT_MUTED};">{role}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_status_html(system_online):
    status_dot = "🟢" if system_online else "🔴"
    status_text = "Hệ thống hoạt động" if system_online else "Hệ thống ngưng"
    return (
        f'<div style="font-size:0.85rem;color:{COLOR_TEXT_MUTED};">{status_dot} {status_text}</div>'
    )


def status_banner_html(level, title, subtitle=""):
    """level: 'safe' | 'monitoring' | 'emergency'"""
    css_class = {"safe": "sw-status-safe", "monitoring": "sw-status-monitoring", "emergency": "sw-status-emergency"}[level]
    return f"""
    <div class="sw-status-banner {css_class}">
        <div class="sw-status-label">{title}</div>
        <div class="sw-status-sub">{subtitle}</div>
    </div>
    """


def badge_html(level, text):
    css_class = {
        "safe": "sw-badge-safe",
        "monitoring": "sw-badge-monitoring",
        "emergency": "sw-badge-emergency",
        "neutral": "sw-badge-neutral",
        "live": "sw-badge-live",
    }[level]
    return f'<span class="sw-badge {css_class}">{text}</span>'
