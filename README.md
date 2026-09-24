# SafeWatch — Giám sát an toàn cá nhân bằng AI (demo)

Demo hợp nhất: phát hiện **ngã quỵ** + **tín hiệu tay SOS** qua webcam.

## Yêu cầu

MediaPipe bản đầy đủ (API `solutions.Pose` / `solutions.Hands` mà code dùng) trên Windows chỉ có wheel cho **Python ≤ 3.12**. Máy dùng sẵn Python 3.12 (cài qua `winget install Python.Python.3.12`), venv `.venv` trong thư mục này được tạo bằng bản 3.12 đó — không dùng Python 3.13 mặc định của máy.

## Chạy demo

**Bản đầy đủ (multi-page, khuyến nghị):**

```bash
cd D:\DeTaiAI\combined-demo-dev
D:\DeTaiAI\Combined_Demo\.venv\Scripts\streamlit.exe run main.py
```

**Bản đơn giản, 1 trang (fallback, giữ nguyên từ trước):**

```bash
D:\DeTaiAI\Combined_Demo\.venv\Scripts\streamlit.exe run app.py
```

(Thư mục này chưa có `.venv` riêng — có thể mượn venv của `Combined_Demo` gốc như câu lệnh trên.)

Trình duyệt sẽ tự mở `http://localhost:8501`. Vào trang **Xem trực tiếp**, bấm "Bắt đầu giám sát" để bắt đầu.

## Cấu trúc

**Lõi nhận diện (dùng chung mọi giao diện):**
- `detection_engine.py` — model, hàm predict, hằng số, mapping nhãn kỹ thuật → văn bản thân thiện.
- `hand_features.py` — chuẩn hoá landmark tay + đọc danh sách nhãn từ `training/hand_labels.json`.
- `event_log.py` — lưu lịch sử cảnh báo (SQLite, `data/events.db`).
- `contacts.py` — lưu người liên hệ khẩn cấp (SQLite, cùng file DB).
- `profile_store.py` — lưu hồ sơ tài khoản (JSON, `data/profile.json`) — demo không có auth thật nên chỉ 1 hồ sơ.
- `theme.py` — màu/font/CSS + component dùng chung (banner trạng thái, badge, sidebar branding).

**Giao diện:**
- `main.py` — entry point mới, multi-page qua `st.navigation` (branding SafeWatch, sidebar).
- `views/` — từng trang: `dashboard.py`, `live_view.py`, `history.py`, `camera_management.py`, `emergency_contacts.py`, `settings.py`, `account.py`, `logout.py`.
- `app.py` — bản gốc 1 trang, vẫn chạy độc lập được (không dùng chung state với `main.py`).

**Model & training:** xem phần "Model" và `training/` bên dưới (không đổi so với trước).

## Cách hoạt động

3 pipeline độc lập, chạy song song (threading):

- **Ngã**: MediaPipe Pose (33 điểm) → sliding window 10 frame → LSTM → vote đa số 5 lần predict → `Fall`/`NotFall`.
- **Tay tĩnh (14 cử chỉ)**: MediaPipe Hands (21 điểm) → chuẩn hoá landmark → DNN+LSTM predict mỗi frame → vote đa số → 1 trong 14 nhãn. Có ngưỡng confidence, dưới ngưỡng hiện "Uncertain"/"Không rõ cử chỉ".
- **Signal for Help theo chuyển động** (thử nghiệm, tắt mặc định): LSTM nhị phân trên chuỗi 30 frame — còn hay báo sai vì chỉ có 60 mẫu train.

**Logic cảnh báo:**
- 3 cử chỉ tay SOS (Need Ambulance/Need Help/Signal For Help) → báo động ngay.
- Ngã → theo dõi im lặng (không báo ngay) trong khoảng thời gian cấu hình được (mặc định 120s). Đứng lên lại trong thời gian đó → huỷ, ghi log "đã phục hồi". Hết giờ vẫn chưa đứng lên → báo động thật + ghi log.
- Mọi cảnh báo được ghi vào `data/events.db`, xem lại ở trang Lịch sử sự cố / Dashboard.

## Model (không đổi so với các bản trước)

- `models/LSTM_model.h5` — phát hiện ngã (giữ nguyên từ bản gốc).
- `models/HandLandMarks_Model_Extended.keras` — model tay 14 nhãn (7 gốc + 7 tự thu: Fist, Peace Sign, Rock On, One Finger, Three Fingers, Call Me, Gun Sign), landmark đã chuẩn hoá, accuracy 99.54%. **Đang được dùng**.
- `models/HandLandMarks_Model_Normalized.keras` — model tay 7 nhãn gốc, chuẩn hoá landmark, accuracy 99.67% trên 7 nhãn đó — giữ lại để so sánh.
- `models/HandLandMarks_Model_300Epochs_new.keras` — model tay bản gốc nhất (raw landmark, 7 nhãn, accuracy 81.21%) — giữ lại để rollback/so sánh.
- `models/SignalForHelpSequenceModel.keras` — model nhị phân nhận diện chuỗi động tác Signal for Help (thử nghiệm).

`training/` chứa các script train/thu data tương ứng — xem comment đầu mỗi file.

## Giới hạn của bản demo (quan trọng khi đọc UI)

- **Không có auth/nhiều tài khoản thật** — trang Tài khoản chỉ là 1 hồ sơ tĩnh lưu JSON.
- **Chỉ hỗ trợ 1 webcam local** — trang Quản lý camera không thêm/xoá camera được, chỉ hiển thị 1 card.
- **Email/SMS/Push notification chưa có backend gửi thật** — các toggle ở Cấu hình > Cảnh báo chỉ là giao diện minh hoạ.
- **Lưu snapshot/video sự cố chưa hỗ trợ** — cần luồng ghi riêng, disable trong Cấu hình > Lưu trữ.
- **Dashboard không "live" thật song song với Live View** — Streamlit chỉ chạy 1 trang/lúc trong 1 session, nên khối trạng thái lớn ở Dashboard hiện đúng trạng thái ghi nhận lần cuối từ trang Xem trực tiếp, không phải cập nhật real-time khi đang ở trang khác.

## Chưa làm

- Đa tín hiệu thật (âm thanh phát hiện tiếng kêu cứu qua mic).
- Gửi thông báo thật (email/SMS/push).
- Ghi snapshot/video lúc xảy ra sự cố.
- Đa camera, đa người dùng thật (auth, phân quyền thật).
