# Demo hợp nhất: Ngã quỵ + Tín hiệu tay SOS

## Yêu cầu

MediaPipe bản đầy đủ (API `solutions.Pose` / `solutions.Hands` mà code dùng) trên Windows chỉ có wheel cho **Python ≤ 3.12**. Máy dùng sẵn Python 3.12 (cài qua `winget install Python.Python.3.12`), venv `.venv` trong thư mục này được tạo bằng bản 3.12 đó — không dùng Python 3.13 mặc định của máy.

## Chạy demo

```bash
cd D:\DeTaiAI\combined-demo-dev
streamlit run app.py
```

(Thư mục này chưa có `.venv` riêng — có thể mượn venv của `Combined_Demo` gốc, ví dụ chạy trực tiếp bằng `D:\DeTaiAI\Combined_Demo\.venv\Scripts\streamlit.exe run app.py`.)

Trình duyệt sẽ tự mở `http://localhost:8501`. Tích "Chạy webcam" ở thanh bên để bắt đầu.

## Cấu trúc

- `app.py` — giao diện Streamlit + pipeline nhận diện.
- `hand_features.py` — hàm chuẩn hoá landmark tay, dùng chung giữa `app.py` và script train, bắt buộc phải khớp nhau.
- `models/` — model đã train sẵn:
  - `LSTM_model.h5` — phát hiện ngã (giữ nguyên từ bản gốc).
  - `HandLandMarks_Model_Normalized.keras` — model tay đã train lại với landmark chuẩn hoá, **đang được `app.py` dùng**. Accuracy 99.67% trên test set so với 81.21% của model cũ (xem `training/compare_models.py`), đặc biệt sửa lỗi nhận nhầm Need Ambulance/Need Help thành OK/Neutral.
  - `HandLandMarks_Model_300Epochs_new.keras` — model tay bản gốc (raw landmark, không chuẩn hoá), giữ lại để so sánh/rollback nếu cần.
- `training/` — script train lại và so sánh model tay:
  - `retrain_hand_model.py` — train model mới từ dataset gốc + chuẩn hoá landmark.
  - `compare_models.py` — so sánh model cũ vs model mới trên cùng test set, không cần train lại.
- `.venv/` — nếu tự tạo, môi trường Python riêng cho demo này (không commit lên git).

## Cách hoạt động

Mỗi frame webcam được đưa qua cả 2 pipeline độc lập:

- **Ngã**: MediaPipe Pose (33 điểm) → sliding window 10 frame gần nhất → LSTM → vote đa số trong 5 lần predict gần nhất → `Fall` / `NotFall`.
- **Tay SOS**: MediaPipe Hands (21 điểm) → chuẩn hoá landmark (`hand_features.normalize_hand_landmarks`) → DNN+LSTM predict ngay mỗi frame có tay → vote đa số trong 5 lần predict gần nhất → 1 trong 7 cử chỉ.

Cảnh báo SOS bật lên khi: `Fall == "Fall"` **hoặc** cử chỉ tay thuộc nhóm `Need Ambulance / Need Help / Signal For Help`, và giữ tối thiểu 2 giây để tránh nhấp nháy.

## Đã cải tiến so với bản gốc (`Combined_Demo`)

1. Bỏ delay ~10 frame vô nghĩa của model tay (trước đây gom 10 frame chỉ để dùng frame cuối).
2. Sliding window cho model ngã thay vì gom khối rời rạc.
3. Vote đa số (debounce) cho cả 2 pipeline, giảm nhận sai do 1 frame nhiễu.
4. Giữ cảnh báo SOS tối thiểu 2 giây, tránh nhấp nháy.
5. Chuẩn hoá landmark tay + train lại model → accuracy 81.21% → 99.67%.

## Chưa làm (cần thu data mới + bạn tự thực hiện)

- Thêm cử chỉ SOS mới ngoài 7 nhãn hiện có.
- Đa tín hiệu (âm thanh, bất động sau ngã, nút vật lý dự phòng).
- Chạy song song pose/hands detection để tăng FPS.
