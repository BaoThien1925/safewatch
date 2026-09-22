# Demo hợp nhất: Ngã quỵ + Tín hiệu tay SOS

## Yêu cầu

MediaPipe bản đầy đủ (API `solutions.Pose` / `solutions.Hands` mà code dùng) trên Windows chỉ có wheel cho **Python ≤ 3.12**. Máy dùng sẵn Python 3.12 (cài qua `winget install Python.Python.3.12`), venv `.venv` trong thư mục này được tạo bằng bản 3.12 đó — không dùng Python 3.13 mặc định của máy.

## Chạy demo

```bash
cd D:\DeTaiAI\Combined_Demo
.venv\Scripts\activate
streamlit run app.py
```

Trình duyệt sẽ tự mở `http://localhost:8501`. Tích "Chạy webcam" ở thanh bên để bắt đầu.

## Cấu trúc

- `app.py` — giao diện Streamlit + pipeline nhận diện (dùng lại model có sẵn từ `NgaQuy/Code/LSTM_model.h5` và `NCKH SV 2024/Code/Hand_Detection/Model_HandDetection/HandLandMarks_Model_300Epochs_new.keras`, không cần train lại).
- `.venv/` — môi trường Python riêng cho demo này.

## Cách hoạt động

Mỗi frame webcam được đưa qua cả 2 pipeline độc lập:

- **Ngã**: MediaPipe Pose (33 điểm) → gom đủ 10 frame → LSTM → `Fall` / `NotFall`
- **Tay SOS**: MediaPipe Hands (21 điểm) → lấy frame mới nhất trong cửa sổ 10 frame → DNN+LSTM → 1 trong 7 cử chỉ

Cảnh báo SOS bật lên khi: `Fall == "Fall"` **hoặc** cử chỉ tay thuộc nhóm `Need Ambulance / Need Help / Signal For Help`.
