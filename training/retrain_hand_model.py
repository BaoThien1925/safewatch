"""Train lại model nhận diện cử chỉ tay với landmark đã chuẩn hoá
(hand_features.normalize_hand_landmarks), dùng lại đúng dataset gốc
(Hand_Landmarks_Dataset_2_1_2025.xlsx, 7 class, 6000 mẫu/class).

Kiến trúc DNN+LSTM giữ nguyên như bản gốc (Train_DNN-LSTM.py) — chỉ đổi
bước tiền xử lý đầu vào từ tọa độ thô sang tọa độ đã chuẩn hoá, để so sánh
công bằng xem chuẩn hoá có thực sự cải thiện độ chính xác/độ ổn định không.

Chạy: python training/retrain_hand_model.py
"""

import io
import os
import sys
import zipfile

# Console Windows mặc định dùng cp1252, không encode được ký tự tiếng Việt có
# dấu -> crash khi print. Ép stdout/stderr sang UTF-8 trước khi print bất cứ
# gì.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import h5py
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Concatenate, Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_features import normalize_hand_landmarks

DATASET_PATH = (
    r"D:\DeTaiAI\NCKH SV 2024\Code\Hand_Detection\Data_HandDetection"
    r"\Data_HandDetection\Hand_Landmarks_Dataset_2_1_2025.xlsx"
)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
OLD_MODEL_PATH = os.path.join(MODELS_DIR, "HandLandMarks_Model_300Epochs_new.keras")
OUTPUT_MODEL_PATH = os.path.join(MODELS_DIR, "HandLandMarks_Model_Normalized.keras")
HAND_LABELS = ["Like", "Dislike", "OK", "Neutral", "Need Ambulance", "Need Help", "Signal For Help"]
RANDOM_STATE = 42


def build_old_hand_model_with_weights():
    """Dựng đúng kiến trúc model cũ (named layers) và load trọng số gốc —
    giống app.py, để đánh giá model cũ trên cùng test set cho công bằng."""
    lstm_input = Input(shape=(21, 3))
    lstm_branch = LSTM(64, return_sequences=True, name="lstm")(lstm_input)
    lstm_branch = LSTM(64, name="lstm_1")(lstm_branch)
    lstm_branch = Dropout(0.5)(lstm_branch)

    dnn_input = Input(shape=(63,))
    dnn_branch = Dense(128, activation="relu", name="dense")(dnn_input)
    dnn_branch = Dropout(0.5)(dnn_branch)
    dnn_branch = Dense(64, activation="relu", name="dense_1")(dnn_branch)

    combined = Concatenate()([lstm_branch, dnn_branch])
    output = Dense(7, activation="softmax", name="dense_2")(combined)
    model = Model(inputs=[lstm_input, dnn_input], outputs=output)

    with zipfile.ZipFile(OLD_MODEL_PATH) as z:
        weights_h5 = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")

    def get(name, idx):
        return np.array(weights_h5["_layer_checkpoint_dependencies\\" + name]["vars"][str(idx)])

    model.get_layer("lstm").set_weights([get("lstm\\cell", 0), get("lstm\\cell", 1), get("lstm\\cell", 2)])
    model.get_layer("lstm_1").set_weights([get("lstm_2\\cell", 0), get("lstm_2\\cell", 1), get("lstm_2\\cell", 2)])
    model.get_layer("dense").set_weights([get("dense", 0), get("dense", 1)])
    model.get_layer("dense_1").set_weights([get("dense_2", 0), get("dense_2", 1)])
    model.get_layer("dense_2").set_weights([get("dense_4", 0), get("dense_4", 1)])
    weights_h5.close()
    return model


def load_normalized_dataset():
    print(f"Đang đọc dataset: {DATASET_PATH}")
    data = pd.read_excel(DATASET_PATH)

    y = data["Label"].values
    X_raw = data.drop(columns=["Label"]).values  # (N, 63)

    print("Đang chuẩn hoá landmark...")
    X_norm = np.array([normalize_hand_landmarks(row) for row in X_raw])
    return X_raw, X_norm, y


def build_hand_model():
    lstm_input = Input(shape=(21, 3))
    lstm_branch = LSTM(64, return_sequences=True)(lstm_input)
    lstm_branch = LSTM(64)(lstm_branch)
    lstm_branch = Dropout(0.5)(lstm_branch)

    dnn_input = Input(shape=(63,))
    dnn_branch = Dense(128, activation="relu")(dnn_input)
    dnn_branch = Dropout(0.5)(dnn_branch)
    dnn_branch = Dense(64, activation="relu")(dnn_branch)

    combined = Concatenate()([lstm_branch, dnn_branch])
    output = Dense(7, activation="softmax")(combined)
    return Model(inputs=[lstm_input, dnn_input], outputs=output)


def main():
    X_raw, X_norm, y = load_normalized_dataset()

    # Cùng random_state + stratify trên cả 2 lần split (chỉ phụ thuộc y, không
    # phụ thuộc X) nên chỉ số test set của bản raw và bản normalized khớp nhau
    # -> so sánh model cũ (raw) với model mới (normalized) trên đúng cùng mẫu.
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    _, X_test_raw, _, y_test_raw = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    assert np.array_equal(y_test, y_test_raw), "Test split không khớp giữa raw và normalized"

    X_train_lstm = X_train.reshape((-1, 21, 3))
    X_test_lstm = X_test.reshape((-1, 21, 3))

    model = build_hand_model()
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        [X_train_lstm, X_train],
        y_train,
        validation_data=([X_test_lstm, X_test], y_test),
        epochs=300,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=2,
    )

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    model.save(OUTPUT_MODEL_PATH)
    print(f"\nĐã lưu model mới: {OUTPUT_MODEL_PATH}")

    loss, accuracy = model.evaluate([X_test_lstm, X_test], y_test, verbose=0)
    print("\n=== Model MỚI (landmark đã chuẩn hoá) ===")
    print(f"Test accuracy: {accuracy:.4f}")
    y_pred_new = np.argmax(model.predict([X_test_lstm, X_test], verbose=0), axis=1)
    print(classification_report(y_test, y_pred_new, target_names=HAND_LABELS))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_new))

    print("\nĐang đánh giá model CŨ (landmark thô) trên cùng test set để so sánh...")
    old_model = build_old_hand_model_with_weights()
    X_test_raw_lstm = X_test_raw.reshape((-1, 21, 3))
    old_loss, old_accuracy = old_model.evaluate([X_test_raw_lstm, X_test_raw], y_test_raw, verbose=0)
    print("\n=== Model CŨ (landmark thô, bản đang chạy trong Combined_Demo) ===")
    print(f"Test accuracy: {old_accuracy:.4f}")
    y_pred_old = np.argmax(old_model.predict([X_test_raw_lstm, X_test_raw], verbose=0), axis=1)
    print(classification_report(y_test_raw, y_pred_old, target_names=HAND_LABELS))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_raw, y_pred_old))

    print("\n=== TỔNG KẾT ===")
    print(f"Model cũ (raw)        : {old_accuracy:.4f}")
    print(f"Model mới (normalized): {accuracy:.4f}")
    print(f"Chênh lệch            : {accuracy - old_accuracy:+.4f}")


if __name__ == "__main__":
    main()
