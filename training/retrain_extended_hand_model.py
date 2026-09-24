"""Gộp dataset gốc (7 nhãn, Excel) với data mới bạn tự thu bằng
collect_hand_data.py (CSV, dùng tên nhãn) thành 1 dataset, rồi train model
DNN+LSTM với N nhãn (N = số dòng trong hand_labels.json), dùng landmark đã
chuẩn hoá (hand_features.normalize_hand_landmarks).

Chạy: python training/retrain_extended_hand_model.py
Cần: đã thu đủ data cho các nhãn mới bằng collect_hand_data.py trước khi
chạy — script này KHÔNG tự quay webcam.
"""

import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Concatenate, Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_features import HAND_LABELS_PATH, load_hand_labels, normalize_hand_landmarks

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")

ORIGINAL_DATASET_PATH = (
    r"D:\DeTaiAI\NCKH SV 2024\Code\Hand_Detection\Data_HandDetection"
    r"\Data_HandDetection\Hand_Landmarks_Dataset_2_1_2025.xlsx"
)
COLLECTED_DATA_GLOB = os.path.join(SCRIPT_DIR, "data", "*.csv")
OUTPUT_MODEL_PATH = os.path.join(MODELS_DIR, "HandLandMarks_Model_Extended.keras")
COMBINED_DATASET_OUT = os.path.join(SCRIPT_DIR, "data", "combined_raw_dataset.csv")

RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 200  # cảnh báo nếu 1 nhãn có quá ít mẫu so với các nhãn khác


def load_original_dataset(label_names):
    print(f"Đang đọc dataset gốc: {ORIGINAL_DATASET_PATH}")
    df = pd.read_excel(ORIGINAL_DATASET_PATH)
    # Label trong file gốc là số 0-6, đúng thứ tự 7 nhãn đầu của hand_labels.json.
    return df


def load_collected_datasets(label_names):
    name_to_idx = {name: i for i, name in enumerate(label_names)}
    frames = []
    csv_paths = sorted(glob.glob(COLLECTED_DATA_GLOB))
    for path in csv_paths:
        print(f"Đang đọc data mới: {path}")
        df = pd.read_csv(path)
        unknown = set(df["Label"].unique()) - set(name_to_idx.keys())
        if unknown:
            raise SystemExit(
                f"File {path} có nhãn chưa khai báo trong {HAND_LABELS_PATH}: {unknown}. "
                f"Thêm nhãn đó vào hand_labels.json trước."
            )
        df["Label"] = df["Label"].map(name_to_idx)
        frames.append(df)
    if not frames:
        print(f"Không thấy file CSV nào ở {COLLECTED_DATA_GLOB} — chỉ dùng dataset gốc (7 nhãn cũ).")
        return None
    return pd.concat(frames, ignore_index=True)


def build_model(num_classes):
    lstm_input = Input(shape=(21, 3))
    lstm_branch = LSTM(64, return_sequences=True)(lstm_input)
    lstm_branch = LSTM(64)(lstm_branch)
    lstm_branch = Dropout(0.5)(lstm_branch)

    dnn_input = Input(shape=(63,))
    dnn_branch = Dense(128, activation="relu")(dnn_input)
    dnn_branch = Dropout(0.5)(dnn_branch)
    dnn_branch = Dense(64, activation="relu")(dnn_branch)

    combined = Concatenate()([lstm_branch, dnn_branch])
    output = Dense(num_classes, activation="softmax")(combined)
    return Model(inputs=[lstm_input, dnn_input], outputs=output)


def main():
    label_names = load_hand_labels()
    print(f"Danh sách nhãn ({len(label_names)}): {label_names}")

    original_df = load_original_dataset(label_names)
    collected_df = load_collected_datasets(label_names)

    if collected_df is not None:
        combined_df = pd.concat([original_df, collected_df], ignore_index=True)
    else:
        combined_df = original_df

    os.makedirs(os.path.dirname(COMBINED_DATASET_OUT), exist_ok=True)
    combined_df.to_csv(COMBINED_DATASET_OUT, index=False)
    print(f"Đã lưu dataset gộp: {COMBINED_DATASET_OUT} ({len(combined_df)} dòng)")

    counts = combined_df["Label"].value_counts().sort_index()
    print("\nSố mẫu theo nhãn:")
    for idx, name in enumerate(label_names):
        n = int(counts.get(idx, 0))
        flag = "  <-- QUÁ ÍT, nên thu thêm" if 0 < n < MIN_SAMPLES_PER_CLASS else ""
        if n == 0:
            flag = "  <-- CHƯA CÓ DATA, model sẽ không nhận diện được nhãn này"
        print(f"  [{idx}] {name}: {n}{flag}")

    missing = [name for idx, name in enumerate(label_names) if counts.get(idx, 0) == 0]
    if missing:
        print(f"\nCẢNH BÁO: các nhãn chưa có data sẽ được loại khỏi lần train này: {missing}")
        keep_idx = [idx for idx, name in enumerate(label_names) if counts.get(idx, 0) > 0]
        combined_df = combined_df[combined_df["Label"].isin(keep_idx)].copy()
        # Re-map nhãn về liên tiếp 0..N-1 để train, nhưng vẫn in tên đúng theo index cũ.
        remap = {old: new for new, old in enumerate(keep_idx)}
        combined_df["Label"] = combined_df["Label"].map(remap)
        label_names = [label_names[i] for i in keep_idx]

    y = combined_df["Label"].values
    X_raw = combined_df.drop(columns=["Label"]).values

    print("\nĐang chuẩn hoá landmark...")
    X_norm = np.array([normalize_hand_landmarks(row) for row in X_raw])

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train_lstm = X_train.reshape((-1, 21, 3))
    X_test_lstm = X_test.reshape((-1, 21, 3))

    model = build_model(len(label_names))
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

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(OUTPUT_MODEL_PATH)
    print(f"\nĐã lưu model: {OUTPUT_MODEL_PATH}")

    loss, accuracy = model.evaluate([X_test_lstm, X_test], y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    y_pred = np.argmax(model.predict([X_test_lstm, X_test], verbose=0), axis=1)
    print(classification_report(y_test, y_pred, target_names=label_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
