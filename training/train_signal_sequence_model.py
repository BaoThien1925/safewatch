"""Train model nhị phân nhận diện CHUỖI động tác Signal for Help, tách biệt
hoàn toàn khỏi model 14 nhãn tĩnh (không đụng, không ảnh hưởng model đó).

Input: chuỗi SIGNAL_SEQUENCE_LENGTH frame landmark tay (đã chuẩn hoá từng
frame bằng hand_features.normalize_hand_landmarks).
Output: 1 xác suất "đúng là chuỗi Signal for Help" (sigmoid).

Cần chạy training/collect_hand_sequence.py trước để có data (cả 2 nhãn
'signal' và 'negative') trong training/data/signal_sequences.csv.

Chạy: python training/train_signal_sequence_model.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hand_features import SIGNAL_SEQUENCE_LENGTH, normalize_hand_sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "signal_sequences.csv")
MODELS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")
OUTPUT_MODEL_PATH = os.path.join(MODELS_DIR, "SignalForHelpSequenceModel.keras")
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 20


def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"Chưa có {DATA_PATH}. Chạy training/collect_hand_sequence.py trước "
            f"để thu data cho cả 2 nhãn 'signal' và 'negative'."
        )
    df = pd.read_csv(DATA_PATH)

    counts = df["Label"].value_counts()
    print("Số mẫu theo nhãn:")
    for label in ["signal", "negative"]:
        n = int(counts.get(label, 0))
        print(f"  {label}: {n}")
        if n < MIN_SAMPLES_PER_CLASS:
            print(f"  CẢNH BÁO: '{label}' chỉ có {n} mẫu, nên thu thêm (khuyến nghị >= {MIN_SAMPLES_PER_CLASS}).")
    if "signal" not in counts or "negative" not in counts:
        raise SystemExit("Cần cả 2 nhãn 'signal' và 'negative' để train model nhị phân.")

    y = (df["Label"] == "signal").astype(int).values
    X_flat = df.drop(columns=["Label"]).values  # (N, SIGNAL_SEQUENCE_LENGTH * 63)
    X_sequences = X_flat.reshape((-1, SIGNAL_SEQUENCE_LENGTH, 63))
    return X_sequences, y


def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(SIGNAL_SEQUENCE_LENGTH, 63)))
    model.add(Dropout(0.3))
    model.add(LSTM(50))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="sigmoid"))
    return model


def main():
    X_sequences, y = load_dataset()

    print("Đang chuẩn hoá landmark từng frame trong mỗi chuỗi...")
    X_norm = np.array([normalize_hand_sequence(seq) for seq in X_sequences])

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=8,
        callbacks=[early_stopping],
        verbose=2,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(OUTPUT_MODEL_PATH)
    print(f"\nĐã lưu model: {OUTPUT_MODEL_PATH}")

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).reshape(-1)
    print(classification_report(y_test, y_pred, target_names=["negative", "signal"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
