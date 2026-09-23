"""So sánh model cũ (landmark thô) vs model mới (landmark đã chuẩn hoá) trên
cùng test set, dùng lại 2 model đã có sẵn (không train lại).

Chạy: python training/compare_models.py
"""

import io
import os
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import h5py
import numpy as np
import pandas as pd
from keras.layers import LSTM, Concatenate, Dense, Dropout, Input
from keras.models import Model, load_model
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
NEW_MODEL_PATH = os.path.join(MODELS_DIR, "HandLandMarks_Model_Normalized.keras")
HAND_LABELS = ["Like", "Dislike", "OK", "Neutral", "Need Ambulance", "Need Help", "Signal For Help"]
RANDOM_STATE = 42


def build_old_hand_model_with_weights():
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
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    print(f"Đang đọc dataset: {DATASET_PATH}")
    data = pd.read_excel(DATASET_PATH)
    y = data["Label"].values
    X_raw = data.drop(columns=["Label"]).values

    print("Đang chuẩn hoá landmark...")
    X_norm = np.array([normalize_hand_landmarks(row) for row in X_raw])

    _, X_test_norm, _, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    _, X_test_raw, _, y_test_raw = train_test_split(X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    assert np.array_equal(y_test, y_test_raw), "Test split không khớp"

    print("\n=== Model MỚI (landmark đã chuẩn hoá) ===")
    new_model = load_model(NEW_MODEL_PATH)
    X_test_norm_lstm = X_test_norm.reshape((-1, 21, 3))
    new_loss, new_acc = new_model.evaluate([X_test_norm_lstm, X_test_norm], y_test, verbose=0)
    y_pred_new = np.argmax(new_model.predict([X_test_norm_lstm, X_test_norm], verbose=0), axis=1)
    print(f"Test accuracy: {new_acc:.4f}")
    print(classification_report(y_test, y_pred_new, target_names=HAND_LABELS))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_new))

    print("\n=== Model CŨ (landmark thô, bản đang chạy trong Combined_Demo) ===")
    old_model = build_old_hand_model_with_weights()
    X_test_raw_lstm = X_test_raw.reshape((-1, 21, 3))
    old_loss, old_acc = old_model.evaluate([X_test_raw_lstm, X_test_raw], y_test_raw, verbose=0)
    y_pred_old = np.argmax(old_model.predict([X_test_raw_lstm, X_test_raw], verbose=0), axis=1)
    print(f"Test accuracy: {old_acc:.4f}")
    print(classification_report(y_test_raw, y_pred_old, target_names=HAND_LABELS))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_raw, y_pred_old))

    print("\n=== TỔNG KẾT ===")
    print(f"Model cũ (raw)        : {old_acc:.4f}")
    print(f"Model mới (normalized): {new_acc:.4f}")
    print(f"Chênh lệch            : {new_acc - old_acc:+.4f}")


if __name__ == "__main__":
    main()
