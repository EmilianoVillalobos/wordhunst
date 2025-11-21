import os
import json
from pathlib import Path

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# ===== CONFIG =====

DATASET_DIR = "dataset"     # root with A, B, C... folders
IMG_SIZE = 32               # image will be resized to IMG_SIZE x IMG_SIZE
TEST_SPLIT = 0.1            # 10% held out for validation
RANDOM_STATE = 42


def load_dataset(dataset_dir=DATASET_DIR, img_size=IMG_SIZE):
    dataset_path = Path(dataset_dir)
    assert dataset_path.is_dir(), f"{dataset_dir} is not a directory"

    X = []
    y = []
    labels = []

    # each subdirectory (A, B, C, ...) is a class
    for class_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
        label = class_dir.name.upper()
        if not (len(label) == 1 and "A" <= label <= "Z"):
            print(f"Skipping non-letter folder: {class_dir}")
            continue

        labels.append(label)
        print(f"Loading letter '{label}' from {class_dir}")

        for img_path in class_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARN] could not read {img_path}")
                continue

            # resize to fixed size
            img_resized = cv2.resize(img, (img_size, img_size))

            # normalize to [0,1]
            img_norm = img_resized.astype("float32") / 255.0

            X.append(img_norm)
            y.append(label)

    X = np.array(X)  # (N, H, W)
    X = X[..., np.newaxis]  # (N, H, W, 1)

    # map labels (letters) to integer indices
    unique_labels = sorted(list(set(y)))
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    y_idx = np.array([label_to_idx[label] for label in y], dtype=np.int32)

    print(f"\nLoaded {len(X)} images across {len(unique_labels)} classes.")
    return X, y_idx, label_to_idx, idx_to_label


def build_model(img_size=IMG_SIZE, num_classes=26):
    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def main():
    # 1) load dataset
    X, y, label_to_idx, idx_to_label = load_dataset()

    # 2) train / val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )

    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}")

    # 3) build model
    num_classes = len(label_to_idx)
    model = build_model(img_size=IMG_SIZE, num_classes=num_classes)

    # 4) train
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # 5) evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {val_acc:.4f}")

    # 6) save model + label map
    os.makedirs("models", exist_ok=True)
    model_path = "models/wordhunt_cnn.h5"
    labelmap_path = "models/label_map.json"

    model.save(model_path)
    with open(labelmap_path, "w") as f:
        json.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f)

    print(f"Saved model to {model_path}")
    print(f"Saved label map to {labelmap_path}")


if __name__ == "__main__":
    main()
