"""Train a frozen MobileNetV2 baseline on the cleaned Office-31 split."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf


from office31_data import (
    CLASSES, CLASS_TO_ID, DATA_DIR, IMAGE_SIZE, MANIFEST,
    load_split, read_manifest, resize_and_pad,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "office31_baseline"

SEED = 42
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001


def build_model():
    """Use ImageNet features and train only the six-class output layer."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="rgb_image",
    )
    x = tf.keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1,
        name="pixel_scaling",
    )(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="feature_average")(x)
    outputs = tf.keras.layers.Dense(
        len(CLASSES), activation="softmax", name="office_classes"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="office31_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Experiment already exists: {OUTPUT_DIR}\n"
            "Preserve it; use a different experiment name for another run."
        )

    tf.keras.utils.set_random_seed(SEED)

    rows = read_manifest()
    train_images, train_labels = load_split(rows, "train")
    val_images, val_labels = load_split(rows, "val")

    counts = Counter(int(label) for label in train_labels)
    class_weights = {
        index: len(train_labels) / (len(CLASSES) * counts[index])
        for index in range(len(CLASSES))
    }

    print(f"\nTraining images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print("Test images are not loaded.")
    print("\nTraining counts and class weights:")
    for index, name in enumerate(CLASSES):
        print(
            f"{name}: {counts[index]} images, "
            f"weight={class_weights[index]:.4f}"
        )

    model = build_model()
    OUTPUT_DIR.mkdir(parents=True)

    save_json(OUTPUT_DIR / "classes.json", CLASSES)
    save_json(
        OUTPUT_DIR / "training_config.json",
        {
            "seed": SEED,
            "epochs_limit": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "colour_order": "RGB",
            "padding": "black, aspect ratio preserved",
            "input_pixel_range": [0, 255],
            "model_scaling": "0..255 to -1..1",
            "backbone": "MobileNetV2",
            "pretrained_weights": "imagenet",
            "backbone_trainable": False,
            "augmentation": False,
            "class_weights": {
                CLASSES[index]: weight
                for index, weight in class_weights.items()
            },
            "selection_metric": "val_loss",
            "patience": 5,
            "split_manifest_sha256": hashlib.sha256(
                MANIFEST.read_bytes()
            ).hexdigest(),
        },
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUTPUT_DIR / "office_classifier.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.CSVLogger(str(OUTPUT_DIR / "training_log.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    save_json(OUTPUT_DIR / "history.json", history_data)

    if not all(
        np.isfinite(values).all() for values in history_data.values()
    ):
        raise RuntimeError(
            "Non-finite training metrics detected. Inspect the saved logs."
        )

    best_index = int(np.argmin(history_data["val_loss"]))
    print("\nTraining complete.")
    print(f"Epochs completed: {len(history_data['loss'])}")
    print(f"Best epoch by validation loss: {best_index + 1}")
    print(f"Best validation loss: {history_data['val_loss'][best_index]:.4f}")
    print(
        "Validation accuracy at that epoch: "
        f"{history_data['val_accuracy'][best_index] * 100:.2f}%"
    )
    print(f"Best model and experiment files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()