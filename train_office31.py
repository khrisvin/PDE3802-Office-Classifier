"""Train a frozen MobileNetV2 baseline on the cleaned Office-31 split."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "office31_split"
MANIFEST = DATA_DIR / "split_manifest.csv"
OUTPUT_DIR = ROOT / "outputs" / "office31_baseline"

CLASSES = ["bottle", "calculator", "keyboard", "mouse", "mug", "stapler"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}

IMAGE_SIZE = 224
SEED = 42
EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001


def resize_and_pad(image):
    """Preserve proportions and centre the image on a black square."""
    height, width = image.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)

    new_width = min(IMAGE_SIZE, max(1, round(width * scale)))
    new_height = min(IMAGE_SIZE, max(1, round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=interpolation
    )

    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    x = (IMAGE_SIZE - new_width) // 2
    y = (IMAGE_SIZE - new_height) // 2
    canvas[y:y + new_height, x:x + new_width] = resized
    return canvas


def read_manifest():
    """Check the recorded split assignments without loading test images."""
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"class", "split", "split_path", "sha256", "group_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Split manifest is missing required columns.")
        rows = list(reader)

    if not rows:
        raise ValueError("Split manifest is empty.")

    seen_paths = set()
    seen_hashes = set()
    group_assignments = {}

    for row in rows:
        if row["class"] not in CLASS_TO_ID:
            raise ValueError(f"Unknown class: {row['class']}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {row['split']}")

        relative = Path(row["split_path"])
        if (
            len(relative.parts) != 3
            or relative.parts[0] != row["split"]
            or relative.parts[1] != row["class"]
        ):
            raise ValueError(f"Invalid split path: {relative}")

        path = (DATA_DIR / relative).resolve()
        if not path.is_relative_to(DATA_DIR.resolve()):
            raise ValueError(f"Path outside dataset: {relative}")

        if row["split_path"] in seen_paths:
            raise ValueError(f"Repeated path: {relative}")
        seen_paths.add(row["split_path"])

        if row["sha256"] in seen_hashes:
            raise ValueError("Repeated image hash in split manifest.")
        seen_hashes.add(row["sha256"])

        group = row["group_id"].strip()
        if not group:
            raise ValueError(f"Missing group: {relative}")

        assignment = (row["class"], row["split"])
        if group in group_assignments and group_assignments[group] != assignment:
            raise ValueError(f"Group crosses classes or splits: {group}")
        group_assignments[group] = assignment

    return rows


def load_split(rows, split):
    """Load and verify training or validation images in manifest order."""
    if split not in {"train", "val"}:
        raise ValueError("This training script only loads train and val.")

    images = []
    labels = []

    for row in rows:
        if row["split"] != split:
            continue

        path = DATA_DIR / row["split_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Image changed since splitting: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")

        image = resize_and_pad(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        labels.append(CLASS_TO_ID[row["class"]])

    if not images:
        raise ValueError(f"No images found for {split}.")

    if set(labels) != set(range(len(CLASSES))):
        raise ValueError(f"Some classes are missing from {split}.")

    return np.stack(images), np.asarray(labels, dtype=np.int32)


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