"""Evaluate the selected Office-31 baseline on the held-out test split."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from train_office31 import (
    CLASSES,
    CLASS_TO_ID,
    DATA_DIR,
    MANIFEST,
    read_manifest,
    resize_and_pad,
)


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "outputs" / "office31_baseline"
MODEL_PATH = EXPERIMENT / "office_classifier.keras"
RESULTS = EXPERIMENT / "test_evaluation"

EXPECTED_COUNTS = {
    "bottle": 3,
    "calculator": 14,
    "keyboard": 8,
    "mouse": 14,
    "mug": 10,
    "stapler": 12,
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if RESULTS.exists():
        raise FileExistsError(
            f"Results already exist: {RESULTS}\n"
            "Preserve the existing evaluation before running again."
        )

    required_files = [
        MODEL_PATH,
        EXPERIMENT / "classes.json",
        EXPERIMENT / "training_config.json",
        EXPERIMENT / "validation_evaluation" / "validation_metrics.json",
        MANIFEST,
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing file: {path}")

    saved_classes = json.loads(
        (EXPERIMENT / "classes.json").read_text(encoding="utf-8")
    )
    if saved_classes != CLASSES:
        raise ValueError("Saved class order differs from the training script.")

    config = json.loads(
        (EXPERIMENT / "training_config.json").read_text(encoding="utf-8")
    )
    validation_metrics = json.loads(
        required_files[3].read_text(encoding="utf-8")
    )

    manifest_hash = file_hash(MANIFEST)
    model_hash = file_hash(MODEL_PATH)

    if manifest_hash != config["split_manifest_sha256"]:
        raise ValueError("The split manifest has changed since training.")

    if manifest_hash != validation_metrics["split_manifest_sha256"]:
        raise ValueError("The split differs from the validation evaluation.")

    if model_hash != validation_metrics["model_sha256"]:
        raise ValueError("The model has changed since validation evaluation.")

    rows = read_manifest()
    test_rows = [row for row in rows if row["split"] == "test"]

    counts = Counter(row["class"] for row in test_rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected test counts: {dict(counts)}")

    images = []
    labels = []

    for row in test_rows:
        path = DATA_DIR / row["split_path"]

        if file_hash(path) != row["sha256"]:
            raise ValueError(f"Image changed since splitting: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")

        image = resize_and_pad(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        images.append(image)
        labels.append(CLASS_TO_ID[row["class"]])

    images = np.stack(images)
    labels = np.asarray(labels, dtype=np.int32)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scores = np.asarray(model.predict(images, batch_size=8, verbose=0))

    if scores.shape != (len(labels), len(CLASSES)):
        raise ValueError(f"Unexpected prediction shape: {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("Predictions contain non-finite values.")
    if (
        np.any(scores < 0)
        or np.any(scores > 1)
        or not np.allclose(scores.sum(axis=1), 1, atol=1e-4)
    ):
        raise ValueError("Expected softmax class scores.")

    predictions = np.argmax(scores, axis=1)
    class_ids = list(range(len(CLASSES)))

    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(
        labels,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        labels,
        predictions,
        labels=class_ids,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    mistakes = int(np.sum(labels != predictions))

    RESULTS.mkdir(parents=True)

    metrics = {
        "experiment": "office31_baseline",
        "split": "test",
        "images": len(labels),
        "correct": int(np.sum(labels == predictions)),
        "mistakes": mistakes,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classes": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "model_sha256": model_hash,
        "split_manifest_sha256": manifest_hash,
    }
    (RESULTS / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (RESULTS / "test_report.txt").write_text(
        f"Test accuracy: {accuracy:.2%}\n"
        f"Test macro F1: {macro_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )

    with (RESULTS / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_path",
            "group_id",
            "true_class",
            "predicted_class",
            "model_score",
            "correct",
        ])

        for row, true_id, predicted_id, probabilities in zip(
            test_rows, labels, predictions, scores
        ):
            writer.writerow([
                row["split_path"],
                row["group_id"],
                CLASSES[int(true_id)],
                CLASSES[int(predicted_id)],
                float(probabilities[predicted_id]),
                bool(true_id == predicted_id),
            ])

    figure, axis = plt.subplots(figsize=(8, 7))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASSES,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        xticks_rotation=45,
    )
    axis.set_title("Office-31 baseline — test")
    figure.tight_layout()
    figure.savefig(
        RESULTS / "test_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nSelected model: Office-31 baseline")
    print(f"Test images: {len(labels)}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"\nClassification report:\n{report}")
    print("Misclassified photos:")

    for row, true_id, predicted_id in zip(
        test_rows, labels, predictions
    ):
        if true_id != predicted_id:
            print(
                f"{row['split_path']}: "
                f"{CLASSES[int(true_id)]} -> {CLASSES[int(predicted_id)]}"
            )

    if mistakes == 0:
        print("None.")

    print(f"\nTotal mistakes: {mistakes}/{len(labels)}")
    print(f"Results saved to: {RESULTS}")
    print("The model was not retrained or modified.")


if __name__ == "__main__":
    main()