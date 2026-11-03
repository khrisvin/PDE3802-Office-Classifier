"""Evaluate the saved Office-31 model on validation images only."""

import csv
import hashlib
import json
from pathlib import Path

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

from office31_data import (
    CLASSES,
    MANIFEST,
    load_split,
    read_manifest,
)


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "outputs" / "office31_baseline"
MODEL_PATH = EXPERIMENT / "office_classifier.keras"
RESULTS = EXPERIMENT / "validation_evaluation"


def main():
    if RESULTS.exists():
        raise FileExistsError(
            f"Results already exist: {RESULTS}\n"
            "Preserve the previous evaluation before running again."
        )

    for filename in (
        "office_classifier.keras",
        "classes.json",
        "training_config.json",
    ):
        if not (EXPERIMENT / filename).is_file():
            raise FileNotFoundError(f"Missing: {EXPERIMENT / filename}")

    saved_classes = json.loads(
        (EXPERIMENT / "classes.json").read_text(encoding="utf-8")
    )
    if saved_classes != CLASSES:
        raise ValueError("Saved class order differs from the training script.")

    config = json.loads(
        (EXPERIMENT / "training_config.json").read_text(encoding="utf-8")
    )
    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if manifest_hash != config["split_manifest_sha256"]:
        raise ValueError("The split manifest has changed since training.")

    rows = read_manifest()
    validation_rows = [row for row in rows if row["split"] == "val"]
    images, labels = load_split(rows, "val")

    if len(validation_rows) != len(labels):
        raise ValueError("Image and manifest counts do not match.")

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
        "split": "val",
        "images": len(labels),
        "correct": int(np.sum(labels == predictions)),
        "mistakes": mistakes,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classes": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "model_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        "split_manifest_sha256": manifest_hash,
    }

    (RESULTS / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (RESULTS / "validation_report.txt").write_text(
        f"Validation accuracy: {accuracy:.2%}\n"
        f"Validation macro F1: {macro_f1:.4f}\n\n"
        + report,
        encoding="utf-8",
    )

    with (RESULTS / "validation_predictions.csv").open(
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
            validation_rows, labels, predictions, scores
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
    axis.set_title("Office-31 baseline — validation")
    figure.tight_layout()
    figure.savefig(
        RESULTS / "validation_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"\nValidation images: {len(labels)}")
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Validation macro F1: {macro_f1:.4f}")
    print(f"\nClassification report:\n{report}")
    print("Misclassified photos:")

    for row, true_id, predicted_id in zip(
        validation_rows, labels, predictions
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
    print("Test images were not loaded.")


if __name__ == "__main__":
    main()