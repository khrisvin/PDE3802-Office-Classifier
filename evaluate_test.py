import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from tensorflow import keras

from dataset import classes, collect_images
from prepare_data import load_images


def main():
    project_folder = Path(__file__).resolve().parent
    model_folder = project_folder / "outputs" / "pilot"
    output_folder = model_folder / "test_evaluation"

    model_path = model_folder / "office_classifier.keras"
    classes_path = model_folder / "classes.json"

    if output_folder.exists():
        raise FileExistsError(
            f"Test results already exist: {output_folder}\n"
            "Review the saved results instead of overwriting them."
        )

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with classes_path.open(encoding="utf-8") as file:
        saved_classes = json.load(file)

    if saved_classes != classes:
        raise ValueError("Saved class order does not match dataset.py.")

    images, actual_labels = load_images("test")
    image_paths, text_labels = collect_images("test")

    expected_labels = np.array(
        [classes.index(label) for label in text_labels],
        dtype=np.int32,
    )

    if len(image_paths) != len(images):
        raise ValueError("Image paths and loaded images have different lengths.")

    if not np.array_equal(actual_labels, expected_labels):
        raise ValueError("Image labels are not in the expected order.")

    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    model = keras.models.load_model(model_path, compile=False)
    probabilities = model.predict(images, batch_size=8, verbose=0)

    if probabilities.shape != (len(images), len(classes)):
        raise ValueError("Unexpected model prediction shape.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain invalid numerical values.")

    predicted_labels = np.argmax(probabilities, axis=1)
    class_ids = list(range(len(classes)))

    accuracy = accuracy_score(actual_labels, predicted_labels)

    macro_f1 = f1_score(
        actual_labels,
        predicted_labels,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=class_ids,
    )

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=class_ids,
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    mistakes = int(np.sum(actual_labels != predicted_labels))

    output_folder.mkdir(parents=True, exist_ok=False)

    print("Selected model: pilot baseline")
    print("Test images:", len(images))
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)
    print("Misclassified photos:")

    with (output_folder / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image",
            "actual",
            "predicted",
            "predicted_probability",
            "correct",
        ])

        for path, actual, predicted, probs in zip(
            image_paths,
            actual_labels,
            predicted_labels,
            probabilities,
        ):
            correct = bool(actual == predicted)

            writer.writerow([
                path.relative_to(project_folder).as_posix(),
                classes[actual],
                classes[predicted],
                float(probs[predicted]),
                correct,
            ])

            if not correct:
                print(
                    f"{path.name}: "
                    f"{classes[actual]} -> {classes[predicted]}"
                )

    if mistakes == 0:
        print("None")

    metrics = {
        "experiment": "pilot",
        "split": "test",
        "model_sha256": model_hash,
        "images": len(images),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "mistakes": mistakes,
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
    }

    with (output_folder / "test_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    (output_folder / "test_report.txt").write_text(
        report, encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    ax.set_title("Selected baseline: held-out test confusion matrix")
    fig.tight_layout()
    fig.savefig(output_folder / "test_confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"\nTotal mistakes: {mistakes}/{len(images)}")
    print("Test results saved to:", output_folder)


if __name__ == "__main__":
    main()