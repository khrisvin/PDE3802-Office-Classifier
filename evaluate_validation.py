import csv
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
    output_folder = project_folder / "outputs" / "pilot"
    model_path = output_folder / "office_classifier.keras"

    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    with (output_folder / "classes.json").open(
        encoding="utf-8"
    ) as file:
        saved_classes = json.load(file)

    if saved_classes != classes:
        raise ValueError("Saved class order does not match dataset.py.")

    images, actual_labels = load_images("val")
    image_paths, _ = collect_images("val")

    model = keras.models.load_model(model_path, compile=False)
    probabilities = model.predict(images, batch_size=8, verbose=0)

    if probabilities.shape != (len(images), len(classes)):
        raise ValueError("Unexpected model prediction shape.")

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

    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Validation macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(report)

    mistakes = 0
    predictions_path = output_folder / "validation_predictions.csv"

    print("Misclassified photos:")

    with predictions_path.open(
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
                mistakes += 1
                print(
                    f"{path.name}: "
                    f"{classes[actual]} -> {classes[predicted]}"
                )

    if mistakes == 0:
        print("None")

    print(f"\nTotal mistakes: {mistakes}/{len(images)}")

    metrics = {
        "split": "val",
        "images": len(images),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "mistakes": mistakes,
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
    }

    with (output_folder / "validation_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=2)

    (output_folder / "validation_report.txt").write_text(
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
    ax.set_title("Pilot validation confusion matrix")
    fig.tight_layout()
    fig.savefig(
        output_folder / "validation_confusion_matrix.png",
        dpi=200,
    )
    plt.close(fig)

    print("Validation results saved to:", output_folder)


if __name__ == "__main__":
    main()