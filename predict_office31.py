"""Predict the class of one image using the saved Office-31 model."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from office31_data import CLASSES, resize_and_pad


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "outputs" / "office31_baseline"
MODEL_PATH = EXPERIMENT_DIR / "office_classifier.keras"
CLASSES_PATH = EXPERIMENT_DIR / "classes.json"


def load_classes():
    if not CLASSES_PATH.is_file():
        raise FileNotFoundError(
            f"Class mapping was not found: {CLASSES_PATH}"
        )

    with CLASSES_PATH.open("r", encoding="utf-8") as file:
        classes = json.load(file)

    if classes != list(CLASSES):
        raise ValueError(
            "The saved class order does not match the Office-31 "
            f"training class order.\n"
            f"Saved: {classes}\n"
            f"Expected: {list(CLASSES)}"
        )

    return classes


def load_image(image_path):
    if not image_path.is_file():
        raise FileNotFoundError(f"Image was not found: {image_path}")

    # imdecode supports Windows paths containing non-ASCII characters.
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    if image_bytes.size == 0:
        raise ValueError(f"The image file is empty: {image_path}")

    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read: {image_path}")

    # Match training: preserve proportions, pad, then convert BGR to RGB.
    padded_bgr = resize_and_pad(image_bgr)
    image_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)

    if image_rgb.shape != (224, 224, 3):
        raise ValueError(
            f"Unexpected image shape: {image_rgb.shape}; "
            "expected (224, 224, 3)."
        )

    # Scaling is already inside the saved model.
    return np.expand_dims(image_rgb, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Classify one image using the Office-31 baseline."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the image to classify.",
    )
    args = parser.parse_args()

    try:
        image_path = args.image.expanduser().resolve()

        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Saved model was not found: {MODEL_PATH}"
            )

        classes = load_classes()
        image_batch = load_image(image_path)

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
        )

        predictions = np.asarray(
            model.predict(image_batch, verbose=0)
        )

        if predictions.shape != (1, len(classes)):
            raise ValueError(
                f"Unexpected prediction shape: {predictions.shape}"
            )

        scores = predictions[0]

        if not np.all(np.isfinite(scores)):
            raise ValueError("The model returned non-finite scores.")

        if np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("Model scores are outside the range 0–1.")

        if not np.isclose(scores.sum(), 1.0, atol=1e-4):
            raise ValueError("Model scores do not sum to approximately 1.")

        ranking = np.argsort(-scores)
        predicted_index = int(ranking[0])

        print("\nModel: Office-31 baseline")
        print(f"Photo: {image_path.name}")
        print(f"Predicted class: {classes[predicted_index]}")
        print(f"Model score: {scores[predicted_index]:.2%}")

        print("\nAll class scores:")
        for index in ranking:
            print(f"{classes[index]:<12}: {scores[index]:.2%}")

        print(
            "\nModel scores are not calibrated probabilities "
            "of correctness."
        )

    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(status=1, message=f"\nError: {error}\n")


if __name__ == "__main__":
    main()