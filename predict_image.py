import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from preprocessing import resize_and_pad


def main():
    parser = argparse.ArgumentParser(
        description="Classify one office-item photo."
    )
    parser.add_argument("image", type=Path, help="Path to the photo")
    args = parser.parse_args()

    project_folder = Path(__file__).resolve().parent
    model_folder = project_folder / "outputs" / "pilot"
    model_path = model_folder / "office_classifier.keras"
    classes_path = model_folder / "classes.json"

    image_path = args.image.expanduser()

    if not image_path.is_absolute():
        image_path = project_folder / image_path

    if not image_path.is_file():
        raise FileNotFoundError(f"Photo not found: {image_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with classes_path.open(encoding="utf-8") as file:
        class_names = json.load(file)

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Cannot read photo: {image_path}")

    processed = resize_and_pad(img)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    batch = np.expand_dims(processed, axis=0)

    model = keras.models.load_model(model_path, compile=False)
    predictions = model.predict(batch, verbose=0)

    if predictions.shape != (1, len(class_names)):
        raise ValueError("Model outputs do not match the saved classes.")

    if not np.isfinite(predictions).all():
        raise ValueError("Model returned invalid numerical values.")

    probabilities = predictions[0]
    predicted_id = int(np.argmax(probabilities))

    print("\nPhoto:", image_path.name)
    print("Predicted class:", class_names[predicted_id])
    print(f"Model score: {probabilities[predicted_id]:.2%}")

    print("\nAll class scores:")
    for index in np.argsort(probabilities)[::-1]:
        print(f"{class_names[index]:10s}: {probabilities[index]:.2%}")


if __name__ == "__main__":
    main()