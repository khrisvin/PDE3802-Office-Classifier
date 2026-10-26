import json
from pathlib import Path

from tensorflow import keras

from dataset import classes
from model import build_model
from prepare_data import load_images


def main():
    project_folder = Path(__file__).resolve().parent
    output_folder = project_folder / "outputs" / "pilot"
    output_folder.mkdir(parents=True, exist_ok=True)

    train_images, train_labels = load_images("train")
    val_images, val_labels = load_images("val")

    print("Training images:", len(train_images))
    print("Validation images:", len(val_images))

    model = build_model()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=30,
        batch_size=8,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=2
    )

    model.save(output_folder / "office_classifier.keras")

    with (output_folder / "classes.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(classes, file, indent=2)

    history_values = {
        name: [float(value) for value in values]
        for name, values in history.history.items()
    }

    with (output_folder / "history.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(history_values, file, indent=2)

    print("\nTraining complete.")
    print("Epochs completed:", len(history.history["loss"]))
    print("Saved model, class names and learning history to:")
    print(output_folder)


if __name__ == "__main__":
    main()