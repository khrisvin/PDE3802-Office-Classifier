import json
from pathlib import Path

from tensorflow import keras

from dataset import classes
from model import build_model
from prepare_data import load_images


def main():
    project_folder = Path(__file__).resolve().parent
    output_folder = project_folder / "outputs" / "augmented_v1"

    if output_folder.exists():
        raise FileExistsError(
            f"Experiment folder already exists: {output_folder}\n"
            "Use a new experiment folder name to preserve previous results."
        )

    train_images, train_labels = load_images("train")
    val_images, val_labels = load_images("val")

    print("Training images:", len(train_images))
    print("Validation images:", len(val_images))

    classifier = build_model()

    augmentation = keras.Sequential(
        [
            keras.layers.RandomRotation(
                factor=0.05,
                fill_mode="constant",
                fill_value=0.0,
                seed=42,
            ),
            keras.layers.RandomZoom(
                height_factor=0.1,
                width_factor=None,
                fill_mode="constant",
                fill_value=0.0,
                seed=43,
            ),
        ],
        name="training_augmentation",
    )

    inputs = keras.Input(shape=(224, 224, 3), name="rgb_input")
    x = augmentation(inputs)
    outputs = classifier(x)

    model = keras.Model(
        inputs,
        outputs,
        name="augmented_office_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    output_folder.mkdir(parents=True, exist_ok=False)

    history = model.fit(
        train_images,
        train_labels,
        validation_data=(val_images, val_labels),
        epochs=30,
        batch_size=8,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=2,
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
    print("Saved experiment to:", output_folder)


if __name__ == "__main__":
    main()