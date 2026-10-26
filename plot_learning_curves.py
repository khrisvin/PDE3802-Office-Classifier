import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_experiment(project_folder, experiment):
    output_folder = project_folder / "outputs" / experiment
    history_path = output_folder / "history.json"

    if not history_path.is_file():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    with history_path.open(encoding="utf-8") as file:
        history = json.load(file)

    required_keys = ["accuracy", "val_accuracy", "loss", "val_loss"]

    for key in required_keys:
        if key not in history or not history[key]:
            raise ValueError(f"Missing or empty history entry: {key}")

    epoch_count = len(history["loss"])

    if any(len(history[key]) != epoch_count for key in required_keys):
        raise ValueError("History entries have different lengths.")

    if any(
        not np.isfinite(np.asarray(history[key], dtype=float)).all()
        for key in required_keys
    ):
        raise ValueError("Training history contains invalid values.")

    epochs = np.arange(1, epoch_count + 1)
    best_index = int(np.argmin(history["val_loss"]))
    best_epoch = best_index + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["accuracy"], label="Training")
    axes[0].plot(epochs, history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(epochs, history["loss"], label="Training")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")

    for ax in axes:
        ax.axvline(
            best_epoch,
            color="grey",
            linestyle="--",
            label=f"Lowest validation loss: epoch {best_epoch}",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"Learning curves: {experiment}")
    fig.tight_layout()

    figure_path = output_folder / "learning_curves.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    print(f"\nExperiment: {experiment}")
    print("Epochs completed:", epoch_count)
    print("Lowest validation loss at epoch:", best_epoch)
    print(f"Validation loss: {history['val_loss'][best_index]:.4f}")
    print(
        "Validation accuracy at that epoch:",
        f"{history['val_accuracy'][best_index]:.2%}",
    )
    print("Saved:", figure_path)


def main():
    project_folder = Path(__file__).resolve().parent

    for experiment in ["pilot", "augmented_v1"]:
        plot_experiment(project_folder, experiment)


if __name__ == "__main__":
    main()