"""Plot the saved Office-31 training and validation history."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT / "outputs" / "office31_baseline"
HISTORY_PATH = EXPERIMENT_DIR / "history.json"
OUTPUT_PATH = EXPERIMENT_DIR / "learning_curves.png"


def main():
    if not HISTORY_PATH.is_file():
        raise FileNotFoundError(
            f"Training history was not found: {HISTORY_PATH}"
        )

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        history = json.load(file)

    required_keys = ("accuracy", "val_accuracy", "loss", "val_loss")
    values = {}

    for key in required_keys:
        if key not in history:
            raise ValueError(f"History is missing: {key}")

        array = np.asarray(history[key], dtype=float)

        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"History entry '{key}' must be a nonempty list.")

        if not np.all(np.isfinite(array)):
            raise ValueError(f"History entry '{key}' contains invalid values.")

        values[key] = array

    lengths = {len(array) for array in values.values()}
    if len(lengths) != 1:
        raise ValueError("History entries have different numbers of epochs.")

    epoch_count = len(values["loss"])
    epochs = np.arange(1, epoch_count + 1)

    best_index = int(np.argmin(values["val_loss"]))
    best_epoch = best_index + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        epochs, values["accuracy"] * 100,
        label="Training accuracy", linewidth=2,
    )
    axes[0].plot(
        epochs, values["val_accuracy"] * 100,
        label="Validation accuracy", linewidth=2,
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)

    axes[1].plot(
        epochs, values["loss"],
        label="Training loss (class-weighted)", linewidth=2,
    )
    axes[1].plot(
        epochs, values["val_loss"],
        label="Validation loss (unweighted)", linewidth=2,
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")

    for axis in axes:
        axis.axvline(
            best_epoch,
            color="grey",
            linestyle="--",
            alpha=0.8,
            label=f"Lowest validation loss: epoch {best_epoch}",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle("Office-31 baseline — learning curves", fontsize=14)

    fig.text(
        0.5,
        0.02,
        "Training loss uses class weights; validation loss does not. "
        "Their magnitudes are not directly comparable.",
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nExperiment: Office-31 baseline")
    print(f"Epochs completed: {epoch_count}")
    print(f"Lowest validation loss at epoch: {best_epoch}")
    print(f"Validation loss: {values['val_loss'][best_index]:.4f}")
    print(
        "Validation accuracy at that epoch: "
        f"{values['val_accuracy'][best_index]:.2%}"
    )
    print(f"Saved: {OUTPUT_PATH}")
    print("No training or test evaluation was performed.")


if __name__ == "__main__":
    main()