"""Shared Office-31 data helpers; importing this module never starts training."""

import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "office31_split"
MANIFEST = DATA_DIR / "split_manifest.csv"
CLASSES = ["bottle", "calculator", "keyboard", "mouse", "mug", "stapler"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SIZE = 224


def resize_and_pad(image):
    """Preserve proportions and centre the image on a black square."""
    height, width = image.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)

    new_width = min(IMAGE_SIZE, max(1, round(width * scale)))
    new_height = min(IMAGE_SIZE, max(1, round(height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=interpolation
    )

    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    x = (IMAGE_SIZE - new_width) // 2
    y = (IMAGE_SIZE - new_height) // 2
    canvas[y:y + new_height, x:x + new_width] = resized
    return canvas


def read_manifest():
    """Check the recorded split assignments without loading test images."""
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"class", "split", "split_path", "sha256", "group_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Split manifest is missing required columns.")
        rows = list(reader)

    if not rows:
        raise ValueError("Split manifest is empty.")

    seen_paths = set()
    seen_hashes = set()
    group_assignments = {}

    for row in rows:
        if row["class"] not in CLASS_TO_ID:
            raise ValueError(f"Unknown class: {row['class']}")
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {row['split']}")

        relative = Path(row["split_path"])
        if (
            len(relative.parts) != 3
            or relative.parts[0] != row["split"]
            or relative.parts[1] != row["class"]
        ):
            raise ValueError(f"Invalid split path: {relative}")

        path = (DATA_DIR / relative).resolve()
        if not path.is_relative_to(DATA_DIR.resolve()):
            raise ValueError(f"Path outside dataset: {relative}")

        if row["split_path"] in seen_paths:
            raise ValueError(f"Repeated path: {relative}")
        seen_paths.add(row["split_path"])

        if row["sha256"] in seen_hashes:
            raise ValueError("Repeated image hash in split manifest.")
        seen_hashes.add(row["sha256"])

        group = row["group_id"].strip()
        if not group:
            raise ValueError(f"Missing group: {relative}")

        assignment = (row["class"], row["split"])
        if group in group_assignments and group_assignments[group] != assignment:
            raise ValueError(f"Group crosses classes or splits: {group}")
        group_assignments[group] = assignment

    return rows


def load_split(rows, split):
    """Load and verify training or validation images in manifest order."""
    if split not in {"train", "val"}:
        raise ValueError("This training script only loads train and val.")

    images = []
    labels = []

    for row in rows:
        if row["split"] != split:
            continue

        path = DATA_DIR / row["split_path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Image changed since splitting: {path}")

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")

        image = resize_and_pad(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        labels.append(CLASS_TO_ID[row["class"]])

    if not images:
        raise ValueError(f"No images found for {split}.")

    if set(labels) != set(range(len(CLASSES))):
        raise ValueError(f"Some classes are missing from {split}.")

    return np.stack(images), np.asarray(labels, dtype=np.int32)


