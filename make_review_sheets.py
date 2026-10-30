"""Create labelled thumbnail sheets for reviewing repeated products."""

import math
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "office31_reviewed"
EXCLUSIONS_FILE = PROJECT_DIR / "exact_duplicate_exclusions.txt"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "office31_review"

CLASSES = ("bottle", "calculator", "keyboard", "mouse", "mug", "stapler")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

COLUMNS = 5
IMAGES_PER_PAGE = 25
CELL_WIDTH = 220
CELL_HEIGHT = 200
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 155


def make_thumbnail(path):
    """Fit an image inside a white cell without stretching it."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    height, width = image.shape[:2]
    scale = min(IMAGE_WIDTH / width, IMAGE_HEIGHT / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (new_width, new_height), interpolation=interpolation
    )

    cell = np.full((CELL_HEIGHT, CELL_WIDTH, 3), 255, dtype=np.uint8)
    x = (CELL_WIDTH - new_width) // 2
    y = 8 + (IMAGE_HEIGHT - new_height) // 2
    cell[y:y + new_height, x:x + new_width] = resized

    label = path.stem.removeprefix("office31_amazon_")
    cv2.putText(
        cell, label, (12, 185),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
    )
    return cell


def main():
    if not EXCLUSIONS_FILE.is_file():
        raise FileNotFoundError(f"Missing file: {EXCLUSIONS_FILE}")

    exclusions = {
        line.strip().replace("\\", "/")
        for line in EXCLUSIONS_FILE.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }

    images_by_class = {}
    available = set()

    for class_name in CLASSES:
        folder = DATA_DIR / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {folder}")

        paths = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        available.update(path.relative_to(DATA_DIR).as_posix() for path in paths)
        images_by_class[class_name] = paths

    missing = exclusions - available
    if missing:
        raise ValueError(f"Excluded files not found: {sorted(missing)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for class_name, paths in images_by_class.items():
        accepted = [
            path for path in paths
            if path.relative_to(DATA_DIR).as_posix() not in exclusions
        ]
        if not accepted:
            raise ValueError(f"No images remain for {class_name}")

        page_count = math.ceil(len(accepted) / IMAGES_PER_PAGE)

        for page_index in range(page_count):
            start = page_index * IMAGES_PER_PAGE
            page_paths = accepted[start:start + IMAGES_PER_PAGE]
            rows = math.ceil(len(page_paths) / COLUMNS)

            sheet = np.full(
                (rows * CELL_HEIGHT + 45, COLUMNS * CELL_WIDTH, 3),
                255,
                dtype=np.uint8,
            )

            title = f"{class_name} - page {page_index + 1} of {page_count}"
            cv2.putText(
                sheet, title, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
            )

            for index, path in enumerate(page_paths):
                row, column = divmod(index, COLUMNS)
                y = 45 + row * CELL_HEIGHT
                x = column * CELL_WIDTH
                sheet[y:y + CELL_HEIGHT, x:x + CELL_WIDTH] = make_thumbnail(path)

            destination = OUTPUT_DIR / f"{class_name}_{page_index + 1:02d}.jpg"
            if not cv2.imwrite(str(destination), sheet):
                raise OSError(f"Could not save: {destination}")

        total += len(accepted)
        print(f"{class_name}: {len(accepted)} images, {page_count} sheets")

    print(f"\nTotal images shown: {total}")
    print(f"Sheets saved to: {OUTPUT_DIR}")
    print("Original images and manifest were not changed.")


if __name__ == "__main__":
    main()