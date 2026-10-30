"""Prepare reviewed Office-31 images without changing the original dataset."""

import csv
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = PROJECT_DIR / "data" / "external" / "office31" / "amazon" / "images"
OUTPUT_DIR = PROJECT_DIR / "data" / "office31_reviewed"

SOURCE_URL = "https://zenodo.org/records/14751811"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# These exclusions come from the team's manual class-label review.
EXCLUDED = {
    "bottle": {"frame_0002", "frame_0020", "frame_0021"},
    "calculator": {"frame_0060", "frame_0073"},
    "keyboard": {"frame_0056", "frame_0064"},
    "mouse": set(),
    "mug": {"frame_0035", "frame_0046", "frame_0080"},
    "stapler": {
        "frame_0010",
        "frame_0013",
        "frame_0025",
        "frame_0057",
        "frame_0058",
        "frame_0088",
        "frame_0093",
    },
}

EXPECTED_COUNTS = {
    "bottle": 36,
    "calculator": 94,
    "keyboard": 100,
    "mouse": 100,
    "mug": 94,
    "stapler": 99,
}


def sha256_file(path):
    """Calculate a fingerprint of the original file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DIR}\n"
            "Nothing was changed. Preserve or move it before running again."
        )

    records = []
    fingerprints = defaultdict(list)

    # Validate every accepted image before creating the output folder.
    for class_name in sorted(EXCLUDED):
        folder = SOURCE_DIR / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing source folder: {folder}")

        paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if len(paths) != EXPECTED_COUNTS[class_name]:
            raise ValueError(
                f"{class_name}: expected {EXPECTED_COUNTS[class_name]} "
                f"source images, found {len(paths)}."
            )

        stems = [path.stem.lower() for path in paths]
        if len(stems) != len(set(stems)):
            raise ValueError(f"Ambiguous repeated filenames in: {folder}")

        missing = EXCLUDED[class_name] - set(stems)
        if missing:
            raise ValueError(
                f"{class_name}: excluded filenames not found: {sorted(missing)}"
            )

        for path in paths:
            if path.stem.lower() in EXCLUDED[class_name]:
                continue

            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unreadable image: {path}")

            height, width = image.shape[:2]
            fingerprint = sha256_file(path)
            destination_name = f"office31_amazon_{path.name}"
            relative_destination = Path(class_name) / destination_name

            record = {
                "class": class_name,
                "source_path": path.relative_to(PROJECT_DIR).as_posix(),
                "output_path": relative_destination.as_posix(),
                "source_url": SOURCE_URL,
                "sha256": fingerprint,
                "width": width,
                "height": height,
                "duplicate_group": "",
                "object_group": "",
                "split": "",
            }
            records.append(record)
            fingerprints[fingerprint].append(record)

    duplicate_groups = [
        group for group in fingerprints.values() if len(group) > 1
    ]

    # Conflicting labels for identical files require manual investigation.
    for group in duplicate_groups:
        labels = {record["class"] for record in group}
        if len(labels) > 1:
            filenames = [record["source_path"] for record in group]
            raise ValueError(
                f"Identical files have different class labels: {filenames}"
            )

    for number, group in enumerate(duplicate_groups, start=1):
        for record in group:
            record["duplicate_group"] = f"duplicate_{number:03d}"

    OUTPUT_DIR.mkdir(parents=True)

    for record in records:
        source = PROJECT_DIR / record["source_path"]
        destination = OUTPUT_DIR / record["output_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    exclusions_path = OUTPUT_DIR / "exclusions.csv"
    with exclusions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "filename_stem", "reason"])
        for class_name, stems in sorted(EXCLUDED.items()):
            for stem in sorted(stems):
                writer.writerow(
                    [class_name, stem, "Incorrect class according to team review"]
                )

    duplicates_path = OUTPUT_DIR / "exact_duplicates.txt"
    with duplicates_path.open("w", encoding="utf-8") as file:
        if not duplicate_groups:
            file.write("No identical file-byte duplicates found.\n")
        else:
            for group in duplicate_groups:
                file.write(f"{group[0]['duplicate_group']}\n")
                for record in group:
                    file.write(f"  {record['output_path']}\n")
                file.write("\n")

    print("\nPreparation complete.")
    for class_name in sorted(EXCLUDED):
        count = sum(record["class"] == class_name for record in records)
        print(f"{class_name}: {count} images")

    print(f"\nTotal copied: {len(records)}")
    print(f"Excluded: {sum(len(items) for items in EXCLUDED.values())}")
    print(f"Exact duplicate groups: {len(duplicate_groups)}")
    print(f"Saved to: {OUTPUT_DIR}")
    print("\nOriginal images were not changed.")
    print("No training, validation or test split has been created.")
    print("Repeated-object and near-duplicate review is still required.")


if __name__ == "__main__":
    main()