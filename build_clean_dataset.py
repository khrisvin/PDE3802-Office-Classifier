"""Build a separate dataset from the team's saved image review.

This script does not create train/validation/test splits.
Visual groups are review-based, not verified physical-object identities.
"""

import csv
import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "external" / "office31" / "amazon" / "images"
DESTINATION = ROOT / "data" / "office31_clean"
REVIEW = ROOT / "external_image_review.txt"

EXPECTED_ORIGINAL = {
    "bottle": 36,
    "calculator": 94,
    "keyboard": 100,
    "mouse": 100,
    "mug": 94,
    "stapler": 99,
}

EXPECTED_RETAINED = {
    "bottle": 23,
    "calculator": 91,
    "keyboard": 54,
    "mouse": 94,
    "mug": 71,
    "stapler": 82,
}

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def fingerprint(path):
    """Identify identical file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_review():
    """Read exclusion decisions and visual groups from the review."""
    exclusions = {}
    groups = {}

    for line in REVIEW.read_text(encoding="utf-8-sig").splitlines():
        parts = [part.strip() for part in line.split("|")]

        if (
            len(parts) == 3
            and parts[0] in EXPECTED_ORIGINAL
            and re.fullmatch(r"frame_\d{4}", parts[1])
        ):
            class_name, frame, reason = parts
            key = (class_name, frame)

            if reason.upper().startswith("KEEP:"):
                continue
            if "pending" in reason.lower():
                raise ValueError(f"Unresolved review decision: {line}")
            if key in exclusions:
                raise ValueError(f"Repeated exclusion: {key}")

            exclusions[key] = reason

        elif len(parts) == 2:
            match = re.fullmatch(r"([a-z]+)_g\d+", parts[0])
            if not match:
                continue

            class_name = match.group(1)
            if class_name not in EXPECTED_ORIGINAL:
                raise ValueError(f"Unknown group class: {class_name}")

            frames = [frame.strip() for frame in parts[1].split(",")]
            for frame in frames:
                if not re.fullmatch(r"frame_\d{4}", frame):
                    raise ValueError(f"Invalid group member: {frame}")

                key = (class_name, frame)
                if key in groups:
                    raise ValueError(f"Image appears in multiple groups: {key}")

                groups[key] = parts[0]

    return exclusions, groups


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not REVIEW.is_file():
        raise FileNotFoundError(f"Missing review: {REVIEW}")

    if DESTINATION.exists():
        raise FileExistsError(
            f"Output already exists: {DESTINATION}\n"
            "Nothing was changed. Preserve it before attempting another build."
        )

    exclusions, groups = read_review()
    sources = {}

    for class_name, expected in EXPECTED_ORIGINAL.items():
        folder = SOURCE / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing source folder: {folder}")

        paths = sorted(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )

        if len(paths) != expected:
            raise ValueError(
                f"{class_name}: expected {expected} originals, found {len(paths)}"
            )

        for path in paths:
            key = (class_name, path.stem.lower())
            if key in sources:
                raise ValueError(f"Ambiguous filename: {key}")
            sources[key] = path

    unknown = (set(exclusions) | set(groups)) - set(sources)
    if unknown:
        raise ValueError(f"Review references missing files: {sorted(unknown)}")

    overlap = set(exclusions) & set(groups)
    if overlap:
        raise ValueError(f"Excluded images appear in retained groups: {overlap}")

    rows = []
    hashes = {}

    # Finish validation before creating output files.
    for (class_name, frame), source_path in sorted(sources.items()):
        if (class_name, frame) in exclusions:
            continue

        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {source_path}")

        digest = fingerprint(source_path)
        if digest in hashes:
            raise ValueError(
                f"Exact duplicate remains: {source_path} and {hashes[digest]}"
            )
        hashes[digest] = source_path

        key = (class_name, frame)
        grouped = key in groups
        group_id = groups.get(key, f"{class_name}_single_{frame}")
        filename = f"office31_amazon_{source_path.name}"

        rows.append({
            "class": class_name,
            "filename": filename,
            "relative_path": f"{class_name}/{filename}",
            "source_path": source_path.relative_to(ROOT).as_posix(),
            "source_url": "https://zenodo.org/records/14751811",
            "sha256": digest,
            "width": image.shape[1],
            "height": image.shape[0],
            "group_id": group_id,
            "group_basis": (
                "visual_product_family"
                if grouped else "singleton_not_verified_independent"
            ),
            "split": "",
        })

    counts = Counter(row["class"] for row in rows)
    if dict(counts) != EXPECTED_RETAINED:
        raise ValueError(
            f"Counts do not match the review.\n"
            f"Expected: {EXPECTED_RETAINED}\nFound: {dict(counts)}"
        )

    DESTINATION.mkdir(parents=True)

    for row in rows:
        target = DESTINATION / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / row["source_path"], target)

        if fingerprint(target) != row["sha256"]:
            raise OSError(f"Copied file failed verification: {target}")

    write_csv(
        DESTINATION / "manifest.csv",
        rows,
        list(rows[0]),
    )

    excluded_rows = [
        {"class": class_name, "frame": frame, "reason": reason}
        for (class_name, frame), reason in sorted(exclusions.items())
    ]
    write_csv(
        DESTINATION / "exclusions.csv",
        excluded_rows,
        ["class", "frame", "reason"],
    )

    shutil.copy2(REVIEW, DESTINATION / "review_snapshot.txt")

    print("\nClean dataset created.")
    for class_name in EXPECTED_RETAINED:
        print(f"{class_name}: {counts[class_name]} images")

    print(f"\nTotal retained: {len(rows)}")
    print(f"Total excluded: {len(exclusions)}")
    print(f"Saved to: {DESTINATION}")
    print("Original images and previous experiments were not changed.")
    print("Visual groups recorded; physical identities are not verified.")
    print("No training, validation or test split created yet.")


if __name__ == "__main__":
    main()