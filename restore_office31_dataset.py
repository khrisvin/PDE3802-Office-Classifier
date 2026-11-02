"""Restore Office-31 images using the existing clean and split manifests.

Uses only Python's standard library.
Preserves manifests, split assignments, models and experiment results.
Existing images are checked and skipped when their hashes match.
Conflicting files cause an error instead of being overwritten.
"""

import csv
import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "office31_clean"
SPLIT_DIR = ROOT / "data" / "office31_split"

CLEAN_MANIFEST = CLEAN_DIR / "manifest.csv"
SPLIT_MANIFEST = SPLIT_DIR / "split_manifest.csv"

EXPECTED = {
    "bottle": {"train": 17, "val": 3, "test": 3},
    "calculator": {"train": 63, "val": 14, "test": 14},
    "keyboard": {"train": 38, "val": 8, "test": 8},
    "mouse": {"train": 66, "val": 14, "test": 14},
    "mug": {"train": 49, "val": 12, "test": 10},
    "stapler": {"train": 58, "val": 12, "test": 12},
}

COMMON_FIELDS = {
    "class", "filename", "relative_path", "source_path",
    "source_url", "sha256", "width", "height",
    "group_id", "group_basis", "split",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path, required):
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    for row in rows:
        if None in row or any(row.get(key) is None for key in required):
            raise ValueError(f"Malformed CSV row in {path}")

    return rows


def safe_path(base, relative):
    """Resolve a manifest path while keeping it inside its base directory."""
    path = PurePosixPath(relative)

    if (
        not relative
        or "\\" in relative
        or ":" in relative
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise ValueError(f"Invalid relative path: {relative!r}")

    resolved = base.joinpath(*path.parts).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"Path escapes its directory: {relative}")

    return resolved


def validate_manifests(clean_rows, split_rows):
    clean_by_path = {}
    hashes = set()
    clean_counts = Counter()

    for row in clean_rows:
        label = row["class"]
        filename = row["filename"]
        relative = row["relative_path"]

        if label not in EXPECTED:
            raise ValueError(f"Unknown class: {label}")

        if not re.fullmatch(r"office31_amazon_frame_\d{4}\.jpg", filename):
            raise ValueError(f"Unexpected filename: {filename}")

        if relative != f"{label}/{filename}":
            raise ValueError(f"Unexpected clean path: {relative}")

        original_name = filename.removeprefix("office31_amazon_")
        expected_source = (
            f"data/external/office31/amazon/images/"
            f"{label}/{original_name}"
        )
        if row["source_path"] != expected_source:
            raise ValueError(f"Unexpected source path: {row['source_path']}")

        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid SHA-256 for: {relative}")

        if relative in clean_by_path:
            raise ValueError(f"Repeated clean path: {relative}")

        if row["sha256"] in hashes:
            raise ValueError(f"Repeated image hash: {relative}")

        if row["split"]:
            raise ValueError("The clean manifest must have blank splits.")

        if not row["group_id"] or not row["group_basis"]:
            raise ValueError(f"Missing group information: {relative}")

        clean_by_path[relative] = row
        hashes.add(row["sha256"])
        clean_counts[label] += 1

    expected_clean = {
        label: sum(counts.values()) for label, counts in EXPECTED.items()
    }
    if dict(clean_counts) != expected_clean:
        raise ValueError(f"Unexpected clean class counts: {clean_counts}")

    seen = set()
    split_counts = Counter()
    groups = {}

    for row in split_rows:
        relative = row["relative_path"]
        label = row["class"]
        split = row["split"]

        if relative not in clean_by_path or relative in seen:
            raise ValueError(f"Unknown or repeated split image: {relative}")

        clean_row = clean_by_path[relative]
        for field in COMMON_FIELDS - {"split"}:
            if row[field] != clean_row[field]:
                raise ValueError(
                    f"Manifests disagree on '{field}' for: {relative}"
                )

        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        if row["split_path"] != f"{split}/{relative}":
            raise ValueError(f"Unexpected split path: {row['split_path']}")

        group = row["group_id"]
        assignment = (label, split)
        if group in groups and groups[group] != assignment:
            raise ValueError(f"Group crosses classes or splits: {group}")

        groups[group] = assignment
        seen.add(relative)
        split_counts[(label, split)] += 1

    if seen != set(clean_by_path):
        raise ValueError("The split manifest does not cover all clean images.")

    for label, counts in EXPECTED.items():
        for split, expected_count in counts.items():
            if split_counts[(label, split)] != expected_count:
                raise ValueError(f"Unexpected count for {label}/{split}")


def main():
    clean_rows = read_manifest(CLEAN_MANIFEST, COMMON_FIELDS)
    split_rows = read_manifest(
        SPLIT_MANIFEST, COMMON_FIELDS | {"split_path"}
    )
    validate_manifests(clean_rows, split_rows)

    jobs = []

    for row in clean_rows:
        jobs.append((
            safe_path(ROOT, row["source_path"]),
            safe_path(CLEAN_DIR, row["relative_path"]),
            row["sha256"],
        ))

    for row in split_rows:
        jobs.append((
            safe_path(ROOT, row["source_path"]),
            safe_path(SPLIT_DIR, row["split_path"]),
            row["sha256"],
        ))

    print("Checking source images and existing destination files...")

    checked_sources = set()
    missing_jobs = []
    existing_count = 0

    # Validate every source and existing destination before copying anything.
    for source, destination, expected_hash in jobs:
        if source not in checked_sources:
            if not source.is_file():
                raise FileNotFoundError(
                    f"Original image not found: {source}\n"
                    "Extract Office-31 so its amazon/images folder is at "
                    "data/external/office31/amazon/images."
                )

            if sha256(source) != expected_hash:
                raise ValueError(f"Source image hash mismatch: {source}")

            checked_sources.add(source)

        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"Destination is not a file: {destination}")

            if sha256(destination) != expected_hash:
                raise ValueError(
                    f"Existing image hash mismatch: {destination}\n"
                    "The file has not been overwritten."
                )

            existing_count += 1
        else:
            missing_jobs.append((source, destination, expected_hash))

    restored_count = 0

    for source, destination, expected_hash in missing_jobs:
        destination.parent.mkdir(parents=True, exist_ok=True)

        # Exclusive creation prevents overwriting an existing file.
        with destination.open("xb") as output:
            try:
                with source.open("rb") as original:
                    shutil.copyfileobj(original, output)
            except BaseException:
                output.close()
                destination.unlink(missing_ok=True)
                raise

        if sha256(destination) != expected_hash:
            destination.unlink()
            raise ValueError(f"Restored image hash mismatch: {destination}")

        restored_count += 1

    print("\nDataset restoration complete.")
    print(f"Original images verified: {len(checked_sources)}")
    print(f"Existing image copies verified: {existing_count}")
    print(f"Missing image copies restored: {restored_count}")

    for label, counts in EXPECTED.items():
        print(
            f"{label}: train={counts['train']}, "
            f"val={counts['val']}, test={counts['test']}"
        )

    print("\nClean images: 415")
    print("Split images: train=291, val=63, test=61")
    print("Recorded groups crossing splits: 0")
    print("Manifests, split assignments and experiment files were unchanged.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"\nError: {error}")