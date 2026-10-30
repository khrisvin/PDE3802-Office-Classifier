"""Split the cleaned Office-31 dataset by recorded visual product groups.

Targets 70% training, 15% validation and 15% testing per class.
Actual image counts vary because groups cannot be divided.
"""

import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "office31_clean"
MANIFEST = SOURCE / "manifest.csv"
OUTPUT = ROOT / "data" / "office31_split"

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
EXPECTED = {
    "bottle": 23,
    "calculator": 91,
    "keyboard": 54,
    "mouse": 94,
    "mug": 71,
    "stapler": 82,
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_groups(groups, class_name):
    """Choose a seeded group allocation close to the target image counts.

    Only group sizes are used. No model results influence the split.
    """
    group_ids = sorted(groups)
    if len(group_ids) < 3:
        raise ValueError(f"{class_name}: at least three groups are required.")

    rng = random.Random(f"{SEED}:{class_name}")
    total = sum(len(items) for items in groups.values())
    targets = {name: total * ratio for name, ratio in RATIOS.items()}

    # Allocate approximately 15% of groups to each holdout split.
    holdout_groups = max(1, round(len(group_ids) * 0.15))
    best_score = float("inf")
    best_assignment = None

    for _ in range(2000):
        shuffled = group_ids.copy()
        rng.shuffle(shuffled)

        allocation = {
            "val": shuffled[:holdout_groups],
            "test": shuffled[holdout_groups:2 * holdout_groups],
            "train": shuffled[2 * holdout_groups:],
        }

        counts = {
            split: sum(len(groups[group]) for group in ids)
            for split, ids in allocation.items()
        }

        score = sum(
            ((counts[split] - targets[split]) / targets[split]) ** 2
            for split in RATIOS
        )

        if score < best_score:
            best_score = score
            best_assignment = {
                group: split
                for split, ids in allocation.items()
                for group in ids
            }

    return best_assignment


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}\n"
            "Nothing was changed. Do not regenerate splits to improve scores."
        )

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        original_fields = reader.fieldnames or []
        rows = list(reader)

    required = {"class", "relative_path", "sha256", "group_id", "split"}
    if not required.issubset(original_fields):
        raise ValueError(f"Manifest must contain: {sorted(required)}")

    counts = Counter(row["class"] for row in rows)
    if dict(counts) != EXPECTED:
        raise ValueError(f"Unexpected dataset counts: {dict(counts)}")

    groups_by_class = defaultdict(lambda: defaultdict(list))
    seen_paths = set()
    seen_hashes = set()
    group_classes = {}

    for row in rows:
        class_name = row["class"]
        relative_path = Path(row["relative_path"])
        source_path = (SOURCE / relative_path).resolve()

        if not source_path.is_relative_to(SOURCE.resolve()):
            raise ValueError(f"Invalid source path: {relative_path}")

        if row["relative_path"] in seen_paths:
            raise ValueError(f"Repeated manifest path: {relative_path}")
        seen_paths.add(row["relative_path"])

        if not source_path.is_file():
            raise FileNotFoundError(f"Missing image: {source_path}")

        digest = file_hash(source_path)
        if digest != row["sha256"]:
            raise ValueError(f"Image changed since preparation: {source_path}")
        if digest in seen_hashes:
            raise ValueError(f"Exact duplicate remains: {source_path}")
        seen_hashes.add(digest)

        group = row["group_id"].strip()
        if not group:
            raise ValueError(f"Missing group for: {relative_path}")
        if row["split"].strip():
            raise ValueError("Source manifest already contains split assignments.")

        if group in group_classes and group_classes[group] != class_name:
            raise ValueError(f"Group spans different classes: {group}")
        group_classes[group] = class_name
        groups_by_class[class_name][group].append(row)

    assignments = {}
    for class_name in sorted(EXPECTED):
        assignments.update(
            assign_groups(groups_by_class[class_name], class_name)
        )

    split_rows = []
    for row in rows:
        updated = dict(row)
        updated["split"] = assignments[row["group_id"].strip()]
        updated["split_path"] = (
            Path(updated["split"]) / row["relative_path"]
        ).as_posix()
        split_rows.append(updated)

    # Validate complete coverage and group separation before copying.
    group_splits = defaultdict(set)
    split_counts = Counter()
    for row in split_rows:
        group_splits[row["group_id"]].add(row["split"])
        split_counts[(row["class"], row["split"])] += 1

    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("A product group crosses split boundaries.")

    for class_name in EXPECTED:
        for split in RATIOS:
            if split_counts[(class_name, split)] == 0:
                raise ValueError(f"Empty class/split: {class_name}/{split}")

    OUTPUT.mkdir(parents=True)

    for row in split_rows:
        source_path = SOURCE / row["relative_path"]
        destination = OUTPUT / row["split_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

        if file_hash(destination) != row["sha256"]:
            raise OSError(f"Copy verification failed: {destination}")

    with (OUTPUT / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=original_fields + ["split_path"]
        )
        writer.writeheader()
        writer.writerows(split_rows)

    settings = (
        f"Seed: {SEED}\n"
        f"Target ratios: {RATIOS}\n"
        f"Source manifest SHA256: {file_hash(MANIFEST)}\n"
        "Split unit: recorded visual product group\n"
        "Allocation selected using group sizes only.\n"
        "Visual groups are not verified physical-object identities.\n"
        "Do not use test results to select models or regenerate splits.\n"
    )
    (OUTPUT / "split_details.txt").write_text(settings, encoding="utf-8")

    print("\nSplit complete.")
    for class_name in sorted(EXPECTED):
        print(
            f"{class_name}: "
            f"train={split_counts[(class_name, 'train')]}, "
            f"val={split_counts[(class_name, 'val')]}, "
            f"test={split_counts[(class_name, 'test')]}"
        )

    print()
    for split in RATIOS:
        total = sum(split_counts[(name, split)] for name in EXPECTED)
        print(f"{split}: {total} images")

    print(f"Total: {len(split_rows)}")
    print("Recorded groups crossing splits: 0")
    print(f"Saved to: {OUTPUT}")
    print("Original datasets and experiments were not changed.")


if __name__ == "__main__":
    main()