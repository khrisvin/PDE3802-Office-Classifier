from pathlib import Path
import shutil

project_folder = Path(__file__).resolve().parent
raw_folder = project_folder / "data" / "raw"
split_folder = project_folder / "data" / "split"

split_plan = {
    "book": {
        "train": "khrisvin",
        "val": "pranay",
        "test": "veedoor"
    },
    "bottle": {
        "train": "pranay",
        "val": "veedoor",
        "test": "khrisvin"
    },
    "keyboard": {
        "train": "veedoor",
        "val": "khrisvin",
        "test": "pranay"
    },
    "mouse": {
        "train": "khrisvin",
        "val": "pranay",
        "test": "veedoor"
    },
    "mug": {
        "train": "pranay",
        "val": "veedoor",
        "test": "khrisvin"
    },
    "stapler": {
        "train": "veedoor",
        "val": "khrisvin",
        "test": "pranay"
    }
}

def main():
    if split_folder.exists():
        raise FileExistsError(
            f"Destination already exists: {split_folder}\n"
            "No files were copied. Check the existing split before continuing."
        )

    copy_jobs = []
    errors = []
    ignored_count = 0

    for class_name, assignments in split_plan.items():
        class_folder = raw_folder / class_name

        if not class_folder.is_dir():
            errors.append(f"Missing folder: {class_folder}")
            continue

        files_by_name = {}

        for path in class_folder.iterdir():
            if path.is_file():
                name = path.name.lower()
                files_by_name.setdefault(name, []).append(path)

        expected_names = set()

        for split_name, photographer in assignments.items():
            for photo_number in range(1, 6):
                filename = (
                    f"{photographer}_{class_name}01_"
                    f"{photo_number:03d}.jpeg"
                )
                expected_names.add(filename)
                matches = files_by_name.get(filename, [])

                if len(matches) == 0:
                    errors.append(f"Missing photo: {class_name}/{filename}")
                elif len(matches) > 1:
                    errors.append(f"Ambiguous filename: {class_name}/{filename}")
                else:
                    source = matches[0]
                    destination = split_folder / split_name / class_name / filename
                    copy_jobs.append((source, destination, split_name))

        ignored_count += sum(
            len(paths)
            for name, paths in files_by_name.items()
            if name not in expected_names
        )

    if errors:
        raise RuntimeError(
            "Dataset checks failed. No files were copied:\n"
            + "\n".join(errors)
        )

    totals = {"train": 0, "val": 0, "test": 0}

    for source, destination, split_name in copy_jobs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        totals[split_name] += 1

    for class_name in split_plan:
        print(f"{class_name}: train=5, val=5, test=5")

    print("\nSplit complete.")

    for split_name, count in totals.items():
        print(f"{split_name}: {count} photos")

    print(f"Total copied: {sum(totals.values())}")
    print(f"Other files ignored: {ignored_count}")
    print("Original photos in data/raw were not changed.")

if __name__ == "__main__":
    main()
