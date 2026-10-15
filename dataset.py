from pathlib import Path

project_folder = Path(__file__).resolve().parent
classes = ["book", "bottle", "keyboard", "mouse", "mug", "stapler"]


def collect_images(split_name):
    split_path = project_folder / "data" / "split" / split_name
    image_paths = []
    labels = []

    for class_name in classes:
        class_folder = split_path / class_name

        if not class_folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {class_folder}")

        files = sorted(class_folder.glob("*.jpeg"))

        if not files:
            raise ValueError(f"No JPEG photos found in: {class_folder}")

        for image_path in files:
            image_paths.append(image_path)
            labels.append(class_name)

    return image_paths, labels


if __name__ == "__main__":
    image_paths, labels = collect_images("train")

    print("Image paths:", len(image_paths))
    print("Labels:", len(labels))
    print("Unique classes:", len(set(labels)))

    print("\nFirst five image-label pairs:")

    for image_path, label in zip(image_paths[:5], labels[:5]):
        print(image_path.name, "->", label)