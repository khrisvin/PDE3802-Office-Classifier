from pathlib import Path
import cv2

project_folder = Path(__file__).resolve().parent
data_folder = project_folder / "data" / "raw"

classes = ["book", "bottle", "keyboard", "mouse", "mug", "stapler"]

def main():
    total = 0
    unreadable = 0

    for class_name in classes:
        class_folder = data_folder / class_name

        if not class_folder.is_dir():
            raise FileNotFoundError(f"Missing folder: {class_folder}")

        image_paths = sorted(class_folder.glob("*.jpeg"))
        count = len(image_paths)
        total += count

        print(f"{class_name}: {count} photos")

        for image_path in image_paths:
            img = cv2.imread(str(image_path))

            if img is None:
                print("  Cannot read:", image_path.name)
                unreadable += 1

    print(f"\nTotal: {total} photos")
    print(f"Readable: {total - unreadable}")
    print(f"Unreadable: {unreadable}")

if __name__ == "__main__":
    main()
