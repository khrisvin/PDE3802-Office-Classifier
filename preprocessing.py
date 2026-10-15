from pathlib import Path
import cv2


def resize_and_pad(img, target_size=224):
    if img is None or img.size == 0:
        raise ValueError("The image is empty or could not be loaded.")

    if not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("Target size must be a positive integer.")

    height, width = img.shape[:2]

    scale = min(target_size / width, target_size / height)

    new_width = max(1, min(target_size, round(width * scale)))
    new_height = max(1, min(target_size, round(height * scale)))

    interpolation = (
        cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        img,
        (new_width, new_height),
        interpolation=interpolation
    )

    left = (target_size - new_width) // 2
    right = target_size - new_width - left

    top = (target_size - new_height) // 2
    bottom = target_size - new_height - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    return padded


if __name__ == "__main__":
    project_folder = Path(__file__).resolve().parent
    image_path = (
        project_folder / "data" / "raw" / "mug"
        / "khrisvin_mug01_001.jpeg"
    )

    img = cv2.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    processed = resize_and_pad(img)

    print("Original shape:", img.shape)
    print("Processed shape:", processed.shape)

    cv2.imshow("Resized and padded image", processed)
    print("Click the image window and press any key to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()