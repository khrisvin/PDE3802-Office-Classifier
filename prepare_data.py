import cv2
import numpy as np

from dataset import collect_images, classes
from preprocessing import resize_and_pad


def load_images(split_name):
    image_paths, labels = collect_images(split_name)

    images = []

    for image_path in image_paths:
        img = cv2.imread(str(image_path))

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        processed = resize_and_pad(img)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        images.append(processed)

    class_to_id = {
        name: index
        for index, name in enumerate(classes)
    }

    numeric_labels = np.array(
        [class_to_id[label] for label in labels],
        dtype=np.int32
    )

    return np.stack(images), numeric_labels


if __name__ == "__main__":
    images, labels = load_images("train")

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Pixel data type:", images.dtype)
    print("Label data type:", labels.dtype)
    print("Colour order: RGB")
    print("First label:", labels[0])

    print("\nClass mapping:")
    for index, name in enumerate(classes):
        print(f"{name} -> {index}")