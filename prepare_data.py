import cv2
import numpy as np

from dataset import collect_images
from preprocessing import resize_and_pad


def load_images(split_name):
    image_paths, labels = collect_images(split_name)
    images = []

    for image_path in image_paths:
        img = cv2.imread(str(image_path))

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        processed = resize_and_pad(img)
        images.append(processed)

    return np.stack(images), np.array(labels)


if __name__ == "__main__":
    images, labels = load_images("train")

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Pixel data type:", images.dtype)
    print("First label:", labels[0])