from pathlib import Path
import cv2

project_folder = Path(__file__).resolve().parent
image_path = (
    project_folder / "data" / "raw" / "mug"
    / "khrisvin_mug01_001.jpeg"
)

img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

height, width = img.shape[:2]

print("Filename:", image_path.name)
print("Height:", height, "pixels")
print("Width:", width, "pixels")
print("Colour channels:", img.shape[2])

scale = min(800 / width, 600 / height, 1.0)
preview = cv2.resize(
    img,
    (round(width * scale), round(height * scale)),
    interpolation=cv2.INTER_AREA
)

cv2.imshow("Our first coursework image", preview)

print("Click the image window and press any key to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()