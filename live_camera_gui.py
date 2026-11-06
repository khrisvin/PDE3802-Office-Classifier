
import json
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "office31_baseline" / "office_classifier.keras"
CLASSES_PATH = ROOT / "outputs" / "office31_baseline" / "classes.json"
CLASS_NAMES = ["bottle", "calculator", "keyboard", "mouse", "mug", "stapler"]


def prepare_image(image):
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a colour image.")

    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Image is empty.")

    size = 224
    scale = min(size / width, size / height)
    new_width = min(size, max(1, round(width * scale)))
    new_height = min(size, max(1, round(height * scale)))
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=method)

    padded = np.zeros((size, size, 3), dtype=np.uint8)
    left = (size - new_width) // 2
    top = (size - new_height) // 2
    padded[top:top + new_height, left:left + new_width] = resized
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0)


class OfficeClassifier:
    def __init__(self, window):
        self.window = window
        self.window.title("Office Object Classifier")
        self.window.geometry("940x630")
        self.window.minsize(830, 570)

        self.camera = None
        self.camera_job = None
        self.current_image = None
        self.preview_photo = None
        self.model = None
        self.classes = None
        self.busy = False
        self.closed = False
        self.session = 0
        self.last_prediction = 0.0
        self.events = queue.Queue()

        self.auto = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Loading model...")
        self.prediction = tk.StringVar(value="No prediction yet")
        self.scores = tk.StringVar(value="")

        self.make_window()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.after(100, self.check_results)
        threading.Thread(target=self.load_model, daemon=True).start()

    def make_window(self):
        ttk.Label(
            self.window,
            text="Office Object Classifier",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(15, 4))

        ttk.Label(
            self.window,
            text="PDE3802 - Live camera classification"
        ).pack(pady=(0, 10))

        main = ttk.Frame(self.window, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=(0, 16))

        self.preview = tk.Label(
            left,
            text="Camera is off\nClick Start Camera or Open Image",
            bg="#e8e8e8",
            width=64,
            height=23
        )
        self.preview.pack(fill="both", expand=True)

        buttons = ttk.Frame(left)
        buttons.pack(pady=12)

        ttk.Button(
            buttons, text="Start Camera", command=self.start_camera
        ).pack(side="left", padx=4)

        ttk.Button(
            buttons, text="Stop Camera", command=self.stop_camera
        ).pack(side="left", padx=4)

        ttk.Button(
            buttons, text="Open Image", command=self.open_image
        ).pack(side="left", padx=4)

        right = ttk.Frame(main, width=245)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ttk.Label(
            right, text="Prediction", font=("Segoe UI", 12, "bold")
        ).pack(anchor="w", pady=(6, 10))

        ttk.Label(
            right,
            textvariable=self.prediction,
            font=("Segoe UI", 15, "bold"),
            wraplength=230
        ).pack(anchor="w", pady=(0, 15))

        ttk.Label(right, text="Class scores").pack(anchor="w")

        ttk.Label(
            right,
            textvariable=self.scores,
            font=("Consolas", 10),
            justify="left"
        ).pack(anchor="w", pady=(10, 18))

        ttk.Button(
            right, text="Classify Now", command=self.classify
        ).pack(fill="x", pady=(4, 8))

        ttk.Checkbutton(
            right,
            text="Automatic classification",
            variable=self.auto
        ).pack(anchor="w", pady=5)

        ttk.Label(
            right,
            text="Automatic mode runs every 2 seconds while the camera is on.",
            wraplength=230
        ).pack(anchor="w", pady=(8, 10))

        ttk.Label(
            right,
            text="Place one object in the green square. Scores are not calibrated probabilities.",
            wraplength=230
        ).pack(anchor="w")

        ttk.Separator(self.window).pack(fill="x")

        ttk.Label(
            self.window, textvariable=self.status, padding=10
        ).pack(fill="x")

    def load_model(self):
        try:
            if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
                raise FileNotFoundError("Saved model or classes.json is missing.")

            classes = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))

            if classes != CLASS_NAMES:
                raise ValueError("Class order does not match the saved model.")

            import tensorflow as tf

            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            self.events.put(("model", model, classes))

        except Exception as error:
            self.events.put(("model_error", str(error)))

    def check_results(self):
        if self.closed:
            return

        try:
            while True:
                kind, *data = self.events.get_nowait()

                if kind == "model":
                    self.model, self.classes = data
                    self.status.set("Model loaded. Camera is ready.")

                elif kind == "model_error":
                    self.status.set("Model could not be loaded.")
                    messagebox.showerror("Model error", data[0])

                elif kind == "prediction":
                    session, values = data
                    self.busy = False

                    if session == self.session:
                        best = int(np.argmax(values))

                        self.prediction.set(
                            f"{self.classes[best].capitalize()} - {values[best]:.1%}"
                        )

                        self.scores.set(
                            "\n".join(
                                f"{name:<12} {score:.1%}"
                                for name, score in zip(self.classes, values)
                            )
                        )

                        self.status.set("Classification complete.")

                elif kind == "prediction_error":
                    session, error = data
                    self.busy = False

                    if session == self.session:
                        self.status.set("Prediction failed: " + error)

        except queue.Empty:
            pass

        self.window.after(100, self.check_results)

    def show_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(rgb)
        picture.thumbnail((625, 460), Image.Resampling.LANCZOS)

        self.preview_photo = ImageTk.PhotoImage(picture)
        self.preview.configure(image=self.preview_photo, text="")

    def start_camera(self):
        self.stop_camera()

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        camera = cv2.VideoCapture(0, backend)

        if not camera.isOpened() and backend != cv2.CAP_ANY:
            camera.release()
            camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            camera.release()

            messagebox.showerror(
                "Camera error",
                "Cannot open webcam. Check camera permissions and close Teams or Zoom."
            )
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        success, frame = camera.read()

        if not success or frame is None:
            camera.release()
            messagebox.showerror(
                "Camera error",
                "Camera opened but no image was received."
            )
            return

        self.camera = camera
        self.auto.set(True)
        self.last_prediction = 0.0
        self.prediction.set("No prediction yet")
        self.scores.set("")
        self.status.set("Camera running. Automatic classification is ON.")

        self.update_camera(first_frame=frame)

    def update_camera(self, first_frame=None):
        self.camera_job = None

        if self.camera is None:
            return

        if first_frame is None:
            success, frame = self.camera.read()
        else:
            success, frame = True, first_frame

        if not success or frame is None:
            self.stop_camera()
            self.status.set("Could not read the camera frame.")
            return

        height, width = frame.shape[:2]
        side = max(1, int(min(width, height) * 0.72))

        x1 = (width - side) // 2
        y1 = (height - side) // 2
        x2 = x1 + side
        y2 = y1 + side

        self.current_image = frame[y1:y2, x1:x2].copy()

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2
        )

        self.show_image(frame)

        now = time.monotonic()

        if self.auto.get() and self.model is not None and not self.busy:
            if now - self.last_prediction >= 2.0:
                self.last_prediction = now
                self.classify()

        self.camera_job = self.window.after(
            45,
            self.update_camera
        )

    def stop_camera(self):
        if self.camera_job is not None:
            self.window.after_cancel(self.camera_job)
            self.camera_job = None

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.current_image = None
        self.auto.set(False)
        self.session += 1

    def classify(self):
        if self.model is None:
            self.status.set("Model is still loading.")
            return

        if self.current_image is None:
            self.status.set("Start the camera or open an image first.")
            return

        if self.busy:
            return

        self.busy = True
        self.status.set("Classifying...")

        image = self.current_image.copy()
        session = self.session

        threading.Thread(
            target=self.predict,
            args=(image, session),
            daemon=True
        ).start()

    def predict(self, image, session):
        try:
            output = np.asarray(
                self.model.predict(
                    prepare_image(image),
                    verbose=0
                )
            )

            if output.shape != (1, len(self.classes)):
                raise ValueError("Unexpected model output shape.")

            values = output[0]

            if not np.all(np.isfinite(values)):
                raise ValueError("Model returned invalid scores.")

            if np.any(values < 0) or np.any(values > 1):
                raise ValueError("Model scores are outside the range 0-1.")

            if not np.isclose(values.sum(), 1.0, atol=1e-4):
                raise ValueError("Model scores do not sum to 1.")

            self.events.put(("prediction", session, values))

        except Exception as error:
            self.events.put(("prediction_error", session, str(error)))

    def open_image(self):
        filename = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if not filename:
            return

        try:
            image_bytes = np.fromfile(filename, dtype=np.uint8)
            image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Cannot read the selected image.")

        except (OSError, ValueError) as error:
            messagebox.showerror("Image error", str(error))
            return

        self.stop_camera()

        self.current_image = image.copy()
        self.show_image(image)
        self.classify()

    def close(self):
        self.closed = True
        self.stop_camera()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    OfficeClassifier(root)
    root.mainloop()
