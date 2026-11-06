
# PDE3802 Office Classifier
**PDE3802 — Artificial Intelligence in Robotics**
**Middlesex University Mauritius | Group Coursework**

This project implements a six-class office-object image classifier using Python, OpenCV and TensorFlow/Keras.

The selected model uses an ImageNet-pretrained MobileNetV2 feature extractor with a trained six-class classification layer.

The application supports image-file classification and a simple desktop interface with live webcam input and automatic predictions.

## Team Members

- Khrisvin Nepaulsing — M01003462
- Veedoor Gopal
- Pranay Poheith

## 1. Project Overview

The proposed project is an AI-assisted office organisation robot for shared offices and meeting rooms.

Its intended purpose is to recognise misplaced office items and assist with organising approved objects.

For example, the proposed robot could identify a disconnected computer mouse, navigate towards it and move it to a designated collection tray after staff approval.

The proposed robotic system would require:

- Object recognition
- Perception
- Localisation
- Navigation
- Object manipulation
- Safety controls

The physical robot has not been built or tested.

The implemented part of this project is an image-classification system that can recognise six types of office objects.

### Implemented Features

- Image classification using MobileNetV2
- Prediction using an image file
- Live webcam input
- Automatic image classification
- Manual image classification
- A simple Tkinter graphical interface
- Display of the predicted class and model scores
- Recorded evaluation results and confusion matrix

### Features Not Implemented

- Physical robot navigation
- Robotic arm manipulation
- Object pickup and placement
- Multiple-object detection
- Object localisation
- Autonomous robot operation

The classifier predicts the class of an image. It does not determine an object's position or whether an object belongs to someone.

## 2. Object Classes

The final Office-31 classifier recognises six categories:

1. Bottle
2. Calculator
3. Keyboard
4. Mouse
5. Mug
6. Stapler

The original pilot experiment used a slightly different set of classes.

| Experiment | Classes | Images |
|---|---|---:|
| Original pilot | Book, Bottle, Keyboard, Mouse, Mug, Stapler | 90 |
| Office-31 classifier | Bottle, Calculator, Keyboard, Mouse, Mug, Stapler | 415 |

Calculator replaces Book in the Office-31 model.

The two experiments use different datasets and class mappings. Their models must not be interchanged.

## 3. Recorded Results

The project includes the original personal-photo experiment and the later Office-31 experiment.

| Experiment | Dataset split | Accuracy | Macro F1 |
|---|---|---:|---:|
| Original baseline | Validation | 76.67% | 0.7286 |
| Original augmentation | Validation | 73.33% | 0.7063 |
| Original baseline | Test | 50.00% | 0.4850 |
| Office-31 | Validation | 98.41% | 0.9872 |
| Office-31 | Test | 96.72% | 0.9744 |

The Office-31 model correctly classified 59 out of 61 test images.

The two incorrect predictions were mouse images classified as staplers.

These results apply to the recorded Office-31 test dataset, not to the live webcam.

The original pilot and Office-31 results are not directly comparable because their datasets, classes and evaluation conditions differ.

## 4. Requirements

The project was developed and tested using Windows and Python.

### Software

- Python 3.12
- Visual Studio Code
- Git
- OpenCV
- NumPy
- TensorFlow
- Keras
- Pillow
- Scikit-learn
- Tkinter

The project uses TensorFlow 2.21.0 for the recorded Office-31 experiment.

A dedicated GPU is not required.

The webcam interface requires access to a working computer camera.

Python dependencies are recorded in `requirements.txt`.

Git can be downloaded from:

https://git-scm.com/downloads

The documented environment setup uses `uv`:

https://docs.astral.sh/uv/getting-started/installation/

## 5. Installation

### Step 1 — Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/khrisvin/PDE3802-Office-Classifier.git
```

Open the project folder:

```powershell
cd PDE3802-Office-Classifier
```

The GUI is a development feature. Use a branch that contains `live_camera_gui.py`.

After the updated code has been pushed, the development branch can be selected with:

```powershell
git switch cleanup/source-review
```

The previously tagged `assessment1-final` version contains the original independently tested classifier and does not include the later GUI feature.

### Step 2 — Create the Python Environment

```powershell
uv venv --python 3.12 .venv
```

Install the project dependencies:

```powershell
uv pip install --python ".\.venv\Scripts\python.exe" -r requirements.txt
```

Check the installed dependencies:

```powershell
uv pip check --python ".\.venv\Scripts\python.exe"
```

### Step 3 — Check the Libraries

```powershell
.\.venv\Scripts\python.exe -c "import cv2, numpy, tensorflow, PIL, tkinter; print('All libraries available')"
```

If the installation is successful, the terminal should display:

```text
All libraries available
```

TensorFlow may display informational CPU or oneDNN messages during startup.

The virtual environment can be used directly without activating it.

## 6. Office-31 Dataset

The final model uses a reviewed subset of the Office-31 dataset.

The images were obtained from:

https://zenodo.org/records/14751811

The archive used was:

`Dataset_Office-31.tar.gz`

Only images from the Amazon domain were used.

### Dataset Preparation

The six selected categories initially contained 523 candidate images.

After reviewing images and excluding unsuitable or duplicate examples, 415 images were retained.

The dataset was divided into:

- 291 training images
- 63 validation images
- 61 test images

### Dataset Distribution

| Class | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Bottle | 17 | 3 | 3 | 23 |
| Calculator | 63 | 14 | 14 | 91 |
| Keyboard | 38 | 8 | 8 | 54 |
| Mouse | 66 | 14 | 14 | 94 |
| Mug | 49 | 12 | 10 | 71 |
| Stapler | 58 | 12 | 12 | 82 |
| **Total** | **291** | **63** | **61** | **415** |

Related image groups were assigned to the same split where identified.

However, this does not guarantee that every visually related product or near-duplicate was detected.

### Downloading the Dataset

Download the Office-31 source archive.

Extract the Amazon folder into:

```text
data/external/office31/amazon/
```

The following example image should exist:

```text
data/external/office31/amazon/images/bottle/frame_0003.jpg
```

Check the path:

```powershell
Test-Path ".\data\external\office31\amazon\images\bottle\frame_0003.jpg"
```

The expected output is:

```text
True
```

### Restoring the Dataset

Run:

```powershell
.\.venv\Scripts\python.exe restore_office31_dataset.py
```

The script checks the original images against the recorded hashes and restores the selected image copies.

For a fresh restoration, the expected results include:

```text
Original images verified: 415
Existing image copies verified: 0
Missing image copies restored: 830
```

The 830 copies are two arrangements of the same 415 source images, not 830 independent photographs.

Running restoration again should report:

```text
Original images verified: 415
Existing image copies verified: 830
Missing image copies restored: 0
```

The recorded dataset assignments are preserved.

See `DATASET_CARD.md` for the full dataset description, exclusions, split details and limitations.

## 7. Image Preprocessing

The images are prepared before being passed to the neural network.

The preprocessing steps are:

1. Read the image using OpenCV.
2. Resize the image while preserving its aspect ratio.
3. Add black padding to produce a 224 × 224 image.
4. Convert the image from BGR to RGB.
5. Pass the image to the trained model.

The input image contains pixel values in the range 0–255.

The saved model already includes an internal rescaling layer that converts the values into the range -1 to 1.

A second rescaling operation should not be applied.

The same preprocessing is used for image-file predictions and the live-camera application.

### Class Order

| Label | Office-31 Class |
|---|---|
| 0 | Bottle |
| 1 | Calculator |
| 2 | Keyboard |
| 3 | Mouse |
| 4 | Mug |
| 5 | Stapler |

The saved model files are:

```text
outputs/office31_baseline/office_classifier.keras
outputs/office31_baseline/classes.json
```

## 8. Image Classification

The classifier can predict the category of an image supplied through the command line.

### Running an Image Prediction

```powershell
.\.venv\Scripts\python.exe predict_office31.py "data\office31_split\val\bottle\office31_amazon_frame_0013.jpg"
```

The recorded prediction is:

```text
Predicted class: bottle
Model score: approximately 99.30%
```

### Example of an Incorrect Prediction

```powershell
.\.venv\Scripts\python.exe predict_office31.py "data\office31_split\val\mouse\office31_amazon_frame_0038.jpg"
```

The recorded prediction is:

```text
Predicted class: stapler
Model score: approximately 67.51%
```

The actual class of this image is Mouse.

This illustrates that the classifier can make mistakes.

### Classify Another Image

```powershell
.\.venv\Scripts\python.exe predict_office31.py "C:\path\to\photo.jpg"
```

Replace the example path with the actual image location.

The program displays the predicted class and scores for all six categories.

The scores are not calibrated probabilities of correctness.

## 9. Live Camera Interface

A simple graphical interface was implemented using Tkinter and OpenCV.

The application allows users to classify office objects through a webcam without entering a command for every prediction.

### Features

- Live camera preview
- Automatic classification
- Manual classification
- Image-file selection
- Predicted class display
- Scores for all six categories
- Start and stop camera buttons

### Running the Interface

Open PowerShell in the project folder and run:

```powershell
.\.venv\Scripts\python.exe live_camera_gui.py
```

The application opens a basic window with a camera preview and prediction information.

The saved Office-31 model is loaded when the application starts.

### Using the Webcam

1. Open the application.
2. Click **Start Camera**.
3. Wait for the camera preview.
4. Place one office object inside the green square.
5. Allow the program to classify the object automatically.
6. Read the predicted class and model scores.

Automatic classification starts when the camera is opened.

The application attempts a new prediction approximately every two seconds, provided the previous prediction has finished.

### Manual Classification

The **Classify Now** button can be used to classify the current camera frame manually.

Automatic classification can be disabled using the checkbox.

### Opening an Image

The **Open Image** button allows the user to select an image stored on the computer.

The complete selected image is used for classification.

### How Automatic Classification Works

OpenCV reads frames from the webcam.

A square region in the centre of each frame is selected.

The image is resized, padded and converted to RGB.

The trained MobileNetV2 classifier processes the image and returns six class scores.

The class with the highest score is displayed.

The green square is a fixed classification region, not a detected object boundary.

The program does not use YOLO, object tracking or multiple-object detection.

### Camera Testing

The live-camera application was tested locally on a Windows computer.

The following functions were reported working:

- Application startup
- Loading the saved model
- Opening an image file
- Starting the webcam
- Automatic classification
- Manual classification
- Displaying class scores
- Stopping and restarting the webcam

Independent testing of the new interface is still outstanding.

The recorded 96.72% Office-31 test accuracy must not be interpreted as measured live-camera accuracy.

## 10. Model Architecture

The classifier uses transfer learning with MobileNetV2.

The model consists of:

- ImageNet-pretrained MobileNetV2 backbone
- Frozen pretrained feature extractor
- Global average pooling layer
- Dense classification layer
- Six softmax outputs

Only the final classification layer was trained.

The classification layer contains:

```text
1,280 × 6 weights + 6 biases = 7,686 parameters
```

### Training Configuration

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss function | Sparse categorical cross-entropy |
| Batch size | 8 |
| Maximum epochs | 30 |
| Random seed | 42 |
| Early stopping | Validation loss |
| Patience | 5 epochs |
| Best weights restored | Yes |
| Office-31 augmentation | None |
| Class weights | Based on training frequencies |

The training completed 30 epochs.

The best recorded validation loss was approximately 0.0401 at epoch 30.

Validation accuracy was 98.41%.

The saved model is included in the repository.

Retraining is not required to run predictions or the live-camera application.

### Training Command

```powershell
.\.venv\Scripts\python.exe train_office31.py
```

The training script prevents existing experiment results from being overwritten.

A new training experiment should preserve the original model, results and configuration.

## 11. Evaluation and Error Analysis

The Office-31 model was evaluated using the recorded validation and test splits.

The evaluation scripts save:

- Accuracy
- Precision
- Recall
- Macro-F1
- Classification reports
- Predictions for individual images
- Confusion matrices

The results are stored in:

```text
outputs/office31_baseline/validation_evaluation/
outputs/office31_baseline/test_evaluation/
```

### Evaluation Commands

```powershell
.\.venv\Scripts\python.exe evaluate_office31_validation.py
```

```powershell
.\.venv\Scripts\python.exe evaluate_office31_test.py
```

These scripts refuse to overwrite existing results.

To reproduce an evaluation, use a separate checkout and preserve the original output folders before running the scripts.

Repeating the same evaluation is a reproducibility check, not a new independent test dataset.

### Validation Results

- Total images: 63
- Correct predictions: 62
- Accuracy: 98.41%
- Macro-F1: 0.9872

One mouse validation image was incorrectly classified as a stapler.

### Test Results

- Total images: 61
- Correct predictions: 59
- Incorrect predictions: 2
- Accuracy: 96.72%
- Macro-F1: 0.9744

### Per-Class Results

| Class | Correct/Total | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Bottle | 3/3 | 1.0000 | 1.0000 | 1.0000 |
| Calculator | 14/14 | 1.0000 | 1.0000 | 1.0000 |
| Keyboard | 8/8 | 1.0000 | 1.0000 | 1.0000 |
| Mouse | 12/14 | 1.0000 | 0.8571 | 0.9231 |
| Mug | 10/10 | 1.0000 | 1.0000 | 1.0000 |
| Stapler | 12/12 | 0.8571 | 1.0000 | 0.9231 |

The two incorrectly classified test images were:

```text
test/mouse/office31_amazon_frame_0036.jpg
test/mouse/office31_amazon_frame_0079.jpg
```

Both were predicted as Stapler.

Possible visual similarities may contribute to these errors, but the exact causes have not been experimentally established.

See `ERROR_ANALYSIS.md` for further information.

## 12. Original Personal-Photo Dataset

The first experiment used 90 images photographed by the group.

The classes were:

- Book
- Bottle
- Keyboard
- Mouse
- Mug
- Stapler

The dataset contained 18 physical objects, with five photographs of each object.

The images were divided into:

- 30 training images
- 30 validation images
- 30 test images

Different views of the same physical object were kept within the same split.

The original images are stored in:

```text
data/raw/
```

### Preparing the Original Dataset

```powershell
.\.venv\Scripts\python.exe check_dataset.py
```

```powershell
.\.venv\Scripts\python.exe split_dataset.py
```

```powershell
.\.venv\Scripts\python.exe prepare_data.py
```

Only run the splitting script when `data/split/` does not already exist.

The recorded experiment files are located in:

```text
outputs/pilot/
outputs/augmented_v1/
```

The selected baseline achieved 50.00% accuracy and a macro-F1 score of 0.4850 on its test set.

The augmentation experiment did not improve the recorded validation accuracy.

The pilot experiment remains part of the project history and is separate from the Office-31 model used in the GUI.

## 13. Main Project Files

| File | Description |
|---|---|
| `predict_office31.py` | Predicts an object class from an image |
| `live_camera_gui.py` | Basic webcam and image-classification interface |
| `office31_data.py` | Shared Office-31 preprocessing and dataset helpers |
| `train_office31.py` | Trains the Office-31 model |
| `evaluate_office31_validation.py` | Evaluates validation images |
| `evaluate_office31_test.py` | Evaluates test images |
| `restore_office31_dataset.py` | Restores recorded dataset images |
| `build_clean_dataset.py` | Builds the reviewed dataset |
| `split_office31.py` | Creates the recorded dataset split |
| `prepare_office31.py` | Prepares external images |
| `plot_office31_learning_curves.py` | Generates training graphs |
| `DATASET_CARD.md` | Dataset information |
| `ERROR_ANALYSIS.md` | Model error analysis |
| `requirements.txt` | Required Python packages |

`office31_data.py` is part of the source-cleanup development branch.

The webcam GUI is a later development feature and is not included in the original `assessment1-final` tag.

## 14. Independent Testing

Pranay Poheith independently tested the original Office-31 classifier using a separate Windows computer.

The tested Git commit was:

```text
9aeaaf8b4a36497e788b83ed3c9253dcd989651f
```

On 4 November 2026, Pranay successfully reproduced the recorded Office-31 test evaluation.

The results were:

- Test images: 61
- Correct predictions: 59
- Accuracy: 96.72%
- Macro-F1: 0.9744
- Misclassified images: 2
- Evaluation exit code: 0

The model was not retrained or modified during this evaluation.

The independently tested version is identified by the Git tag:

```text
assessment1-final
```

The later source-cleanup branch and the new webcam interface are separate developments.

The webcam interface has been tested locally, but it has not yet undergone independent installation testing by another group member.

If the new interface becomes part of the submitted version, the updated code should be independently tested and identified using a new Git commit and submission tag.

## 15. Limitations

The classifier has several limitations.

- It recognises only six classes.
- It cannot detect multiple objects independently.
- It cannot determine an object's location in a scene.
- It does not reject unknown objects.
- Camera lighting and background can affect predictions.
- The Office-31 dataset contains mostly product photographs.
- The model may struggle with unfamiliar office environments.
- Some test classes contain few images.
- Similar objects may be confused.
- Model scores are not calibrated correctness probabilities.
- The model has not been integrated into a physical robot.

The green rectangle displayed in the webcam interface only indicates the selected image region.

It does not automatically detect or track the position of an object.

The system also does not infer object ownership or permission to move an item.

Future improvements could include a larger dataset, new independently collected webcam test images, unknown-object handling, object localisation and integration with a mobile robot.

## 16. Troubleshooting

### Python Package Not Found

Install the required dependencies:

```powershell
uv pip install --python ".\.venv\Scripts\python.exe" -r requirements.txt
```

### Webcam Does Not Open

Check that:

- A webcam is connected.
- Windows camera permissions are enabled.
- Other applications are not using the camera.
- The correct Python environment is being used.

In Windows, camera permissions can be checked under:

**Settings → Privacy & security → Camera**

The current GUI uses camera index 0.

### Saved Model Not Found

Check that these files exist:

```text
outputs/office31_baseline/office_classifier.keras
outputs/office31_baseline/classes.json
```

The GUI must be saved inside the main project folder.

### Missing Dataset Images

Download the original Office-31 archive and restore the selected images using:

```powershell
.\.venv\Scripts\python.exe restore_office31_dataset.py
```

Do not change recorded image hashes simply to bypass dataset verification.

### Incorrect Prediction

The model may make mistakes because of lighting, background, object appearance or similarities between categories.

Try placing one clearly visible object inside the green square.

An incorrect prediction does not necessarily indicate a programming error.

### Existing Evaluation Files

The evaluation scripts deliberately prevent existing results from being overwritten.

Preserve the recorded results and use a separate checkout for new reproduction tests.

### TensorFlow Warnings

TensorFlow may display information about CPU instructions, oneDNN or unavailable GPU support.

These messages do not necessarily indicate an error.

Check whether the program loads the model and produces a prediction successfully.

## 17. References

Office-31 dataset:

https://zenodo.org/records/14751811

TensorFlow transfer learning:

https://www.tensorflow.org/tutorials/images/transfer_learning

Keras MobileNet documentation:

https://keras.io/api/applications/mobilenet/

OpenCV documentation:

https://docs.opencv.org/

Scikit-learn evaluation metrics:

https://scikit-learn.org/stable/modules/model_evaluation.html

Python Tkinter documentation:

https://docs.python.org/3/library/tkinter.html

uv documentation:

https://docs.astral.sh/uv/

## 18. Dataset Sources and Reuse

The original 90-image dataset was collected by the group.

The Office-31 images were downloaded from an external research dataset and reviewed for the coursework experiment.

The downloaded images must not be described as photographs taken by the group.

The source archive and applicable reuse conditions should be consulted before redistributing the external images.

The GitHub repository documents the image source, reviewed selections, split assignments and dataset restoration process.

ImageNet pretraining is separate from the two coursework datasets.
