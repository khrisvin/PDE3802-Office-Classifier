# PDE3802 Office Classifier

Group coursework for Artificial Intelligence in Robotics.

## Team

- Khrisvin Nepaulsing — M01003462
- Veedoor Gopal
- Pranay Poheith

## Project

This project classifies a single office item in a photograph into one of
six classes: book, bottle, keyboard, mouse, mug or stapler.

The implemented classifier uses a pretrained MobileNetV2 feature extractor
and a trainable six-class classification layer.

The wider robot design is documented separately. This code does not
implement object localisation, navigation or grasping.

## Dataset

The pilot dataset contains 90 original photographs: 15 per class.

Each class contains three different physical objects, with five photos
per object. All photos of the same physical object stay in one split.

| Split | Images per class | Total images |
|---|---:|---:|
| Training | 5 | 30 |
| Validation | 5 | 30 |
| Test | 5 | 30 |

Original photographs are stored in `data/raw/<class>/`.

`split_dataset.py` reproduces the fixed pilot split in `data/split/`.
The generated split folders are excluded from Git.

The test set has not yet been evaluated.

## Setup on Windows

Install Git and uv, then open PowerShell and run:

```powershell
git clone https://github.com/khrisvin/PDE3802-Office-Classifier.git
cd PDE3802-Office-Classifier
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

If the project is already on your computer, run the environment commands
from its existing folder instead of cloning it again.

The recorded environment uses TensorFlow 2.21.0 and CPU execution.
Package versions are recorded in `requirements.txt`.

## Check and prepare the dataset

```powershell
& ".\.venv\Scripts\python.exe" check_dataset.py
& ".\.venv\Scripts\python.exe" split_dataset.py
& ".\.venv\Scripts\python.exe" prepare_data.py
```

Expected dataset check: 90 readable images and zero unreadable images.

Run `split_dataset.py` only when `data/split/` does not already exist.
The script refuses to overwrite an existing split.

Expected prepared training data:

- Images: `(30, 224, 224, 3)`
- Labels: `(30,)`
- Colour order: RGB
- Pixel type: uint8
- Label type: int32

Images are resized proportionally and padded to 224 × 224.
Pixel scaling from 0–255 to −1–1 happens inside the model.

## Predict one image

The saved baseline model is included, so retraining is not required.

```powershell
& ".\.venv\Scripts\python.exe" predict_image.py "data\split\val\book\pranay_book01_001.jpeg"
```

Expected predicted class for this validation example: `book`.

To classify another photo:

```powershell
& ".\.venv\Scripts\python.exe" predict_image.py "C:\path\to\photo.jpeg"
```

The program prints the predicted class and all six model scores.
A high score does not guarantee a correct prediction.

The classifier always selects one of its six known classes. It does not
currently reject unknown objects or handle multiple items separately.

## Training experiments

### Baseline

```powershell
& ".\.venv\Scripts\python.exe" train.py
```

- Frozen MobileNetV2 pretrained on ImageNet.
- Global average pooling and six-class softmax output.
- 7,686 trainable parameters.
- Adam learning rate: 0.001.
- Batch size: 8.
- Maximum epochs: 30.
- Early stopping monitors validation loss with patience 5.
- Best weights are restored.

The first training run may download the pretrained MobileNetV2 weights.

Outputs are saved in `outputs/pilot/`.
Running this script again overwrites the saved baseline artifacts.
Preserve the existing results before retraining.

### Rotation and zoom augmentation

```powershell
& ".\.venv\Scripts\python.exe" train_augmented.py
```

This experiment adds random rotations up to ±18 degrees and small
proportional zooms during training only. It uses the same original
training images and unchanged validation split.

Outputs are saved in `outputs/augmented_v1/`.
The script refuses to run if that experiment folder already exists.
The recorded results are included; use a new output folder for another run.

## Validation evaluation

```powershell
& ".\.venv\Scripts\python.exe" evaluate_validation.py
& ".\.venv\Scripts\python.exe" evaluate_augmented.py
```

Each evaluation saves:

- Validation accuracy and macro F1.
- Per-class classification report.
- Confusion matrix image.
- Per-image predictions.
- Misclassified filenames printed in the terminal.

| Experiment | Validation accuracy | Macro F1 | Errors |
|---|---:|---:|---:|
| Baseline | 76.67% | 0.7286 | 7/30 |
| Rotation and zoom augmentation | 73.33% | 0.7063 | 8/30 |

The baseline is currently selected because it performed better on both
validation accuracy and macro F1.

The augmentation run stopped after epoch 28 and restored epoch 23,
which had its lowest validation loss.

## Limitations

This is a small pilot dataset with only one physical training object
per class. Five views of one object are not five independent objects.

Both models misclassified all five validation stapler images as mouse.
Those five images show one physical stapler.

Augmentation did not improve this recorded experiment. It adds variation
to existing images but does not introduce new physical object designs.

Validation results were used for model selection. They are not final
test results or proof of reliable performance across office environments.

## Troubleshooting

- Missing package: install `requirements.txt` using the project interpreter.
- Missing image: check the path, filename and extension.
- Missing split: run `split_dataset.py` once.
- Existing split: reuse it instead of rerunning the split script.
- Missing model: check `outputs/pilot/office_classifier.keras`.
- oneDNN or CPU information messages: these alone do not indicate failure.
- Native Windows GPU warning: the recorded project runs on CPU.