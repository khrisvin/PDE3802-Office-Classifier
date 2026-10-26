# PDE3802 Office Classifier

Group coursework for Artificial Intelligence in Robotics.

## Team

- Khrisvin Nepaulsing — M01003462
- Veedoor Gopal
- Pranay Poheith

## Project overview

This project classifies one office item in a photograph into one of
six classes: book, bottle, keyboard, mouse, mug or stapler.

It uses MobileNetV2 pretrained on ImageNet as a frozen feature extractor,
followed by a trainable six-class classification layer.

The wider autonomous robot design is documented separately. This
implementation does not perform object localisation, navigation or grasping.

## Current status

- Original dataset collected and checked.
- Physical-object training, validation and test splits implemented.
- Image preprocessing implemented.
- Baseline and augmentation experiments completed.
- Single-image prediction implemented.
- Validation and held-out test evaluation completed.
- Models, metrics, predictions and learning curves saved.

The selected baseline achieved 50.00% accuracy and 0.4850 macro F1
on the held-out test set. This is a pilot prototype with limited
generalisation, not a reliable deployment-ready classifier.

## Dataset

The dataset contains 90 original photographs across six classes.

Each group member photographed one different physical object per class,
taking five photographs of each object.

| Split | Images per class | Total images | Physical objects |
|---|---:|---:|---:|
| Training | 5 | 30 | 6 |
| Validation | 5 | 30 | 6 |
| Test | 5 | 30 | 6 |
| Total | 15 | 90 | 18 |

All photographs of the same physical object remain in one split.

Original images are stored in `data/raw/<class>/`.
The generated split is stored in `data/split/<split>/<class>/`.

See [DATASET_CARD.md](DATASET_CARD.md) for collection details,
split assignments and limitations.

## Setup on Windows

Install Git and uv, then run these commands in PowerShell:

```powershell
git clone https://github.com/khrisvin/PDE3802-Office-Classifier.git
cd PDE3802-Office-Classifier
uv venv --python 3.12 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

If the project is already on your computer, run the environment commands
from the existing project folder instead of cloning it again.

The recorded experiments used TensorFlow 2.21.0 on CPU.
Installed package versions are recorded in `requirements.txt`.

## Check and prepare the dataset

Check the original photographs:

```powershell
& ".\.venv\Scripts\python.exe" check_dataset.py
```

Expected result:

- 15 images per class.
- 90 images in total.
- 90 readable images.
- Zero unreadable images.

Create the fixed pilot split:

```powershell
& ".\.venv\Scripts\python.exe" split_dataset.py
```

Run this only when `data/split/` does not already exist.
The script refuses to overwrite an existing split and does not modify
the original images.

The split directory is excluded from Git and must be generated after
a fresh clone if evaluation or training is required.

Check the prepared training arrays:

```powershell
& ".\.venv\Scripts\python.exe" prepare_data.py
```

Expected output includes:

```text
Images shape: (30, 224, 224, 3)
Labels shape: (30,)
Pixel data type: uint8
Label data type: int32
Colour order: RGB
First label: 0
```

## Preprocessing and class labels

Images are resized proportionally and padded with black pixels to
224 × 224. OpenCV's BGR channel order is converted to RGB.

The data loader returns uint8 pixel values in the range 0–255.
A layer inside the model scales these values to −1–1 before MobileNetV2.
Do not scale the images again before passing them to the saved model.

| Class | Numeric label |
|---|---:|
| book | 0 |
| bottle | 1 |
| keyboard | 2 |
| mouse | 3 |
| mug | 4 |
| stapler | 5 |

The saved `classes.json` file preserves the output class order.

## Predict one image

The saved baseline model is included, so retraining is not required.

After generating the dataset split, run:

```powershell
& ".\.venv\Scripts\python.exe" predict_image.py "data\split\val\book\pranay_book01_001.jpeg"
```

The recorded prediction for this example is `book`, with a model score
of approximately 59.43%. Small numerical differences may occur.

To classify another photograph:

```powershell
& ".\.venv\Scripts\python.exe" predict_image.py "C:\path\to\photo.jpeg"
```

The program prints the predicted class and all six class scores.

Use an image containing one clearly visible target item.
The classifier always selects one of its six classes. It does not
currently reject unknown objects or locate multiple items separately.

A model score is not a guarantee that the prediction is correct.

## Baseline model

The baseline consists of:

- MobileNetV2 pretrained on ImageNet, with its original classifier removed.
- Frozen MobileNetV2 weights.
- Global average pooling.
- A six-class dense layer with softmax activation.

Only the final classification layer is trained:
1,280 × 6 weights plus 6 biases = 7,686 trainable parameters.

Training settings:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Sparse categorical cross-entropy |
| Batch size | 8 |
| Maximum epochs | 30 |
| Random seed | 42 |
| Early-stopping monitor | Validation loss |
| Early-stopping patience | 5 epochs |
| Restore best weights | Yes |

Build and inspect the model:

```powershell
& ".\.venv\Scripts\python.exe" model.py
```

Train the baseline:

```powershell
& ".\.venv\Scripts\python.exe" train.py
```

The first run may download pretrained MobileNetV2 weights.

Outputs are saved in `outputs/pilot/`.
Running `train.py` again overwrites the baseline model and training history.
Preserve the recorded artifacts before retraining. Existing evaluation
files would not automatically update to match a newly trained model.

## Augmentation experiment

The augmentation experiment uses the same original data split and
training settings, with:

- Random rotations up to ±18 degrees.
- Small random zooms that preserve image proportions.
- Augmentation active during training only.

It starts with a fresh classification layer and the pretrained
MobileNetV2 feature extractor.

```powershell
& ".\.venv\Scripts\python.exe" train_augmented.py
```

Outputs are saved in `outputs/augmented_v1/`.

The script refuses to run when that experiment folder already exists.
Recorded artifacts are included. Use a separate experiment folder
if conducting another run.

Augmentation produces transformed views of existing training images.
It does not add new physical objects or increase the original dataset
count of 90 photographs.

## Validation evaluation and model selection

```powershell
& ".\.venv\Scripts\python.exe" evaluate_validation.py
& ".\.venv\Scripts\python.exe" evaluate_augmented.py
```

These scripts load saved models without retraining and evaluate the
unchanged validation set.

| Experiment | Selected epoch | Validation accuracy | Macro F1 | Errors |
|---|---:|---:|---:|---:|
| Baseline | 30 | 76.67% | 0.7286 | 7/30 |
| Rotation and zoom augmentation | 23 | 73.33% | 0.7063 | 8/30 |

The augmentation run completed 28 epochs and restored epoch 23,
which had its lowest validation loss.

The baseline was selected before test evaluation because it achieved
higher validation accuracy and macro F1.

These results show that the tested augmentation settings did not
improve this experiment. They do not establish that augmentation
is generally ineffective.

Both models classified all five validation stapler photographs as mouse.

## Learning curves

Generate graphs from the saved training histories:

```powershell
& ".\.venv\Scripts\python.exe" plot_learning_curves.py
```

This does not retrain the models.

Each experiment folder contains `learning_curves.png`.
The dashed line marks the epoch with the lowest validation loss.

Training accuracy in the augmented experiment is measured on transformed
images. Validation accuracy is measured on unchanged images.

## Held-out test evaluation

The selected baseline was evaluated on the test set after model selection.

| Metric | Result |
|---|---:|
| Test images | 30 |
| Correct predictions | 15 |
| Incorrect predictions | 15 |
| Accuracy | 50.00% |
| Macro F1 | 0.4850 |

| Class | Correct / total | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| book | 1/5 | 1.0000 | 0.2000 | 0.3333 |
| bottle | 2/5 | 0.6667 | 0.4000 | 0.5000 |
| keyboard | 5/5 | 1.0000 | 1.0000 | 1.0000 |
| mouse | 2/5 | 0.2500 | 0.4000 | 0.3077 |
| mug | 5/5 | 0.6250 | 1.0000 | 0.7692 |
| stapler | 0/5 | 0.0000 | 0.0000 | 0.0000 |

Recorded test artifacts are in `outputs/pilot/test_evaluation/`:

- `test_metrics.json`
- `test_report.txt`
- `test_predictions.csv`
- `test_confusion_matrix.png`

The metrics include the evaluated model's SHA-256 fingerprint.

The evaluation command is:

```powershell
& ".\.venv\Scripts\python.exe" evaluate_test.py
```

The script refuses to overwrite an existing test-results folder.
The recorded results are already included in the repository.

To independently reproduce the evaluation, first preserve the existing
results outside that folder and use the unchanged saved baseline model
and fixed split. Reproduction must not be confused with tuning against
test results.

## Error analysis

Accuracy fell from 76.67% on validation to 50.00% on test.

The test errors were:

- Four book photos predicted as mouse, bottle or stapler.
- Three bottle photos predicted as mug.
- Three mouse photos predicted as stapler.
- All five stapler photos predicted as mouse.

Keyboard and mug each achieved five correct predictions, but each
class's five test images depict only one physical object.

The results indicate limited generalisation in this pilot.
Limited object diversity and differences in background, lighting,
framing or object appearance are possible contributors. Their
individual effects have not been isolated experimentally.

## Limitations and future work

- Only one physical object per class is available for training.
- Views of the same object are correlated observations.
- The evaluation covers few distinct objects.
- Unknown objects are forced into one of the six classes.
- Image classification alone cannot determine object position or grasp safety.
- The prototype has no implemented robot control or safety system.

Future work should collect more distinct physical objects and reserve
a fresh independent test set before further model development.

The existing test set has now been inspected. It must not be described
as untouched in subsequent development experiments.

## Main files

| File | Purpose |
|---|---|
| `check_image.py` | Preview and inspect an example image |
| `check_dataset.py` | Count and check original images |
| `split_dataset.py` | Reproduce the fixed physical-object split |
| `dataset.py` | Collect image paths and labels |
| `preprocessing.py` | Resize and pad images |
| `prepare_data.py` | Load RGB image arrays and numeric labels |
| `model.py` | Build the baseline model |
| `train.py` | Train the baseline |
| `train_augmented.py` | Train the augmentation experiment |
| `evaluate_validation.py` | Evaluate baseline validation performance |
| `evaluate_augmented.py` | Evaluate augmented validation performance |
| `evaluate_test.py` | Evaluate the selected baseline on test data |
| `predict_image.py` | Classify one image file |
| `plot_learning_curves.py` | Plot saved training histories |
| `DATASET_CARD.md` | Document the dataset and its limitations |

## Troubleshooting

- Missing package: install `requirements.txt` using the project interpreter.
- Missing image: check the path, filename and extension.
- Missing split: run `split_dataset.py` once.
- Existing split: reuse it instead of rerunning the split script.
- Missing model: check `outputs/pilot/office_classifier.keras`.
- Existing experiment folder: preserve its contents and use a new folder
  for a new experiment.
- Existing test-results folder: review the saved results; the script
  intentionally prevents overwriting.
- oneDNN or CPU information messages alone do not indicate failure.
- The recorded setup uses CPU execution on native Windows.
- An optimizer-state loading warning may appear for the augmented model.
  Prediction completed in the recorded evaluation; this does not verify
  that its saved optimizer state is suitable for resuming training.

## References

- TensorFlow transfer learning:
  https://www.tensorflow.org/tutorials/images/transfer_learning
- Keras MobileNet models:
  https://keras.io/api/applications/mobilenet/
- OpenCV documentation:
  https://docs.opencv.org/
- Scikit-learn classification metrics:
  https://scikit-learn.org/stable/modules/model_evaluation.html