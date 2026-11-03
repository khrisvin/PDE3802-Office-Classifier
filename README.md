# PDE3802 Office Classifier

Group coursework for PDE3802 Artificial Intelligence in Robotics at Middlesex University Mauritius.

This project implements six-class, single-item image classification using transfer learning with MobileNetV2.

## Team

- Khrisvin Nepaulsing - M01003462
- Veedoor Gopal
- Pranay Poheith

## Project purpose

The proposed application is an assistant for rented shared offices and meeting rooms.

Between bookings, with staff authorisation and the room empty, the proposed robot would support checking and organising approved office equipment. It could report misplaced items and situations requiring staff attention.

The implementation in this repository is the image-classification component. It predicts a category for an entire image.

Object detection, ownership identification, position estimation, navigation and manipulation are proposed functions. They have not been demonstrated by this classifier.

A class prediction does not establish ownership or permission to move an object. Personal belongings, connected equipment and items with uncertain contents would require staff review.

Faster room turnaround and reduced staff workload are proposed benefits, not measured outcomes.

## Experiments

Two separate datasets and experiment histories are preserved.

### Current Office-31 experiment

The current model recognises:

- bottle
- calculator
- keyboard
- mouse
- mug
- stapler

It uses 415 reviewed images from the Amazon domain of Office-31.

### Original personal-photo pilot

The original pilot recognises:

- book
- bottle
- keyboard
- mouse
- mug
- stapler

It uses 90 photographs collected by the group.

Calculator replaces book in the Office-31 experiment. The models and their class mappings must not be interchanged.

## Recorded results

| Dataset and experiment | Split | Correct / total | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| Personal-photo baseline | Validation | 23/30 | 76.67% | 0.7286 |
| Personal-photo augmentation | Validation | 22/30 | 73.33% | 0.7063 |
| Selected personal-photo baseline | Test | 15/30 | 50.00% | 0.4850 |
| Office-31 baseline | Validation | 62/63 | 98.41% | 0.9872 |
| Office-31 baseline | Test | 59/61 | 96.72% | 0.9744 |

The personal-photo and Office-31 test scores are not directly comparable. The experiments use different images, class sets, split methods and capture conditions.

The Office-31 result measures performance on a reviewed subset of product photographs. It does not establish performance on cluttered office scenes, live camera input or a physical robot.

Both test sets have been evaluated and their errors inspected. Neither should be described as untouched in future development.

## Requirements

The documented setup uses:

- Windows PowerShell.
- Git.
- uv.
- Python 3.12, selected through uv.
- Python packages listed in `requirements.txt`.

The recorded experiments used TensorFlow 2.21.0 on CPU. A GPU is not required for the documented prediction workflow.

Official installation instructions:

- Git: https://git-scm.com/downloads
- uv: https://docs.astral.sh/uv/getting-started/installation/

For a private repository, the person cloning it needs access through their own GitHub account.

## Setup on Windows

Open PowerShell and check that Git and uv are available:

```powershell
git --version
uv --version
```

Clone the repository into a new folder:

```powershell
git clone https://github.com/khrisvin/PDE3802-Office-Classifier.git
Set-Location ".\PDE3802-Office-Classifier"
```

Create the virtual environment and install dependencies:

```powershell
uv venv --python 3.12 .venv

uv pip install --python ".\.venv\Scripts\python.exe" -r requirements.txt
```

Check dependency consistency:

```powershell
uv pip check --python ".\.venv\Scripts\python.exe"
```

Check the main imports:

```powershell
& ".\.venv\Scripts\python.exe" -c "import tensorflow as tf; import cv2; import numpy as np; print('TensorFlow:', tf.__version__); print('OpenCV:', cv2.__version__); print('NumPy:', np.__version__)"
```

The commands in this README use the virtual environment's Python directly. Activating the environment is optional.

Do not copy another person's `.venv` folder. Each computer should create its own environment.

## Restore the Office-31 dataset

Downloaded Office-31 photographs and their generated copies are excluded from Git.

The repository includes the manifests, review records and restoration script needed to recreate the recorded dataset from the source images.

### 1. Download the source archive

Download `Dataset_Office-31.tar.gz` from:

https://zenodo.org/records/14751811

### 2. Extract the images

Place the extracted `amazon` folder at:

```text
data/external/office31/amazon/
```

This example file must exist:

```text
data/external/office31/amazon/images/bottle/frame_0003.jpg
```

Check the path:

```powershell
Test-Path ".\data\external\office31\amazon\images\bottle\frame_0003.jpg"
```

Expected output:

```text
True
```

Avoid accidentally creating an extra nested `amazon/amazon/` folder.

The archive also contains DSLR and webcam domains. This experiment uses only the Amazon domain.

### 3. Restore the recorded image copies

Run:

```powershell
& ".\.venv\Scripts\python.exe" restore_office31_dataset.py
```

The script reads the committed manifests and:

- Verifies the SHA-256 hashes of the 415 required original images.
- Restores missing clean and split image copies.
- Verifies existing copies rather than overwriting them.
- Rejects files whose hashes do not match.
- Preserves recorded split assignments and experiment files.

When both destination image collections are missing, the expected output includes:

```text
Original images verified: 415
Existing image copies verified: 0
Missing image copies restored: 830
```

The 830 copies consist of 415 clean images and the same 415 images distributed across training, validation and test folders. They are not 830 independent examples.

### 4. Check repeat restoration

Run the restoration command again:

```powershell
& ".\.venv\Scripts\python.exe" restore_office31_dataset.py
```

Once all copies exist and match their hashes, the expected output includes:

```text
Original images verified: 415
Existing image copies verified: 830
Missing image copies restored: 0
```

Missing-image restoration was exercised in a separate temporary directory on the development computer.

Use the restoration script after cloning. Do not rerun cleaning or splitting just to recreate missing image files. The original construction scripts refuse existing destination directories, including directories already containing committed manifests.

## Office-31 dataset composition

The six selected Amazon categories originally contained 523 images. Review and duplicate exclusions removed 108 images, leaving 415.

| Class | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| bottle | 17 | 3 | 3 | 23 |
| calculator | 63 | 14 | 14 | 91 |
| keyboard | 38 | 8 | 8 | 54 |
| mouse | 66 | 14 | 14 | 94 |
| mug | 49 | 12 | 10 | 71 |
| stapler | 58 | 12 | 12 | 82 |
| **Total** | **291** | **63** | **61** | **415** |

Recorded visual groups remain within one split.

These groups represent visually related products, not verified physical-object identities. No recorded group crosses splits, but this does not prove that every near-duplicate or related product has been identified.

The manifests record source paths, image hashes and group information. The split manifest also records the assigned split and destination path.

See [DATASET_CARD.md](DATASET_CARD.md) for sources, review scope, grouping and limitations.

## Preprocessing and class order

Images are:

1. Read using OpenCV.
2. Resized proportionally and padded with black pixels to 224 x 224.
3. Converted from BGR to RGB.
4. Supplied to the model as pixel values from 0 to 255.

An internal model layer scales pixels to the range -1 to 1. Do not apply this scaling a second time.

Single-image prediction uses the same resizing and colour conversion.

| Numeric label | Office-31 model | Original pilot model |
|---|---|---|
| 0 | bottle | book |
| 1 | calculator | bottle |
| 2 | keyboard | keyboard |
| 3 | mouse | mouse |
| 4 | mug | mug |
| 5 | stapler | stapler |

Each experiment's `classes.json` records its output order.

## Predict one image with the Office-31 model

The saved model is included, so retraining is unnecessary.

The Office-31 model and class mapping are stored at:

```text
outputs/office31_baseline/office_classifier.keras
outputs/office31_baseline/classes.json
```

### Known correct example

After restoring the dataset, run:

```powershell
& ".\.venv\Scripts\python.exe" predict_office31.py "data\office31_split\val\bottle\office31_amazon_frame_0013.jpg"
```

Recorded result:

```text
Predicted class: bottle
Model score: approximately 99.30%
```

### Known incorrect example

Run:

```powershell
& ".\.venv\Scripts\python.exe" predict_office31.py "data\office31_split\val\mouse\office31_amazon_frame_0038.jpg"
```

Recorded result:

```text
Predicted class: stapler
Model score: approximately 67.51%
```

The image is correctly labelled as mouse. It was retained because a model mistake is not a reason to remove a valid evaluation image.

Small numerical differences may occur across environments.

### Another image

```powershell
& ".\.venv\Scripts\python.exe" predict_office31.py "C:\path\to\photo.jpg"
```

Replace the example path with the actual image path.

The program prints the predicted class and all six scores.

Use an image with one clearly visible target item. The classifier always selects one of its six classes; it has no unknown-object rejection mechanism.

Scores are not calibrated probabilities of correctness. A high score does not guarantee a correct prediction.

The two documented examples demonstrate program behaviour. They are not a new independent accuracy evaluation.

## Office-31 model and training

Architecture:

- ImageNet-pretrained MobileNetV2 without its original classification head.
- Frozen backbone, called in inference mode.
- Global average pooling.
- Dense layer with six softmax outputs.

Only the classification layer is trained:

```text
1,280 x 6 weights + 6 biases = 7,686 trainable parameters
```

| Setting | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 0.001 |
| Loss | Sparse categorical cross-entropy |
| Batch size | 8 |
| Maximum epochs | 30 |
| Random seed | 42 |
| Early-stopping monitor | Validation loss |
| Patience | 5 epochs |
| Best weights restored | Yes |
| Augmentation | None |
| Training class weights | N / (6 x class training count) |

Class weights are calculated from the training split only.

Training loss is class-weighted; validation loss is unweighted.

The recorded run completed 30 epochs. Epoch 30 had the lowest validation loss, approximately 0.0401, with 98.41% validation accuracy.

The training command is:

```powershell
& ".\.venv\Scripts\python.exe" train_office31.py
```

The script refuses to overwrite `outputs/office31_baseline/`.

The recorded model already exists, so training is not required for prediction, dataset restoration or fresh installation testing.

For a separate training experiment, explicitly configure a new output directory and preserve the recorded model, configuration and evaluations. A fresh clone alone does not provide an empty experiment directory because the saved artifacts are included.

Downloading pretrained weights may be necessary for a new training run.

A fixed seed does not guarantee bit-identical training across platforms and library versions.

## Learning curves

Generate the Office-31 chart from its saved history:

```powershell
& ".\.venv\Scripts\python.exe" plot_office31_learning_curves.py
```

Output:

```text
outputs/office31_baseline/learning_curves.png
```

This does not retrain the model.

Training accuracy reached 100% at epoch 6. Validation accuracy finished at 98.41%, while validation loss continued decreasing.

Training and validation loss magnitudes are not directly comparable because only training loss uses class weights.

![Office-31 learning curves](outputs/office31_baseline/learning_curves.png)

## Office-31 evaluation

Evaluation scripts load the saved model without retraining:

```powershell
& ".\.venv\Scripts\python.exe" evaluate_office31_validation.py

& ".\.venv\Scripts\python.exe" evaluate_office31_test.py
```

The repository already contains the recorded evaluation results:

```text
outputs/office31_baseline/validation_evaluation/
outputs/office31_baseline/test_evaluation/
```

The evaluation scripts refuse to overwrite existing result directories. Therefore, the commands above are not necessary for the normal prediction workflow.

For independent reproduction:

1. Use a separate checkout.
2. Preserve the existing evaluation folders in a backup location outside their original paths.
3. Keep the saved model and split manifests unchanged.
4. Restore the dataset.
5. Run validation evaluation first.
6. Run test evaluation second.
7. Compare the reproduced results with the preserved records.

The test script checks that the saved model and split-manifest fingerprints match the validation record.

Each evaluation produces metrics, a classification report, per-image predictions and a confusion matrix.

Reproducing the same evaluation is a reproducibility check. It does not create a new untouched test set.

### Validation results

- Images: 63.
- Correct predictions: 62.
- Accuracy: 98.41%.
- Macro F1: 0.9872.
- Error: mouse `frame_0038` predicted as stapler.

### Test results

| Class | Correct / total | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| bottle | 3/3 | 1.0000 | 1.0000 | 1.0000 |
| calculator | 14/14 | 1.0000 | 1.0000 | 1.0000 |
| keyboard | 8/8 | 1.0000 | 1.0000 | 1.0000 |
| mouse | 12/14 | 1.0000 | 0.8571 | 0.9231 |
| mug | 10/10 | 1.0000 | 1.0000 | 1.0000 |
| stapler | 12/12 | 0.8571 | 1.0000 | 0.9231 |

Overall accuracy: 96.72% (59/61).

Macro F1: 0.9744.

Both test mistakes were mouse images predicted as stapler:

- `test/mouse/office31_amazon_frame_0036.jpg`
- `test/mouse/office31_amazon_frame_0079.jpg`

The model was not retrained or modified during this evaluation.

Shapes and backgrounds may contribute to mistakes, but their causes have not been verified experimentally.

See [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md).

## Original personal-photo pilot

The original dataset contains 90 personally captured images of 18 physical objects across six classes:

- book
- bottle
- keyboard
- mouse
- mug
- stapler

Each contributor photographed one distinct object per class from five views.

All views of a physical object stay in one split. Each split contains 30 images: five images of one physical object per class.

Contributors appear across the splits through different classes and objects. This is not a photographer-disjoint split.

Original images are in:

```text
data/raw/
```

To prepare the pilot after cloning:

```powershell
& ".\.venv\Scripts\python.exe" check_dataset.py

& ".\.venv\Scripts\python.exe" split_dataset.py

& ".\.venv\Scripts\python.exe" prepare_data.py
```

Run `split_dataset.py` only when `data/split/` does not already exist.

The baseline and rotation/zoom experiments are preserved in:

```text
outputs/pilot/
outputs/augmented_v1/
```

The augmented run completed 28 epochs and selected epoch 23 by validation loss.

Its validation accuracy was 73.33%, compared with 76.67% for the baseline. The baseline was selected before test evaluation.

The selected baseline achieved 50.00% test accuracy and 0.4850 macro F1. All five test stapler images were predicted as mouse.

These results demonstrate limited generalisation in the pilot. The tested augmentation settings did not improve the recorded validation run.

To use the original model:

```powershell
& ".\.venv\Scripts\python.exe" predict_image.py "data\split\val\book\pranay_book01_001.jpeg"
```

To plot its saved histories:

```powershell
& ".\.venv\Scripts\python.exe" plot_learning_curves.py
```

Historical training scripts are:

```text
train.py
train_augmented.py
```

The pilot `train.py` now refuses an existing experiment folder. Pilot validation scripts also refuse existing evaluation artifacts. Preserve historical results when preparing any separate run.

## Main Office-31 files

| File | Purpose |
|---|---|
| `requirements.txt` | Record Python dependencies |
| `prepare_office31.py` | Initial image preparation and duplicate reporting |
| `make_review_sheets.py` | Generate thumbnail sheets for visual review |
| `external_image_review.txt` | Record exclusions and visual groups |
| `exact_duplicate_exclusions.txt` | Record initial redundant image exclusions |
| `build_clean_dataset.py` | Construct the reviewed clean dataset |
| `split_office31.py` | Create the original recorded grouped split |
| `restore_office31_dataset.py` | Restore images from committed manifests |
| `office31_data.py` | Shared Office-31 classes, preprocessing and manifest/image loading |
| `train_office31.py` | Train the Office-31 baseline |
| `evaluate_office31_validation.py` | Evaluate validation images |
| `evaluate_office31_test.py` | Evaluate the fixed model on test images |
| `predict_office31.py` | Predict one image |
| `plot_office31_learning_curves.py` | Plot saved training history |
| `DATASET_CARD.md` | Document both datasets |
| `ERROR_ANALYSIS.md` | Document Office-31 errors and limitations |

The construction scripts document dataset development. Restoration is the normal setup route for the recorded dataset.

## Fresh installation testing

### Current status

Dataset restoration was tested in a separate temporary directory on the development computer.

Pranay Poheith reported successful independent installation and selected prediction checks on 2 November 2026 using commit `9aeaaf8`. The transcript confirms cloning, required-file checks and prediction exit codes. Supporting screenshots for restoration summaries and prediction outputs are pending review.

That test predates the source cleanup. The revised prediction workflow needs a follow-up check before this revision is described as independently tested.

### Test procedure

Use a fresh clone and follow this README without copying the development computer's virtual environment.

Record the tested commit:

```powershell
git rev-parse HEAD
```

Then check:

1. Dependency installation completes.
2. Dependency consistency checks pass.
3. TensorFlow, OpenCV and NumPy import successfully.
4. Original dataset paths are correct.
5. Restoration produces the recorded image counts.
6. Running restoration again verifies existing copies without replacing them.
7. The bottle example produces the expected class.
8. The known mouse example reproduces the documented model error.
9. A missing image produces a clear error instead of a prediction.

For the missing-image check:

```powershell
& ".\.venv\Scripts\python.exe" predict_office31.py "data\image_that_does_not_exist.jpg"

Write-Host "Exit code: $LASTEXITCODE"
```

Record the actual behaviour and exit code.

Keep:

- Tested commit identifier.
- Operating system and Python version.
- Installation commands and console output.
- Prediction outputs.
- Screenshots where useful.
- Missing files, unclear instructions and any changes required.

A known misclassification is a model limitation, not an installation failure.

This installation test uses the existing model. It does not require retraining or repeating held-out evaluation.

Update this section only after the independent test has been completed and its evidence reviewed.

## Limitations

- Office-31 images are mainly product photographs, unlike cluttered office scenes.
- Visual groups are not verified physical-object identities.
- Undetected near-duplicates or related products may remain.
- Cleaning excluded some ambiguous and multi-item photographs, narrowing the evaluation scope.
- Classes are imbalanced.
- Only three bottles are present in each Office-31 evaluation split.
- The personal-photo pilot contains very few distinct physical objects.
- Multiple views of one object are correlated observations.
- There is no unknown-object rejection or score calibration.
- Classification does not establish ownership.
- Classification does not provide object locations or grasp poses.
- Navigation and manipulation have not been demonstrated.
- Neither dataset demonstrates safe autonomous robot operation.
- Test results have been inspected and must not become targets for further tuning.

Future development should use broader capture conditions and reserve a fresh independent test set before making further model choices.

## Camera status

The documented, evidenced prediction workflow uses image files.

Successful live-camera operation has not yet been confirmed. No webcam accuracy, frame rate or Raspberry Pi inference speed is claimed.

Any future camera demonstration should be reported separately from the saved-image evaluation.

## Troubleshooting

### Git or uv is not recognised

Install the missing tool and reopen PowerShell. Confirm that `git --version` and `uv --version` work before continuing.

### Access denied when cloning

Check that the repository URL is correct and that your GitHub account has access if the repository is private.

### Missing Python package

Install dependencies using the project interpreter:

```powershell
uv pip install --python ".\.venv\Scripts\python.exe" -r requirements.txt
```

### Missing Office-31 images

Check the extracted source path and run:

```powershell
& ".\.venv\Scripts\python.exe" restore_office31_dataset.py
```

### Hash mismatch

Check the source archive and the affected file.

Do not change a recorded hash simply to bypass validation.

### Missing Office-31 model or class mapping

Check these paths:

```text
outputs/office31_baseline/office_classifier.keras
outputs/office31_baseline/classes.json
```

If they are absent from a fresh clone, report the repository problem. Retraining is not the normal installation fix.

### Missing pilot split

Run `split_dataset.py` once when `data/split/` does not exist.

### Existing experiment or evaluation directory

Inspect the saved results and preserve them before conducting a separate run.

An existing output directory is an intentional overwrite safeguard.

### Wrong prediction

A readable image receiving the wrong class is a model error, not automatically a program failure.

Use the recorded error analysis to understand known limitations.

### TensorFlow CPU or oneDNN messages

Informational messages alone do not indicate failure.

Check whether the program completes and produces results. The recorded experiments ran successfully on CPU.

## Sources and reuse

The personal-photo pilot and downloaded Office-31 images have different origins.

The pilot photographs were collected by the group. Office-31 source images were downloaded from the linked archive and must not be described as personally collected.

The Office-31 archive and generated image copies are excluded from Git. Dataset access and reuse should follow the source's terms.

No blanket licence for the external images is granted by this repository.

ImageNet pretraining is separate from both coursework datasets.

## References

- Office-31 archive used: https://zenodo.org/records/14751811
- TensorFlow transfer learning: https://www.tensorflow.org/tutorials/images/transfer_learning
- Keras MobileNet models: https://keras.io/api/applications/mobilenet/
- OpenCV documentation: https://docs.opencv.org/
- Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- uv documentation: https://docs.astral.sh/uv/