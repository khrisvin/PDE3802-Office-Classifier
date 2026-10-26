# Office Items Pilot Dataset Card

## Purpose

This dataset supports a six-class, single-item image classifier for
PDE3802 Artificial Intelligence in Robotics coursework.

It is a small educational pilot dataset, not a production benchmark.

## Contributors

- Khrisvin Nepaulsing
- Veedoor Gopal
- Pranay Poheith

The photographs were taken personally by the group members.

## Classes and labels

| Class | Numeric label |
|---|---:|
| book | 0 |
| bottle | 1 |
| keyboard | 2 |
| mouse | 3 |
| mug | 4 |
| stapler | 5 |

Labels are assigned through the containing class folder.

Each photograph has one image-level label. There are no bounding-box
annotations or segmentation masks.

## Dataset size

The dataset contains 90 original photographs.

Each contributor supplied five photographs of one physical object
per class. Contributors used different physical objects.

Therefore, each class contains:
- 3 physical objects.
- 5 photographs per physical object.
- 15 photographs in total.

Across all six classes, there are 18 physical objects.

All 90 images passed the OpenCV readability check.

## File organisation

Original photographs are stored in:

data/raw/<class>/

Example filename:

khrisvin_mug01_001.jpeg

The filename identifies:
- Contributor: khrisvin.
- Class: mug.
- Physical object number for that contributor and class: 01.
- Photograph number: 001.

Object identity must include the contributor, class and object number.
The number 01 alone is not a globally unique object identifier.

## Training, validation and test split

All photographs of a physical object remain in the same split.

| Class | Training contributor | Validation contributor | Test contributor |
|---|---|---|---|
| book | Khrisvin | Pranay | Veedoor |
| bottle | Pranay | Veedoor | Khrisvin |
| keyboard | Veedoor | Khrisvin | Pranay |
| mouse | Khrisvin | Pranay | Veedoor |
| mug | Pranay | Veedoor | Khrisvin |
| stapler | Veedoor | Khrisvin | Pranay |

Each split contains five images per class, giving:
- Training: 30 images from 6 physical objects.
- Validation: 30 images from 6 physical objects.
- Test: 30 images from 6 physical objects.

The split is fixed and reproduced by split_dataset.py.
The script copies images into data/split/ without changing originals.

Validation data is used for model selection and error analysis.
The test set has not yet been evaluated.

## Preprocessing

Images are:
1. Read using OpenCV.
2. Resized proportionally and padded with black pixels to 224 × 224.
3. Converted from BGR to RGB.
4. Matched with integer labels from 0 to 5.

Pixel values remain in the range 0–255 when loaded.
The model scales them to −1–1 before MobileNetV2 feature extraction.

The same resizing and colour conversion are used for prediction.

## Augmentation experiment

A separate experiment applies random rotations up to ±18 degrees and
small proportional zooms during training only.

Validation and test images are not randomly augmented.

Augmentation does not add new physical objects or change the original
dataset count of 90 photographs.

## Recorded validation results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Baseline MobileNetV2 transfer learning | 76.67% | 0.7286 |
| Rotation and zoom augmentation | 73.33% | 0.7063 |

The baseline was selected based on these validation results.

Both models misclassified all five images of the validation stapler
as mouse. These five photographs show the same physical stapler.

These are validation results, not final test results.

## Limitations and potential bias

- Only one physical object per class is available for training.
- Multiple photographs of the same object are correlated observations.
- Background, lighting, object size and viewpoint vary between contributors.
- The model may learn background or appearance cues that do not generalise.
- Different object designs are poorly represented.
- Small evaluation sets make performance estimates uncertain.
- Results do not establish reliability in an unfamiliar office environment.
- The classifier has no unknown-object class.
- The dataset does not support evaluating object localisation or grasping.

Additional distinct objects and more varied capture conditions would
strengthen future experiments.

## Availability and reuse

Original images are included in the project repository under data/raw/.

No separate dataset reuse licence is declared in this card.
Contact the contributors before reusing the photographs outside
the coursework.