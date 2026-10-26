# Office Items Pilot Dataset Card

## Purpose

This dataset supports a six-class, single-item image classifier for
PDE3802 Artificial Intelligence in Robotics coursework.

It is a small educational pilot dataset, not a production benchmark.

## Contributors and source

- Khrisvin Nepaulsing — M01003462
- Veedoor Gopal
- Pranay Poheith

The group members personally photographed the objects.
The 90-image coursework dataset contains no downloaded web photographs.

The model separately uses MobileNetV2 weights pretrained on ImageNet.
ImageNet pretraining is distinct from the group's collected dataset.

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

Each image has one class label. There are no bounding-box annotations
or segmentation masks.

## Dataset composition

Each contributor supplied five photographs of one physical object
per class. Contributors used different physical objects.

| Class | Physical objects | Images |
|---|---:|---:|
| book | 3 | 15 |
| bottle | 3 | 15 |
| keyboard | 3 | 15 |
| mouse | 3 | 15 |
| mug | 3 | 15 |
| stapler | 3 | 15 |
| Total | 18 | 90 |

All 90 images passed the OpenCV readability check.

Photographs vary in viewpoint, framing, background and lighting.
Capture conditions were not standardised across contributors.

The dataset is balanced by image count, but has very limited
physical-object diversity.

## File organisation

Original photographs are stored in:

data/raw/<class>/

Example:

data/raw/mug/khrisvin_mug01_001.jpeg

The filename identifies:

- Contributor: khrisvin.
- Class: mug.
- Physical object number for that contributor and class: 01.
- Photograph number: 001.

Object identity combines contributor, class and object number.
The number 01 alone is not a globally unique object identifier.

The split script handles filename case differences and lowercases
the copied filenames.

## Split method

The split is fixed and grouped by physical object.
All five photos of each physical object stay in one split.

| Class | Training contributor | Validation contributor | Test contributor |
|---|---|---|---|
| book | Khrisvin | Pranay | Veedoor |
| bottle | Pranay | Veedoor | Khrisvin |
| keyboard | Veedoor | Khrisvin | Pranay |
| mouse | Khrisvin | Pranay | Veedoor |
| mug | Pranay | Veedoor | Khrisvin |
| stapler | Veedoor | Khrisvin | Pranay |

Each assignment refers to that contributor's object01 for the class.

| Split | Images per class | Total images | Physical objects |
|---|---:|---:|---:|
| Training | 5 | 30 | 6 |
| Validation | 5 | 30 | 6 |
| Test | 5 | 30 | 6 |

`split_dataset.py` reproduces this assignment by copying files into:

data/split/<split>/<class>/

Original photographs remain unchanged.
The generated split folders are excluded from Git.

The script implements the fixed original pilot split. It does not
automatically incorporate future object02 photographs.

Grouping prevents different views of the same physical object from
appearing in both training and evaluation.

This is not a photographer-disjoint split: each contributor appears
across the splits through different classes and objects.

## Use of the splits

Training images are used to fit the classification layer.

Validation images were used for:
- Early stopping.
- Comparing the baseline and augmented model.
- Model selection.
- Inspecting errors.

The baseline was selected before test evaluation.

The held-out test set was then evaluated, and its results and errors
were inspected. It is no longer untouched for future development.

## Preprocessing

Images are:

1. Read using OpenCV.
2. Resized proportionally.
3. Padded with black pixels to 224 × 224.
4. Converted from BGR to RGB.
5. Matched with integer labels from 0 to 5.

Loaded image arrays use uint8 values from 0 to 255.
Labels use int32.

A layer inside the model converts the pixel scale to −1–1 before
MobileNetV2 feature extraction.

Single-image prediction uses the same resizing and colour conversion.

## Augmentation experiment

A separate experiment applied:

- Random rotations up to ±18 degrees.
- Small proportional random zooms.

These transformations were active during training only.
Validation and test evaluation used unchanged images apart from
the standard preprocessing.

Augmented views were generated during training rather than saved
as additional original photographs.

Augmentation did not add new physical objects or change the original
dataset count of 90 images.

## Validation results

| Model | Accuracy | Macro F1 | Correct images |
|---|---:|---:|---:|
| Baseline MobileNetV2 transfer learning | 76.67% | 0.7286 | 23/30 |
| Rotation and zoom augmentation | 73.33% | 0.7063 | 22/30 |

The baseline was selected based on these validation results.

Both models classified all five validation stapler photographs as mouse.
These photographs show one physical stapler.

The tested augmentation settings did not improve this recorded run.

## Held-out test results

The selected baseline achieved:

- Accuracy: 50.00%.
- Macro F1: 0.4850.
- Correct predictions: 15/30.
- Incorrect predictions: 15/30.

| Class | Correct test images |
|---|---:|
| book | 1/5 |
| bottle | 2/5 |
| keyboard | 5/5 |
| mouse | 2/5 |
| mug | 5/5 |
| stapler | 0/5 |

Main test errors:

- Four book images were classified as mouse, bottle or stapler.
- Three bottle images were classified as mug.
- Three mouse images were classified as stapler.
- All five stapler images were classified as mouse.

Full results are stored in:

outputs/pilot/test_evaluation/

The saved metrics include a SHA-256 fingerprint identifying the
evaluated model file.

## Limitations and potential bias

- Only one physical object per class is available for training.
- Each evaluation split contains only one physical object per class.
- Five views of one object are correlated observations, not five
  independent examples of different objects.
- Background, lighting, size in the frame and viewpoint vary across images.
- The model may rely on background or appearance cues that do not generalise.
- Object designs are poorly represented.
- Balanced class counts do not guarantee representative data.
- Perfect results on one class's five images do not establish reliability
  across other objects of that class.
- The classifier has no unknown-object category.
- The dataset does not support evaluation of object localisation,
  navigation or grasping.

The drop from validation to test performance demonstrates limited
generalisation in this pilot. The individual causes have not been
isolated experimentally.

## Intended use and future development

The dataset is intended for coursework experiments and demonstrations
of single-item classification.

It is insufficient to establish safe or reliable autonomous robot
operation.

Future work should add distinct physical objects, broader capture
conditions and a fresh independent test set reserved before further
model development.

Changing settings after inspecting the existing test errors and
evaluating again on the same objects would constitute further
development, not a new untouched test evaluation.

## Availability and reuse

Original images are included in the project repository under data/raw/.

No separate dataset reuse licence is declared in this card.
Contact the contributors before reusing the photographs outside
the coursework.