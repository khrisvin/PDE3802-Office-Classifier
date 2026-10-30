# Office Classifier Dataset Card

## Purpose

This project uses two separate educational datasets for PDE3802 Artificial Intelligence in Robotics coursework:

1. A reviewed six-class Office-31 Amazon subset.
2. An original six-class personal-photo pilot.

The datasets have different sources, class sets and grouping methods. They are not combined in the recorded experiments.

Each image has one class label. Neither dataset provides bounding boxes or segmentation masks.

## Project contributors

- Khrisvin Nepaulsing — M01003462
- Veedoor Gopal
- Pranay Poheith

The contributors personally captured the pilot photographs. They reviewed and prepared the external Office-31 subset; they did not create its original photographs.

Both model families use ImageNet-pretrained MobileNetV2 weights. Pretraining is separate from the coursework image datasets.

## Dataset A: reviewed Office-31 subset

### Source and scope

Download source used:

https://zenodo.org/records/14751811

Archive: `Dataset_Office-31.tar.gz`.

Only the Amazon domain was used. DSLR and webcam images were not used in this experiment.

The selected classes are bottle, calculator, keyboard, mouse, mug and stapler. Calculator replaces the pilot's book class.

This is a project-specific reviewed subset, not a claim to reproduce a standard Office-31 benchmark protocol.

### Class mapping and composition

| Class | Label | Original selected images | Excluded | Retained |
|---|---:|---:|---:|---:|
| bottle | 0 | 36 | 13 | 23 |
| calculator | 1 | 94 | 3 | 91 |
| keyboard | 2 | 100 | 46 | 54 |
| mouse | 3 | 100 | 6 | 94 |
| mug | 4 | 94 | 23 | 71 |
| stapler | 5 | 99 | 17 | 82 |
| Total | — | 523 | 108 | 415 |

Labels follow the reviewed class assignment. The saved model's `classes.json` preserves numeric output order.

### Review and exclusions

Review occurred before creating the recorded split and training the Office-31 model.

The process included:

- Initial manual exclusions.
- Image readability checks.
- Exact SHA-256 duplicate checks.
- Thumbnail-sheet inspection.
- Further exclusions and visual-family grouping.

The initial review retained 506 images. Five exact duplicate groups contained six redundant copies; excluding these left 500 images for the subsequent review stage.

The final clean dataset contains 415 images, with 108 exclusions relative to the original 523 selected images.

Exclusion reasons included incorrect or uncertain class membership, accessory-only images, exact duplicates and some multi-item or composite images.

Exclusion does not always mean an image was definitively mislabelled. Some examples were excluded because their identity was uncertain or their presentation fell outside the chosen scope.

The review was not a uniform requirement that every image contain only one object with no accessories. Some single target items with packaging, receivers, cables or small accessories remain.

Electric and heavy-duty staplers, trackball-style pointing devices and some travel drinkware were retained. The review records document individual decisions.

The original downloaded images were not changed.

### Review records

- `external_image_review.txt`
- `exact_duplicate_exclusions.txt`
- `data/office31_clean/exclusions.csv`
- `data/office31_clean/review_snapshot.txt`

The final manifests, rather than an informal description of class appearance, define the recorded dataset.

### Visual groups

Visually related images were assigned shared group identifiers.

The manifest distinguishes:

- `visual_product_family`: images grouped through visual review.
- `singleton_not_verified_independent`: an image without an assigned multi-image group.

These labels do not establish verified physical-object identity. Different product photographs may show the same design, related products or different physical examples.

Singleton status does not prove independence from all other images.

All images within each recorded group remain in one split. Unrecognised near-duplicates or related products may remain across splits.

### Split method

The split was generated with seed 42 and approximate 70% training, 15% validation and 15% test targets.

Allocation respected recorded groups. Candidate allocations were assessed using image counts, not model performance.

Group sizes mean the final counts do not exactly match the target percentages.

| Class | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| bottle | 17 | 3 | 3 | 23 |
| calculator | 63 | 14 | 14 | 91 |
| keyboard | 38 | 8 | 8 | 54 |
| mouse | 66 | 14 | 14 | 94 |
| mug | 49 | 12 | 10 | 71 |
| stapler | 58 | 12 | 12 | 82 |
| Total | 291 | 63 | 61 | 415 |

Recorded groups crossing splits: zero.

This check applies to recorded groups only; it is not a guarantee that all forms of data leakage have been eliminated.

### File organisation and provenance

Original source example:

```text
data/external/office31/amazon/images/bottle/frame_0003.jpg
```

Clean copy:

```text
data/office31_clean/bottle/office31_amazon_frame_0003.jpg
```

Split copy:

```text
data/office31_split/train/bottle/office31_amazon_frame_0003.jpg
```

The clean manifest records:

- Class and filename.
- Relative destination and original source paths.
- Source URL.
- SHA-256 image hash.
- Image width and height.
- Group identifier and grouping basis.

The split manifest additionally records the split and split path.

The authoritative files are:

- `data/office31_clean/manifest.csv`
- `data/office31_split/split_manifest.csv`
- `data/office31_split/split_details.txt`

### Restoration

External images and generated image copies are excluded from Git. Manifests and review records are included.

After downloading and extracting the original archive into the documented source location, run:

```powershell
& ".\.venv\Scripts\python.exe" restore_office31_dataset.py
```

The script checks original image hashes, restores missing copies and verifies existing copies. It rejects conflicting files without overwriting them.

The same 415 images exist as clean copies and split copies. These are two storage layouts, not additional observations.

Restoration preserves the recorded split rather than generating a new random allocation.

### Use of the splits

Training images were used to fit the classification layer.

Class weights were calculated using training counts only:

```text
weight for class = total training images / (6 × class training images)
```

Validation images were used for checkpoint selection, early stopping and error inspection.

The saved model was then evaluated on the test split. The test evaluation checked that the model and split fingerprints matched the validation record.

Both evaluation sets have now been inspected. Future tuning based on test outcomes would make that test set part of the development process.

Correctly labelled errors were retained.

### Recorded results

| Split | Images | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Validation | 63 | 62 | 98.41% | 0.9872 |
| Test | 61 | 59 | 96.72% | 0.9744 |

Validation error:

- Mouse `frame_0038` predicted as stapler.

Test errors:

- Mouse `frame_0036` predicted as stapler.
- Mouse `frame_0079` predicted as stapler.

The cause of these predictions has not been established experimentally.

Results are stored under:

- `outputs/office31_baseline/validation_evaluation/`
- `outputs/office31_baseline/test_evaluation/`

See `ERROR_ANALYSIS.md` for further discussion.

### Limitations

- Mainly product photographs, often against simple backgrounds.
- Limited evidence about real office lighting, clutter, occlusion and viewpoints.
- Class imbalance, especially the small bottle category.
- Only three bottle images in validation and three in test.
- Visual-family grouping without verified physical identities.
- Possible unrecognised near-duplicates or related products.
- Manual review may introduce selection bias.
- Exclusion of ambiguous and multi-item examples narrows the evaluation scope.
- A small test set and potentially correlated images limit generalisation claims.
- Pretraining-data overlap with external product photographs has not been audited.

The recorded accuracy describes this particular reviewed split, not every Office-31 image or real-world office object.

### Availability and reuse

The repository records the source link, review decisions and manifests. The external photographs are not redistributed in the repository.

This card does not establish or grant a reuse licence for the external images. Consult the source's applicable terms and attribution requirements before reuse or redistribution.

## Dataset B: original personal-photo pilot

### Source

The group personally photographed all 90 pilot images.

The statement that the dataset contains no downloaded web photographs applies to this pilot only.

### Classes and composition

Each contributor photographed one distinct physical object per class from five views.

| Class | Label | Physical objects | Images |
|---|---:|---:|---:|
| book | 0 | 3 | 15 |
| bottle | 1 | 3 | 15 |
| keyboard | 2 | 3 | 15 |
| mouse | 3 | 3 | 15 |
| mug | 4 | 3 | 15 |
| stapler | 5 | 3 | 15 |
| Total | — | 18 | 90 |

All 90 images passed the OpenCV readability check.

Viewpoint, framing, backgrounds and lighting vary. Capture conditions were not standardised.

### File naming

Original images are in `data/raw/<class>/`.

Example:

```text
data/raw/mug/khrisvin_mug01_001.jpeg
```

The name identifies contributor, class, contributor-specific object number and photograph number.

Physical-object identity combines contributor, class and object number. The number `01` alone is not globally unique.

The split script handles filename case differences and lowercases copied filenames.

### Fixed physical-object split

All five views of each physical object remain together.

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

`split_dataset.py` creates `data/split/<split>/<class>/` without changing original photographs.

The generated split is excluded from Git. The script reproduces the original assignment and does not automatically incorporate future object02 images.

Grouping prevents views of the same recorded physical object crossing splits. This is not a photographer-disjoint split.

### Augmentation experiment

The separate augmented pilot used random rotations up to ±18 degrees and small random zooms during training only.

Augmentation produced transformed views of existing images. It did not add physical objects or increase the original dataset size.

### Results and split use

Validation was used for early stopping, comparison and model selection.

| Model | Validation accuracy | Macro F1 | Correct |
|---|---:|---:|---:|
| Baseline | 76.67% | 0.7286 | 23/30 |
| Rotation and zoom augmentation | 73.33% | 0.7063 | 22/30 |

The baseline was selected before test evaluation.

Both models classified all five validation stapler photographs as mouse. These images show one physical stapler.

The selected baseline achieved 50.00% test accuracy and 0.4850 macro F1.

| Class | Correct test images |
|---|---:|
| book | 1/5 |
| bottle | 2/5 |
| keyboard | 5/5 |
| mouse | 2/5 |
| mug | 5/5 |
| stapler | 0/5 |
| Total | 15/30 |

Test errors included four book images, three bottle images, three mouse images and all five stapler images.

Artifacts are stored in `outputs/pilot/test_evaluation/`.

The test set has now been inspected and is no longer untouched for further development.

### Limitations

- Only one physical object per class is available for training.
- Each evaluation split contains only one physical object per class.
- Five views of one object are correlated observations.
- Background, framing and appearance may influence predictions.
- Balanced image counts do not provide broad object diversity.
- Correct predictions on one object do not establish reliability across its class.

The validation-to-test performance drop demonstrates limited generalisation in this pilot. Individual causes have not been isolated experimentally.

### Availability and reuse

Original pilot photographs are included under `data/raw/`.

No separate dataset reuse licence is declared here. Contact the contributors before reusing these photographs outside the coursework.

## Shared preprocessing

Both experiments use:

1. OpenCV image loading.
2. Proportional resizing.
3. Black padding to 224 × 224.
4. BGR-to-RGB conversion.
5. Integer class labels.

Loaded images use uint8 pixel values from 0 to 255. Labels use int32.

An internal model layer scales pixels to −1–1 before MobileNetV2 feature extraction. Prediction uses the corresponding training preprocessing and saved class order.

## Intended use

These datasets support coursework experiments in single-item classification.

They do not support claims about object localisation, navigation, grasping or safe autonomous operation.

The classifier has no unknown-object category, and its softmax scores are not calibrated probabilities of correctness.

Future development should expand physical-object diversity and capture conditions, and reserve a fresh independent test set before further model selection.

## Interpreting the two experiments

The original pilot's 50.00% test accuracy and Office-31's 96.72% test accuracy measure different tasks and distributions.

The higher Office-31 score must not be presented as a controlled improvement on the original personal-photo test set.

The personal-photo pilot was not merged into the Office-31 training or evaluation split.