# Error Analysis

## Experiment

A frozen ImageNet-pretrained MobileNetV2 backbone was used with
a trained six-class output layer. Training used class weights.

Classes: bottle, calculator, keyboard, mouse, mug and stapler.

The cleaned Office-31 Amazon subset contained 415 images:
291 training, 63 validation and 61 test images.

Recorded visual product groups were kept within one split.
These groups are not verified physical-object identities.

## Validation results

- Correct predictions: 62/63
- Accuracy: 98.41%
- Macro F1: 0.9872

Misclassified image:
- mouse/frame_0038: mouse predicted as stapler.

The image was inspected and its mouse label was confirmed.
Its angular, elongated shape may resemble some staplers,
but the cause of the prediction was not verified.

## Test results

- Correct predictions: 59/61
- Accuracy: 96.72%
- Macro F1: 0.9744

Misclassified images:
- mouse/frame_0036: mouse predicted as stapler.
- mouse/frame_0079: mouse predicted as stapler.

Mouse recall was 85.71%: 12 of 14 mice were recognised.
Stapler precision was 85.71%: 12 of 14 stapler predictions
were correct.

Both test errors followed the same class-confusion direction
as the validation error. Their causes have not been verified.

The model was evaluated without retraining or changing the
test split. Misclassified images were retained.

## Limitations

- Results describe this cleaned product-photo subset.
- Performance on real office camera images is not established.
- Only three bottle images were present in the test set.
- Visual grouping may have missed related products.
- Dataset cleaning narrowed the types of images evaluated.
- The original personal-photo experiment achieved 50% test
  accuracy. It used different images and included book instead
  of calculator, so the two results are not directly comparable.