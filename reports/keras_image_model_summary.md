# Keras Image Neural Network

The model uses only the automotive part image.

The text description and test split were not used.

## Dataset

- Training samples: 36
- Validation samples: 12

## Architecture

- Image rescaling
- Three convolutional layers
- Two max-pooling layers
- Global average pooling
- Dense layer: 32 units
- Dropout: 0.2
- Output classes: 3

## Training

- Maximum epochs: 60
- Completed epochs: 26
- Best epoch: 18
- Best validation loss: 1.0986
- Trainable parameters: 25763

## Validation results

- Accuracy: 0.3333
- Macro F1: 0.1667

## Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Keras text neural network | 0.5000 | 0.4444 |
| Keras image neural network | 0.3333 | 0.1667 |

## Notes

The label describes the relationship between the image and the text description.

The image-only model cannot observe the description.

The same image is paired with all three labels.

The generated development dataset is used only to validate the training and evaluation pipeline.
