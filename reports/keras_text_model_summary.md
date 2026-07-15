# Keras Text Neural Network

The model uses only the text description.

The image input and test split were not used.

## Dataset

- Training samples: 36
- Validation samples: 12

## Architecture

- Text vectorization
- Token embedding: 32 dimensions
- Global average pooling
- Dense layer: 32 units
- Dropout: 0.2
- Output classes: 3

## Training

- Vocabulary size: 20
- Maximum epochs: 60
- Completed epochs: 47
- Best epoch: 39
- Best validation loss: 1.0380
- Trainable parameters: 1795

## Validation results

- Accuracy: 0.5000
- Macro F1: 0.4444

## Notes

The label describes the relationship between an image and a text description.

A text-only model cannot directly observe that relationship because it does not receive the image.

The generated development dataset is used only to check the training and evaluation pipeline.
