# Keras Multimodal Neural Network

The model uses both the automotive part image and the text description.

The test split was not used.

## Dataset

- Training samples: 36
- Validation samples: 12

## Architecture

### Text branch

- Text vectorization
- Token embedding: 32 dimensions
- Global average pooling
- Text feature layer: 32 units

### Image branch

- Image rescaling
- Three convolutional layers
- Two max-pooling layers
- Global average pooling
- Image feature layer: 32 units

### Fusion

- Text and image feature concatenation
- Fusion layer: 64 units
- Dropout: 0.25
- Output classes: 3

## Training

- Vocabulary size: 20
- Maximum epochs: 80
- Completed epochs: 31
- Best epoch: 21
- Best validation loss: 1.0935
- Parameters: 31715

## Validation results

- Accuracy: 0.1667
- Macro F1: 0.1439

## Model comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Keras text neural network | 0.5000 | 0.4444 |
| Keras image neural network | 0.3333 | 0.1667 |
| Keras multimodal neural network | 0.1667 | 0.1439 |

## Notes

The multimodal model can use both sides of the image-text relationship.

The generated development dataset is used only to validate the complete training pipeline.

Final conclusions will require a larger dataset with real automotive part images.
