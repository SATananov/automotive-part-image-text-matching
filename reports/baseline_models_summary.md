# Baseline Models

The baseline models were trained on the development training split and evaluated on the validation split.

The test split was not used.

## Dataset

- Training samples: 90
- Validation samples: 30
- Training label distribution: {'MATCH': 30, 'MISMATCH': 30, 'PARTIAL_MATCH': 30}

## Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 0.3333 | 0.1667 |
| TF-IDF + Logistic Regression | 0.5000 | 0.3889 |
| Image pixels + Logistic Regression | 0.3333 | 0.1667 |

## Interpretation

The majority model provides the minimum reference performance for the balanced three-class task.

The text model measures how much label information can be learned from the description alone.

The image model measures whether image features alone can predict the relationship label.

Because the same image is paired with all three labels, an image-only model should not reliably solve the task.

These results are based on a small generated development dataset and are used only to validate the pipeline.
