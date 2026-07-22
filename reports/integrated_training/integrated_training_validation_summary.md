# Integrated Training Baselines and Validation Comparison

- Status: **PASS**
- Readiness: **VALIDATION_COMPARISON_COMPLETE**
- Training input: `data/processed/integrated_train.csv` (180 samples, 60 groups)
- Validation input: `data/processed/integrated_validation.csv` (60 samples, 20 groups)
- Keras backend recorded for this run: `tensorflow`
- The locked test split was not loaded as a dataset or evaluated.
- Test evaluation remains prohibited by the committed lock.

## Validation comparison

| Rank | Model | Accuracy | Macro F1 | Development macro F1 | Change |
|---:|---|---:|---:|---:|---:|
| 1 | Keras Multimodal Neural Network | 0.5333 | 0.5208 | 0.7696 | -0.2488 |
| 2 | TF-IDF + Logistic Regression | 0.4167 | 0.3300 | 0.3889 | -0.0589 |
| 3 | Keras Text Neural Network | 0.4167 | 0.3300 | 0.3889 | -0.0589 |
| 4 | Majority baseline | 0.3333 | 0.1667 | 0.1667 | +0.0000 |
| 5 | Image pixels + Logistic Regression | 0.3333 | 0.1667 | 0.1667 | +0.0000 |
| 6 | Keras Image Neural Network | 0.3333 | 0.1667 | 0.1667 | +0.0000 |

## Interpretation

The six models were trained only on the integrated training split and compared only on the integrated validation split.

The comparison is a model-selection checkpoint. It is not a final test result, and it does not unlock either test CSV.

The integrated data combines the generated development samples with approved open-license images while preserving physical-part group isolation.
