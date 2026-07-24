# Locked Evaluation Plan

- Step: **011.0**
- Base checkpoint: `9237824`
- Readiness target: `FULL_COURSE_COVERAGE_ARCHITECTURE_READY_TEST_LOCKED`

## Data boundary

Training input is limited to `data/processed/integrated_train.csv`.
Model comparison and selection are limited to `data/processed/integrated_validation.csv`.
The following files are prohibited experiment inputs:

- `data/processed/integrated_test.csv`
- `data/external/integrated/external_test.csv`

The registry therefore records `test_split_allowed: false`, `test_split_path: null`, and `final_test_evaluation_authorized: false` for every experiment.

## Selection policy

Validation Macro F1 is the default selection metric. Ties are resolved by seed stability, parameter count, inference time, and architectural simplicity in that order.
Accuracy, class-wise precision/recall/F1, confusion matrices, ROC AUC where meaningful, training time, inference time, and parameter counts remain supporting evidence.
A result is not selection-eligible when it is a failure diagnostic, explainability-only analysis, annotation protocol, or synthesis report.

## Fair-comparison rules

1. Use the committed grouped train and validation split without regrouping after seeing results.
2. Fit vocabularies, scalers, feature extractors, and thresholds on train data only unless a threshold is explicitly selected on validation and documented.
3. Keep split, preprocessing, metrics, and seed policy fixed when comparing architectures.
4. Record resolved configuration, software versions, parameter counts, timing, and saved evidence.
5. Do not promote a model from a single favorable seed when the experiment is marked as a retained comparison.

## Closed final-test gate

Step 011.0 does not authorize final test evaluation. Any future one-shot evaluation requires a separate explicit checkpoint after the model recipe, preprocessing, thresholds, metrics, and reporting procedure are frozen. Until that checkpoint, no test metrics may be generated, inspected, or reported.
