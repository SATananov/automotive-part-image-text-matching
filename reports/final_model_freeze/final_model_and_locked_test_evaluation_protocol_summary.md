# Final Model and Locked-Test Evaluation Protocol Freeze

- Status: **PASS**
- Readiness: **FINAL_MODEL_AND_EVALUATION_PROTOCOL_FROZEN_TEST_LOCKED**
- Final model: **Keras Multimodal Neural Network** (`keras_multimodal`)
- Model family: `reference_multimodal`
- Step 010.4 decision: `REFERENCE_RETAINED`
- The frozen object is the architecture, preprocessing, training, and checkpoint-selection recipe.
- No serialized weights are claimed because Step 010.3 did not commit a model file.
- The locked test CSV files were not opened, parsed, trained on, predicted on, or evaluated in Step 010.5.
- Final test authorization remains `false`.

## Frozen validation evidence

- Validation accuracy: `0.5333`
- Validation macro F1: `0.5208`
- Best epoch: `93`
- Selection checkpoint: `be7a3fb`

## One-shot future evaluation

- Primary dataset: `data/processed/integrated_test.csv`
- The external locked test file is not a second model-selection benchmark.
- Every metric and slice is fixed before test access.
- A separate controlled Step 010.6 authorization is mandatory.

## Locked-test contract

- `data/external/integrated/external_test.csv`: `88e5ebe557934bea2b613c3fd5f8cd96c66a534bcfc0e1612da29afb154811f8`
- `data/processed/integrated_test.csv`: `bb234229961b1d6d81f034f906d509ac48b06640a7d7c3359df477089b0cad62`

## Frozen metrics

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `per_class_precision_recall_f1_support`
- `confusion_matrix`
- `per_category_accuracy_and_macro_f1`
- `per_source_accuracy_and_macro_f1`
- `prediction_ledger`

Step 010.5 changes no test authorization state. The final evaluation cannot run until a later step verifies this freeze and creates a separate explicit authorization artifact.
