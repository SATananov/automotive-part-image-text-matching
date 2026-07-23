# Validation Error Analysis and Controlled Model Improvement

- Status: **PASS**
- Readiness: **MODEL_IMPROVEMENT_DECISION_COMPLETE**
- Decision: **REFERENCE_RETAINED**
- Selected candidate: **Reference Multimodal**
- Step 010.3 incumbent validation: accuracy `0.5333`, macro F1 `0.5208`
- Selected controlled aggregate: accuracy `0.4167`, macro F1 `0.3333`
- Keras backend: `tensorflow`
- Three fixed seeds were used for every candidate.
- Only integrated train and integrated validation were loaded.
- The locked test split was not loaded, evaluated, or used for selection.

## Controlled experiment comparison

| Rank | Candidate | Mean seed macro F1 | Aggregate macro F1 | Accuracy | Worst-class F1 |
|---:|---|---:|---:|---:|---:|
| 1 | Relation-Aware Multimodal | 0.4646 | 0.4879 | 0.5000 | 0.4375 |
| 2 | Reference Multimodal | 0.3583 | 0.3333 | 0.4167 | 0.0000 |
| 3 | Regularized Relation-Aware Multimodal | 0.3485 | 0.3300 | 0.4167 | 0.0000 |

## Validation error analysis

- Validation errors: 35 / 60
- High-confidence errors: 0
- Exact image hash overlap across train/validation: 0
- Exact description overlap across train/validation: 14

The model-selection gate requires a stable mean-seed gain, no material accuracy or worst-class regression against the controlled reference, and no more than 0.01 aggregate macro-F1 regression against the Step 010.3 incumbent. Completing this step does not authorize test use.
