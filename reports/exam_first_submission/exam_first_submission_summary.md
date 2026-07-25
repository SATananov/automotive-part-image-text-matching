# Step 011.5.1 — Single-Model Evidence Consistency Correction

Status: **PASS**

Readiness: `SINGLE_MODEL_EVIDENCE_CONSISTENT_TEST_LOCKED`

## Purpose

The focused teacher-facing submission now derives its aggregate score,
confusion matrix, error table, source error rates, class analysis, and concrete
examples from one exact frozen `keras_multimodal` validation-prediction
artifact.

## Primary result

- retained model: `keras_multimodal`;
- prediction artifact: `reports/integrated_training/keras_multimodal/validation_predictions.csv`;
- grouped validation samples: 60;
- independent physical-part groups: 20;
- validation accuracy: 0.5333;
- validation macro F1: 0.5208;
- correct predictions: 32;
- validation errors analyzed: 28;
- `PARTIAL_MATCH` recovered: 12/20;
- `PARTIAL_MATCH` errors: 8;
- generated-image error rate: 40.0%;
- real-image error rate: 53.3%;
- `MATCH` recall: 0.30.

## Evidence separation

The historical Step 010.4 controlled retraining with 35 errors remains
preserved as supplementary stability evidence. It is not used for the primary
0.5333 accuracy / 0.5208 macro-F1 claim.

## Safety boundary

- model training performed: false;
- locked test CSV files opened: false;
- test split used: false;
- final test evaluation authorized: false;
- production final model changed: false;
- model selection changed: false.
