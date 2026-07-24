
# Step 011.2 — Transformers & Sequence Modelling Experimental Suite

This suite maps the ten sequence-modelling exercise slots to the
`automotive-part-image-text-matching` project.

| ID | Project implementation | State |
|---|---|---|
| SEQ-001 | Inspect committed automotive-part descriptions | Complete |
| SEQ-002 | Deterministic train/validation text loader | Complete |
| SEQ-003 | Train-only vocabulary, tokenization, padding | Complete |
| SEQ-004 | Dense embedding baseline | Complete |
| SEQ-005 | TF-IDF + logistic-regression baseline | Complete |
| SEQ-006 | TextCNN, GRU, and LSTM comparison | Complete |
| SEQ-007 | Small Transformer encoder | Complete |
| SEQ-008 | Metrics, confusion matrices, ROC, and error analysis | Complete |
| SEQ-009 | Two-head attention inspection | Complete |
| SEQ-010 | Pretrained transformer | Deferred — explicit approval required |

## Reproducibility

The suite uses seeds 42, 43, and 44 for neural comparisons. Vocabulary is fit
on the training descriptions only. Reports, figures, an executed notebook,
and an integrity manifest are committed; model weights are not committed.

## Locked evaluation boundary

The only authorized inputs are `data/processed/integrated_train.csv` and
`data/processed/integrated_validation.csv`. No test CSV is loaded or scored.
The frozen final model remains unchanged. Readiness is
`SEQUENCE_EXPERIMENTAL_SUITE_CORE_COMPLETE_PRETRAINED_GATE_TEST_LOCKED`.
