# SEQ-009 — Inspect attention heads

This directory is the stable per-exercise evidence index reserved by the Step 011.0 course-coverage mapping.

- Execution status: **COMPLETED**
- Recorded training runs: **0**
- Configuration: `configs/course_coverage/experiments/SEQ-009.json`
- Executed notebook: `notebooks/course_coverage/02_sequence_model_comparison.ipynb`
- Execution registry: `data/experiment_registry/sequence_execution_registry.json`
- Suite summary: `reports/course_coverage/sequence/sequence_suite_summary.md`

## Primary evidence

- `reports/course_coverage/sequence/attention_evidence.json`
- `reports/course_coverage/sequence/attention_token_summary.csv`
- `reports/course_coverage/sequence/figures/attention_correct_head_1.png`
- `reports/course_coverage/sequence/figures/attention_incorrect_head_1.png`

## Safety contract

- Evidence is derived from train and validation data only.
- The locked test split was not used.
- Final test evaluation was not authorized.
- The production final model was not changed by this exercise suite.
