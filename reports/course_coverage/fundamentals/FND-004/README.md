# FND-004 — Overfit one batch

This directory is the stable per-exercise evidence index reserved by the Step 011.0 course-coverage mapping.

- Execution status: **COMPLETED**
- Recorded training runs: **1**
- Configuration: `configs/course_coverage/experiments/FND-004.json`
- Executed notebook: `notebooks/course_coverage/01_fundamentals_experiments.ipynb`
- Execution registry: `data/experiment_registry/fundamentals_execution_registry.json`
- Suite summary: `reports/course_coverage/fundamentals/fundamentals_suite_summary.md`

## Primary evidence

- `reports/course_coverage/fundamentals/overfit_result.json`
- `reports/course_coverage/fundamentals/overfit_history.csv`
- `reports/course_coverage/fundamentals/figures/overfit_learning_curves.png`

## Safety contract

- Evidence is derived from train and validation data only.
- The locked test split was not used.
- Final test evaluation was not authorized.
- The production final model was not changed by this exercise suite.
