# VIS-008 — Failure-driven data augmentation

- Step: `011.3A`
- Status: **COMPLETED**
- Exercise problem: 8 of 9
- Test split used: `false`
- Final test evaluation authorized: `false`
- Production final model changed: `false`

## Evidence

- [`reports/course_coverage/vision/augmentation_runs.csv`](../../../../reports/course_coverage/vision/augmentation_runs.csv)
- [`reports/course_coverage/vision/augmentation_comparison.csv`](../../../../reports/course_coverage/vision/augmentation_comparison.csv)
- [`reports/course_coverage/vision/failure_to_augmentation_matrix.csv`](../../../../reports/course_coverage/vision/failure_to_augmentation_matrix.csv)

## Requirement

Identify real failures, test augmentations individually, document label validity, then evaluate a justified combined pipeline.

## Safety boundary

Only committed train and validation splits are in scope. The locked test split remains unopened.
