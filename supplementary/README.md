# Supplementary evidence

The folders below support the main exam story but are not the recommended
starting point for assessment.

## Course exercises

- `notebooks/course_coverage/`
- `docs/course_coverage/`
- `reports/course_coverage/fundamentals/`
- `reports/course_coverage/sequence/`
- `reports/course_coverage/vision/`

## Historical research notebooks

- `notebooks/01_development_experiment.ipynb`
- `notebooks/02_final_exam_project.ipynb`
- `notebooks/03_final_exam_submission.ipynb`

The historical Step 010.4 / Step 011.4 error analysis reports 35 errors from a
separate controlled retraining. It is preserved as stability and audit
evidence. It must not be combined with the primary frozen `keras_multimodal`
run, which has 32 correct predictions and 28 errors.

## Engineering and audit evidence

- `tests/`
- `src/verification/`
- `reports/project_quality/`
- `reports/notebook_quality_audit/`
- `reports/exam_submission_readiness/`
- `reports/final_submission/`
- `reports/final_model_freeze/`

## Dataset provenance

- `data/external/`
- `data/real/`
- `reports/external_dataset/`
- `reports/real_dataset/`

The standard `src/`, `data/`, `reports/`, and `tests/` directories remain in
place so the project stays reproducible and conventional. They were not
physically moved, because their stable paths protect imports, manifests,
notebook links, and historical evidence.
