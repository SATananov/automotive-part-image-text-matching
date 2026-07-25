# Exam project: start here

## One question

**Does a multimodal neural network that combines an automotive-part image and a short description classify their relationship better than image-only and text-only neural baselines?**

The main submission is one focused, executed notebook:

- [`01_focused_deep_learning_project.ipynb`](01_focused_deep_learning_project.ipynb)

It presents the data, grouped split, model comparison, concrete successes,
concrete errors, limitations, and the exact reproduction boundary.

## Main result

The frozen multimodal model reaches **0.5333 accuracy**
and **0.5208 macro F1** on 60 validation samples
from 20 independent physical-part groups. It ranks above the text-only and
image-only neural baselines.

All primary metrics and errors are derived from one exact prediction artifact:
`reports/integrated_training/keras_multimodal/validation_predictions.csv`.

The model makes **28 errors** and correctly recovers
**12/20 `PARTIAL_MATCH` cases**. Its
weakest class recall is `MATCH` at
**0.30**. Real open-license
images have a **53.3%** error rate, compared with
**40.0%** for generated images.

## Scientific boundary

- Split isolation is by `part_group_id`, not by individual row.
- Train and validation have no physical-part group overlap.
- Selection and analysis use committed validation evidence only.
- The locked test split has not been used.
- No model training is performed by this exam-facing layer.
- The retained production model and selection decision are unchanged.
- The historical 35-error Step 010.4 retraining is supporting evidence only.

## Supporting pages

- [Reproduce the evidence](reproduction.md)
- [Prepare for the oral defense](defense_notes.md)
- [See honest course-topic alignment](course_alignment.md)
- [Browse supplementary engineering evidence](../supplementary/README.md)

The previous full rubric notebook remains available at
[`notebooks/03_final_exam_submission.ipynb`](../notebooks/03_final_exam_submission.ipynb),
but it is historical supporting evidence rather than the source of the focused
primary metrics.
