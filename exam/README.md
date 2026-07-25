# Exam project: start here

## One question

**Does a multimodal neural network that combines an automotive-part image and a short description classify their relationship better than image-only and text-only neural baselines?**

The main submission is one focused, executed notebook:

- [`01_focused_deep_learning_project.ipynb`](01_focused_deep_learning_project.ipynb)

It presents the data, grouped split, model comparison, concrete successes,
concrete errors, limitations, and the exact reproduction boundary.

## Main result

The retained multimodal model reaches **0.5333 accuracy**
and **0.5208 macro F1** on 60 validation samples
from 20 independent physical-part groups. It ranks above the text-only and
image-only neural baselines.

This is not presented as a solved problem. The retained model makes
**35 errors**, including
**20 errors involving the difficult
`PARTIAL_MATCH` class**. Real open-license images are harder than generated
images.

## Scientific boundary

- Split isolation is by `part_group_id`, not by individual row.
- Train and validation have no physical-part group overlap.
- Selection and analysis use committed validation evidence only.
- The locked test split has not been used.
- No model training is performed by this exam-facing layer.
- The retained production model and selection decision are unchanged.

## Supporting pages

- [Reproduce the evidence](reproduction.md)
- [Prepare for the oral defense](defense_notes.md)
- [See honest course-topic alignment](course_alignment.md)
- [Browse supplementary engineering evidence](../supplementary/README.md)

The previous full rubric notebook remains available at
[`notebooks/03_final_exam_submission.ipynb`](../notebooks/03_final_exam_submission.ipynb),
but it is now supporting evidence rather than the first document a reviewer
must read.
