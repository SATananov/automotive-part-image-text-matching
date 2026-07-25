# Notebook catalogue

## Recommended exam entry point

The primary teacher-facing notebook is now:

- [`exam/01_focused_deep_learning_project.ipynb`](../exam/01_focused_deep_learning_project.ipynb)

It presents one research question, grouped-split protection, the primary model
comparison, concrete successes, errors from one frozen prediction artifact,
limitations, and reproduction.

## Supporting notebooks

- `notebooks/01_development_experiment.ipynb` — early development evidence;
- `notebooks/02_final_exam_project.ipynb` — historical full project notebook;
- `notebooks/03_final_exam_submission.ipynb` — Step 011.4 rubric-alignment and historical controlled-retraining evidence;
- `notebooks/course_coverage/` — Fundamentals, sequence, vision, ranking, and controlled experiments.

The historical notebooks are supporting evidence and are not intended to
replace or redefine the focused primary metrics.

<details>
<summary>Historical notebook catalogue and verification commands</summary>

# Jupyter notebooks

## Final exam submission

The main teacher-facing presentation is the committed executed notebook:

- [Open `03_final_exam_submission.ipynb` directly on GitHub](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/03_final_exam_submission.ipynb)
- Repository path: `notebooks/03_final_exam_submission.ipynb`
- Submission deadline: **11 August 2026, 16:00**

Build and verify it from the repository root:

```powershell
python -m src.project_cli build-final-submission-notebook
python -m src.project_cli verify-final-submission
```

Open it locally with:

```powershell
python -m jupyter notebook notebooks/03_final_exam_submission.ipynb
```

The notebook is aligned directly to the eight exam categories:

- problem statement and real-world significance;
- readable article layout;
- modular and tested Python code;
- previous research and six formal references;
- data acquisition, licensing, cleaning, formatting and group isolation;
- automated testing, controlled failure tests and locked-test safeguards;
- saved tables, confusion matrices, error plots and course-suite figures;
- conclusions, limitations, self-assessment and defense summary.

It includes a dedicated **Deep Learning Error Analysis and Failure Diagnostics** section with PARTIAL_MATCH confusion, domain shift, category errors, representative mistakes, controlled training failures and explicit explainability boundaries.

The notebook reads committed validation and report artifacts only. It does not retrain models, open a locked test CSV, authorize final test evaluation or change the production model.

## Historical quality gate

Step 010.7 remains the immutable notebook execution, visual QA, numeric consistency and citation audit for `02_final_exam_project.ipynb`.

```powershell
python -m src.project_cli run-notebook-quality-audit
python -m src.project_cli verify-notebook-quality-audit
```

## Historical and specialist notebooks

The following notebooks remain committed as research evidence:

- `01_development_experiment.ipynb` — generated development baseline;
- [`02_final_exam_project.ipynb`](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/02_final_exam_project.ipynb) — Step 010.6/010.7 historical final narrative;
- `course_coverage/01_fundamentals_experiments.ipynb` — 10/10 fundamentals tasks;
- `course_coverage/02_sequence_model_comparison.ipynb` — sequence core experiments;
- `course_coverage/03_vision_model_comparison.ipynb` — vision representation and augmentation;
- `course_coverage/04_scoring_ranking_explainability.ipynb` — compatibility, ranking and occlusion.

## Verification

Run the current teacher-facing gate and the complete project verification:

```powershell
python -m src.project_cli verify-final-submission
python -m src.project_cli verify-project
python -m pytest -q
```

The final submission checklist, self-assessment, Deep Learning error report and defense guide are under `reports/final_submission/`.

</details>
