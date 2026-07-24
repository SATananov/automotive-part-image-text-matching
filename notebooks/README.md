# Jupyter notebooks

## Final exam project

The main exam presentation is:

```text
notebooks/02_final_exam_project.ipynb
```

Build and execute it reproducibly from the repository root:

```powershell
python -m src.project_cli build-final-exam-notebook
python -m src.project_cli verify-final-exam-notebook
```

Open the committed executed notebook with:

```powershell
python -m jupyter notebook notebooks/02_final_exam_project.ipynb
```

The final notebook integrates the complete validation-only research narrative through Step 010.5:

- problem statement, motivation, research question and hypothesis;
- related work and formal references;
- generated and reviewed open-license data;
- cleaning, licensing, grouped splitting and leakage safeguards;
- six classical and neural model families;
- development versus integrated validation results;
- confusion matrix, per-class metrics and validation error analysis;
- controlled model improvement and `REFERENCE_RETAINED` decision;
- final model recipe and one-shot protocol freeze;
- testing, reproducibility, limitations and conclusion.

The notebook reads committed training, validation and report artifacts only. It does not retrain models, change model selection, open locked test CSV files, or authorize final test evaluation.

## Development experiment

The earlier development presentation remains available at:

```text
notebooks/01_development_experiment.ipynb
```

It documents the generated development experiment and remains useful as historical evidence. The final exam notebook supersedes it as the main submission document.

## Step 010.7 quality gate

Run the notebook execution, visual-output, numeric-consistency and citation
audit from the repository root:

```powershell
python -m src.project_cli run-notebook-quality-audit
python -m src.project_cli verify-notebook-quality-audit
```

The audit re-executes the notebook without training, compares a scientific
output fingerprint, validates all six saved figures, checks the retained
model's 28 validation errors against its committed predictions and confusion
matrix, and confirms six numbered primary or official references. The locked
test split remains unused and unauthorized.
