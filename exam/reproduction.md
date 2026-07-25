# Reproduction

Run commands from the repository root.

## Rebuild the focused exam layer

```powershell
python -m src.project_cli build-exam-first-submission
python -m src.project_cli verify-exam-first-submission
```

These commands read committed train and validation artifacts, derive every
focused metric and error table from the frozen `keras_multimodal` prediction
file, rebuild the notebook and reports, and verify their hashes. They do not
train a model, open a locked test CSV, authorize final test evaluation, or
change the retained model.

## Open the notebook

```powershell
python -m jupyter notebook exam/01_focused_deep_learning_project.ipynb
```

## Verify the wider repository

```powershell
python -m src.project_cli verify-project
python -m pytest -q
```

TensorFlow-dependent tests require the locked environment in
`requirements-lock.txt`. Archive copies without `.git` cannot independently
prove branch, push, or commit-count state; the Step 011.5.1 status records the
declared source checkpoint and exact source-archive SHA-256.
