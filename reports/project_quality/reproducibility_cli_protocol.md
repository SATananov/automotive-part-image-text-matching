# Step 008.2 - Reproducibility, CLI and Real Dataset Protocol Hardening

## Status

Completed.

## Changes

- Added `src/project_cli.py` as the supported command-line entry point.
- Added lazy command imports so CLI help and non-neural commands do not load TensorFlow.
- Documented the complete reproducible workflow and direct module invocation rules in `README.md`.
- Replaced the truncated real-dataset collection protocol with complete naming, annotation, approval, privacy, leakage, and validation rules.
- Normalized `requirements-lock.txt` from UTF-16 LE to plain UTF-8.
- Added `src/verification/development_pipeline.py` for deterministic integrity checks.
- Added automated tests for CLI registration, Markdown fence balance, documentation coverage, protocol schema coverage, and lock-file encoding.

## Supported CLI

```powershell
python -m src.project_cli --help
python -m src.project_cli validate-development-data
python -m src.project_cli create-grouped-split
python -m src.project_cli run-baselines
python -m src.project_cli train-text
python -m src.project_cli train-image
python -m src.project_cli train-multimodal
python -m src.project_cli verify-development-pipeline
```

## Reproducibility policy

- Commands are run from the repository root with `python -m`.
- The split remains grouped by physical part.
- Validation results are used during development.
- The test split remains closed until the model and evaluation procedure are fixed.
- Real photographs follow one synchronized collection and approval protocol.
