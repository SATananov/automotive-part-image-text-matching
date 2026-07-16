# Automotive Part Image-Text Matching

This project studies whether a model can determine if a photograph of an automotive part matches a short text description.

The task is a three-class classification problem:

- `MATCH` - the image and description represent the same part category;
- `PARTIAL_MATCH` - the description represents a different category from the same automotive system;
- `MISMATCH` - the description represents a category from a different automotive system.

The implemented comparison includes classical baselines, text-only and image-only neural networks, and a multimodal neural network.

## Current development results

The current category-balanced development split contains 150 samples from 50 physical part groups. The test split remains untouched.

| Model | Validation accuracy | Validation macro F1 |
|---|---:|---:|
| Majority baseline | 0.3333 | 0.1667 |
| TF-IDF + Logistic Regression | 0.5000 | 0.3889 |
| Image pixels + Logistic Regression | 0.3333 | 0.1667 |
| Keras text model | 0.5000 | 0.3889 |
| Keras image model | 0.3333 | 0.1667 |
| Keras multimodal model | **0.7667** | **0.7696** |

These values describe the generated development dataset. They are not final real-dataset results.

## Project structure

- `data/development/` - deterministic generated images and metadata;
- `data/processed/` - grouped train, validation, and test split files;
- `data/real/` - real-data annotation templates and approved processed images;
- `src/` - reusable project modules and command-line entry point;
- `models/` - saved model artifacts;
- `reports/` - dataset, training, and evaluation reports;
- `notebooks/` - optional exploratory notebooks;
- `app/` - reserved demonstration application directory;
- `tests/` - automated tests.

## Environment setup

Run the commands from the repository root.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

The smaller `requirements.txt` file contains the direct project dependencies. `requirements-lock.txt` records the complete tested environment in UTF-8.

## Project command line

Use the package command line instead of executing files under `src/` directly:

```powershell
python -m src.project_cli --help
```

Available workflows:

```powershell
python -m src.project_cli environment
python -m src.project_cli create-development-data
python -m src.project_cli validate-development-data
python -m src.project_cli create-grouped-split
python -m src.project_cli run-baselines
python -m src.project_cli train-text
python -m src.project_cli train-image
python -m src.project_cli train-multimodal
python -m src.project_cli verify-step-008-2
python -m src.project_cli validate-real-data
python -m src.project_cli verify-step-009
python -m src.project_cli review-real-intake
python -m src.project_cli apply-real-intake
python -m src.project_cli verify-step-009-1
```

The command modules are imported only when selected. Displaying CLI help or running a non-neural workflow does not import TensorFlow unnecessarily.

## Reproducible development workflow

The following sequence rebuilds the generated dataset and all development results:

```powershell
python -m src.project_cli create-development-data
python -m src.project_cli validate-development-data
python -m src.project_cli create-grouped-split
python -m src.project_cli run-baselines
python -m src.project_cli train-text
python -m src.project_cli train-image
python -m src.project_cli train-multimodal
python -m pytest -q
python -m src.project_cli verify-step-008-2
```

Neural-network training can produce small numeric differences across hardware and TensorFlow builds. Random seeds and deterministic TensorFlow operations are configured where supported.

## Direct module execution

Individual modules can also be executed with `python -m`:

```powershell
python -m src.validate_development_dataset
python -m src.create_grouped_split
python -m src.run_baseline_models
python -m src.train_keras_text_model
python -m src.train_keras_image_model
python -m src.train_multimodal_model
```

Do not use commands such as `python src/validate_development_dataset.py`. The source files use package imports and are intended to run from the repository root as modules.

## Split and evaluation policy

All rows from the same `part_group_id` remain in one split. This prevents photographs or descriptions of the same physical part from crossing between train, validation, and test data.

The current models are selected and compared only on the validation split. The test split must remain unused until the final model and evaluation procedure are fixed.

## Real dataset intake

The generated dataset is only for pipeline development. The final experiment requires real photographs collected and annotated according to:

```text
reports/real_dataset/collection_protocol.md
```

The real dataset uses a separate `real_` identifier namespace and a dedicated directory tree:

```text
data/real/originals/                  local untouched photographs
data/real/staging/                    local temporary review files
data/real/processed/images/           approved reproducible images
data/real/annotations/part_groups.csv physical-part annotations
data/real/annotations/images.csv      image annotations
data/real/processed/real_image_manifest.csv generated intake manifest
```

Original and staging photographs are excluded from Git. Approved processed images, annotation tables, and the generated manifest remain separate from `data/development/`.

Validate the current intake and regenerate the approved-image manifest with:

```powershell
python -m src.project_cli validate-real-data
```

The validator checks schemas, identifiers, category-family mappings, approval relationships, safe paths, readable images, SHA-256 duplicates, repeated views, and overlap with development identifiers or image content. Empty annotation templates are accepted as an `EMPTY_FOUNDATION` state before collection begins.

Verify the complete Step 009 foundation with:

```powershell
python -m src.project_cli verify-step-009
```

## Real sample intake and approval workflow

Step 009.1 adds a controlled queue for the first real photographs. Copy a candidate photograph to `data/real/staging/`, rename it to its intake identifier such as `intake_000001.jpg`, and add one row to:

```text
data/real/annotations/sample_intake.csv
```

Each queue row has one of three decisions:

- `pending` - review the image and metadata but do not modify the dataset;
- `approved` - normalize the photograph to an EXIF-free RGB PNG and add it to the approved annotations;
- `rejected` - record the decision and reason without adding an image to the dataset.

Always review the queue first:

```powershell
python -m src.project_cli review-real-intake
```

The review checks identifier and path safety, category-family mapping, description semantics, image readability and dimensions, luminance and contrast warnings, duplicate hashes, existing annotation conflicts, and development-data overlap. It does not modify annotations or processed images.

After checking `reports/real_dataset/sample_intake_review.md`, apply the explicit decisions with:

```powershell
python -m src.project_cli apply-real-intake
```

The apply command is transactional. It updates `part_groups.csv`, `images.csv`, `approval_log.csv`, the approved image manifest, and the remaining queue only if the final real-dataset validation passes. Any failure restores the previous files and removes newly created processed images.

Approved photographs are written to `data/real/processed/images/<image_id>.png`. The source file remains in the ignored staging directory until it is removed manually. Rejected and approved rows are removed from the queue after a successful apply; pending rows remain.

Verify the complete workflow with:

```powershell
python -m src.project_cli verify-step-009-1
```

## Tests

Run the complete test suite from the repository root:

```powershell
python -m pytest -q
```

The Step 008.2 integrity verifier checks CLI module paths, documentation commands, Markdown fences, the real-data protocol, and UTF-8 lock-file encoding:

```powershell
python -m src.project_cli verify-step-008-2
```

The Step 009 verifier checks the real-data directory boundary, annotation and manifest schemas, CLI registration, Git ignore policy, and the current intake validation state:

```powershell
python -m src.project_cli verify-step-009
```

The Step 009.1 verifier checks the sample queue and approval-log schemas, CLI documentation, transactional safeguards, and the current review state:

```powershell
python -m src.project_cli verify-step-009-1
```

## First real sample batch preparation and dry run

Step 009.2 defines a balanced first collection batch without adding invented
real samples to the approved dataset. The committed plan is:

```text
data/real/annotations/first_batch_plan.csv
```

It reserves 20 planned images from 10 physical parts: one physical part from
each configured category, with `front` and `detail` views. The plan uses
`batch_001`, intake IDs `intake_000001` through `intake_000020`, and the
`real_<category>_001` group namespace.

Prepare or refresh the plan report and queue preview with:

```powershell
python -m src.project_cli prepare-first-real-batch
```

The command scans the expected staging paths, validates category balance,
identifiers, views, metadata consistency, live-queue conflicts, and any files
that have already been captured. It writes
`data/real/processed/first_batch_queue_preview.csv` but does not change
`sample_intake.csv`, approve an image, or modify the real dataset.

Place photographs under `data/real/staging/` with the exact filenames listed
in the plan. JPEG is the expected capture format for this first batch. Missing
files produce the valid `AWAITING_CAPTURE` preparation state.

Run the controlled intake simulation with:

```powershell
python -m src.project_cli dry-run-first-real-batch
```

The dry run reviews captured files, simulates approval and PNG normalization
inside temporary storage, checks prospective annotations and duplicate
protection, and compares live-state fingerprints before and after the
simulation. It does not approve, queue, process, move, or delete real files.
Manual review and the Step 009.1 commands remain mandatory before any actual
approval.

Verify all Step 009.2 safeguards with:

```powershell
python -m src.project_cli verify-step-009-2
```
