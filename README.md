# Automotive Part Image-Text Matching

> **Exam submission:** [Open the executed final notebook on GitHub](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/02_final_exam_project.ipynb) · [Read the submission checklist](reports/exam_submission_readiness/submission_checklist.md) · [View the release readiness summary](reports/exam_submission_readiness/exam_submission_readiness_summary.md)

This project studies whether a model can determine if a photograph of an automotive part matches a short text description.

The task is a three-class classification problem:

- `MATCH` - the image and description represent the same part category;
- `PARTIAL_MATCH` - the description represents a different category from the same automotive system;
- `MISMATCH` - the description represents a category from a different automotive system.

The implemented comparison includes classical baselines, text-only and image-only neural networks, and a multimodal neural network.

## Integrated validation results

The final model selection uses the integrated grouped validation split: 60 samples from 20 independent physical-part groups, balanced across the three relationship labels. The locked test split remains unused and unauthorized.

| Model | Integrated validation accuracy | Integrated validation macro F1 |
|---|---:|---:|
| Majority baseline | 0.3333 | 0.1667 |
| TF-IDF + Logistic Regression | 0.4167 | 0.3300 |
| Image pixels + Logistic Regression | 0.3333 | 0.1667 |
| Keras text model | 0.4167 | 0.3300 |
| Keras image model | 0.3333 | 0.1667 |
| Keras multimodal model | **0.5333** | **0.5208** |

The multimodal model is the retained final recipe. The earlier generated-development result was higher (`0.7667` accuracy and `0.7696` macro F1), but it is reported only as development evidence rather than the final validation result.

## Project structure

- `data/development/` - deterministic generated images and metadata;
- `data/processed/` - grouped train, validation, and test split files;
- `data/real/` - real-data annotation templates and approved processed images;
- `src/` - reusable project modules and command-line entry point;
- `models/` - saved model artifacts;
- `reports/` - dataset, training, and evaluation reports;
- `notebooks/` - Jupyter presentation notebooks for the development experiment;
- `app/` - reserved demonstration application directory;
- `tests/` - automated tests.

## Jupyter notebooks

The main exam presentation is the committed executed notebook:

- [Open `notebooks/02_final_exam_project.ipynb` directly on GitHub](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/02_final_exam_project.ipynb)
- Local path: `notebooks/02_final_exam_project.ipynb`

Build, verify, and open it from the repository root:

```powershell
python -m src.project_cli build-final-exam-notebook
python -m src.project_cli verify-final-exam-notebook
python -m jupyter notebook notebooks/02_final_exam_project.ipynb
```

The notebook integrates the complete validation-only research narrative: problem definition, previous research, data acquisition and licensing, grouped splitting, six model families, development and integrated validation results, error analysis, controlled model selection, final model freeze, testing, limitations, conclusion, and references. It contains saved tables and visualizations and does not retrain models or access the locked test split.

The earlier development notebook remains available as historical evidence:

```text
notebooks/01_development_experiment.ipynb
```

## Full course exercise coverage roadmap

Step 011.0 maps all 29 tasks from the supplied Deep Learning Fundamentals, Transformers and Sequence Modelling, and Vision Models exercises to controlled experiments for this automotive image-text matching problem. This checkpoint defines architecture, metrics, evidence paths, resource gates, and execution policy; it does not claim that the new experiments have already run.

- [Full course coverage matrix](docs/course_coverage/full_course_coverage_matrix.md)
- [Locked evaluation plan](docs/course_coverage/locked_evaluation_plan.md)
- [Experiment execution policy](docs/course_coverage/experiment_execution_policy.md)
- [Machine-readable experiment registry](data/experiment_registry/course_coverage_registry.json)
- [Planned course-coverage notebooks](notebooks/course_coverage/README.md)

Every Step 011 experiment is limited to the committed train and validation splits. The test split remains locked, unused, and unauthorized. Human agreement metrics are planned only after genuine independent annotations exist; simulated annotators are not accepted as evidence.

Build and verify the planning checkpoint with:

```powershell
python -m src.project_cli build-course-coverage-architecture
python -m src.project_cli verify-course-coverage-architecture
```

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
python -m src.project_cli verify-development-pipeline
python -m src.project_cli build-course-coverage-architecture
python -m src.project_cli verify-course-coverage-architecture
python -m src.project_cli validate-real-data
python -m src.project_cli verify-real-dataset-foundation
python -m src.project_cli review-real-intake
python -m src.project_cli apply-real-intake
python -m src.project_cli verify-sample-intake
python -m src.project_cli prepare-first-real-batch
python -m src.project_cli dry-run-first-real-batch
python -m src.project_cli verify-first-batch-preparation
python -m src.project_cli stage-first-real-batch-capture
python -m src.project_cli verify-capture-staging
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
python -m src.project_cli verify-development-pipeline
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
python -m src.project_cli verify-real-dataset-foundation
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
python -m src.project_cli verify-sample-intake
```

## Tests

Run the complete test suite from the repository root:

```powershell
python -m pytest -q
```

The Step 008.2 integrity verifier checks CLI module paths, documentation commands, Markdown fences, the real-data protocol, and UTF-8 lock-file encoding:

```powershell
python -m src.project_cli verify-development-pipeline
```

The Step 009 verifier checks the real-data directory boundary, annotation and manifest schemas, CLI registration, Git ignore policy, and the current intake validation state:

```powershell
python -m src.project_cli verify-real-dataset-foundation
```

The Step 009.1 verifier checks the sample queue and approval-log schemas, CLI documentation, transactional safeguards, and the current review state:

```powershell
python -m src.project_cli verify-sample-intake
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
python -m src.project_cli verify-first-batch-preparation
```
## First real batch capture, staging and review readiness

Step 009.3 turns the committed `batch_001` plan into a controlled local
capture workflow without approving any image. Step 009.4 adds the supported
file-naming and local-import boundary. Place descriptively named JPEG or PNG
files in:

```text
data/real/capture_inbox/batch_001/
```

Use names such as `real_starter_001_front.jpg` and run
`import-first-real-batch`. The importer copies the original bytes into
`data/real/originals/batch_001/`. Then run the capture and staging command from
the repository root:

```powershell
python -m src.project_cli stage-first-real-batch-capture
```

The command applies EXIF orientation, converts each source to a reproducible
RGB JPEG staging file, checks exact duplicates and development overlap, and
never overwrites a conflicting staging destination. New staging writes are
transactional. Original photographs remain unchanged in the ignored originals
directory.

The command generates:

```text
data/real/processed/first_batch_capture_inventory.csv
data/real/processed/first_batch_review_queue_draft.csv
reports/real_dataset/first_batch_capture_readiness.md
```

The review queue draft contains only `pending` decisions. It is separate from
`sample_intake.csv`; the live queue, approval log, manifest, and approved image
directory are not modified. Review the inventory and report manually before
copying acceptable draft rows into the live queue. Actual approval still uses
`review-real-intake` followed by `apply-real-intake`.

With no local photographs, the correct state is `AWAITING_CAPTURE`. A partial
batch is `CAPTURE_IN_PROGRESS`. When all 20 planned files pass review, the
state becomes `READY_FOR_MANUAL_QUEUE_IMPORT`.

Verify the Step 009.3 safeguards with:

```powershell
python -m src.project_cli verify-capture-staging
```

## First real batch file naming and local import

Use descriptive filenames for the 20 first-batch photographs. Do not name local photographs with internal `intake_` identifiers. The required pattern is:

```text
real_<part_category>_001_<view>.jpg
```

Examples:

```text
real_starter_001_front.jpg
real_starter_001_detail.jpg
real_brake_disc_001_front.jpg
real_air_filter_001_detail.jpg
```

The exact filename-to-intake mapping is committed in:

```text
data/real/annotations/first_batch_capture_file_map.csv
```

The complete capture checklist is:

```text
reports/real_dataset/first_batch_capture_checklist.md
```

Place renamed JPEG or PNG files in the ignored local inbox:

```text
data/real/capture_inbox/batch_001/
```

Then run:

```powershell
python -m src.project_cli import-first-real-batch
```

The importer copies original bytes into `data/real/originals/batch_001/` without image conversion. It blocks unclear filenames, duplicate content, unreadable images, multiple extensions for one planned photograph, and conflicts with an existing original. Writes are transactional and do not modify staging, annotations, the live queue, approval log, or approved manifest.

Review:

```text
data/real/processed/first_batch_local_import_inventory.csv
reports/real_dataset/first_batch_local_import_readiness.md
```

When the readiness is `READY_FOR_STAGING`, continue with:

```powershell
python -m src.project_cli stage-first-real-batch-capture
```

The staging workflow accepts the new descriptive filenames and still supports the earlier technical intake stems for backward compatibility. Verify the naming and import safeguards with:

```powershell
python -m src.project_cli verify-local-capture-import
```

## First real batch operator guide and capture session

Use the practical operator guide before photographing the first batch:

```text
reports/real_dataset/first_batch_operator_guide.md
```

Prepare or refresh the capture-session worksheet with:

```powershell
python -m src.project_cli prepare-first-real-batch-session
```

The command groups the 20 planned photographs into 10 physical-part pairs, reports the exact missing `front` and `detail` filenames, selects the next capture, and writes:

```text
data/real/processed/first_batch_capture_session.csv
reports/real_dataset/first_batch_capture_session_readiness.json
reports/real_dataset/first_batch_capture_session_readiness.md
```

The preparation command does not copy, convert, approve, queue, or delete photographs. It fingerprints the local inbox, immutable originals, staging, annotations, live queue, approval log, and manifest before and after the scan. Safe readiness values are `AWAITING_CAPTURE`, `CAPTURE_SESSION_IN_PROGRESS`, `READY_FOR_LOCAL_IMPORT`, and `READY_FOR_STAGING`.

Verify the operator guide and session-preparation safeguards with:

```powershell
python -m src.project_cli verify-capture-session
```

## First real batch capture dashboard and progress tracking

Build the local operator dashboard at any point during capture, import,
staging, review, or approval:

```powershell
python -m src.project_cli build-first-real-batch-dashboard
```

The self-contained dashboard is written to:

```text
reports/real_dataset/first_batch_capture_dashboard.html
```

Its machine-readable and review outputs are:

```text
data/real/processed/first_batch_capture_progress.csv
reports/real_dataset/first_batch_capture_dashboard.json
reports/real_dataset/first_batch_capture_progress_summary.md
```

The dashboard tracks every planned photograph from `AWAITING_CAPTURE` through
`APPROVED_DATASET`, shows the next required action, and reports overall and
per-category progress. It does not copy, convert, queue, approve, reject, or
delete data. Input fingerprints must remain unchanged.

Verify the dashboard and progress safeguards with:

```powershell
python -m src.project_cli verify-capture-dashboard
```

## First-batch capture execution and live progress

Run a safe operator cycle after adding one or more planned photographs to the
local capture inbox:

```powershell
python -m src.project_cli run-first-real-batch-capture-session
```

The cycle imports valid captures, stages valid originals, refreshes the session
worksheet, and writes a live dashboard under the Git-ignored
`data/real/runtime/first_batch_capture/` directory. It rolls back originals
and staging when a downstream operation fails and proves that the live queue,
approval log, annotations, manifest, and tracked reports remain unchanged.

Use the read-only refresh command between capture actions:

```powershell
python -m src.project_cli refresh-first-real-batch-live-progress
```

Open `data/real/runtime/first_batch_capture/live_dashboard.html` to see the
latest pipeline progress. These commands never queue or approve samples.
Verify the execution safeguards with:

```powershell
python -m src.project_cli verify-capture-execution
```

## First batch review queue and manual decision preparation

When the live capture dashboard reports review-ready staged images, activate only the validated pending draft rows with:

```powershell
python -m src.project_cli activate-first-real-batch-review-queue
```

The command is transactional and idempotent. It checks the canonical first-batch plan, staged image review, duplicate safeguards, existing queue rows, and the approval log. It may update only `data/real/annotations/sample_intake.csv`; it never creates approval or rejection decisions.

Prepare the runtime operator workbook with:

```powershell
python -m src.project_cli prepare-first-real-batch-manual-decisions
```

The workbook is stored under `data/real/runtime/first_batch_review/` and preserves operator entries between refreshes. Edit only the operator decision, rejection reason, and operator notes columns. The preparation command is read-only with respect to the live queue and approved dataset.

Verify these safeguards with:

```powershell
python -m src.project_cli verify-review-queue
```

## Manual decision validation and controlled application

Step 009.9 validates the operator decisions in the runtime workbook before any
live queue or approved-dataset change. Build the fingerprinted application plan
with:

```powershell
python -m src.project_cli validate-first-real-batch-manual-decisions
```

Application is allowed only when the validation readiness is
`READY_TO_APPLY`. Apply the exact validated decisions with:

```powershell
python -m src.project_cli apply-first-real-batch-manual-decisions
```

The apply command rejects stale workbook, queue, or canonical-plan
fingerprints. It delegates the actual approvals and rejections to the existing
transactional intake workflow and adds an outer rollback snapshot for the live
queue, annotations, approval log, manifest, tracked reports, and processed
images.

Verify Step 009.9 with:

```powershell
python -m src.project_cli verify-manual-decisions
```

## First real dataset capture and approved sample ingestion

Step 010 joins the capture, staging, review, and controlled decision layers into
the first operational real-dataset workflow.

After placing descriptively named photographs in
`data/real/capture_inbox/batch_001/`, run:

```powershell
python -m src.project_cli run-first-real-dataset-capture
```

This command may import and stage local photographs, activate validated pending
review rows, and refresh the runtime manual-decision workbook. It fingerprints
the approved dataset and never creates automatic decisions.

When validation reports `READY_TO_APPLY`, run:

```powershell
python -m src.project_cli finalize-first-real-dataset-ingestion
```

The finalization command requires the Step 009.9 fingerprinted plan, delegates
writes to the controlled transactional layer, and audits the approval log,
annotations, manifest, processed images, category coverage, front/detail pairs,
and remaining queue.

A complete 20-image batch becomes `FIRST_BATCH_INGESTED`. Rejected photographs
produce `RECAPTURE_REQUIRED` while valid approved samples remain ingested.

Verify Step 010 with:

```powershell
python -m src.project_cli verify-real-dataset-ingestion
```

## Open-license internet image collection

Step 010.1 adds a separate external development-image collection. It does not
replace or modify the real warehouse-photo workflow.

Collect five Wikimedia Commons candidates for every project category with:

```powershell
python -m src.project_cli collect-open-license-images
```

The collector stores the source file page, download URL, creator or credit,
license name, license URL, local SHA-256, dimensions, and modification note in:

```text
data/external/open_license/open_license_manifest.csv
```

Every new candidate remains `pending`. Build the local review gallery with:

```powershell
python -m src.project_cli build-open-license-review-gallery
```

Open `reports/external_dataset/open_license_review_gallery.html`, then edit only
the operator columns in `data/external/open_license/open_license_review.csv`.

Validate the files, metadata, licenses, hashes, and manual decisions with:

```powershell
python -m src.project_cli validate-open-license-images
```

The collection becomes `READY_FOR_EXTERNAL_DATASET` only when each of the ten
categories has at least five manually approved images.

Verify Step 010.1 with:

```powershell
python -m src.project_cli verify-open-license-dataset
```

## External dataset integration and training readiness

Step 010.2 converts the 50 manually approved open-license images into 150
image-text samples. Each approved image becomes one independent external part
group with `MATCH`, `PARTIAL_MATCH`, and `MISMATCH` descriptions.

Build the external-only grouped split and the integrated development + external
split with:

```powershell
python -m src.project_cli integrate-external-dataset
```

The deterministic external split uses three groups per category for training,
one for validation, and one for the locked test split. All samples from the
same `part_group_id` remain together.

Validate the approved-image catalog, generated samples, group isolation,
integrated split composition, image hashes, and test-lock fingerprints with:

```powershell
python -m src.project_cli validate-external-training-readiness
```

Verify the complete Step 010.2 workflow with:

```powershell
python -m src.project_cli verify-external-dataset-integration
```

Training-ready inputs are `data/processed/integrated_train.csv` and
`data/processed/integrated_validation.csv`. The integrated test split remains
fingerprinted and locked. Step 010.2 does not train or evaluate a model.

## Integrated training baselines and validation comparison

Step 010.3 trains three classical references and three Keras neural models on
`data/processed/integrated_train.csv`. It compares all six models only on
`data/processed/integrated_validation.csv`.

Run the complete integrated training workflow with:

```powershell
python -m src.project_cli run-integrated-training-validation
```

The workflow checks the canonical UTF-8/LF fingerprints of both locked test
CSVs before and after training. The locked test split was not loaded as model data, used for model fitting,
used for model selection, or evaluated.

Generated metrics, predictions, confusion matrices, neural training histories,
and the ranked six-model comparison are written below:

```text
reports/integrated_training/
```

Verify the outputs and the locked-test policy with:

```powershell
python -m src.project_cli verify-integrated-training-validation
```

The test split remains locked until the final model and evaluation procedure
are fixed explicitly in a later controlled step.

## Validation error analysis and controlled model improvement

Step 010.4 analyzes the integrated validation errors and compares one frozen
reference architecture with two predefined multimodal improvements. Every
candidate is trained with the same three fixed seeds, early-stopping policy,
training split, validation split, and model-selection rules.

Run the workflow with:

```powershell
python -m src.project_cli run-validation-error-analysis-model-improvement
```

The workflow loads only `data/processed/integrated_train.csv` and
`data/processed/integrated_validation.csv`. It records exact-image and text
overlap diagnostics, per-class and per-category errors, high-confidence errors,
candidate disagreements, multi-seed stability, and the controlled selection
decision below:

```text
reports/validation_model_improvement/
```

The locked test split was not loaded, evaluated, used for candidate ranking, or
used by the acceptance gate. Step 010.4 never authorizes final test evaluation.

Verify the complete workflow with:

```powershell
python -m src.project_cli verify-validation-model-improvement
```

The selection decision may accept a stable validation improvement or retain the
reference model when the predefined acceptance thresholds are not met.

## Project verification

Run the complete set of integrity, dataset, workflow, and test-lock
verifications from the repository root:

```powershell
python -m src.project_cli verify-project
```

<!-- FINAL_MODEL_FREEZE_START -->
## Final model and locked-test evaluation protocol freeze

Step 010.5 freezes the selected Step 010.3 `keras_multimodal` reference model recipe after the Step 010.4 `REFERENCE_RETAINED` decision. The frozen contract includes the architecture, preprocessing, label order, random seed, optimizer, early-stopping checkpoint rule, environment fingerprint, final metrics, and one-shot reporting procedure.

Create or refresh the committed freeze artifacts with:

    python -m src.project_cli freeze-final-model-evaluation-protocol

Verify the complete freeze and closed authorization gate with:

    python -m src.project_cli verify-final-model-freeze

The locked test CSV files were not opened, parsed, trained on, predicted on, or evaluated by this workflow. Test authorization remains `false`. Protocol freeze alone does not unlock either test file; any future one-shot final evaluation requires a separate explicit authorization step after submission review.
<!-- FINAL_MODEL_FREEZE_END -->

<!-- FINAL_EXAM_NOTEBOOK_START -->
## Final exam notebook and research narrative

Step 010.6 creates and verifies the executed final exam notebook without changing the frozen model or opening the locked test split.

```powershell
python -m src.project_cli build-final-exam-notebook
python -m src.project_cli verify-final-exam-notebook
```

The final notebook is generated from committed train, validation, model-comparison, error-analysis, and model-freeze artifacts. It includes related work, clear visualizations, limitations, and formal references. The committed notebook contains saved outputs for direct review on GitHub.

Current Step 010.6 policy:

- model retraining: `false`;
- model selection change: `false`;
- locked test CSV access: `false`;
- test split used: `false`;
- final test evaluation authorized: `false`.
<!-- FINAL_EXAM_NOTEBOOK_END -->

<!-- NOTEBOOK_QUALITY_AUDIT_START -->
## Notebook Execution, Visual QA and Citation Audit

Step 010.7 re-executes the committed final exam notebook from the project
environment and applies a separate quality gate to the saved outputs.

```powershell
python -m src.project_cli run-notebook-quality-audit
python -m src.project_cli verify-notebook-quality-audit
```

The audit verifies:

- deterministic scientific outputs after a fresh notebook execution;
- sequential execution counts with no error outputs or transient timestamps;
- six readable, non-blank and uniquely fingerprinted figures;
- numeric consistency between the narrative, predictions, metrics and
  confusion matrix;
- correct separation between the retained Step 010.3 model's 28 validation
  errors and the 35 errors from the separate Step 010.4 controlled reference
  rerun;
- six inline-numbered citations linked to primary papers or official
  documentation;
- unchanged locked-test artifacts and a closed final-test authorization gate.

Current Step 010.7 policy:

- model retraining: `false`;
- model selection change: `false`;
- locked test CSV access: `false`;
- test split used: `false`;
- final test evaluation authorized: `false`.
<!-- NOTEBOOK_QUALITY_AUDIT_END -->

<!-- EXAM_SUBMISSION_READINESS_START -->
## Exam submission readiness and clean release checkpoint

Step 010.8 audits the repository as a final SoftUni submission package and creates a deterministic readiness record without training a model or opening the locked test split.

```powershell
python -m src.project_cli build-exam-submission-readiness
python -m src.project_cli verify-exam-submission-readiness
python -m pytest -q
python -m src.project_cli verify-project
```

The committed release evidence is under:

```text
reports/exam_submission_readiness/
```

It contains the submission checklist, clean-clone protocol, readiness summary, machine-readable status and normalized SHA-256 release manifest. The packaged PowerShell clean-clone verifier should be run after the Step 010.8 commit is pushed to `main`.

Current Step 010.8 policy:

- model retraining: `false`;
- model selection change: `false`;
- locked test CSV access: `false`;
- test split used: `false`;
- final test evaluation authorized: `false`.
<!-- EXAM_SUBMISSION_READINESS_END -->
