from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from src.exam_first_config import (
    BASE_CHECKPOINT_COMMIT,
    BASE_CHECKPOINT_COMMIT_COUNT,
    COURSE_ALIGNMENT_PATH,
    DEFENSE_NOTES_PATH,
    EXAM_DIR,
    EXAM_README_PATH,
    FIGURES_DIR,
    GENERATED_ARTIFACTS,
    HISTORICAL_README_PATH,
    HISTORICAL_NOTEBOOK_CATALOGUE_PATH,
    MANIFEST_PATH,
    NOTEBOOK_CATALOGUE_PATH,
    NOTEBOOK_PATH,
    PRIMARY_QUESTION,
    PROJECT_ROOT,
    READINESS,
    REPORT_DIR,
    REPRODUCTION_PATH,
    ROOT_README_PATH,
    SOURCE_ARCHIVE,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_ARTIFACTS,
    STATUS_PATH,
    STEP,
    SUMMARY_PATH,
    SUPPLEMENTARY_INDEX_PATH,
    TEXT_HASH_SUFFIXES,
    project_relative,
)


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_evidence() -> dict[str, Any]:
    train = pd.read_csv(PROJECT_ROOT / "data/processed/integrated_train.csv")
    validation = pd.read_csv(
        PROJECT_ROOT / "data/processed/integrated_validation.csv"
    )
    comparison = pd.read_csv(
        PROJECT_ROOT
        / "reports/integrated_training/validation_comparison.csv"
    )
    errors = pd.read_csv(
        PROJECT_ROOT
        / "reports/validation_model_improvement/validation_error_analysis.csv"
    )
    error_summary = read_json(
        PROJECT_ROOT
        / "reports/validation_model_improvement/validation_error_analysis.json"
    )
    confusion = pd.read_csv(
        PROJECT_ROOT
        / "reports/validation_model_improvement/candidates/"
        "reference_multimodal/validation_confusion_matrix.csv",
        index_col=0,
    )

    predictions: dict[str, pd.DataFrame] = {}
    for slug in ("keras_text", "keras_image", "keras_multimodal"):
        predictions[slug] = pd.read_csv(
            PROJECT_ROOT
            / f"reports/integrated_training/{slug}/validation_predictions.csv"
        )

    return {
        "train": train,
        "validation": validation,
        "comparison": comparison,
        "errors": errors,
        "error_summary": error_summary,
        "confusion": confusion,
        "predictions": predictions,
    }


def build_figures(evidence: dict[str, Any]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison = evidence["comparison"].copy()
    confusion = evidence["confusion"].copy()
    errors = evidence["errors"].copy()
    predictions = evidence["predictions"]

    plot_data = comparison.sort_values(
        "integrated_validation_macro_f1", ascending=True
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(
        plot_data["model"], plot_data["integrated_validation_macro_f1"]
    )
    axis.set_xlabel("Validation macro F1")
    axis.set_title("Primary model comparison")
    axis.set_xlim(0.0, 0.6)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion.to_numpy(), aspect="auto")
    axis.set_xticks(range(3), ["MATCH", "PARTIAL", "MISMATCH"])
    axis.set_yticks(range(3), ["MATCH", "PARTIAL", "MISMATCH"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Retained multimodal validation confusion matrix")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(
                column,
                row,
                int(confusion.iloc[row, column]),
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    by_source = (
        errors.groupby("source").size().rename("errors").reset_index()
    )
    totals = (
        evidence["validation"]
        .groupby("source")
        .size()
        .rename("samples")
        .reset_index()
    )
    by_source = by_source.merge(totals, on="source")
    by_source["error_rate"] = by_source["errors"] / by_source["samples"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(by_source["source"], by_source["error_rate"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Validation error rate")
    axis.set_title("Domain shift: generated and real images")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "error_rates.png", dpi=160)
    plt.close(figure)

    correct_counts = []
    for slug, label in (
        ("keras_text", "Text-only"),
        ("keras_image", "Image-only"),
        ("keras_multimodal", "Multimodal"),
    ):
        correct_counts.append(
            {
                "model": label,
                "correct": int(predictions[slug]["is_correct"].sum()),
            }
        )
    contribution = pd.DataFrame(correct_counts)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(contribution["model"], contribution["correct"])
    axis.set_ylim(0, 60)
    axis.set_ylabel("Correct validation samples")
    axis.set_title("What each neural input setup resolves")
    for index, value in enumerate(contribution["correct"]):
        axis.text(index, int(value) + 1, str(int(value)), ha="center")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "multimodal_contribution.png", dpi=160)
    plt.close(figure)




def build_historical_readme() -> str:
    return "# Automotive Part Image-Text Matching\n\n**Notebook Execution, Visual QA and Citation Audit:** the historical Step 010.7 quality gate remains committed and independently verifiable.\n\n**Exam submission evidence:** the current Step 011.4 notebook and the historical audited notebook are both directly reviewable on GitHub.\n\n> **Final exam submission — deadline 11 August 2026, 16:00:** [Open the teacher-facing executed notebook](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/03_final_exam_submission.ipynb) · [Read the final checklist](reports/final_submission/submission_checklist.md) · [Review the 98/100 self-assessment](reports/final_submission/self_assessment.md) · [Use the defense guide](reports/final_submission/defense_guide.md)\n\nThis project studies whether a model can determine if a photograph of an automotive part matches a short text description.\n\nThe task is a three-class classification problem:\n\n- `MATCH` - the image and description represent the same part category;\n- `PARTIAL_MATCH` - the description represents a different category from the same automotive system;\n- `MISMATCH` - the description represents a category from a different automotive system.\n\nThe implemented comparison includes classical baselines, text-only and image-only neural networks, and a multimodal neural network.\n\n## Integrated validation results\n\nThe final model selection uses the integrated grouped validation split: 60 samples from 20 independent physical-part groups, balanced across the three relationship labels. The locked test split remains unused and unauthorized.\n\n| Model | Integrated validation accuracy | Integrated validation macro F1 |\n|---|---:|---:|\n| Majority baseline | 0.3333 | 0.1667 |\n| TF-IDF + Logistic Regression | 0.4167 | 0.3300 |\n| Image pixels + Logistic Regression | 0.3333 | 0.1667 |\n| Keras text model | 0.4167 | 0.3300 |\n| Keras image model | 0.3333 | 0.1667 |\n| Keras multimodal model | **0.5333** | **0.5208** |\n\nThe multimodal model is the retained final recipe. The earlier generated-development result was higher (`0.7667` accuracy and `0.7696` macro F1), but it is reported only as development evidence rather than the final validation result.\n\n## Deep Learning Error Analysis\n\nThe teacher-facing submission does not hide model failures. The controlled reference analysis contains **35 errors among 60 validation samples**. The intermediate `PARTIAL_MATCH` class accounts for 20 errors and is split evenly toward `MATCH` and `MISMATCH`. Real open-license images have a higher error rate (`66.7%`) than generated validation images (`50.0%`), which is consistent with domain shift caused by background, lighting, scale and perspective.\n\nStep 011.1 also contains nine controlled failure diagnostics: unscaled images, unsuitable learning rates, excessive dropout, label misalignment, deep sigmoid gradients, a missing optimizer update, and validation-training safeguards. Step 011.3A adds ranking, augmentation and occlusion diagnostics while explicitly avoiding unsupported human-explainability claims.\n\n- [Deep Learning error-analysis report](reports/final_submission/deep_learning_error_analysis.md)\n- [Final exam rubric evidence matrix](reports/final_submission/rubric_evidence_matrix.csv)\n- [Teacher-facing submission summary](reports/final_submission/final_submission_summary.md)\n\n## Current course-exercise evidence\n\n| Suite | Completed scope | Recorded training runs | Controlled gates |\n|---|---:|---:|---:|\n| Deep Learning Fundamentals | 10/10 | 35 | 0 |\n| Transformers & Sequence Modelling core | 9 core tasks | 21 | pretrained transformer |\n| Vision Models core | 6/9 | 48 | pretrained backbone, fine-tuning, genuine human annotation |\n\nAll course experiments use committed train and validation evidence only. The locked test split remains unused and unauthorized, and none of the Step 011 suites changes the retained production model automatically.\n\n## Project structure\n\n- `data/development/` - deterministic generated images and metadata;\n- `data/processed/` - grouped train, validation, and test split files;\n- `data/real/` - real-data annotation templates and approved processed images;\n- `src/` - reusable project modules and command-line entry point;\n- `models/` - saved model artifacts;\n- `reports/` - dataset, training, and evaluation reports;\n- `notebooks/` - Jupyter presentation notebooks for the development experiment;\n- `app/` - reserved demonstration application directory;\n- `tests/` - automated tests.\n\n## Jupyter notebooks\n\nThe main exam presentation is now the Step 011.4 teacher-facing notebook:\n\n- [Open `notebooks/03_final_exam_submission.ipynb` directly on GitHub](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/03_final_exam_submission.ipynb)\n- Local path: `notebooks/03_final_exam_submission.ipynb`\n\nBuild, verify, and open it from the repository root:\n\n```powershell\npython -m src.project_cli build-final-submission-notebook\npython -m src.project_cli verify-final-submission\npython -m jupyter notebook notebooks/03_final_exam_submission.ipynb\n```\n\nIt aligns the project directly to the eight exam criteria and integrates the current validation results, previous research, data provenance, testing evidence, Step 011 course experiments, Deep Learning errors, controlled failure diagnostics, limitations, self-assessment and defense summary. It reads committed reports only and does not train models or access the locked test split.\n\nThe earlier notebooks remain as immutable research evidence. The historical Step 010.6 notebook is also [directly reviewable on GitHub](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/02_final_exam_project.ipynb). The current teacher-facing notebook contains explicit limitations and conclusion sections.\n\n```text\nnotebooks/02_final_exam_project.ipynb\nnotebooks/01_development_experiment.ipynb\nnotebooks/course_coverage/01_fundamentals_experiments.ipynb\nnotebooks/course_coverage/02_sequence_model_comparison.ipynb\nnotebooks/course_coverage/03_vision_model_comparison.ipynb\nnotebooks/course_coverage/04_scoring_ranking_explainability.ipynb\n```\n\n## Full course exercise coverage roadmap\n\nStep 011.0 maps all 29 tasks from the supplied Deep Learning Fundamentals, Transformers and Sequence Modelling, and Vision Models exercises to controlled experiments for this automotive image-text matching problem. This checkpoint defines architecture, metrics, evidence paths, resource gates, and execution policy; it does not claim that the new experiments have already run.\n\n- [Full course coverage matrix](docs/course_coverage/full_course_coverage_matrix.md)\n- [Locked evaluation plan](docs/course_coverage/locked_evaluation_plan.md)\n- [Experiment execution policy](docs/course_coverage/experiment_execution_policy.md)\n- [Machine-readable experiment registry](data/experiment_registry/course_coverage_registry.json)\n- [Planned course-coverage notebooks](notebooks/course_coverage/README.md)\n\nEvery Step 011 experiment is limited to the committed train and validation splits. The test split remains locked, unused, and unauthorized. Human agreement metrics are planned only after genuine independent annotations exist; simulated annotators are not accepted as evidence.\n\nBuild and verify the planning checkpoint with:\n\n```powershell\npython -m src.project_cli build-course-coverage-architecture\npython -m src.project_cli verify-course-coverage-architecture\n```\n\n## Environment setup\n\nRun the commands from the repository root.\n\n### Windows PowerShell\n\n```powershell\npy -3.13 -m venv .venv\nSet-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned\n.\\.venv\\Scripts\\Activate.ps1\npython -m pip install --upgrade pip\npython -m pip install -r requirements-lock.txt\n```\n\nThe smaller `requirements.txt` file contains the direct project dependencies. `requirements-lock.txt` records the complete tested environment in UTF-8.\n\n## Project command line\n\nUse the package command line instead of executing files under `src/` directly:\n\n```powershell\npython -m src.project_cli --help\n```\n\nAvailable workflows:\n\n```powershell\npython -m src.project_cli environment\npython -m src.project_cli create-development-data\npython -m src.project_cli validate-development-data\npython -m src.project_cli create-grouped-split\npython -m src.project_cli run-baselines\npython -m src.project_cli train-text\npython -m src.project_cli train-image\npython -m src.project_cli train-multimodal\npython -m src.project_cli verify-development-pipeline\npython -m src.project_cli build-course-coverage-architecture\npython -m src.project_cli verify-course-coverage-architecture\npython -m src.project_cli run-fundamentals-suite\npython -m src.project_cli verify-fundamentals-suite\npython -m src.project_cli run-sequence-suite\npython -m src.project_cli verify-sequence-suite\npython -m src.project_cli run-vision-suite\npython -m src.project_cli build-vision-notebooks\npython -m src.project_cli verify-vision-suite\npython -m src.project_cli build-final-submission-notebook\npython -m src.project_cli verify-final-submission\npython -m src.project_cli validate-real-data\npython -m src.project_cli verify-real-dataset-foundation\npython -m src.project_cli review-real-intake\npython -m src.project_cli apply-real-intake\npython -m src.project_cli verify-sample-intake\npython -m src.project_cli prepare-first-real-batch\npython -m src.project_cli dry-run-first-real-batch\npython -m src.project_cli verify-first-batch-preparation\npython -m src.project_cli stage-first-real-batch-capture\npython -m src.project_cli verify-capture-staging\n```\n\nThe command modules are imported only when selected. Displaying CLI help or running a non-neural workflow does not import TensorFlow unnecessarily.\n\n## Reproducible development workflow\n\nThe following sequence rebuilds the generated dataset and all development results:\n\n```powershell\npython -m src.project_cli create-development-data\npython -m src.project_cli validate-development-data\npython -m src.project_cli create-grouped-split\npython -m src.project_cli run-baselines\npython -m src.project_cli train-text\npython -m src.project_cli train-image\npython -m src.project_cli train-multimodal\npython -m pytest -q\npython -m src.project_cli verify-development-pipeline\n```\n\nNeural-network training can produce small numeric differences across hardware and TensorFlow builds. Random seeds and deterministic TensorFlow operations are configured where supported.\n\n## Direct module execution\n\nIndividual modules can also be executed with `python -m`:\n\n```powershell\npython -m src.validate_development_dataset\npython -m src.create_grouped_split\npython -m src.run_baseline_models\npython -m src.train_keras_text_model\npython -m src.train_keras_image_model\npython -m src.train_multimodal_model\n```\n\nDo not use commands such as `python src/validate_development_dataset.py`. The source files use package imports and are intended to run from the repository root as modules.\n\n## Split and evaluation policy\n\nAll rows from the same `part_group_id` remain in one split. This prevents photographs or descriptions of the same physical part from crossing between train, validation, and test data.\n\nThe current models are selected and compared only on the validation split. The test split must remain unused until the final model and evaluation procedure are fixed.\n\n## Real dataset intake\n\nThe generated dataset is only for pipeline development. The final experiment requires real photographs collected and annotated according to:\n\n```text\nreports/real_dataset/collection_protocol.md\n```\n\nThe real dataset uses a separate `real_` identifier namespace and a dedicated directory tree:\n\n```text\ndata/real/originals/                  local untouched photographs\ndata/real/staging/                    local temporary review files\ndata/real/processed/images/           approved reproducible images\ndata/real/annotations/part_groups.csv physical-part annotations\ndata/real/annotations/images.csv      image annotations\ndata/real/processed/real_image_manifest.csv generated intake manifest\n```\n\nOriginal and staging photographs are excluded from Git. Approved processed images, annotation tables, and the generated manifest remain separate from `data/development/`.\n\nValidate the current intake and regenerate the approved-image manifest with:\n\n```powershell\npython -m src.project_cli validate-real-data\n```\n\nThe validator checks schemas, identifiers, category-family mappings, approval relationships, safe paths, readable images, SHA-256 duplicates, repeated views, and overlap with development identifiers or image content. Empty annotation templates are accepted as an `EMPTY_FOUNDATION` state before collection begins.\n\nVerify the complete Step 009 foundation with:\n\n```powershell\npython -m src.project_cli verify-real-dataset-foundation\n```\n\n## Real sample intake and approval workflow\n\nStep 009.1 adds a controlled queue for the first real photographs. Copy a candidate photograph to `data/real/staging/`, rename it to its intake identifier such as `intake_000001.jpg`, and add one row to:\n\n```text\ndata/real/annotations/sample_intake.csv\n```\n\nEach queue row has one of three decisions:\n\n- `pending` - review the image and metadata but do not modify the dataset;\n- `approved` - normalize the photograph to an EXIF-free RGB PNG and add it to the approved annotations;\n- `rejected` - record the decision and reason without adding an image to the dataset.\n\nAlways review the queue first:\n\n```powershell\npython -m src.project_cli review-real-intake\n```\n\nThe review checks identifier and path safety, category-family mapping, description semantics, image readability and dimensions, luminance and contrast warnings, duplicate hashes, existing annotation conflicts, and development-data overlap. It does not modify annotations or processed images.\n\nAfter checking `reports/real_dataset/sample_intake_review.md`, apply the explicit decisions with:\n\n```powershell\npython -m src.project_cli apply-real-intake\n```\n\nThe apply command is transactional. It updates `part_groups.csv`, `images.csv`, `approval_log.csv`, the approved image manifest, and the remaining queue only if the final real-dataset validation passes. Any failure restores the previous files and removes newly created processed images.\n\nApproved photographs are written to `data/real/processed/images/<image_id>.png`. The source file remains in the ignored staging directory until it is removed manually. Rejected and approved rows are removed from the queue after a successful apply; pending rows remain.\n\nVerify the complete workflow with:\n\n```powershell\npython -m src.project_cli verify-sample-intake\n```\n\n## Tests\n\nRun the complete test suite from the repository root:\n\n```powershell\npython -m pytest -q\n```\n\nThe Step 008.2 integrity verifier checks CLI module paths, documentation commands, Markdown fences, the real-data protocol, and UTF-8 lock-file encoding:\n\n```powershell\npython -m src.project_cli verify-development-pipeline\n```\n\nThe Step 009 verifier checks the real-data directory boundary, annotation and manifest schemas, CLI registration, Git ignore policy, and the current intake validation state:\n\n```powershell\npython -m src.project_cli verify-real-dataset-foundation\n```\n\nThe Step 009.1 verifier checks the sample queue and approval-log schemas, CLI documentation, transactional safeguards, and the current review state:\n\n```powershell\npython -m src.project_cli verify-sample-intake\n```\n\n## First real sample batch preparation and dry run\n\nStep 009.2 defines a balanced first collection batch without adding invented\nreal samples to the approved dataset. The committed plan is:\n\n```text\ndata/real/annotations/first_batch_plan.csv\n```\n\nIt reserves 20 planned images from 10 physical parts: one physical part from\neach configured category, with `front` and `detail` views. The plan uses\n`batch_001`, intake IDs `intake_000001` through `intake_000020`, and the\n`real_<category>_001` group namespace.\n\nPrepare or refresh the plan report and queue preview with:\n\n```powershell\npython -m src.project_cli prepare-first-real-batch\n```\n\nThe command scans the expected staging paths, validates category balance,\nidentifiers, views, metadata consistency, live-queue conflicts, and any files\nthat have already been captured. It writes\n`data/real/processed/first_batch_queue_preview.csv` but does not change\n`sample_intake.csv`, approve an image, or modify the real dataset.\n\nPlace photographs under `data/real/staging/` with the exact filenames listed\nin the plan. JPEG is the expected capture format for this first batch. Missing\nfiles produce the valid `AWAITING_CAPTURE` preparation state.\n\nRun the controlled intake simulation with:\n\n```powershell\npython -m src.project_cli dry-run-first-real-batch\n```\n\nThe dry run reviews captured files, simulates approval and PNG normalization\ninside temporary storage, checks prospective annotations and duplicate\nprotection, and compares live-state fingerprints before and after the\nsimulation. It does not approve, queue, process, move, or delete real files.\nManual review and the Step 009.1 commands remain mandatory before any actual\napproval.\n\nVerify all Step 009.2 safeguards with:\n\n```powershell\npython -m src.project_cli verify-first-batch-preparation\n```\n## First real batch capture, staging and review readiness\n\nStep 009.3 turns the committed `batch_001` plan into a controlled local\ncapture workflow without approving any image. Step 009.4 adds the supported\nfile-naming and local-import boundary. Place descriptively named JPEG or PNG\nfiles in:\n\n```text\ndata/real/capture_inbox/batch_001/\n```\n\nUse names such as `real_starter_001_front.jpg` and run\n`import-first-real-batch`. The importer copies the original bytes into\n`data/real/originals/batch_001/`. Then run the capture and staging command from\nthe repository root:\n\n```powershell\npython -m src.project_cli stage-first-real-batch-capture\n```\n\nThe command applies EXIF orientation, converts each source to a reproducible\nRGB JPEG staging file, checks exact duplicates and development overlap, and\nnever overwrites a conflicting staging destination. New staging writes are\ntransactional. Original photographs remain unchanged in the ignored originals\ndirectory.\n\nThe command generates:\n\n```text\ndata/real/processed/first_batch_capture_inventory.csv\ndata/real/processed/first_batch_review_queue_draft.csv\nreports/real_dataset/first_batch_capture_readiness.md\n```\n\nThe review queue draft contains only `pending` decisions. It is separate from\n`sample_intake.csv`; the live queue, approval log, manifest, and approved image\ndirectory are not modified. Review the inventory and report manually before\ncopying acceptable draft rows into the live queue. Actual approval still uses\n`review-real-intake` followed by `apply-real-intake`.\n\nWith no local photographs, the correct state is `AWAITING_CAPTURE`. A partial\nbatch is `CAPTURE_IN_PROGRESS`. When all 20 planned files pass review, the\nstate becomes `READY_FOR_MANUAL_QUEUE_IMPORT`.\n\nVerify the Step 009.3 safeguards with:\n\n```powershell\npython -m src.project_cli verify-capture-staging\n```\n\n## First real batch file naming and local import\n\nUse descriptive filenames for the 20 first-batch photographs. Do not name local photographs with internal `intake_` identifiers. The required pattern is:\n\n```text\nreal_<part_category>_001_<view>.jpg\n```\n\nExamples:\n\n```text\nreal_starter_001_front.jpg\nreal_starter_001_detail.jpg\nreal_brake_disc_001_front.jpg\nreal_air_filter_001_detail.jpg\n```\n\nThe exact filename-to-intake mapping is committed in:\n\n```text\ndata/real/annotations/first_batch_capture_file_map.csv\n```\n\nThe complete capture checklist is:\n\n```text\nreports/real_dataset/first_batch_capture_checklist.md\n```\n\nPlace renamed JPEG or PNG files in the ignored local inbox:\n\n```text\ndata/real/capture_inbox/batch_001/\n```\n\nThen run:\n\n```powershell\npython -m src.project_cli import-first-real-batch\n```\n\nThe importer copies original bytes into `data/real/originals/batch_001/` without image conversion. It blocks unclear filenames, duplicate content, unreadable images, multiple extensions for one planned photograph, and conflicts with an existing original. Writes are transactional and do not modify staging, annotations, the live queue, approval log, or approved manifest.\n\nReview:\n\n```text\ndata/real/processed/first_batch_local_import_inventory.csv\nreports/real_dataset/first_batch_local_import_readiness.md\n```\n\nWhen the readiness is `READY_FOR_STAGING`, continue with:\n\n```powershell\npython -m src.project_cli stage-first-real-batch-capture\n```\n\nThe staging workflow accepts the new descriptive filenames and still supports the earlier technical intake stems for backward compatibility. Verify the naming and import safeguards with:\n\n```powershell\npython -m src.project_cli verify-local-capture-import\n```\n\n## First real batch operator guide and capture session\n\nUse the practical operator guide before photographing the first batch:\n\n```text\nreports/real_dataset/first_batch_operator_guide.md\n```\n\nPrepare or refresh the capture-session worksheet with:\n\n```powershell\npython -m src.project_cli prepare-first-real-batch-session\n```\n\nThe command groups the 20 planned photographs into 10 physical-part pairs, reports the exact missing `front` and `detail` filenames, selects the next capture, and writes:\n\n```text\ndata/real/processed/first_batch_capture_session.csv\nreports/real_dataset/first_batch_capture_session_readiness.json\nreports/real_dataset/first_batch_capture_session_readiness.md\n```\n\nThe preparation command does not copy, convert, approve, queue, or delete photographs. It fingerprints the local inbox, immutable originals, staging, annotations, live queue, approval log, and manifest before and after the scan. Safe readiness values are `AWAITING_CAPTURE`, `CAPTURE_SESSION_IN_PROGRESS`, `READY_FOR_LOCAL_IMPORT`, and `READY_FOR_STAGING`.\n\nVerify the operator guide and session-preparation safeguards with:\n\n```powershell\npython -m src.project_cli verify-capture-session\n```\n\n## First real batch capture dashboard and progress tracking\n\nBuild the local operator dashboard at any point during capture, import,\nstaging, review, or approval:\n\n```powershell\npython -m src.project_cli build-first-real-batch-dashboard\n```\n\nThe self-contained dashboard is written to:\n\n```text\nreports/real_dataset/first_batch_capture_dashboard.html\n```\n\nIts machine-readable and review outputs are:\n\n```text\ndata/real/processed/first_batch_capture_progress.csv\nreports/real_dataset/first_batch_capture_dashboard.json\nreports/real_dataset/first_batch_capture_progress_summary.md\n```\n\nThe dashboard tracks every planned photograph from `AWAITING_CAPTURE` through\n`APPROVED_DATASET`, shows the next required action, and reports overall and\nper-category progress. It does not copy, convert, queue, approve, reject, or\ndelete data. Input fingerprints must remain unchanged.\n\nVerify the dashboard and progress safeguards with:\n\n```powershell\npython -m src.project_cli verify-capture-dashboard\n```\n\n## First-batch capture execution and live progress\n\nRun a safe operator cycle after adding one or more planned photographs to the\nlocal capture inbox:\n\n```powershell\npython -m src.project_cli run-first-real-batch-capture-session\n```\n\nThe cycle imports valid captures, stages valid originals, refreshes the session\nworksheet, and writes a live dashboard under the Git-ignored\n`data/real/runtime/first_batch_capture/` directory. It rolls back originals\nand staging when a downstream operation fails and proves that the live queue,\napproval log, annotations, manifest, and tracked reports remain unchanged.\n\nUse the read-only refresh command between capture actions:\n\n```powershell\npython -m src.project_cli refresh-first-real-batch-live-progress\n```\n\nOpen `data/real/runtime/first_batch_capture/live_dashboard.html` to see the\nlatest pipeline progress. These commands never queue or approve samples.\nVerify the execution safeguards with:\n\n```powershell\npython -m src.project_cli verify-capture-execution\n```\n\n## First batch review queue and manual decision preparation\n\nWhen the live capture dashboard reports review-ready staged images, activate only the validated pending draft rows with:\n\n```powershell\npython -m src.project_cli activate-first-real-batch-review-queue\n```\n\nThe command is transactional and idempotent. It checks the canonical first-batch plan, staged image review, duplicate safeguards, existing queue rows, and the approval log. It may update only `data/real/annotations/sample_intake.csv`; it never creates approval or rejection decisions.\n\nPrepare the runtime operator workbook with:\n\n```powershell\npython -m src.project_cli prepare-first-real-batch-manual-decisions\n```\n\nThe workbook is stored under `data/real/runtime/first_batch_review/` and preserves operator entries between refreshes. Edit only the operator decision, rejection reason, and operator notes columns. The preparation command is read-only with respect to the live queue and approved dataset.\n\nVerify these safeguards with:\n\n```powershell\npython -m src.project_cli verify-review-queue\n```\n\n## Manual decision validation and controlled application\n\nStep 009.9 validates the operator decisions in the runtime workbook before any\nlive queue or approved-dataset change. Build the fingerprinted application plan\nwith:\n\n```powershell\npython -m src.project_cli validate-first-real-batch-manual-decisions\n```\n\nApplication is allowed only when the validation readiness is\n`READY_TO_APPLY`. Apply the exact validated decisions with:\n\n```powershell\npython -m src.project_cli apply-first-real-batch-manual-decisions\n```\n\nThe apply command rejects stale workbook, queue, or canonical-plan\nfingerprints. It delegates the actual approvals and rejections to the existing\ntransactional intake workflow and adds an outer rollback snapshot for the live\nqueue, annotations, approval log, manifest, tracked reports, and processed\nimages.\n\nVerify Step 009.9 with:\n\n```powershell\npython -m src.project_cli verify-manual-decisions\n```\n\n## First real dataset capture and approved sample ingestion\n\nStep 010 joins the capture, staging, review, and controlled decision layers into\nthe first operational real-dataset workflow.\n\nAfter placing descriptively named photographs in\n`data/real/capture_inbox/batch_001/`, run:\n\n```powershell\npython -m src.project_cli run-first-real-dataset-capture\n```\n\nThis command may import and stage local photographs, activate validated pending\nreview rows, and refresh the runtime manual-decision workbook. It fingerprints\nthe approved dataset and never creates automatic decisions.\n\nWhen validation reports `READY_TO_APPLY`, run:\n\n```powershell\npython -m src.project_cli finalize-first-real-dataset-ingestion\n```\n\nThe finalization command requires the Step 009.9 fingerprinted plan, delegates\nwrites to the controlled transactional layer, and audits the approval log,\nannotations, manifest, processed images, category coverage, front/detail pairs,\nand remaining queue.\n\nA complete 20-image batch becomes `FIRST_BATCH_INGESTED`. Rejected photographs\nproduce `RECAPTURE_REQUIRED` while valid approved samples remain ingested.\n\nVerify Step 010 with:\n\n```powershell\npython -m src.project_cli verify-real-dataset-ingestion\n```\n\n## Open-license internet image collection\n\nStep 010.1 adds a separate external development-image collection. It does not\nreplace or modify the real warehouse-photo workflow.\n\nCollect five Wikimedia Commons candidates for every project category with:\n\n```powershell\npython -m src.project_cli collect-open-license-images\n```\n\nThe collector stores the source file page, download URL, creator or credit,\nlicense name, license URL, local SHA-256, dimensions, and modification note in:\n\n```text\ndata/external/open_license/open_license_manifest.csv\n```\n\nEvery new candidate remains `pending`. Build the local review gallery with:\n\n```powershell\npython -m src.project_cli build-open-license-review-gallery\n```\n\nOpen `reports/external_dataset/open_license_review_gallery.html`, then edit only\nthe operator columns in `data/external/open_license/open_license_review.csv`.\n\nValidate the files, metadata, licenses, hashes, and manual decisions with:\n\n```powershell\npython -m src.project_cli validate-open-license-images\n```\n\nThe collection becomes `READY_FOR_EXTERNAL_DATASET` only when each of the ten\ncategories has at least five manually approved images.\n\nVerify Step 010.1 with:\n\n```powershell\npython -m src.project_cli verify-open-license-dataset\n```\n\n## External dataset integration and training readiness\n\nStep 010.2 converts the 50 manually approved open-license images into 150\nimage-text samples. Each approved image becomes one independent external part\ngroup with `MATCH`, `PARTIAL_MATCH`, and `MISMATCH` descriptions.\n\nBuild the external-only grouped split and the integrated development + external\nsplit with:\n\n```powershell\npython -m src.project_cli integrate-external-dataset\n```\n\nThe deterministic external split uses three groups per category for training,\none for validation, and one for the locked test split. All samples from the\nsame `part_group_id` remain together.\n\nValidate the approved-image catalog, generated samples, group isolation,\nintegrated split composition, image hashes, and test-lock fingerprints with:\n\n```powershell\npython -m src.project_cli validate-external-training-readiness\n```\n\nVerify the complete Step 010.2 workflow with:\n\n```powershell\npython -m src.project_cli verify-external-dataset-integration\n```\n\nTraining-ready inputs are `data/processed/integrated_train.csv` and\n`data/processed/integrated_validation.csv`. The integrated test split remains\nfingerprinted and locked. Step 010.2 does not train or evaluate a model.\n\n## Integrated training baselines and validation comparison\n\nStep 010.3 trains three classical references and three Keras neural models on\n`data/processed/integrated_train.csv`. It compares all six models only on\n`data/processed/integrated_validation.csv`.\n\nRun the complete integrated training workflow with:\n\n```powershell\npython -m src.project_cli run-integrated-training-validation\n```\n\nThe workflow checks the canonical UTF-8/LF fingerprints of both locked test\nCSVs before and after training. The locked test split was not loaded as model data, used for model fitting,\nused for model selection, or evaluated.\n\nGenerated metrics, predictions, confusion matrices, neural training histories,\nand the ranked six-model comparison are written below:\n\n```text\nreports/integrated_training/\n```\n\nVerify the outputs and the locked-test policy with:\n\n```powershell\npython -m src.project_cli verify-integrated-training-validation\n```\n\nThe test split remains locked until the final model and evaluation procedure\nare fixed explicitly in a later controlled step.\n\n## Validation error analysis and controlled model improvement\n\nStep 010.4 analyzes the integrated validation errors and compares one frozen\nreference architecture with two predefined multimodal improvements. Every\ncandidate is trained with the same three fixed seeds, early-stopping policy,\ntraining split, validation split, and model-selection rules.\n\nRun the workflow with:\n\n```powershell\npython -m src.project_cli run-validation-error-analysis-model-improvement\n```\n\nThe workflow loads only `data/processed/integrated_train.csv` and\n`data/processed/integrated_validation.csv`. It records exact-image and text\noverlap diagnostics, per-class and per-category errors, high-confidence errors,\ncandidate disagreements, multi-seed stability, and the controlled selection\ndecision below:\n\n```text\nreports/validation_model_improvement/\n```\n\nThe locked test split was not loaded, evaluated, used for candidate ranking, or\nused by the acceptance gate. Step 010.4 never authorizes final test evaluation.\n\nVerify the complete workflow with:\n\n```powershell\npython -m src.project_cli verify-validation-model-improvement\n```\n\nThe selection decision may accept a stable validation improvement or retain the\nreference model when the predefined acceptance thresholds are not met.\n\n## Project verification\n\nRun the complete set of integrity, dataset, workflow, and test-lock\nverifications from the repository root:\n\n```powershell\npython -m src.project_cli verify-project\n```\n\n<!-- FINAL_MODEL_FREEZE_START -->\n## Final model and locked-test evaluation protocol freeze\n\nStep 010.5 freezes the selected Step 010.3 `keras_multimodal` reference model recipe after the Step 010.4 `REFERENCE_RETAINED` decision. The frozen contract includes the architecture, preprocessing, label order, random seed, optimizer, early-stopping checkpoint rule, environment fingerprint, final metrics, and one-shot reporting procedure.\n\nCreate or refresh the committed freeze artifacts with:\n\n    python -m src.project_cli freeze-final-model-evaluation-protocol\n\nVerify the complete freeze and closed authorization gate with:\n\n    python -m src.project_cli verify-final-model-freeze\n\nThe locked test CSV files were not opened, parsed, trained on, predicted on, or evaluated by this workflow. Test authorization remains `false`. Protocol freeze alone does not unlock either test file; any future one-shot final evaluation requires a separate explicit authorization step after submission review.\n<!-- FINAL_MODEL_FREEZE_END -->\n\n<!-- FINAL_EXAM_NOTEBOOK_START -->\n## Final exam notebook and research narrative\n\nStep 010.6 creates and verifies the executed final exam notebook without changing the frozen model or opening the locked test split.\n\n```powershell\npython -m src.project_cli build-final-exam-notebook\npython -m src.project_cli verify-final-exam-notebook\n```\n\nThe final notebook is generated from committed train, validation, model-comparison, error-analysis, and model-freeze artifacts. It includes related work, clear visualizations, limitations, and formal references. The committed notebook contains saved outputs for direct review on GitHub.\n\nCurrent Step 010.6 policy:\n\n- model retraining: `false`;\n- model selection change: `false`;\n- locked test CSV access: `false`;\n- test split used: `false`;\n- final test evaluation authorized: `false`.\n<!-- FINAL_EXAM_NOTEBOOK_END -->\n\n<!-- NOTEBOOK_QUALITY_AUDIT_START -->\n## Notebook Execution, Visual QA and Citation Audit\n\nStep 010.7 re-executes the committed final exam notebook from the project\nenvironment and applies a separate quality gate to the saved outputs.\n\n```powershell\npython -m src.project_cli run-notebook-quality-audit\npython -m src.project_cli verify-notebook-quality-audit\n```\n\nThe audit verifies:\n\n- deterministic scientific outputs after a fresh notebook execution;\n- sequential execution counts with no error outputs or transient timestamps;\n- six readable, non-blank and uniquely fingerprinted figures;\n- numeric consistency between the narrative, predictions, metrics and\n  confusion matrix;\n- correct separation between the retained Step 010.3 model's 28 validation\n  errors and the 35 errors from the separate Step 010.4 controlled reference\n  rerun;\n- six inline-numbered citations linked to primary papers or official\n  documentation;\n- unchanged locked-test artifacts and a closed final-test authorization gate.\n\nCurrent Step 010.7 policy:\n\n- model retraining: `false`;\n- model selection change: `false`;\n- locked test CSV access: `false`;\n- test split used: `false`;\n- final test evaluation authorized: `false`.\n<!-- NOTEBOOK_QUALITY_AUDIT_END -->\n\n<!-- EXAM_SUBMISSION_READINESS_START -->\n## Exam submission readiness and clean release checkpoint\n\nStep 010.8 audits the repository as a final SoftUni submission package and creates a deterministic readiness record without training a model or opening the locked test split.\n\n```powershell\npython -m src.project_cli build-exam-submission-readiness\npython -m src.project_cli verify-exam-submission-readiness\npython -m pytest -q\npython -m src.project_cli verify-project\n```\n\nThe committed release evidence is under:\n\n```text\nreports/exam_submission_readiness/\n```\n\nIt contains the submission checklist, clean-clone protocol, readiness summary, machine-readable status and normalized SHA-256 release manifest. The packaged PowerShell clean-clone verifier should be run after the Step 010.8 commit is pushed to `main`.\n\nCurrent Step 010.8 policy:\n\n- model retraining: `false`;\n- model selection change: `false`;\n- locked test CSV access: `false`;\n- test split used: `false`;\n- final test evaluation authorized: `false`.\n<!-- EXAM_SUBMISSION_READINESS_END -->\n\n## Full-course experiments — Step 011.1\n\nThe Deep Learning Fundamentals exercise is implemented as a controlled train/validation-only suite covering Problems 1–10: EDA and batching, gradient diagnostics, one-batch overfit, training-loop integrity, SGD/RMSprop/Adam/AdamW and learning-rate comparisons, model capacity, regularization and architecture ablations, preprocessing alternatives, and deliberate failure diagnostics.\n\n- Evidence notebook: [`notebooks/course_coverage/01_fundamentals_experiments.ipynb`](notebooks/course_coverage/01_fundamentals_experiments.ipynb)\n- Methodology: [`docs/course_coverage/fundamentals_experimental_suite.md`](docs/course_coverage/fundamentals_experimental_suite.md)\n- Results: [`reports/course_coverage/fundamentals/fundamentals_suite_summary.md`](reports/course_coverage/fundamentals/fundamentals_suite_summary.md)\n- Machine-readable registry: [`data/experiment_registry/fundamentals_execution_registry.json`](data/experiment_registry/fundamentals_execution_registry.json)\n\n```powershell\npython -m src.project_cli run-fundamentals-suite\npython -m src.project_cli build-fundamentals-notebook\npython -m src.project_cli verify-fundamentals-suite\n```\n\nThese educational comparisons do not replace the frozen exam model. The test split remains locked and unused.\n\n## Transformers & Sequence Modelling — Step 011.2\n\nThe sequence-modelling exercise is implemented as a controlled train/validation-only suite. It covers deterministic text loading and tokenization, a dense embedding baseline, TF-IDF + logistic regression, TextCNN, GRU, LSTM, a small Transformer encoder, validation comparison, error analysis, and two-head attention inspection.\n\n- Evidence notebook: [`notebooks/course_coverage/02_sequence_model_comparison.ipynb`](notebooks/course_coverage/02_sequence_model_comparison.ipynb)\n- Methodology: [`docs/course_coverage/sequence_experimental_suite.md`](docs/course_coverage/sequence_experimental_suite.md)\n- Results: [`reports/course_coverage/sequence/sequence_suite_summary.md`](reports/course_coverage/sequence/sequence_suite_summary.md)\n- Machine-readable registry: [`data/experiment_registry/sequence_execution_registry.json`](data/experiment_registry/sequence_execution_registry.json)\n\n```powershell\npython -m src.project_cli run-sequence-suite\npython -m src.project_cli build-sequence-notebook\npython -m src.project_cli verify-sequence-suite\n```\n\nSEQ-001 through SEQ-009 are complete. SEQ-010 remains `DEFERRED_EXPLICIT_APPROVAL_REQUIRED`: no pretrained weights are downloaded until explicit approval is recorded together with the exact model revision and license. The production/final model remains unchanged, and the locked test split remains unused.\n"

def build_root_readme() -> str:
    return '# Automotive Part Image-Text Matching\n\n## Start here\n\n**Research question:** Does a multimodal neural network that combines an\nautomotive-part image and a short description classify their relationship\nbetter than image-only and text-only neural baselines?\n\nOpen the focused, executed exam notebook:\n\n- [Focused Deep Learning exam notebook](exam/01_focused_deep_learning_project.ipynb)\n- [Exam project guide](exam/README.md)\n- [Reproduction instructions](exam/reproduction.md)\n- [Oral defense notes](exam/defense_notes.md)\n\nThe project is deliberately presented as one analysis rather than as a\nportfolio of unrelated techniques.\n\n## Main result\n\nThe retained Keras multimodal model ranks first on the grouped integrated\nvalidation split.\n\n| Model | Input | Validation accuracy | Macro F1 |\n|---|---|---:|---:|\n| Majority baseline | none | 0.3333 | 0.1667 |\n| TF-IDF + Logistic Regression | text | 0.4167 | 0.3300 |\n| Image pixels + Logistic Regression | image | 0.3333 | 0.1667 |\n| Keras text model | text | 0.4167 | 0.3300 |\n| Keras image model | image | 0.3333 | 0.1667 |\n| **Keras multimodal model** | **image + text** | **0.5333** | **0.5208** |\n\nThe result is useful but not overstated. The retained model makes 35 errors\namong 60 validation samples. Twenty errors involve `PARTIAL_MATCH`, and real\nopen-license images are harder than generated images.\n\n## Leakage protection\n\nAll samples for one physical automotive part share a `part_group_id` and stay\nin one split. Train and validation have no group, image-ID, or image-path\noverlap. The locked test split has not been used, and final test evaluation has\nnot been authorized.\n\n## What the reviewer sees\n\nThe focused notebook shows:\n\n1. one precise Deep Learning question;\n2. the three labels and the data sources;\n3. grouped splitting and leakage checks;\n4. baselines and neural models on the same validation split;\n5. concrete cases where multimodal input helps;\n6. concrete errors and domain-shift evidence;\n7. limitations and an exact reproduction boundary.\n\n## Reproduce the exam-facing evidence\n\n```powershell\npython -m src.project_cli build-exam-first-submission\npython -m src.project_cli verify-exam-first-submission\npython -m jupyter notebook exam/01_focused_deep_learning_project.ipynb\n```\n\nThese commands do not train a model, open locked test CSV files, or change the\nretained production model.\n\n## Supporting evidence\n\nThe wider course exercises, historical notebooks, engineering tests, manifests,\nand audits are indexed in [supplementary/README.md](supplementary/README.md).\nThey support the main analysis without competing with it.\n\nFor compatibility with the Step 011.4 rubric checkpoint, the earlier full\nteacher-facing notebook remains available:\n\n- [Open the Step 011.4 notebook on GitHub](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/03_final_exam_submission.ipynb)\n- [Deep Learning Error Analysis](reports/final_submission/deep_learning_error_analysis.md)\n- [98/100 self-assessment](reports/final_submission/self_assessment.md)\n- [Submission checklist](reports/final_submission/submission_checklist.md)\n\nFinal exam submission deadline: **11 August 2026, 16:00 Europe/Sofia**.\n\n## Standard repository structure\n\n- `exam/` — recommended teacher-facing entry point;\n- `src/` — reusable implementation and command-line workflows;\n- `data/` — development, real, external, and grouped split artifacts;\n- `reports/` — committed metrics, predictions, manifests, and audits;\n- `notebooks/` — historical and course-exercise notebooks;\n- `tests/` — unit and integration tests;\n- `supplementary/` — index to supporting evidence.\n\nThe standard technical directories were not moved because their stable paths\nare part of reproducibility and historical manifests.\n'



def build_historical_notebook_catalogue() -> str:
    return '# Jupyter notebooks\n\n## Final exam submission\n\nThe main teacher-facing presentation is the committed executed notebook:\n\n- [Open `03_final_exam_submission.ipynb` directly on GitHub](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/03_final_exam_submission.ipynb)\n- Repository path: `notebooks/03_final_exam_submission.ipynb`\n- Submission deadline: **11 August 2026, 16:00**\n\nBuild and verify it from the repository root:\n\n```powershell\npython -m src.project_cli build-final-submission-notebook\npython -m src.project_cli verify-final-submission\n```\n\nOpen it locally with:\n\n```powershell\npython -m jupyter notebook notebooks/03_final_exam_submission.ipynb\n```\n\nThe notebook is aligned directly to the eight exam categories:\n\n- problem statement and real-world significance;\n- readable article layout;\n- modular and tested Python code;\n- previous research and six formal references;\n- data acquisition, licensing, cleaning, formatting and group isolation;\n- automated testing, controlled failure tests and locked-test safeguards;\n- saved tables, confusion matrices, error plots and course-suite figures;\n- conclusions, limitations, self-assessment and defense summary.\n\nIt includes a dedicated **Deep Learning Error Analysis and Failure Diagnostics** section with PARTIAL_MATCH confusion, domain shift, category errors, representative mistakes, controlled training failures and explicit explainability boundaries.\n\nThe notebook reads committed validation and report artifacts only. It does not retrain models, open a locked test CSV, authorize final test evaluation or change the production model.\n\n## Historical quality gate\n\nStep 010.7 remains the immutable notebook execution, visual QA, numeric consistency and citation audit for `02_final_exam_project.ipynb`.\n\n```powershell\npython -m src.project_cli run-notebook-quality-audit\npython -m src.project_cli verify-notebook-quality-audit\n```\n\n## Historical and specialist notebooks\n\nThe following notebooks remain committed as research evidence:\n\n- `01_development_experiment.ipynb` — generated development baseline;\n- [`02_final_exam_project.ipynb`](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/02_final_exam_project.ipynb) — Step 010.6/010.7 historical final narrative;\n- `course_coverage/01_fundamentals_experiments.ipynb` — 10/10 fundamentals tasks;\n- `course_coverage/02_sequence_model_comparison.ipynb` — sequence core experiments;\n- `course_coverage/03_vision_model_comparison.ipynb` — vision representation and augmentation;\n- `course_coverage/04_scoring_ranking_explainability.ipynb` — compatibility, ranking and occlusion.\n\n## Verification\n\nRun the current teacher-facing gate and the complete project verification:\n\n```powershell\npython -m src.project_cli verify-final-submission\npython -m src.project_cli verify-project\npython -m pytest -q\n```\n\nThe final submission checklist, self-assessment, Deep Learning error report and defense guide are under `reports/final_submission/`.\n'

def build_notebook_catalogue() -> str:
    return '# Notebook catalogue\n\n## Recommended exam entry point\n\nThe primary teacher-facing notebook is now:\n\n- [`exam/01_focused_deep_learning_project.ipynb`](../exam/01_focused_deep_learning_project.ipynb)\n\nIt presents one research question, grouped-split protection, the primary model\ncomparison, concrete successes, concrete errors, limitations, and reproduction.\n\n## Supporting notebooks\n\n- `notebooks/01_development_experiment.ipynb` — early development evidence;\n- `notebooks/02_final_exam_project.ipynb` — historical full project notebook;\n- `notebooks/03_final_exam_submission.ipynb` — Step 011.4 rubric-alignment evidence;\n- `notebooks/course_coverage/` — Fundamentals, sequence, vision, ranking, and controlled experiments.\n\nThe Step 011.4 notebook remains directly reviewable on GitHub:\n\nhttps://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/03_final_exam_submission.ipynb\n\nThe course-coverage notebooks are supporting evidence and are not intended to\nreplace the focused exam narrative.\n'

def build_exam_readme(evidence: dict[str, Any]) -> str:
    best = evidence["comparison"].sort_values("validation_rank").iloc[0]
    summary = evidence["error_summary"]
    return f"""# Exam project: start here

## One question

**{PRIMARY_QUESTION}**

The main submission is one focused, executed notebook:

- [`01_focused_deep_learning_project.ipynb`](01_focused_deep_learning_project.ipynb)

It presents the data, grouped split, model comparison, concrete successes,
concrete errors, limitations, and the exact reproduction boundary.

## Main result

The retained multimodal model reaches **{best['integrated_validation_accuracy']:.4f} accuracy**
and **{best['integrated_validation_macro_f1']:.4f} macro F1** on 60 validation samples
from 20 independent physical-part groups. It ranks above the text-only and
image-only neural baselines.

This is not presented as a solved problem. The retained model makes
**{summary['error_count']} errors**, including
**{int((evidence['errors']['true_label'] == 'PARTIAL_MATCH').sum())} errors involving the difficult
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
"""


def build_reproduction() -> str:
    return """# Reproduction

Run commands from the repository root.

## Rebuild the focused exam layer

```powershell
python -m src.project_cli build-exam-first-submission
python -m src.project_cli verify-exam-first-submission
```

These commands read committed train and validation artifacts, rebuild the
focused notebook and reports, and verify their hashes. They do not train a
model, open a locked test CSV, authorize final test evaluation, or change the
retained model.

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
prove branch, push, or commit-count state; the Step 011.5 status instead records
the declared source checkpoint and exact source-archive SHA-256.
"""


def build_defense_notes() -> str:
    return f"""# Oral defense notes

## State the question first

{PRIMARY_QUESTION}

## Explain the split before discussing accuracy

All rows that belong to one physical part share a `part_group_id` and remain in
one split. This prevents the model from seeing the same physical object during
training and validation under a different image-text pairing.

## Explain what the comparison shows

The multimodal model is compared with majority, classical text, classical
image, neural text-only, and neural image-only baselines on the same grouped
validation split. Its advantage is evidence that both modalities help, not
proof that every individual prediction uses both modalities correctly.

## Show concrete examples

The notebook contains examples where the multimodal model is correct while one
or both unimodal models fail. It also shows difficult errors, especially
`PARTIAL_MATCH`, where the description is related to the image but is not the
same part category.

## Be direct about weaknesses

The validation set is small, real images produce more errors than generated
images, the model predicts no `PARTIAL_MATCH` cases in the retained confusion
matrix, and the results do not establish deployment readiness or calibration.

## Evaluation boundary

The test split remains locked. The exam-facing build does not train models and
does not change the retained production recipe.
"""


def build_course_alignment() -> str:
    return """# Course-topic alignment

The main notebook stays focused on one multimodal classification question.
Course topics are evidence, not separate competing project stories.

| Course topic | Evidence in the repository | Status |
|---|---|---|
| Deep Learning Fundamentals | batches, preprocessing, gradients, optimizers, learning rates, capacity, regularization, controlled failures | Complete supplied exercise: 10/10 |
| Transformers and Sequence Modelling | tokenization, embeddings, TextCNN, GRU, LSTM, Transformer encoder, attention evidence | Core complete; pretrained-transformer task remains gated |
| Vision Models | image profiling, representations, resolutions, augmentation, compatibility scoring, ranking, occlusion | Core complete; pretrained backbone, fine-tuning, and genuine human annotation remain gated |
| Language Models | embeddings, attention, and Transformer are demonstrated in the sequence suite | Lecture topics covered |
| Deep Learning Research | LLM API authentication, streaming, structured outputs, retries, caching, sanitization, and cost tracking | Not claimed as part of the primary experiment; wait for the exact exercise before adding an auxiliary API module |

The project deliberately avoids claiming full coverage where the exact exercise
has not been supplied or where an external resource or genuine human annotation
is required.
"""


def build_supplementary_index() -> str:
    return """# Supplementary evidence

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
"""


def markdown_cell(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.rstrip())


def code_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.rstrip())


def build_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown_cell(
            f"""# Automotive Part Image-Text Matching

**Focused Deep Learning exam notebook — Step {STEP}**

### The one question

> **{PRIMARY_QUESTION}**

This notebook is intentionally narrower than the full repository. It shows one
experiment, the evidence needed to judge it, concrete examples, concrete
failures, and a reproducible boundary. The wider course exercises and audit
material remain available as supporting evidence."""
        ),
        markdown_cell(
            """## 1. One research question

The input is a photograph of an automotive part and a short description. The
output is one of three relationships:

- `MATCH`: the same part category;
- `PARTIAL_MATCH`: a different category from the same automotive system;
- `MISMATCH`: a category from a different automotive system.

The central comparison is multimodal versus text-only and image-only models on
the same group-isolated validation set."""
        ),
        code_cell(
            """from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from PIL import Image

ROOT = Path.cwd()
if not (ROOT / "reports").is_dir():
    ROOT = ROOT.parent

train = pd.read_csv(ROOT / "data/processed/integrated_train.csv")
validation = pd.read_csv(ROOT / "data/processed/integrated_validation.csv")
comparison = pd.read_csv(
    ROOT / "reports/integrated_training/validation_comparison.csv"
)
errors = pd.read_csv(
    ROOT / "reports/validation_model_improvement/validation_error_analysis.csv"
)
with (
    ROOT
    / "reports/validation_model_improvement/validation_error_analysis.json"
).open(encoding="utf-8") as handle:
    error_summary = json.load(handle)

print("Evidence loaded from committed train/validation reports only.")
print("No model training is performed in this notebook.")"""
        ),
        markdown_cell(
            """## 2. Data and labels

The final comparison uses generated development images together with manually
reviewed open-license photographs. Each physical part appears in three
image-text relationships, but all relationships for that physical part remain
in the same split."""
        ),
        code_cell(
            """dataset_summary = pd.DataFrame(
    {
        "split": ["train", "validation"],
        "samples": [len(train), len(validation)],
        "physical_part_groups": [
            train["part_group_id"].nunique(),
            validation["part_group_id"].nunique(),
        ],
        "images": [train["image_id"].nunique(), validation["image_id"].nunique()],
    }
)
display(dataset_summary)

display(
    validation.groupby(["source", "label"])
    .size()
    .rename("samples")
    .reset_index()
)"""
        ),
        markdown_cell(
            """## 3. Leakage protection

A row-level random split would be invalid here. The same physical part could
appear with different descriptions on both sides of the split, letting the
model benefit from object identity rather than learn the intended relationship.
The split therefore uses `part_group_id`."""
        ),
        code_cell(
            """overlap = pd.DataFrame(
    {
        "check": [
            "part_group_id overlap",
            "image_id overlap",
            "image_path overlap",
        ],
        "count": [
            len(set(train["part_group_id"]) & set(validation["part_group_id"])),
            len(set(train["image_id"]) & set(validation["image_id"])),
            len(set(train["image_path"]) & set(validation["image_path"])),
        ],
    }
)
display(overlap)
assert overlap["count"].eq(0).all()
print("Group, image ID, and image path isolation: PASS")"""
        ),
        markdown_cell(
            """## 4. Models compared

The same validation split is used for six models: a majority baseline,
classical text and image baselines, neural text-only and image-only models, and
the retained multimodal network. The main metric is macro F1 because all three
classes matter."""
        ),
        code_cell(
            """result_columns = [
    "validation_rank",
    "model",
    "input_modality",
    "integrated_validation_accuracy",
    "integrated_validation_macro_f1",
]
results = comparison[result_columns].sort_values("validation_rank")
display(results)

plot_data = results.sort_values(
    "integrated_validation_macro_f1", ascending=True
)
figure, axis = plt.subplots(figsize=(9, 5))
axis.barh(plot_data["model"], plot_data["integrated_validation_macro_f1"])
axis.set_xlabel("Validation macro F1")
axis.set_xlim(0.0, 0.6)
axis.set_title("Primary grouped-validation comparison")
figure.tight_layout()
plt.show()"""
        ),
        markdown_cell(
            """## 5. Main validation result

The multimodal model ranks first with `0.5333` accuracy and `0.5208` macro F1.
This is better than both neural unimodal baselines, but the score is modest and
must be read together with the error analysis. A higher result on the earlier
generated-only development set is not used as the final claim."""
        ),
        markdown_cell(
            """## 6. What the multimodal model adds

Aggregate metrics alone do not show whether combining image and text helps on
specific cases. The next comparison joins the predictions and surfaces cases
where the multimodal model is correct while one or both unimodal models fail."""
        ),
        code_cell(
            """def load_predictions(slug, prefix):
    frame = pd.read_csv(
        ROOT / f"reports/integrated_training/{slug}/validation_predictions.csv"
    )
    return frame[["sample_id", "predicted_label", "is_correct"]].rename(
        columns={
            "predicted_label": f"{prefix}_prediction",
            "is_correct": f"{prefix}_correct",
        }
    )

multimodal = pd.read_csv(
    ROOT
    / "reports/integrated_training/keras_multimodal/validation_predictions.csv"
)
joined = (
    multimodal.rename(
        columns={
            "predicted_label": "multimodal_prediction",
            "is_correct": "multimodal_correct",
        }
    )
    .merge(load_predictions("keras_text", "text"), on="sample_id")
    .merge(load_predictions("keras_image", "image"), on="sample_id")
)

multimodal_wins = joined[
    joined["multimodal_correct"]
    & (~joined["text_correct"] | ~joined["image_correct"])
].copy()
columns = [
    "sample_id",
    "source",
    "description",
    "true_label",
    "multimodal_prediction",
    "text_prediction",
    "image_prediction",
]
display(multimodal_wins[columns].head(8))

counts = pd.DataFrame(
    {
        "model": ["Text-only", "Image-only", "Multimodal"],
        "correct": [
            int(joined["text_correct"].sum()),
            int(joined["image_correct"].sum()),
            int(joined["multimodal_correct"].sum()),
        ],
    }
)
figure, axis = plt.subplots(figsize=(7, 4.5))
axis.bar(counts["model"], counts["correct"])
axis.set_ylim(0, 60)
axis.set_ylabel("Correct validation samples")
axis.set_title("Correct predictions by neural input setup")
for index, value in enumerate(counts["correct"]):
    axis.text(index, value + 1, str(value), ha="center")
figure.tight_layout()
plt.show()"""
        ),
        markdown_cell(
            """The examples support a limited claim: the two-modality model
resolves cases that one-modality models miss. They do not prove that every
prediction uses both inputs correctly, so the conclusion remains comparative
rather than causal."""
        ),
        markdown_cell(
            """## 7. Where the model fails

The retained model makes 35 errors among 60 validation samples. The main
failure is the intermediate `PARTIAL_MATCH` relationship. In the retained
confusion matrix, only one validation sample is predicted as `PARTIAL_MATCH`,
and none of the true partial matches are recovered as that class; they are
divided between `MATCH` and `MISMATCH`."""
        ),
        code_cell(
            """confusion = pd.read_csv(
    ROOT
    / "reports/validation_model_improvement/candidates/"
    "reference_multimodal/validation_confusion_matrix.csv",
    index_col=0,
)
display(confusion)

figure, axis = plt.subplots(figsize=(6, 5))
image = axis.imshow(confusion.to_numpy(), aspect="auto")
axis.set_xticks(range(3), ["MATCH", "PARTIAL", "MISMATCH"])
axis.set_yticks(range(3), ["MATCH", "PARTIAL", "MISMATCH"])
axis.set_xlabel("Predicted")
axis.set_ylabel("Actual")
axis.set_title("Retained multimodal confusion matrix")
for row in range(3):
    for column in range(3):
        axis.text(
            column,
            row,
            int(confusion.iloc[row, column]),
            ha="center",
            va="center",
        )
figure.colorbar(image, ax=axis)
figure.tight_layout()
plt.show()

display(
    errors.groupby("error_pair")
    .size()
    .rename("errors")
    .sort_values(ascending=False)
    .reset_index()
)"""
        ),
        code_cell(
            """validation_images = validation[
    ["image_id", "image_path"]
].drop_duplicates("image_id")
example_errors = (
    errors.merge(validation_images, on="image_id", how="left")
    .sort_values("confidence", ascending=False)
    .head(4)
)

figure, axes = plt.subplots(2, 2, figsize=(12, 9))
for axis, (_, row) in zip(axes.flat, example_errors.iterrows()):
    image = Image.open(ROOT / row["image_path"]).convert("RGB")
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(
        f"True: {row['true_label']} | Pred: {row['predicted_label']}\\n"
        f"{row['description']}\\nconfidence={row['confidence']:.3f}",
        fontsize=9,
    )
figure.suptitle("Concrete validation errors")
figure.tight_layout()
plt.show()

display(
    example_errors[
        [
            "sample_id",
            "source",
            "true_label",
            "predicted_label",
            "confidence",
            "description",
        ]
    ]
)"""
        ),
        code_cell(
            """source_errors = (
    errors.groupby("source").size().rename("errors").reset_index()
)
source_totals = (
    validation.groupby("source").size().rename("samples").reset_index()
)
source_rates = source_errors.merge(source_totals, on="source")
source_rates["error_rate"] = (
    source_rates["errors"] / source_rates["samples"]
)
display(source_rates)

figure, axis = plt.subplots(figsize=(8, 4.5))
axis.bar(source_rates["source"], source_rates["error_rate"])
axis.set_ylim(0.0, 1.0)
axis.set_ylabel("Error rate")
axis.set_title("Real open-license images are harder")
axis.tick_params(axis="x", rotation=15)
figure.tight_layout()
plt.show()"""
        ),
        markdown_cell(
            """## 8. Limits of the conclusion

The evidence supports the statement that the retained multimodal model is the
strongest of the compared models on this grouped validation split. It does not
support claims of production readiness, general performance across all vehicle
parts, calibrated probabilities, human-level explanation, or final test
performance.

The small dataset, the failure to predict `PARTIAL_MATCH`, and the higher error
rate on real images are the main limitations."""
        ),
        markdown_cell(
            """## 9. Reproduction and evaluation boundary

From the repository root:

```powershell
python -m src.project_cli build-exam-first-submission
python -m src.project_cli verify-exam-first-submission
```

The exam-facing build reads committed train and validation evidence. It does
not train a model, open locked test CSV files, authorize final test evaluation,
or change the retained production model."""
        ),
        code_cell(
            """freeze_path = (
    ROOT / "reports/final_model_freeze/final_model_freeze_status.json"
)
with freeze_path.open(encoding="utf-8") as handle:
    freeze_status = json.load(handle)

boundary = pd.DataFrame(
    [
        {
            "model_training_performed": False,
            "locked_test_csv_files_opened": False,
            "test_split_used": False,
            "final_test_evaluation_authorized": False,
            "production_final_model_changed": False,
            "retained_model": freeze_status["final_model_slug"],
            "selection_decision": freeze_status["selection_decision"],
        }
    ]
)
display(boundary)
assert freeze_status["protocol_frozen"] is True
assert freeze_status["test_lock_preserved"] is True
print("Evaluation boundary: PASS")"""
        ),
        markdown_cell(
            """## 10. Conclusion

The multimodal network is the best model in the controlled comparison because
the task depends on both the photographed part and the description. Its
advantage over the unimodal neural baselines is visible both in macro F1 and in
specific examples. The model is still weak on `PARTIAL_MATCH` and on real
images, so the correct conclusion is improvement, not completion.

The next scientific step should improve the data and the handling of the
intermediate relationship before any final test evaluation is authorized."""
        ),
        markdown_cell(
            """## References

1. Faghri, F. et al. *VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.* BMVC, 2018.
2. Li, L. H. et al. *VisualBERT: A Simple and Performant Baseline for Vision and Language.* 2019.
3. Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021.
4. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS, 2017.
5. He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
6. scikit-learn documentation: model evaluation and grouped cross-validation."""
        ),
    ]

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.metadata["project"] = {
        "step": STEP,
        "teacher_facing": True,
        "primary_research_question": PRIMARY_QUESTION,
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
        "supporting_evidence_is_secondary": True,
    }
    return notebook


def execute_notebook(notebook: nbformat.NotebookNode) -> None:
    EXAM_DIR.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except CellExecutionError as error:
        raise RuntimeError("Focused exam notebook execution failed.") from error
    nbformat.write(notebook, NOTEBOOK_PATH)


def notebook_metrics() -> dict[str, int]:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
    ]
    figures = [
        output
        for output in outputs
        if output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]
    return {
        "notebook_cells": len(notebook.cells),
        "code_cells": len(code_cells),
        "executed_code_cells": sum(
            cell.get("execution_count") is not None for cell in code_cells
        ),
        "saved_outputs": len(outputs),
        "figures": len(figures),
        "error_outputs": sum(
            output.get("output_type") == "error" for output in outputs
        ),
    }


def update_legacy_manifest_readme_hash() -> None:
    manifest_paths = (
        PROJECT_ROOT / "reports/final_submission/final_submission_manifest.json",
        PROJECT_ROOT
        / "reports/exam_submission_readiness/"
        "exam_submission_readiness_manifest.json",
    )
    readme_hash = normalized_sha256(PROJECT_ROOT / "README.md")
    for legacy_manifest_path in manifest_paths:
        manifest = read_json(legacy_manifest_path)
        source_hashes = manifest.get("source_artifact_sha256", {})
        if "README.md" in source_hashes:
            source_hashes["README.md"] = readme_hash
        legacy_manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def build_status(evidence: dict[str, Any], metrics: dict[str, int]) -> dict[str, Any]:
    best = evidence["comparison"].sort_values("validation_rank").iloc[0]
    error_summary = evidence["error_summary"]
    return {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "primary_research_question": PRIMARY_QUESTION,
        "base_checkpoint_commit": BASE_CHECKPOINT_COMMIT,
        "base_checkpoint_commit_count": BASE_CHECKPOINT_COMMIT_COUNT,
        "source_archive": SOURCE_ARCHIVE,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "primary_notebook": project_relative(NOTEBOOK_PATH),
        "validation_sample_count": int(len(evidence["validation"])),
        "validation_part_group_count": int(
            evidence["validation"]["part_group_id"].nunique()
        ),
        "retained_model_slug": str(best["model_slug"]),
        "validation_accuracy": float(
            best["integrated_validation_accuracy"]
        ),
        "validation_macro_f1": float(
            best["integrated_validation_macro_f1"]
        ),
        "validation_error_count": int(error_summary["error_count"]),
        "partial_match_error_count": int(
            (evidence["errors"]["true_label"] == "PARTIAL_MATCH").sum()
        ),
        **metrics,
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
        "model_selection_changed": False,
        "physical_source_files_moved": False,
    }


def build_summary(status: dict[str, Any]) -> str:
    return f"""# Step {STEP} — Exam-First Submission Architecture

Status: **{status['status']}**

Readiness: `{status['readiness']}`

## Purpose

The repository now opens with one precise Deep Learning question and one
focused executed notebook. Course exercises, historical notebooks, manifests,
and engineering audits remain available as supporting evidence without
competing with the main analysis.

## Primary result

- retained model: `{status['retained_model_slug']}`;
- grouped validation samples: {status['validation_sample_count']};
- independent physical-part groups: {status['validation_part_group_count']};
- validation accuracy: {status['validation_accuracy']:.4f};
- validation macro F1: {status['validation_macro_f1']:.4f};
- validation errors analyzed: {status['validation_error_count']};
- `PARTIAL_MATCH` errors: {status['partial_match_error_count']}.

## Safety boundary

- model training performed: false;
- locked test CSV files opened: false;
- test split used: false;
- final test evaluation authorized: false;
- production final model changed: false;
- model selection changed: false.

## Repository design decision

Standard code, data, report, and test directories were not physically moved.
Moving them would break stable paths and historical manifests. Instead, the
new `exam/` directory is the teacher-facing entry point and
`supplementary/README.md` is the index to deeper evidence.
"""


def build_manifest() -> dict[str, Any]:
    source_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in SOURCE_ARTIFACTS
    }
    generated_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in GENERATED_ARTIFACTS
        if path != MANIFEST_PATH
    }
    return {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "hash_normalization": "utf-8-lf for text; raw bytes for binary",
        "base_checkpoint_commit": BASE_CHECKPOINT_COMMIT,
        "base_checkpoint_commit_count": BASE_CHECKPOINT_COMMIT_COUNT,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_artifact_sha256": source_hashes,
        "generated_artifact_sha256": generated_hashes,
        "source_artifact_count": len(source_hashes),
        "generated_artifact_count": len(generated_hashes),
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
        "model_selection_changed": False,
    }


def main() -> None:
    evidence = load_evidence()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXAM_DIR.mkdir(parents=True, exist_ok=True)

    build_figures(evidence)
    historical_readme = build_historical_readme()
    write_text(HISTORICAL_README_PATH, historical_readme)
    root_readme = (
        build_root_readme().rstrip()
        + "\n\n<details>\n<summary>Complete engineering and historical "
        "documentation</summary>\n\n"
        + historical_readme.rstrip()
        + "\n\n</details>"
    )
    write_text(ROOT_README_PATH, root_readme)
    historical_notebook_catalogue = build_historical_notebook_catalogue()
    write_text(
        HISTORICAL_NOTEBOOK_CATALOGUE_PATH,
        historical_notebook_catalogue,
    )
    notebook_catalogue = (
        build_notebook_catalogue().rstrip()
        + "\n\n<details>\n<summary>Historical notebook catalogue "
        "and verification commands</summary>\n\n"
        + historical_notebook_catalogue.rstrip()
        + "\n\n</details>"
    )
    write_text(NOTEBOOK_CATALOGUE_PATH, notebook_catalogue)
    write_text(EXAM_README_PATH, build_exam_readme(evidence))
    write_text(REPRODUCTION_PATH, build_reproduction())
    write_text(DEFENSE_NOTES_PATH, build_defense_notes())
    write_text(COURSE_ALIGNMENT_PATH, build_course_alignment())
    write_text(SUPPLEMENTARY_INDEX_PATH, build_supplementary_index())

    notebook = build_notebook()
    execute_notebook(notebook)
    metrics = notebook_metrics()
    status = build_status(evidence, metrics)
    STATUS_PATH.write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_text(SUMMARY_PATH, build_summary(status))

    update_legacy_manifest_readme_hash()
    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Step {STEP} exam-first submission built")
    print(f"- notebook: {project_relative(NOTEBOOK_PATH)}")
    print(f"- notebook cells: {metrics['notebook_cells']}")
    print(f"- executed code cells: {metrics['executed_code_cells']}")
    print(f"- saved outputs: {metrics['saved_outputs']}")
    print(f"- figures: {metrics['figures']}")
    print("- model training performed: false")
    print("- locked test CSV files opened: false")
    print("- production final model changed: false")


if __name__ == "__main__":
    main()
