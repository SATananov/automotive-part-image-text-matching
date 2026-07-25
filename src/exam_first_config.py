from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

STEP = "011.5"
BASE_CHECKPOINT_COMMIT = "285327cc7e1304f6c7e64e8513f0857b13790303"
BASE_CHECKPOINT_COMMIT_COUNT = 42
SOURCE_ARCHIVE = (
    "automotive-part-image-text-matching_"
    "CLEAN_STEP011_4_285327cc_20260725_060717.zip"
)
SOURCE_ARCHIVE_SHA256 = (
    "5fba05dd7027412ce0ac131e152c47b165c6ccfe4952d9454b3b61868717a00c"
)
READINESS = "EXAM_FIRST_SUBMISSION_FOCUSED_TEST_LOCKED"

ROOT_README_PATH = PROJECT_ROOT / "README.md"
NOTEBOOK_CATALOGUE_PATH = PROJECT_ROOT / "notebooks" / "README.md"
EXAM_DIR = PROJECT_ROOT / "exam"
NOTEBOOK_PATH = EXAM_DIR / "01_focused_deep_learning_project.ipynb"
EXAM_README_PATH = EXAM_DIR / "README.md"
REPRODUCTION_PATH = EXAM_DIR / "reproduction.md"
DEFENSE_NOTES_PATH = EXAM_DIR / "defense_notes.md"
COURSE_ALIGNMENT_PATH = EXAM_DIR / "course_alignment.md"
SUPPLEMENTARY_INDEX_PATH = PROJECT_ROOT / "supplementary" / "README.md"
HISTORICAL_README_PATH = (
    PROJECT_ROOT / "supplementary" / "README_STEP0114_FULL.md"
)
HISTORICAL_NOTEBOOK_CATALOGUE_PATH = (
    PROJECT_ROOT / "supplementary" / "NOTEBOOKS_README_STEP0114_FULL.md"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "exam_first_submission"
STATUS_PATH = REPORT_DIR / "exam_first_submission_status.json"
SUMMARY_PATH = REPORT_DIR / "exam_first_submission_summary.md"
MANIFEST_PATH = REPORT_DIR / "exam_first_submission_manifest.json"
FIGURES_DIR = REPORT_DIR / "figures"

PRIMARY_QUESTION = (
    "Does a multimodal neural network that combines an automotive-part "
    "image and a short description classify their relationship better "
    "than image-only and text-only neural baselines?"
)

REQUIRED_NOTEBOOK_HEADINGS = (
    "# Automotive Part Image-Text Matching",
    "## 1. One research question",
    "## 2. Data and labels",
    "## 3. Leakage protection",
    "## 4. Models compared",
    "## 5. Main validation result",
    "## 6. What the multimodal model adds",
    "## 7. Where the model fails",
    "## 8. Limits of the conclusion",
    "## 9. Reproduction and evaluation boundary",
    "## 10. Conclusion",
    "## References",
)

FORBIDDEN_NOTEBOOK_CODE_TOKENS = (
    "integrated_test.csv",
    "external_test.csv",
    "model.fit(",
    "keras.fit(",
    "tensorflow.keras.models.load_model",
    "load_model(",
)

SOURCE_ARTIFACTS = (
    PROJECT_ROOT / "src" / "project_cli.py",
    PROJECT_ROOT / "src" / "exam_first_config.py",
    PROJECT_ROOT / "src" / "build_exam_first_submission.py",
    PROJECT_ROOT
    / "src"
    / "verification"
    / "exam_first_submission.py",
    PROJECT_ROOT / "tests" / "test_exam_first_submission.py",
    PROJECT_ROOT / "data" / "processed" / "integrated_train.csv",
    PROJECT_ROOT / "data" / "processed" / "integrated_validation.csv",
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "validation_comparison.csv",
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "keras_text"
    / "validation_predictions.csv",
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "keras_image"
    / "validation_predictions.csv",
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "keras_multimodal"
    / "validation_predictions.csv",
    PROJECT_ROOT
    / "reports"
    / "validation_model_improvement"
    / "validation_error_analysis.csv",
    PROJECT_ROOT
    / "reports"
    / "validation_model_improvement"
    / "validation_error_analysis.json",
    PROJECT_ROOT
    / "reports"
    / "validation_model_improvement"
    / "candidates"
    / "reference_multimodal"
    / "validation_confusion_matrix.csv",
    PROJECT_ROOT
    / "reports"
    / "final_model_freeze"
    / "final_model_freeze_status.json",
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "fundamentals"
    / "fundamentals_suite_status.json",
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "sequence"
    / "sequence_suite_status.json",
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "vision"
    / "vision_suite_status.json",
)

GENERATED_ARTIFACTS = (
    ROOT_README_PATH,
    NOTEBOOK_CATALOGUE_PATH,
    NOTEBOOK_PATH,
    EXAM_README_PATH,
    REPRODUCTION_PATH,
    DEFENSE_NOTES_PATH,
    COURSE_ALIGNMENT_PATH,
    SUPPLEMENTARY_INDEX_PATH,
    HISTORICAL_README_PATH,
    HISTORICAL_NOTEBOOK_CATALOGUE_PATH,
    STATUS_PATH,
    SUMMARY_PATH,
    MANIFEST_PATH,
    FIGURES_DIR / "model_comparison.png",
    FIGURES_DIR / "confusion_matrix.png",
    FIGURES_DIR / "error_rates.png",
    FIGURES_DIR / "multimodal_contribution.png",
)

TEXT_HASH_SUFFIXES = {
    ".csv",
    ".html",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".txt",
    ".yaml",
    ".yml",
}


def project_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()
