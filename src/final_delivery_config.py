from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

STEP = "011.6"
BASE_CHECKPOINT_COMMIT = "065a2dc32c7bafcf801c526c7b52354c1273efcc"
BASE_CHECKPOINT_COMMIT_COUNT = 44
EXPECTED_COMMIT_COUNT_AFTER_STEP = 45
SOURCE_ARCHIVE = (
    "automotive-part-image-text-matching_"
    "CLEAN_STEP011_5_1_065a2dc3_20260725_105043.zip"
)
SOURCE_ARCHIVE_SHA256 = (
    "ec4b1d8d799c0140b1e226c56790081b2b84f80daeb8cb1209d7b780287f1aca"
)
READINESS = "FINAL_SUBMISSION_LOCKED_SINGLE_ENTRY_POINT_TEST_LOCKED"
SUBMISSION_DEADLINE = "2026-08-11 16:00 Europe/Sofia"
REPOSITORY_URL = (
    "https://github.com/SATananov/"
    "automotive-part-image-text-matching"
)

CANONICAL_NOTEBOOK_PATH = (
    PROJECT_ROOT / "exam" / "01_focused_deep_learning_project.ipynb"
)
CANONICAL_NOTEBOOK_RELATIVE = (
    "exam/01_focused_deep_learning_project.ipynb"
)
CANONICAL_NOTEBOOK_GITHUB_URL = (
    REPOSITORY_URL + "/blob/main/" + CANONICAL_NOTEBOOK_RELATIVE
)

REPORT_DIR = PROJECT_ROOT / "reports" / "final_delivery"
STATUS_PATH = REPORT_DIR / "final_delivery_status.json"
MANIFEST_PATH = REPORT_DIR / "final_delivery_manifest.json"
CHECKLIST_PATH = REPORT_DIR / "final_submission_checklist.md"
TEACHER_ENTRY_POINT_PATH = REPORT_DIR / "teacher_entry_point.md"
GITHUB_RENDER_REVIEW_PATH = REPORT_DIR / "github_render_review.md"
SUBMISSION_BOUNDARY_PATH = REPORT_DIR / "submission_boundary.md"
ENTRY_POINT_MAP_PATH = REPORT_DIR / "entry_point_map.json"

ROOT_README_PATH = PROJECT_ROOT / "README.md"
EXAM_README_PATH = PROJECT_ROOT / "exam" / "README.md"
NOTEBOOK_CATALOGUE_PATH = PROJECT_ROOT / "notebooks" / "README.md"
SUPPLEMENTARY_INDEX_PATH = PROJECT_ROOT / "supplementary" / "README.md"
HISTORICAL_FINAL_CHECKLIST_PATH = (
    PROJECT_ROOT / "reports" / "final_submission" / "submission_checklist.md"
)
HISTORICAL_READINESS_CHECKLIST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "exam_submission_readiness"
    / "submission_checklist.md"
)

EXAM_FIRST_STATUS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "exam_first_submission"
    / "exam_first_submission_status.json"
)
CONSISTENCY_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "exam_first_submission"
    / "single_model_consistency_report.json"
)
PRIMARY_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "keras_multimodal"
    / "validation_predictions.csv"
)
PRIMARY_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "keras_multimodal"
    / "validation_metrics.json"
)
PRIMARY_CONFUSION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "keras_multimodal"
    / "validation_confusion_matrix.csv"
)
FINAL_MODEL_STATUS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_model_freeze"
    / "final_model_freeze_status.json"
)

SOURCE_ARTIFACTS = (
    CANONICAL_NOTEBOOK_PATH,
    EXAM_FIRST_STATUS_PATH,
    CONSISTENCY_REPORT_PATH,
    PRIMARY_PREDICTIONS_PATH,
    PRIMARY_METRICS_PATH,
    PRIMARY_CONFUSION_PATH,
    FINAL_MODEL_STATUS_PATH,
    ROOT_README_PATH,
    EXAM_README_PATH,
    NOTEBOOK_CATALOGUE_PATH,
    SUPPLEMENTARY_INDEX_PATH,
    HISTORICAL_FINAL_CHECKLIST_PATH,
    HISTORICAL_READINESS_CHECKLIST_PATH,
)

IMPLEMENTATION_ARTIFACTS = (
    PROJECT_ROOT / "src" / "project_cli.py",
    PROJECT_ROOT / "src" / "final_delivery_config.py",
    PROJECT_ROOT / "src" / "build_final_delivery.py",
    PROJECT_ROOT / "src" / "verification" / "final_delivery.py",
    PROJECT_ROOT / "src" / "verification" / "exam_first_submission.py",
    PROJECT_ROOT / "src" / "verification" / "exam_submission_readiness.py",
    PROJECT_ROOT
    / "src"
    / "verification"
    / "final_submission_rubric_alignment.py",
    PROJECT_ROOT / "src" / "verification" / "vision_experimental_suite.py",
    PROJECT_ROOT / "src" / "verification" / "project_verification.py",
    PROJECT_ROOT / "tests" / "test_exam_submission_readiness.py",
    PROJECT_ROOT / "tests" / "test_final_delivery.py",
)

GENERATED_ARTIFACTS = (
    STATUS_PATH,
    CHECKLIST_PATH,
    TEACHER_ENTRY_POINT_PATH,
    GITHUB_RENDER_REVIEW_PATH,
    SUBMISSION_BOUNDARY_PATH,
    ENTRY_POINT_MAP_PATH,
    MANIFEST_PATH,
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
