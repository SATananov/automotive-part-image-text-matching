from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

STEP = "010.8"
BASE_COMMIT = "74fab18"
REPOSITORY_URL = (
    "https://github.com/SATananov/"
    "automotive-part-image-text-matching"
)
FINAL_NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "02_final_exam_project.ipynb"
)
FINAL_NOTEBOOK_GITHUB_URL = (
    REPOSITORY_URL
    + "/blob/main/notebooks/02_final_exam_project.ipynb"
)

READINESS_DIR = PROJECT_ROOT / "reports" / "exam_submission_readiness"
STATUS_PATH = READINESS_DIR / "exam_submission_readiness_status.json"
MANIFEST_PATH = READINESS_DIR / "exam_submission_readiness_manifest.json"
CHECKLIST_PATH = READINESS_DIR / "submission_checklist.md"
CLEAN_CLONE_PATH = READINESS_DIR / "clean_clone_reproducibility.md"
SUMMARY_PATH = READINESS_DIR / "exam_submission_readiness_summary.md"

READINESS = "EXAM_SUBMISSION_READY_FOR_CLEAN_RELEASE_TEST_LOCKED"

EXPECTED_DIRECT_REQUIREMENTS = (
    "tensorflow",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "pillow",
    "jupyter",
    "ipykernel",
    "pytest",
)

REQUIRED_SOURCE_ARTIFACTS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "requirements-lock.txt",
    FINAL_NOTEBOOK_PATH,
    PROJECT_ROOT
    / "reports"
    / "final_exam_notebook"
    / "final_exam_notebook_status.json",
    PROJECT_ROOT
    / "reports"
    / "notebook_quality_audit"
    / "notebook_quality_audit_status.json",
    PROJECT_ROOT
    / "reports"
    / "final_model_freeze"
    / "final_model_freeze_status.json",
    PROJECT_ROOT / "data" / "processed" / "integrated_test_lock.json",
)

REQUIRED_OUTPUTS = (
    STATUS_PATH,
    MANIFEST_PATH,
    CHECKLIST_PATH,
    CLEAN_CLONE_PATH,
    SUMMARY_PATH,
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
