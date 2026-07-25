from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

STEP = "011.4"
BASE_CHECKPOINT = "18d27b0b"
SUBMISSION_DEADLINE = "2026-08-11 16:00 Europe/Sofia"
REPOSITORY_URL = (
    "https://github.com/SATananov/"
    "automotive-part-image-text-matching"
)
FINAL_SUBMISSION_NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "03_final_exam_submission.ipynb"
)
FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL = (
    REPOSITORY_URL
    + "/blob/main/notebooks/03_final_exam_submission.ipynb"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "final_submission"
STATUS_PATH = REPORT_DIR / "final_submission_status.json"
MANIFEST_PATH = REPORT_DIR / "final_submission_manifest.json"
SUMMARY_PATH = REPORT_DIR / "final_submission_summary.md"
SELF_ASSESSMENT_PATH = REPORT_DIR / "self_assessment.md"
CHECKLIST_PATH = REPORT_DIR / "submission_checklist.md"
DEFENSE_GUIDE_PATH = REPORT_DIR / "defense_guide.md"
ERROR_ANALYSIS_PATH = REPORT_DIR / "deep_learning_error_analysis.md"
RUBRIC_MATRIX_PATH = REPORT_DIR / "rubric_evidence_matrix.csv"
ERROR_SUMMARY_PATH = REPORT_DIR / "deep_learning_error_summary.json"
FIGURES_DIR = REPORT_DIR / "figures"

READINESS = "FINAL_EXAM_RUBRIC_ALIGNED_DL_ERROR_ANALYSIS_TEST_LOCKED"

RUBRIC_MAX_POINTS = {
    "Problem statement": 10,
    "Layout": 20,
    "Code quality": 20,
    "Previous research": 10,
    "Data": 10,
    "Testing": 10,
    "Visualization": 10,
    "Communication": 10,
}

RUBRIC_SELF_ASSESSMENT = {
    "Problem statement": 10,
    "Layout": 20,
    "Code quality": 20,
    "Previous research": 10,
    "Data": 10,
    "Testing": 10,
    "Visualization": 9,
    "Communication": 9,
}

REQUIRED_NOTEBOOK_HEADINGS = (
    "# Automotive Part Image-Text Matching",
    "## 1. Executive Summary",
    "## 2. Problem Statement and Real-World Motivation",
    "## 3. Formal Task Definition and Evaluation Metrics",
    "## 4. Previous Research",
    "## 5. Data Acquisition, Licensing, Cleaning, and Grouped Splitting",
    "## 6. Model Families and Experimental Design",
    "## 7. Integrated Validation Results",
    "## 8. Deep Learning Error Analysis and Failure Diagnostics",
    "## 9. Course Exercise Evidence: Fundamentals, Sequence, and Vision",
    "## 10. Testing, Reproducibility, and Locked-Test Policy",
    "## 11. Limitations, Ethics, and Legal Compliance",
    "## 12. Self-Assessment Against the Exam Rubric",
    "## 13. Conclusions and Defense Summary",
    "## References",
)

REFERENCE_TITLES = (
    "Visual Semantic Embedding",
    "VisualBERT",
    "Learning Transferable Visual Models From Natural Language Supervision",
    "Deep Residual Learning for Image Recognition",
    "Attention Is All You Need",
    "scikit-learn model evaluation documentation",
)

FORBIDDEN_NOTEBOOK_CODE_TOKENS = (
    "integrated_test.csv",
    "external_test.csv",
    "final_test_evaluation",
    "model.fit(",
    "keras.fit(",
    "tensorflow.keras.models.load_model",
)

SOURCE_ARTIFACTS = (
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / "requirements-lock.txt",
    PROJECT_ROOT / "data" / "processed" / "integrated_validation.csv",
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "validation_comparison.csv",
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
    / "course_coverage"
    / "fundamentals"
    / "failure_diagnostics.csv",
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "fundamentals"
    / "fundamentals_suite_status.json",
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "sequence"
    / "model_comparison.csv",
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
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "vision"
    / "ranking_metrics.json",
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "vision"
    / "explainability_summary.json",
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "vision"
    / "augmentation_comparison.csv",
    PROJECT_ROOT
    / "reports"
    / "final_model_freeze"
    / "final_model_freeze_status.json",
)

GENERATED_ARTIFACTS = (
    FINAL_SUBMISSION_NOTEBOOK_PATH,
    STATUS_PATH,
    MANIFEST_PATH,
    SUMMARY_PATH,
    SELF_ASSESSMENT_PATH,
    CHECKLIST_PATH,
    DEFENSE_GUIDE_PATH,
    ERROR_ANALYSIS_PATH,
    RUBRIC_MATRIX_PATH,
    ERROR_SUMMARY_PATH,
    FIGURES_DIR / "integrated_model_comparison.png",
    FIGURES_DIR / "reference_confusion_matrix.png",
    FIGURES_DIR / "errors_by_source_and_category.png",
    FIGURES_DIR / "controlled_failure_diagnostics.png",
    FIGURES_DIR / "course_suite_summary.png",
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
