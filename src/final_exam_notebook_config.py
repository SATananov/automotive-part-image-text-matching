from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

BASE_CHECKPOINT_COMMIT = "d517668"
NOTEBOOK_READINESS = "FINAL_EXAM_NOTEBOOK_INTEGRATED_TEST_LOCKED"
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
FINAL_EXAM_NOTEBOOK_PATH = NOTEBOOK_ROOT / "02_final_exam_project.ipynb"
FINAL_EXAM_REPORT_ROOT = PROJECT_ROOT / "reports" / "final_exam_notebook"
FINAL_EXAM_NOTEBOOK_STATUS_PATH = (
    FINAL_EXAM_REPORT_ROOT / "final_exam_notebook_status.json"
)
FINAL_EXAM_NOTEBOOK_MANIFEST_PATH = (
    FINAL_EXAM_REPORT_ROOT / "final_exam_notebook_manifest.json"
)
FINAL_EXAM_NOTEBOOK_SUMMARY_PATH = (
    FINAL_EXAM_REPORT_ROOT / "final_exam_notebook_summary.md"
)

INTEGRATED_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "integrated_train.csv"
INTEGRATED_VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "integrated_validation.csv"
)
INTEGRATED_COMPARISON_PATH = (
    PROJECT_ROOT / "reports" / "integrated_training" / "validation_comparison.csv"
)
INTEGRATED_STATUS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "integrated_training_run_status.json"
)
MULTIMODAL_METRICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "keras_multimodal"
    / "validation_metrics.json"
)
MULTIMODAL_CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "keras_multimodal"
    / "validation_confusion_matrix.csv"
)
MULTIMODAL_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "keras_multimodal"
    / "validation_predictions.csv"
)
CONTROLLED_EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation_model_improvement"
    / "controlled_experiment_comparison.csv"
)
ERROR_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation_model_improvement"
    / "validation_error_analysis.csv"
)
MODEL_SELECTION_DECISION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation_model_improvement"
    / "model_selection_decision.json"
)
FINAL_MODEL_SPECIFICATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_model_freeze"
    / "final_model_specification.json"
)
FINAL_EVALUATION_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_model_freeze"
    / "final_evaluation_protocol.json"
)
FINAL_MODEL_FREEZE_STATUS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_model_freeze"
    / "final_model_freeze_status.json"
)
FINAL_TEST_AUTHORIZATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_model_freeze"
    / "final_test_authorization.json"
)
LOCKED_TEST_CONTRACT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_model_freeze"
    / "locked_test_contract.json"
)

LOCKED_TEST_PATH_TOKENS = (
    "integrated_test.csv",
    "external_test.csv",
    "development_test.csv",
)
FORBIDDEN_NOTEBOOK_CODE_TOKENS = (
    *LOCKED_TEST_PATH_TOKENS,
    "model.fit(",
    "model.predict(",
    "tensorflow",
    "import keras",
    "from keras",
)

REQUIRED_NOTEBOOK_HEADINGS = (
    "# Automotive Part Image-Text Matching",
    "## 1. Abstract",
    "## 2. Problem statement and motivation",
    "## 3. Research question and hypothesis",
    "## 4. Related work",
    "## 5. Dataset construction, licensing and ethics",
    "## 6. Data cleaning and grouped split",
    "## 7. Models and experimental design",
    "## 8. Development and integrated validation results",
    "## 9. Validation error analysis",
    "## 10. Controlled model improvement and selection",
    "## 11. Final model and locked-test protocol freeze",
    "## 12. Testing and reproducibility",
    "## 13. Limitations and threats to validity",
    "## 14. Conclusion and future work",
    "## 15. References",
)

REFERENCE_TITLES = (
    "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives",
    "VisualBERT: A Simple and Performant Baseline for Vision and Language",
    "Learning Transferable Visual Models From Natural Language Supervision",
    "The Functional API",
)

BASE_VERIFIED_TEST_COUNT = 275
BASE_VERIFIED_WARNING_COUNT = 154
