from __future__ import annotations

from src.real_dataset_config import PROJECT_ROOT

VALIDATION_IMPROVEMENT_ROOT = (
    PROJECT_ROOT / "reports" / "validation_model_improvement"
)
ERROR_ANALYSIS_CSV_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "validation_error_analysis.csv"
)
ERROR_ANALYSIS_JSON_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "validation_error_analysis.json"
)
DATA_DIAGNOSTICS_JSON_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "validation_data_diagnostics.json"
)
DISAGREEMENT_CSV_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "candidate_disagreement_analysis.csv"
)
DISAGREEMENT_JSON_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "candidate_disagreement_analysis.json"
)
EXPERIMENT_COMPARISON_CSV_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "controlled_experiment_comparison.csv"
)
EXPERIMENT_COMPARISON_JSON_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "controlled_experiment_comparison.json"
)
SELECTION_DECISION_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "model_selection_decision.json"
)
EXPERIMENT_REGISTRY_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "experiment_registry.json"
)
VALIDATION_IMPROVEMENT_SUMMARY_PATH = (
    VALIDATION_IMPROVEMENT_ROOT
    / "validation_error_analysis_and_model_improvement_summary.md"
)
VALIDATION_IMPROVEMENT_STATUS_PATH = (
    VALIDATION_IMPROVEMENT_ROOT / "validation_model_improvement_status.json"
)

CANDIDATE_TITLES = {
    "reference_multimodal": "Reference Multimodal",
    "relation_aware_multimodal": "Relation-Aware Multimodal",
    "regularized_relation_multimodal": (
        "Regularized Relation-Aware Multimodal"
    ),
}
CANDIDATE_DESCRIPTIONS = {
    "reference_multimodal": (
        "Exact Step 010.3 multimodal architecture, retrained under the "
        "same controlled multi-seed protocol."
    ),
    "relation_aware_multimodal": (
        "Adds elementwise product and absolute-difference relation features "
        "between equal-width text and image representations."
    ),
    "regularized_relation_multimodal": (
        "Uses relation features with stronger dropout, a smaller fusion "
        "layer, and a lower learning rate to reduce overfitting."
    ),
}
CANDIDATE_DIRECTORIES = {
    slug: VALIDATION_IMPROVEMENT_ROOT / "candidates" / slug
    for slug in CANDIDATE_TITLES
}
CANDIDATE_METRIC_PATHS = {
    slug: directory / "validation_metrics.json"
    for slug, directory in CANDIDATE_DIRECTORIES.items()
}
CANDIDATE_PREDICTION_PATHS = {
    slug: directory / "validation_predictions.csv"
    for slug, directory in CANDIDATE_DIRECTORIES.items()
}
CANDIDATE_CONFUSION_MATRIX_PATHS = {
    slug: directory / "validation_confusion_matrix.csv"
    for slug, directory in CANDIDATE_DIRECTORIES.items()
}
CANDIDATE_HISTORY_PATHS = {
    slug: directory / "training_history_by_seed.csv"
    for slug, directory in CANDIDATE_DIRECTORIES.items()
}
CANDIDATE_ARCHITECTURE_PATHS = {
    slug: directory / "model_architecture.txt"
    for slug, directory in CANDIDATE_DIRECTORIES.items()
}

EXPERIMENT_SEEDS = (42, 52, 62)
VALIDATION_IMPROVEMENT_MAX_EPOCHS = 60
VALIDATION_IMPROVEMENT_PATIENCE = 8
VALIDATION_IMPROVEMENT_BATCH_SIZE = 32
MINIMUM_MACRO_F1_GAIN = 0.01
MAXIMUM_INCUMBENT_MACRO_F1_REGRESSION = 0.01
MAXIMUM_ACCURACY_REGRESSION = 0.02
MAXIMUM_WORST_CLASS_F1_REGRESSION = 0.05
HIGH_CONFIDENCE_ERROR_THRESHOLD = 0.60
