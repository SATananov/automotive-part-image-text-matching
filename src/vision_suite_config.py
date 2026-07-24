from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

STEP = "011.3A"
READINESS = (
    "VISION_EXPERIMENTAL_SUITE_CORE_COMPLETE_"
    "PRETRAINED_AND_HUMAN_GATES_TEST_LOCKED"
)
BASE_CHECKPOINT = "040a637"

CONFIG_ROOT = PROJECT_ROOT / "configs" / "course_coverage"
SUITE_CONFIG_PATH = CONFIG_ROOT / "vision_suite.json"
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
EXPERIMENT_CONFIG_PATHS = tuple(
    EXPERIMENT_CONFIG_DIR / f"VIS-{number:03d}.json"
    for number in range(1, 10)
)

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "integrated_train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "integrated_validation.csv"

REPORT_ROOT = PROJECT_ROOT / "reports" / "course_coverage" / "vision"
FIGURE_ROOT = REPORT_ROOT / "figures"
STATUS_PATH = REPORT_ROOT / "vision_suite_status.json"
SUMMARY_PATH = REPORT_ROOT / "vision_suite_summary.md"
MANIFEST_PATH = REPORT_ROOT / "vision_suite_manifest.json"
IMAGE_PROFILE_PATH = REPORT_ROOT / "image_profile.json"
IMAGE_INVENTORY_PATH = REPORT_ROOT / "image_inventory.csv"
ANNOTATION_REVIEW_PATH = REPORT_ROOT / "annotation_review.csv"
REPRESENTATIVE_IMAGES_PATH = REPORT_ROOT / "representative_images.csv"
REPRESENTATION_RUNS_PATH = REPORT_ROOT / "representation_resolution_runs.csv"
REPRESENTATION_COMPARISON_PATH = REPORT_ROOT / "representation_resolution_comparison.csv"
AUGMENTATION_RUNS_PATH = REPORT_ROOT / "augmentation_runs.csv"
AUGMENTATION_COMPARISON_PATH = REPORT_ROOT / "augmentation_comparison.csv"
FAILURE_AUGMENTATION_MATRIX_PATH = REPORT_ROOT / "failure_to_augmentation_matrix.csv"
COMPATIBILITY_RUNS_PATH = REPORT_ROOT / "compatibility_training_runs.csv"
COMPATIBILITY_COMPARISON_PATH = REPORT_ROOT / "compatibility_strategy_comparison.csv"
COMPATIBILITY_PREDICTIONS_PATH = REPORT_ROOT / "compatibility_validation_predictions.csv"
SCORE_DISTRIBUTIONS_PATH = REPORT_ROOT / "score_distributions.csv"
RANKING_TRIPLETS_PATH = REPORT_ROOT / "ranking_triplets.csv"
EQUAL_PAIR_EVALUATION_PATH = REPORT_ROOT / "equal_pair_evaluation.csv"
RANKING_METRICS_PATH = REPORT_ROOT / "ranking_metrics.json"
OCCLUSION_RESULTS_PATH = REPORT_ROOT / "occlusion_results.csv"
REGION_PERTURBATION_PATH = REPORT_ROOT / "region_perturbation_summary.csv"
EXPLAINABILITY_SUMMARY_PATH = REPORT_ROOT / "explainability_summary.json"
PRETRAINED_BACKBONE_GATE_PATH = REPORT_ROOT / "pretrained_backbone_gate.json"
FINE_TUNING_GATE_PATH = REPORT_ROOT / "fine_tuning_gate.json"
HUMAN_ANNOTATION_GATE_PATH = REPORT_ROOT / "human_annotation_gate.json"
NOTEBOOK_AUDIT_PATH = REPORT_ROOT / "notebook_execution_audit.json"

EXECUTION_REGISTRY_JSON_PATH = (
    PROJECT_ROOT / "data" / "experiment_registry" / "vision_execution_registry.json"
)
EXECUTION_REGISTRY_CSV_PATH = (
    PROJECT_ROOT / "data" / "experiment_registry" / "vision_execution_registry.csv"
)

VISION_NOTEBOOK_PATH = (
    PROJECT_ROOT / "notebooks" / "course_coverage" / "03_vision_model_comparison.ipynb"
)
SCORING_NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "course_coverage"
    / "04_scoring_ranking_explainability.ipynb"
)
DOCUMENTATION_PATH = (
    PROJECT_ROOT / "docs" / "course_coverage" / "vision_experimental_suite.md"
)
CURRENT_STATUS_PATH = PROJECT_ROOT / "notebooks" / "course_coverage" / "CURRENT_STATUS.md"

OCCLUSION_FIGURE_PATHS = tuple(
    FIGURE_ROOT / f"occlusion_example_{number:02d}.png"
    for number in range(1, 9)
)

FIGURE_PATHS = {
    "dimension_scatter": FIGURE_ROOT / "image_dimension_scatter.png",
    "brightness_contrast": FIGURE_ROOT / "brightness_contrast_distribution.png",
    "category_source": FIGURE_ROOT / "category_source_distribution.png",
    "representative_gallery": FIGURE_ROOT / "representative_image_gallery.png",
    "representation_macro_f1": FIGURE_ROOT / "representation_macro_f1.png",
    "augmentation_macro_f1": FIGURE_ROOT / "augmentation_macro_f1.png",
    "compatibility_scores": FIGURE_ROOT / "compatibility_score_distribution.png",
    "ranking_margins": FIGURE_ROOT / "ranking_margin_distribution.png",
}

VISION_IDS = tuple(f"VIS-{number:03d}" for number in range(1, 10))
COMPLETED_VISION_IDS = ("VIS-001", "VIS-003", "VIS-004", "VIS-005", "VIS-008", "VIS-009")
DEFERRED_VISION_IDS = ("VIS-002", "VIS-006", "VIS-007")

RANDOM_SEEDS = (42, 43, 44)
RESOLUTIONS = (32, 48, 64)
REPRESENTATIONS = (
    "global_pool",
    "intermediate_fixed_conv",
    "multi_stage_fixed_conv",
)
AUGMENTATION_POLICIES = (
    "none",
    "brightness",
    "center_crop",
    "jpeg_compression",
    "combined",
)
COMPATIBILITY_STRATEGIES = (
    "ordinal_ridge",
    "class_probability_expected_score",
)

LOCK_FLAGS = {
    "locked_test_csv_files_opened": False,
    "test_split_used": False,
    "final_test_evaluation_authorized": False,
    "production_final_model_changed": False,
    "pretrained_weights_downloaded": False,
    "synthetic_human_agreement_reported": False,
}

TEXT_HASH_SUFFIXES = {
    ".csv", ".html", ".ipynb", ".json", ".md", ".py", ".txt", ".yaml", ".yml"
}

SOURCE_PATHS = (
    SUITE_CONFIG_PATH,
    *EXPERIMENT_CONFIG_PATHS,
    PROJECT_ROOT / "src" / "vision_suite_config.py",
    PROJECT_ROOT / "src" / "run_vision_experimental_suite.py",
    PROJECT_ROOT / "src" / "build_vision_experiment_notebooks.py",
    PROJECT_ROOT / "src" / "verification" / "vision_experimental_suite.py",
    PROJECT_ROOT / "tests" / "test_vision_experimental_suite.py",
    DOCUMENTATION_PATH,
    CURRENT_STATUS_PATH,
    PROJECT_ROOT / "src" / "verification" / "project_verification.py",
)

GENERATED_PATHS = (
    STATUS_PATH,
    SUMMARY_PATH,
    IMAGE_PROFILE_PATH,
    IMAGE_INVENTORY_PATH,
    ANNOTATION_REVIEW_PATH,
    REPRESENTATIVE_IMAGES_PATH,
    REPRESENTATION_RUNS_PATH,
    REPRESENTATION_COMPARISON_PATH,
    AUGMENTATION_RUNS_PATH,
    AUGMENTATION_COMPARISON_PATH,
    FAILURE_AUGMENTATION_MATRIX_PATH,
    COMPATIBILITY_RUNS_PATH,
    COMPATIBILITY_COMPARISON_PATH,
    COMPATIBILITY_PREDICTIONS_PATH,
    SCORE_DISTRIBUTIONS_PATH,
    RANKING_TRIPLETS_PATH,
    EQUAL_PAIR_EVALUATION_PATH,
    RANKING_METRICS_PATH,
    OCCLUSION_RESULTS_PATH,
    REGION_PERTURBATION_PATH,
    EXPLAINABILITY_SUMMARY_PATH,
    PRETRAINED_BACKBONE_GATE_PATH,
    FINE_TUNING_GATE_PATH,
    HUMAN_ANNOTATION_GATE_PATH,
    NOTEBOOK_AUDIT_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXECUTION_REGISTRY_CSV_PATH,
    VISION_NOTEBOOK_PATH,
    SCORING_NOTEBOOK_PATH,
    *FIGURE_PATHS.values(),
    *OCCLUSION_FIGURE_PATHS,
)


def project_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()
