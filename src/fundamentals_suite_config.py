from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

STEP = "011.1"
READINESS = "FUNDAMENTALS_EXPERIMENTAL_SUITE_COMPLETE_TEST_LOCKED"
BASE_CHECKPOINT = "STEP_011_0_FULL_COURSE_COVERAGE_ARCHITECTURE"

CONFIG_ROOT = PROJECT_ROOT / "configs" / "course_coverage"
SUITE_CONFIG_PATH = CONFIG_ROOT / "fundamentals_suite.json"
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
EXPERIMENT_CONFIG_PATHS = tuple(
    EXPERIMENT_CONFIG_DIR / f"FND-{number:03d}.json"
    for number in range(1, 11)
)

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "integrated_train.csv"
VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "integrated_validation.csv"
)

REPORT_ROOT = PROJECT_ROOT / "reports" / "course_coverage" / "fundamentals"
FIGURE_ROOT = REPORT_ROOT / "figures"
STATUS_PATH = REPORT_ROOT / "fundamentals_suite_status.json"
SUMMARY_PATH = REPORT_ROOT / "fundamentals_suite_summary.md"
MANIFEST_PATH = REPORT_ROOT / "fundamentals_suite_manifest.json"
DATASET_PROFILE_PATH = REPORT_ROOT / "dataset_profile.json"
REPRESENTATIVE_EXAMPLES_PATH = REPORT_ROOT / "representative_examples.csv"
BATCH_CONTRACT_PATH = REPORT_ROOT / "batch_contract.json"
SAMPLE_BATCH_PATH = REPORT_ROOT / "sample_batch.csv"
BASELINE_DIAGNOSTIC_PATH = REPORT_ROOT / "baseline_gradient_diagnostic.json"
BASELINE_HISTORY_PATH = REPORT_ROOT / "baseline_small_batch_history.csv"
BASELINE_ARCHITECTURE_PATH = REPORT_ROOT / "baseline_model_architecture.txt"
OVERFIT_RESULT_PATH = REPORT_ROOT / "overfit_result.json"
OVERFIT_HISTORY_PATH = REPORT_ROOT / "overfit_history.csv"
TRAINING_LOOP_AUDIT_PATH = REPORT_ROOT / "training_loop_audit.json"
EXPERIMENT_COMPARISON_CSV_PATH = REPORT_ROOT / "experiment_comparison.csv"
EXPERIMENT_COMPARISON_JSON_PATH = REPORT_ROOT / "experiment_comparison.json"
TRAINING_HISTORIES_PATH = REPORT_ROOT / "training_histories.csv"
VALIDATION_PREDICTIONS_PATH = REPORT_ROOT / "validation_predictions.csv"
CONFUSION_MATRICES_PATH = REPORT_ROOT / "confusion_matrices.json"
OPTIMIZER_COMPARISON_PATH = REPORT_ROOT / "optimizer_comparison.csv"
OPTIMIZER_STABILITY_PATH = REPORT_ROOT / "optimizer_stability.json"
CAPACITY_COMPARISON_PATH = REPORT_ROOT / "capacity_comparison.csv"
CAPACITY_PROBABILITY_TRACKING_PATH = (
    REPORT_ROOT / "capacity_probability_tracking.csv"
)
ARCHITECTURE_COMPARISON_PATH = REPORT_ROOT / "architecture_comparison.csv"
PREPROCESSING_COMPARISON_PATH = REPORT_ROOT / "preprocessing_comparison.csv"
FAILURE_DIAGNOSTICS_PATH = REPORT_ROOT / "failure_diagnostics.csv"
FAILURE_PREVENTION_PATH = REPORT_ROOT / "failure_prevention_checklist.md"
NOTEBOOK_AUDIT_PATH = REPORT_ROOT / "notebook_execution_audit.json"

EXECUTION_REGISTRY_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "experiment_registry"
    / "fundamentals_execution_registry.json"
)
EXECUTION_REGISTRY_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "experiment_registry"
    / "fundamentals_execution_registry.csv"
)

NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "course_coverage"
    / "01_fundamentals_experiments.ipynb"
)
DOCUMENTATION_PATH = (
    PROJECT_ROOT
    / "docs"
    / "course_coverage"
    / "fundamentals_experimental_suite.md"
)

FIGURE_PATHS = {
    "label_distribution": FIGURE_ROOT / "eda_label_distribution.png",
    "text_length_distribution": FIGURE_ROOT / "eda_text_length_distribution.png",
    "overfit_curves": FIGURE_ROOT / "overfit_learning_curves.png",
    "optimizer_comparison": FIGURE_ROOT / "optimizer_macro_f1.png",
    "capacity_tradeoff": FIGURE_ROOT / "capacity_tradeoff.png",
    "architecture_comparison": FIGURE_ROOT / "architecture_comparison.png",
    "preprocessing_comparison": FIGURE_ROOT / "preprocessing_comparison.png",
    "failure_signatures": FIGURE_ROOT / "failure_signatures.png",
}

FUNDAMENTALS_IDS = tuple(f"FND-{number:03d}" for number in range(1, 11))
LOCK_FLAGS = {
    "locked_test_csv_files_opened": False,
    "test_split_used": False,
    "final_test_evaluation_authorized": False,
    "production_final_model_changed": False,
}
TEXT_HASH_SUFFIXES = {
    ".csv", ".html", ".ipynb", ".json", ".md", ".py",
    ".txt", ".yaml", ".yml",
}

GENERATED_PATHS = (
    STATUS_PATH,
    SUMMARY_PATH,
    DATASET_PROFILE_PATH,
    REPRESENTATIVE_EXAMPLES_PATH,
    BATCH_CONTRACT_PATH,
    SAMPLE_BATCH_PATH,
    BASELINE_DIAGNOSTIC_PATH,
    BASELINE_HISTORY_PATH,
    BASELINE_ARCHITECTURE_PATH,
    OVERFIT_RESULT_PATH,
    OVERFIT_HISTORY_PATH,
    TRAINING_LOOP_AUDIT_PATH,
    EXPERIMENT_COMPARISON_CSV_PATH,
    EXPERIMENT_COMPARISON_JSON_PATH,
    TRAINING_HISTORIES_PATH,
    VALIDATION_PREDICTIONS_PATH,
    CONFUSION_MATRICES_PATH,
    OPTIMIZER_COMPARISON_PATH,
    OPTIMIZER_STABILITY_PATH,
    CAPACITY_COMPARISON_PATH,
    CAPACITY_PROBABILITY_TRACKING_PATH,
    ARCHITECTURE_COMPARISON_PATH,
    PREPROCESSING_COMPARISON_PATH,
    FAILURE_DIAGNOSTICS_PATH,
    FAILURE_PREVENTION_PATH,
    NOTEBOOK_AUDIT_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXECUTION_REGISTRY_CSV_PATH,
    NOTEBOOK_PATH,
    *FIGURE_PATHS.values(),
)


def project_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()
