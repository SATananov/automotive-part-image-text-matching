from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

STEP = "011.2"
READINESS = (
    "SEQUENCE_EXPERIMENTAL_SUITE_CORE_COMPLETE_"
    "PRETRAINED_GATE_TEST_LOCKED"
)
BASE_CHECKPOINT = "cfb6bba"

CONFIG_ROOT = PROJECT_ROOT / "configs" / "course_coverage"
SUITE_CONFIG_PATH = CONFIG_ROOT / "sequence_suite.json"
EXPERIMENT_CONFIG_DIR = CONFIG_ROOT / "experiments"
EXPERIMENT_CONFIG_PATHS = tuple(
    EXPERIMENT_CONFIG_DIR / f"SEQ-{number:03d}.json"
    for number in range(1, 11)
)

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "integrated_train.csv"
VALIDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "integrated_validation.csv"
)

REPORT_ROOT = PROJECT_ROOT / "reports" / "course_coverage" / "sequence"
FIGURE_ROOT = REPORT_ROOT / "figures"
STATUS_PATH = REPORT_ROOT / "sequence_suite_status.json"
SUMMARY_PATH = REPORT_ROOT / "sequence_suite_summary.md"
MANIFEST_PATH = REPORT_ROOT / "sequence_suite_manifest.json"
TEXT_PROFILE_PATH = REPORT_ROOT / "text_profile.json"
REPRESENTATIVE_EXAMPLES_PATH = REPORT_ROOT / "representative_examples.csv"
LOADER_CONTRACT_PATH = REPORT_ROOT / "text_loader_contract.json"
SAMPLE_BATCH_PATH = REPORT_ROOT / "sample_text_batch.csv"
TOKENIZATION_SUMMARY_PATH = REPORT_ROOT / "tokenization_summary.json"
TOKENIZATION_EXAMPLES_PATH = REPORT_ROOT / "tokenization_examples.csv"
VOCABULARY_PATH = REPORT_ROOT / "vocabulary.json"
MODEL_COMPARISON_CSV_PATH = REPORT_ROOT / "model_comparison.csv"
MODEL_COMPARISON_JSON_PATH = REPORT_ROOT / "model_comparison.json"
TRAINING_RUNS_PATH = REPORT_ROOT / "training_runs.csv"
TRAINING_HISTORIES_PATH = REPORT_ROOT / "training_histories.csv"
VALIDATION_PREDICTIONS_PATH = REPORT_ROOT / "validation_predictions.csv"
CONFUSION_MATRICES_PATH = REPORT_ROOT / "confusion_matrices.json"
ROC_CURVES_PATH = REPORT_ROOT / "roc_curves.json"
ERROR_ANALYSIS_PATH = REPORT_ROOT / "validation_error_analysis.csv"
ATTENTION_EVIDENCE_PATH = REPORT_ROOT / "attention_evidence.json"
ATTENTION_TOKEN_SUMMARY_PATH = REPORT_ROOT / "attention_token_summary.csv"
PRETRAINED_GATE_PATH = REPORT_ROOT / "pretrained_transformer_gate.json"
NOTEBOOK_AUDIT_PATH = REPORT_ROOT / "notebook_execution_audit.json"

EXECUTION_REGISTRY_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "experiment_registry"
    / "sequence_execution_registry.json"
)
EXECUTION_REGISTRY_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "experiment_registry"
    / "sequence_execution_registry.csv"
)

NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "course_coverage"
    / "02_sequence_model_comparison.ipynb"
)
DOCUMENTATION_PATH = (
    PROJECT_ROOT
    / "docs"
    / "course_coverage"
    / "sequence_experimental_suite.md"
)

FIGURE_PATHS = {
    "text_length_distribution": FIGURE_ROOT / "text_length_distribution.png",
    "label_distribution": FIGURE_ROOT / "label_distribution.png",
    "model_macro_f1": FIGURE_ROOT / "model_macro_f1.png",
    "complexity_tradeoff": FIGURE_ROOT / "complexity_tradeoff.png",
    "roc_curves": FIGURE_ROOT / "roc_curves.png",
    "attention_correct_head_1": FIGURE_ROOT / "attention_correct_head_1.png",
    "attention_correct_head_2": FIGURE_ROOT / "attention_correct_head_2.png",
    "attention_incorrect_head_1": FIGURE_ROOT / "attention_incorrect_head_1.png",
    "attention_incorrect_head_2": FIGURE_ROOT / "attention_incorrect_head_2.png",
}

SEQUENCE_IDS = tuple(f"SEQ-{number:03d}" for number in range(1, 11))
COMPLETED_SEQUENCE_IDS = tuple(f"SEQ-{number:03d}" for number in range(1, 10))
DEFERRED_SEQUENCE_IDS = ("SEQ-010",)
LOCK_FLAGS = {
    "locked_test_csv_files_opened": False,
    "test_split_used": False,
    "final_test_evaluation_authorized": False,
    "production_final_model_changed": False,
    "pretrained_weights_downloaded": False,
}
TEXT_HASH_SUFFIXES = {
    ".csv", ".html", ".ipynb", ".json", ".md", ".py",
    ".txt", ".yaml", ".yml",
}

GENERATED_PATHS = (
    STATUS_PATH,
    SUMMARY_PATH,
    TEXT_PROFILE_PATH,
    REPRESENTATIVE_EXAMPLES_PATH,
    LOADER_CONTRACT_PATH,
    SAMPLE_BATCH_PATH,
    TOKENIZATION_SUMMARY_PATH,
    TOKENIZATION_EXAMPLES_PATH,
    VOCABULARY_PATH,
    MODEL_COMPARISON_CSV_PATH,
    MODEL_COMPARISON_JSON_PATH,
    TRAINING_RUNS_PATH,
    TRAINING_HISTORIES_PATH,
    VALIDATION_PREDICTIONS_PATH,
    CONFUSION_MATRICES_PATH,
    ROC_CURVES_PATH,
    ERROR_ANALYSIS_PATH,
    ATTENTION_EVIDENCE_PATH,
    ATTENTION_TOKEN_SUMMARY_PATH,
    PRETRAINED_GATE_PATH,
    NOTEBOOK_AUDIT_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXECUTION_REGISTRY_CSV_PATH,
    NOTEBOOK_PATH,
    *FIGURE_PATHS.values(),
)


def project_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()
