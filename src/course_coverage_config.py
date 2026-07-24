from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

STEP = "011.0"
BASE_COMMIT = "9237824"
READINESS = "FULL_COURSE_COVERAGE_ARCHITECTURE_READY_TEST_LOCKED"

CONFIG_DIR = PROJECT_ROOT / "configs" / "course_coverage"
MAPPING_PATH = CONFIG_DIR / "course_exercise_mapping.json"
DEFAULTS_PATH = CONFIG_DIR / "experiment_defaults.json"
RESOURCE_TIERS_PATH = CONFIG_DIR / "resource_tiers.json"

REGISTRY_DIR = PROJECT_ROOT / "data" / "experiment_registry"
REGISTRY_JSON_PATH = REGISTRY_DIR / "course_coverage_registry.json"
REGISTRY_CSV_PATH = REGISTRY_DIR / "course_coverage_registry.csv"

DOCS_DIR = PROJECT_ROOT / "docs" / "course_coverage"
MATRIX_PATH = DOCS_DIR / "full_course_coverage_matrix.md"
LOCKED_PLAN_PATH = DOCS_DIR / "locked_evaluation_plan.md"
EXECUTION_POLICY_PATH = DOCS_DIR / "experiment_execution_policy.md"

NOTEBOOK_PLAN_DIR = PROJECT_ROOT / "notebooks" / "course_coverage"
NOTEBOOK_PLAN_PATH = NOTEBOOK_PLAN_DIR / "README.md"

REPORT_DIR = PROJECT_ROOT / "reports" / "course_coverage"
READINESS_PATH = REPORT_DIR / "course_coverage_architecture_readiness.json"
MANIFEST_PATH = REPORT_DIR / "course_coverage_release_manifest.json"
SUMMARY_PATH = REPORT_DIR / "course_coverage_architecture_summary.md"

EXPECTED_EXPERIMENT_COUNTS = {
    "deep_learning_fundamentals": 10,
    "transformers_and_sequence_modelling": 10,
    "vision_models": 9,
}
EXPECTED_EXPERIMENT_COUNT = sum(EXPECTED_EXPERIMENT_COUNTS.values())
EXPECTED_RESOURCE_TIERS = {
    "TIER_0", "TIER_1", "TIER_2", "TIER_3", "TIER_4"
}
ALLOWED_EXECUTION_STATUSES = {
    "PLANNED", "READY", "RUNNING", "COMPLETED",
    "FAILED_DIAGNOSTIC", "REJECTED", "RETAINED",
}
PROHIBITED_TEST_INPUTS = {
    "data/processed/integrated_test.csv",
    "data/external/integrated/external_test.csv",
}
ALLOWED_TRAIN_SPLIT = "data/processed/integrated_train.csv"
ALLOWED_VALIDATION_SPLIT = "data/processed/integrated_validation.csv"

SOURCE_PATHS = (
    MAPPING_PATH,
    DEFAULTS_PATH,
    RESOURCE_TIERS_PATH,
)
GENERATED_PATHS = (
    REGISTRY_JSON_PATH,
    REGISTRY_CSV_PATH,
    MATRIX_PATH,
    LOCKED_PLAN_PATH,
    EXECUTION_POLICY_PATH,
    NOTEBOOK_PLAN_PATH,
    READINESS_PATH,
    SUMMARY_PATH,
)
REQUIRED_PATHS = SOURCE_PATHS + GENERATED_PATHS + (MANIFEST_PATH,)
TEXT_HASH_SUFFIXES = {
    ".csv", ".html", ".ipynb", ".json", ".md", ".py",
    ".txt", ".yaml", ".yml",
}


def project_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()
