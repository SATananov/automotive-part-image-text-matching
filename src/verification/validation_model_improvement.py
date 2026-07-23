from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.integrated_training_config import (
    INTEGRATED_TEST_LOCK_PATH,
    INTEGRATED_TEST_PATH,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import PROJECT_ROOT
from src.validate_external_training_readiness import (
    project_relative_path,
    sha256_canonical_csv,
)
from src.validation_model_improvement_config import (
    CANDIDATE_ARCHITECTURE_PATHS,
    CANDIDATE_CONFUSION_MATRIX_PATHS,
    CANDIDATE_HISTORY_PATHS,
    CANDIDATE_METRIC_PATHS,
    CANDIDATE_PREDICTION_PATHS,
    CANDIDATE_TITLES,
    DATA_DIAGNOSTICS_JSON_PATH,
    DISAGREEMENT_CSV_PATH,
    DISAGREEMENT_JSON_PATH,
    ERROR_ANALYSIS_CSV_PATH,
    ERROR_ANALYSIS_JSON_PATH,
    EXPERIMENT_COMPARISON_CSV_PATH,
    EXPERIMENT_COMPARISON_JSON_PATH,
    EXPERIMENT_REGISTRY_PATH,
    EXPERIMENT_SEEDS,
    SELECTION_DECISION_PATH,
    VALIDATION_IMPROVEMENT_ROOT,
    VALIDATION_IMPROVEMENT_STATUS_PATH,
    VALIDATION_IMPROVEMENT_SUMMARY_PATH,
)

README_PATH = PROJECT_ROOT / "README.md"
WORKFLOW_PATH = (
    PROJECT_ROOT
    / "src"
    / "run_validation_error_analysis_and_model_improvement.py"
)
CONFIG_PATH = PROJECT_ROOT / "src" / "validation_model_improvement_config.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_validation_model_improvement.py"


def read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(
            f"Missing validation improvement artifact: "
            f"{path.relative_to(PROJECT_ROOT)}."
        )
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(
            f"Cannot read {path.relative_to(PROJECT_ROOT)}: {error}."
        )
        return {}
    if not isinstance(payload, dict):
        errors.append(
            f"Expected JSON object: {path.relative_to(PROJECT_ROOT)}."
        )
        return {}
    return payload


def validate_structure() -> list[str]:
    required = (
        WORKFLOW_PATH,
        CONFIG_PATH,
        TEST_PATH,
        ERROR_ANALYSIS_CSV_PATH,
        ERROR_ANALYSIS_JSON_PATH,
        DATA_DIAGNOSTICS_JSON_PATH,
        DISAGREEMENT_CSV_PATH,
        DISAGREEMENT_JSON_PATH,
        EXPERIMENT_COMPARISON_CSV_PATH,
        EXPERIMENT_COMPARISON_JSON_PATH,
        SELECTION_DECISION_PATH,
        EXPERIMENT_REGISTRY_PATH,
        VALIDATION_IMPROVEMENT_SUMMARY_PATH,
        VALIDATION_IMPROVEMENT_STATUS_PATH,
        *CANDIDATE_METRIC_PATHS.values(),
        *CANDIDATE_PREDICTION_PATHS.values(),
        *CANDIDATE_CONFUSION_MATRIX_PATHS.values(),
        *CANDIDATE_HISTORY_PATHS.values(),
        *CANDIDATE_ARCHITECTURE_PATHS.values(),
    )
    return [
        "Missing validation improvement file: "
        f"{path.relative_to(PROJECT_ROOT)}."
        for path in required
        if not path.is_file()
    ]


def validate_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    run_spec = COMMANDS.get(
        "run-validation-error-analysis-model-improvement"
    )
    verify_spec = COMMANDS.get("verify-validation-model-improvement")
    if run_spec is None:
        errors.append("Validation improvement run command is missing.")
    elif not run_spec.requires_tensorflow:
        errors.append("Validation improvement run command is not TensorFlow-gated.")
    if verify_spec is None:
        errors.append("Validation improvement verification command is missing.")

    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (README_PATH, VALIDATION_IMPROVEMENT_SUMMARY_PATH)
        if path.is_file()
    )
    for fragment in (
        "run-validation-error-analysis-model-improvement",
        "verify-validation-model-improvement",
        "integrated_train.csv",
        "integrated_validation.csv",
        "locked test split",
        "was not loaded",
        "fixed seeds",
    ):
        if fragment not in combined:
            errors.append(
                "Validation improvement documentation is missing "
                f"'{fragment}'."
            )
    return errors


def validate_source_safeguards() -> list[str]:
    if not WORKFLOW_PATH.is_file():
        return ["Validation improvement workflow is missing."]
    source = WORKFLOW_PATH.read_text(encoding="utf-8-sig")
    required = (
        "read_test_lock",
        "locked_test_fingerprints",
        "load_integrated_datasets",
        "EXPERIMENT_SEEDS",
        "test_split_used",
        "test_evaluation_permitted",
        "final_test_evaluation_authorized",
        "A locked test artifact changed during Step 010.4",
    )
    errors = [
        f"Validation improvement safeguard is missing: {fragment}."
        for fragment in required
        if fragment not in source
    ]
    for forbidden in (
        "pd.read_csv(INTEGRATED_TEST_PATH",
        "read_integrated_split(INTEGRATED_TEST_PATH",
        "model.fit(INTEGRATED_TEST_PATH",
        "extract_image_arrays(INTEGRATED_TEST_PATH",
    ):
        if forbidden in source:
            errors.append(
                "Validation improvement source loads the locked test split."
            )
    return errors


def validate_test_lock() -> list[str]:
    errors: list[str] = []
    lock = read_json(INTEGRATED_TEST_LOCK_PATH, errors)
    if not lock:
        return errors
    if lock.get("test_locked") is not True:
        errors.append("Integrated test lock is not closed.")
    if lock.get("test_evaluation_permitted") is not False:
        errors.append("Integrated test lock permits evaluation.")
    training_inputs = {str(value) for value in lock.get("training_inputs", [])}
    expected = {
        project_relative_path(INTEGRATED_TRAIN_PATH),
        project_relative_path(INTEGRATED_VALIDATION_PATH),
    }
    if training_inputs != expected:
        errors.append("Test lock authorizes unexpected training inputs.")
    if project_relative_path(INTEGRATED_TEST_PATH) in training_inputs:
        errors.append("Locked test path is authorized as a training input.")
    if INTEGRATED_TEST_PATH.is_file():
        if sha256_canonical_csv(INTEGRATED_TEST_PATH) != lock.get(
            "integrated_test_sha256"
        ):
            errors.append("Integrated test fingerprint differs from the lock.")
    return errors


def validate_candidate_artifacts() -> list[str]:
    errors: list[str] = []
    for slug in CANDIDATE_TITLES:
        metrics = read_json(CANDIDATE_METRIC_PATHS[slug], errors)
        if not metrics:
            continue
        if metrics.get("candidate_slug") != slug:
            errors.append(f"Candidate metric slug differs: {slug}.")
        if metrics.get("evaluation_split") != "integrated_validation":
            errors.append(f"Candidate uses a non-validation split: {slug}.")
        if metrics.get("training_sample_count") != 180:
            errors.append(f"Candidate training count differs: {slug}.")
        if metrics.get("sample_count") != 60:
            errors.append(f"Candidate validation count differs: {slug}.")
        if metrics.get("seeds") != list(EXPERIMENT_SEEDS):
            errors.append(f"Candidate seeds differ: {slug}.")
        if len(metrics.get("seed_results", [])) != len(EXPERIMENT_SEEDS):
            errors.append(f"Candidate seed result count differs: {slug}.")
        if metrics.get("test_split_used") is not False:
            errors.append(f"Candidate reports test use: {slug}.")
        if metrics.get("test_evaluation_permitted") is not False:
            errors.append(f"Candidate permits test evaluation: {slug}.")
        for name in (
            "accuracy",
            "macro_f1",
            "mean_seed_accuracy",
            "mean_seed_macro_f1",
            "worst_class_f1",
        ):
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                errors.append(f"Invalid {name} for candidate {slug}.")

        try:
            predictions = pd.read_csv(CANDIDATE_PREDICTION_PATHS[slug])
        except Exception as error:
            errors.append(f"Cannot read predictions for {slug}: {error}.")
            continue
        if len(predictions) != 60:
            errors.append(f"Candidate prediction count differs: {slug}.")
        if set(predictions.get("true_label", [])) != {
            "MATCH",
            "PARTIAL_MATCH",
            "MISMATCH",
        }:
            errors.append(f"Candidate true label set differs: {slug}.")
    return errors


def validate_comparison_and_decision() -> list[str]:
    errors: list[str] = []
    try:
        comparison = pd.read_csv(EXPERIMENT_COMPARISON_CSV_PATH)
    except Exception as error:
        return [f"Cannot read controlled experiment comparison: {error}."]
    if len(comparison) != len(CANDIDATE_TITLES):
        errors.append("Controlled comparison candidate count differs.")
    if set(comparison.get("candidate_slug", [])) != set(CANDIDATE_TITLES):
        errors.append("Controlled comparison candidate set differs.")
    if set(comparison.get("validation_rank", [])) != set(
        range(1, len(CANDIDATE_TITLES) + 1)
    ):
        errors.append("Controlled comparison ranks are invalid.")
    if set(comparison.get("test_split_used", [])) not in (
        {False},
        {"False"},
        {"false"},
    ):
        errors.append("Controlled comparison reports test use.")

    payload = read_json(EXPERIMENT_COMPARISON_JSON_PATH, errors)
    if payload and payload.get("status") != "PASS":
        errors.append("Controlled comparison JSON is not PASS.")

    decision = read_json(SELECTION_DECISION_PATH, errors)
    if decision:
        if decision.get("status") != "PASS":
            errors.append("Model selection decision is not PASS.")
        if decision.get("decision") not in {
            "IMPROVEMENT_ACCEPTED",
            "REFERENCE_RETAINED",
        }:
            errors.append("Model selection decision is invalid.")
        if decision.get("selected_candidate_slug") not in CANDIDATE_TITLES:
            errors.append("Selected candidate is unknown.")
        if decision.get("test_split_used") is not False:
            errors.append("Model selection decision reports test use.")
        if decision.get("final_test_evaluation_authorized") is not False:
            errors.append("Model selection decision authorizes final test use.")
    return errors


def validate_analysis_artifacts() -> list[str]:
    errors: list[str] = []
    diagnostics = read_json(DATA_DIAGNOSTICS_JSON_PATH, errors)
    if diagnostics:
        if diagnostics.get("train_validation_group_overlap") != 0:
            errors.append("Data diagnostics reports group overlap.")
        if diagnostics.get("test_split_used") is not False:
            errors.append("Data diagnostics reports test use.")

    errors_payload = read_json(ERROR_ANALYSIS_JSON_PATH, errors)
    if errors_payload:
        if errors_payload.get("validation_sample_count") != 60:
            errors.append("Error analysis validation count differs.")
        if errors_payload.get("test_split_used") is not False:
            errors.append("Error analysis reports test use.")

    disagreement = read_json(DISAGREEMENT_JSON_PATH, errors)
    if disagreement:
        if disagreement.get("sample_count") != 60:
            errors.append("Disagreement analysis sample count differs.")
        if disagreement.get("candidate_count") != len(CANDIDATE_TITLES):
            errors.append("Disagreement candidate count differs.")
        if disagreement.get("test_split_used") is not False:
            errors.append("Disagreement analysis reports test use.")
    return errors


def validate_status() -> list[str]:
    errors: list[str] = []
    status = read_json(VALIDATION_IMPROVEMENT_STATUS_PATH, errors)
    if not status:
        return errors
    if status.get("status") != "PASS":
        errors.append("Validation improvement status is not PASS.")
    if status.get("readiness") != "MODEL_IMPROVEMENT_DECISION_COMPLETE":
        errors.append("Validation improvement readiness is incomplete.")
    if status.get("candidate_count") != len(CANDIDATE_TITLES):
        errors.append("Validation improvement candidate count differs.")
    if status.get("seeds_per_candidate") != len(EXPERIMENT_SEEDS):
        errors.append("Validation improvement seed count differs.")
    if status.get("test_split_used") is not False:
        errors.append("Validation improvement status reports test use.")
    if status.get("final_test_evaluation_authorized") is not False:
        errors.append("Validation improvement status authorizes test use.")
    if status.get("locked_test_fingerprints_unchanged") is not True:
        errors.append("Locked test fingerprints were not preserved.")
    return errors


def validate_semantic_names() -> list[str]:
    errors: list[str] = []
    for root in (
        PROJECT_ROOT / "src",
        VALIDATION_IMPROVEMENT_ROOT,
    ):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower().startswith("step_"):
                errors.append(
                    "Validation improvement artifact uses a technical step "
                    f"filename: {path.relative_to(PROJECT_ROOT)}."
                )
    return errors


def build_verification_report() -> dict[str, Any]:
    checks = {
        "structure": validate_structure(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "source_safeguards": validate_source_safeguards(),
        "test_lock": validate_test_lock(),
        "candidate_artifacts": validate_candidate_artifacts(),
        "comparison_and_decision": validate_comparison_and_decision(),
        "analysis_artifacts": validate_analysis_artifacts(),
        "current_status": validate_status(),
        "semantic_filenames": validate_semantic_names(),
    }
    errors = [error for group in checks.values() for error in group]
    return {
        "status": "PASS" if not errors else "FAIL",
        "checks": {
            name: "PASS" if not check_errors else "FAIL"
            for name, check_errors in checks.items()
        },
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()
    print("Validation error analysis and model improvement verification")
    for name, status in report["checks"].items():
        print(f"- {name}: {status}")
    print(f"Status: {report['status']}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
