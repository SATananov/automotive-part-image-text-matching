from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.integrated_training_config import (
    DEVELOPMENT_METRIC_PATHS,
    INTEGRATED_COMPARISON_CSV_PATH,
    INTEGRATED_COMPARISON_JSON_PATH,
    INTEGRATED_METRIC_PATHS,
    INTEGRATED_RUN_STATUS_PATH,
    INTEGRATED_SUMMARY_PATH,
    INTEGRATED_TEST_LOCK_PATH,
    INTEGRATED_TEST_PATH,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
    MODEL_TITLES,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import PROJECT_ROOT
from src.validate_external_training_readiness import (
    project_relative_path,
    sha256_canonical_csv,
)

README_PATH = PROJECT_ROOT / "README.md"
WORKFLOW_PATH = (
    PROJECT_ROOT / "src" / "run_integrated_training_validation.py"
)
CONFIG_PATH = PROJECT_ROOT / "src" / "integrated_training_config.py"
TEST_PATH = (
    PROJECT_ROOT / "tests" / "test_integrated_training_validation.py"
)


def read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(
            f"Missing integrated training artifact: "
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
        INTEGRATED_COMPARISON_CSV_PATH,
        INTEGRATED_COMPARISON_JSON_PATH,
        INTEGRATED_SUMMARY_PATH,
        INTEGRATED_RUN_STATUS_PATH,
        *INTEGRATED_METRIC_PATHS.values(),
        *DEVELOPMENT_METRIC_PATHS.values(),
    )
    return [
        "Missing integrated training file: "
        f"{path.relative_to(PROJECT_ROOT)}."
        for path in required
        if not path.is_file()
    ]


def validate_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    run_command = COMMANDS.get(
        "run-integrated-training-validation"
    )
    verify_command = COMMANDS.get(
        "verify-integrated-training-validation"
    )
    if run_command is None:
        errors.append("Integrated training CLI command is missing.")
    elif not run_command.requires_tensorflow:
        errors.append(
            "Integrated training CLI command is not TensorFlow-gated."
        )
    if verify_command is None:
        errors.append("Integrated verification CLI command is missing.")

    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (README_PATH, INTEGRATED_SUMMARY_PATH)
        if path.is_file()
    )
    required_fragments = (
        "run-integrated-training-validation",
        "verify-integrated-training-validation",
        "integrated_train.csv",
        "integrated_validation.csv",
        "locked test split",
        "was not loaded",
    )
    for fragment in required_fragments:
        if fragment not in combined:
            errors.append(
                "Integrated training documentation is missing "
                f"'{fragment}'."
            )
    return errors


def validate_source_safeguards() -> list[str]:
    if not WORKFLOW_PATH.is_file():
        return ["Integrated training workflow is missing."]
    source = WORKFLOW_PATH.read_text(encoding="utf-8-sig")
    required = (
        "read_test_lock",
        "locked_test_fingerprints",
        "load_integrated_datasets",
        "test_split_used",
        "test_evaluation_permitted",
        "INTEGRATED_TRAIN_PATH",
        "INTEGRATED_VALIDATION_PATH",
        "A locked test artifact changed during training",
    )
    errors = [
        f"Integrated training safeguard is missing: {fragment}."
        for fragment in required
        if fragment not in source
    ]
    forbidden = (
        "pd.read_csv(INTEGRATED_TEST_PATH",
        "read_integrated_split(INTEGRATED_TEST_PATH",
        "model.fit(INTEGRATED_TEST_PATH",
    )
    for fragment in forbidden:
        if fragment in source:
            errors.append(
                "Integrated training source loads the locked test split."
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
    if lock.get("hash_normalization") != "utf-8-lf":
        errors.append("Integrated test lock lacks canonical hashing.")

    training_inputs = {
        str(value)
        for value in lock.get("training_inputs", [])
    }
    expected_inputs = {
        project_relative_path(INTEGRATED_TRAIN_PATH),
        project_relative_path(INTEGRATED_VALIDATION_PATH),
    }
    if training_inputs != expected_inputs:
        errors.append(
            "Integrated test lock authorizes unexpected training inputs."
        )
    if project_relative_path(INTEGRATED_TEST_PATH) in training_inputs:
        errors.append("Integrated test path appears in training inputs.")

    if INTEGRATED_TEST_PATH.is_file():
        actual = sha256_canonical_csv(INTEGRATED_TEST_PATH)
        if actual != lock.get("integrated_test_sha256"):
            errors.append(
                "Integrated test fingerprint differs from the lock."
            )
    return errors


def validate_metric_artifacts() -> list[str]:
    errors: list[str] = []
    for model_slug, path in INTEGRATED_METRIC_PATHS.items():
        metrics = read_json(path, errors)
        if not metrics:
            continue
        if metrics.get("model_slug") != model_slug:
            errors.append(
                f"Integrated metric slug differs for {model_slug}."
            )
        if metrics.get("model") != MODEL_TITLES[model_slug]:
            errors.append(
                f"Integrated model title differs for {model_slug}."
            )
        if metrics.get("evaluation_split") != "integrated_validation":
            errors.append(
                f"Integrated metric uses a non-validation split: "
                f"{model_slug}."
            )
        if metrics.get("test_split_used") is not False:
            errors.append(
                f"Integrated metric reports test use: {model_slug}."
            )
        if metrics.get("test_evaluation_permitted") is not False:
            errors.append(
                f"Integrated metric permits test evaluation: {model_slug}."
            )
        if metrics.get("training_sample_count") != 180:
            errors.append(
                f"Integrated training sample count differs: {model_slug}."
            )
        if metrics.get("sample_count") != 60:
            errors.append(
                f"Integrated validation sample count differs: "
                f"{model_slug}."
            )
        for metric_name in ("accuracy", "macro_f1"):
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                errors.append(
                    f"Invalid {metric_name} for {model_slug}."
                )
    return errors


def validate_comparison() -> list[str]:
    errors: list[str] = []
    if not INTEGRATED_COMPARISON_CSV_PATH.is_file():
        return ["Integrated validation comparison CSV is missing."]
    try:
        comparison = pd.read_csv(
            INTEGRATED_COMPARISON_CSV_PATH
        )
    except Exception as error:
        return [f"Cannot read validation comparison CSV: {error}."]

    if len(comparison) != len(MODEL_TITLES):
        errors.append("Validation comparison does not contain six models.")
    if set(comparison.get("model_slug", [])) != set(MODEL_TITLES):
        errors.append("Validation comparison model set is incomplete.")
    if set(comparison.get("validation_rank", [])) != set(
        range(1, len(MODEL_TITLES) + 1)
    ):
        errors.append("Validation comparison ranks are invalid.")
    test_values = set(comparison.get("test_split_used", []))
    if test_values not in ({False}, {"False"}, {"false"}):
        errors.append("Validation comparison reports test use.")

    payload = read_json(INTEGRATED_COMPARISON_JSON_PATH, errors)
    if payload:
        if payload.get("status") != "PASS":
            errors.append("Validation comparison JSON is not PASS.")
        if payload.get("test_split_used") is not False:
            errors.append("Validation comparison JSON reports test use.")
        if len(payload.get("models", [])) != len(MODEL_TITLES):
            errors.append(
                "Validation comparison JSON model count is invalid."
            )
    return errors


def validate_current_status() -> list[str]:
    errors: list[str] = []
    status = read_json(INTEGRATED_RUN_STATUS_PATH, errors)
    if not status:
        return errors
    if status.get("status") != "PASS":
        errors.append("Integrated training status is not PASS.")
    if status.get("readiness") != (
        "VALIDATION_COMPARISON_COMPLETE"
    ):
        errors.append("Integrated training readiness is incomplete.")
    if status.get("model_count") != len(MODEL_TITLES):
        errors.append("Integrated training model count is invalid.")
    if status.get("test_split_used") is not False:
        errors.append("Integrated training status reports test use.")
    if status.get("locked_test_fingerprints_unchanged") is not True:
        errors.append("Locked test fingerprints were not preserved.")
    return errors


def validate_semantic_names() -> list[str]:
    errors: list[str] = []
    for root in (
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "reports" / "integrated_training",
    ):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower().startswith("step_"):
                errors.append(
                    "Integrated training artifact uses a technical step "
                    f"filename: {path.relative_to(PROJECT_ROOT)}."
                )
    return errors


def build_verification_report() -> dict[str, Any]:
    checks = {
        "structure": validate_structure(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "source_safeguards": validate_source_safeguards(),
        "test_lock": validate_test_lock(),
        "metric_artifacts": validate_metric_artifacts(),
        "validation_comparison": validate_comparison(),
        "current_status": validate_current_status(),
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
    print("Integrated training validation verification")
    for name, status in report["checks"].items():
        print(f"- {name}: {status}")
    print(f"Status: {report['status']}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
