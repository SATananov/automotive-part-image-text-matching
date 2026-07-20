from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Callable

from src.external_dataset_integration_config import (
    APPROVED_EXTERNAL_CATALOG_PATH,
    EXTERNAL_INTEGRATION_JSON_PATH,
    EXTERNAL_INTEGRATION_MARKDOWN_PATH,
    EXTERNAL_METADATA_PATH,
    EXTERNAL_SPLIT_MANIFEST_PATH,
    EXTERNAL_TEST_PATH,
    EXTERNAL_TRAIN_PATH,
    EXTERNAL_TRAINING_READINESS_JSON_PATH,
    EXTERNAL_TRAINING_READINESS_MARKDOWN_PATH,
    EXTERNAL_VALIDATION_PATH,
    INTEGRATED_SPLIT_MANIFEST_PATH,
    INTEGRATED_TEST_LOCK_PATH,
    INTEGRATED_TEST_PATH,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import PROJECT_ROOT
from src.integrate_external_dataset import (
    project_relative_path,
)
from src.validate_external_training_readiness import (
    validate_external_training_readiness,
)

REQUIRED_FILES = (
    PROJECT_ROOT / "src" / "external_dataset_integration_config.py",
    PROJECT_ROOT / "src" / "integrate_external_dataset.py",
    PROJECT_ROOT / "src" / "validate_external_training_readiness.py",
    PROJECT_ROOT / "src" / "verification" / "external_dataset_integration.py",
    PROJECT_ROOT / "tests" / "test_external_dataset_integration.py",
    PROJECT_ROOT
    / "reports"
    / "external_dataset"
    / "external_dataset_integration.md",
    APPROVED_EXTERNAL_CATALOG_PATH,
    EXTERNAL_METADATA_PATH,
    EXTERNAL_TRAIN_PATH,
    EXTERNAL_VALIDATION_PATH,
    EXTERNAL_TEST_PATH,
    EXTERNAL_SPLIT_MANIFEST_PATH,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
    INTEGRATED_TEST_PATH,
    INTEGRATED_SPLIT_MANIFEST_PATH,
    INTEGRATED_TEST_LOCK_PATH,
    EXTERNAL_INTEGRATION_JSON_PATH,
    EXTERNAL_INTEGRATION_MARKDOWN_PATH,
    EXTERNAL_TRAINING_READINESS_JSON_PATH,
    EXTERNAL_TRAINING_READINESS_MARKDOWN_PATH,
)

EXPECTED_COMMANDS = {
    "integrate-external-dataset": (
        "src.integrate_external_dataset"
    ),
    "validate-external-training-readiness": (
        "src.validate_external_training_readiness"
    ),
    "verify-external-dataset-integration": "src.verification.external_dataset_integration",
}


def source_contains(
    source: str,
    marker: str,
) -> bool:
    if marker in source:
        return True

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    return any(
        marker in node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    )


def check_structure() -> list[str]:
    return [
        f"Missing file: {path.relative_to(PROJECT_ROOT)}"
        for path in REQUIRED_FILES
        if not path.is_file()
    ]


def check_cli_and_documentation() -> list[str]:
    errors: list[str] = []

    for command, module_name in EXPECTED_COMMANDS.items():
        command_spec = COMMANDS.get(command)
        if command_spec is None:
            errors.append(f"Missing CLI command: {command}.")
        elif command_spec.module != module_name:
            errors.append(
                f"CLI command {command} uses "
                f"{command_spec.module}; expected {module_name}."
            )

    readme = (
        PROJECT_ROOT / "README.md"
    ).read_text(encoding="utf-8-sig")

    step_document = (
        PROJECT_ROOT
        / "reports"
        / "external_dataset"
        / "external_dataset_integration.md"
    ).read_text(encoding="utf-8-sig")

    for command in EXPECTED_COMMANDS:
        command_text = f"python -m src.project_cli {command}"
        if command_text not in readme:
            errors.append(
                f"README is missing: {command_text}."
            )
        if command_text not in step_document:
            errors.append(
                f"External dataset integration documentation is missing: "
                f"{command_text}."
            )

    return errors


def check_safeguards() -> list[str]:
    errors: list[str] = []

    integration_source = (
        PROJECT_ROOT
        / "src"
        / "integrate_external_dataset.py"
    ).read_text(encoding="utf-8-sig")

    validation_source = (
        PROJECT_ROOT
        / "src"
        / "validate_external_training_readiness.py"
    ).read_text(encoding="utf-8-sig")

    integration_markers = (
        "READY_FOR_EXTERNAL_DATASET",
        "operator_decision",
        "approved",
        "part_group_id",
        "external_group_",
        "test_locked",
        "test_evaluation_permitted",
        "restore_files",
        "INTEGRATED_TRAIN_PATH",
        "INTEGRATED_VALIDATION_PATH",
        "INTEGRATED_TEST_PATH",
    )

    for marker in integration_markers:
        if not source_contains(
            integration_source,
            marker,
        ):
            errors.append(
                f"Integration safeguard marker is missing: "
                f"{marker}."
            )

    validation_markers = (
        "Rejected audit candidates leaked",
        "all three labels",
        "integrated grouped split",
        "A locked test path appears in training inputs",
        "READY_FOR_TRAINING",
    )

    for marker in validation_markers:
        if not source_contains(
            validation_source,
            marker,
        ):
            errors.append(
                f"Training-readiness safeguard marker is missing: "
                f"{marker}."
            )

    return errors


def check_test_lock() -> list[str]:
    errors: list[str] = []

    if not INTEGRATED_TEST_LOCK_PATH.is_file():
        return [
            "The integrated test-lock file is missing."
        ]

    try:
        lock = json.loads(
            INTEGRATED_TEST_LOCK_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        return [
            f"Cannot read the integrated test lock: {error}."
        ]

    if lock.get("test_locked") is not True:
        errors.append(
            "The integrated test split is not locked."
        )

    if lock.get("test_evaluation_permitted") is not False:
        errors.append(
            "The integrated test lock permits evaluation."
        )

    training_inputs = {
        str(value)
        for value in lock.get(
            "training_inputs",
            [],
        )
    }

    expected_inputs = {
        project_relative_path(INTEGRATED_TRAIN_PATH),
        project_relative_path(INTEGRATED_VALIDATION_PATH),
    }
    if training_inputs != expected_inputs:
        errors.append(
            "Training inputs are not exactly train and validation."
        )

    forbidden = {
        project_relative_path(EXTERNAL_TEST_PATH),
        project_relative_path(INTEGRATED_TEST_PATH),
    }
    if training_inputs & forbidden:
        errors.append(
            "A locked test path appears in training inputs."
        )

    return errors


def check_current_state() -> list[str]:
    try:
        report = validate_external_training_readiness()
    except Exception as error:
        return [
            f"Current External dataset integration validation failed: {error}."
        ]

    errors: list[str] = []

    if report.get("status") != "PASS":
        errors.append(
            "External dataset integration validation status is not PASS."
        )

    if report.get("readiness") != "READY_FOR_TRAINING":
        errors.append(
            "External dataset integration is not READY_FOR_TRAINING."
        )

    if report.get("approved_external_images") != 50:
        errors.append(
            "External dataset integration does not contain 50 approved external images."
        )

    if report.get("external_samples") != 150:
        errors.append(
            "External dataset integration does not contain 150 external samples."
        )

    if report.get("test_locked") is not True:
        errors.append(
            "External dataset integration current state does not report a locked test split."
        )

    if report.get("test_evaluation_permitted") is not False:
        errors.append(
            "External dataset integration current state permits test evaluation."
        )

    return errors


def run_check(
    name: str,
    callback: Callable[[], list[str]],
) -> tuple[str, list[str]]:
    try:
        return name, callback()
    except Exception as error:
        return name, [
            f"Unexpected verifier error: {error}."
        ]


def main() -> None:
    checks = [
        run_check("structure", check_structure),
        run_check(
            "cli_and_documentation",
            check_cli_and_documentation,
        ),
        run_check(
            "integration_safeguards",
            check_safeguards,
        ),
        run_check(
            "test_lock",
            check_test_lock,
        ),
        run_check(
            "current_state",
            check_current_state,
        ),
    ]

    failed = False

    print("External dataset integration verification")
    for name, errors in checks:
        status = "PASS" if not errors else "FAIL"
        print(f"- {name}: {status}")
        for error in errors:
            print(f"  - {error}")
        failed = failed or bool(errors)

    print(f"Status: {'FAIL' if failed else 'PASS'}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
