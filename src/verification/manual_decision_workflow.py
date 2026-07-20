from __future__ import annotations

import ast
from typing import Callable

from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_REVIEW_RUNTIME_DIRECTORY,
    PROJECT_ROOT,
)
from src.validate_first_batch_manual_decisions import (
    APPLICATION_PLAN_COLUMNS,
    APPLICATION_PLAN_PATH,
    build_manual_decision_application_plan,
)

REQUIRED_FILES = (
    PROJECT_ROOT / "src" / "validate_first_batch_manual_decisions.py",
    PROJECT_ROOT / "src" / "apply_first_batch_manual_decisions.py",
    PROJECT_ROOT / "src" / "verification" / "manual_decision_workflow.py",
    PROJECT_ROOT
    / "tests"
    / "test_first_batch_manual_decision_application.py",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_manual_decision_validation_and_controlled_application.md",
)

EXPECTED_COMMANDS = {
    "validate-first-real-batch-manual-decisions": (
        "src.validate_first_batch_manual_decisions"
    ),
    "apply-first-real-batch-manual-decisions": (
        "src.apply_first_batch_manual_decisions"
    ),
    "verify-manual-decisions": "src.verification.manual_decision_workflow",
}

SAFE_READINESS_VALUES = {
    "AWAITING_QUEUE_ACTIVATION",
    "MANUAL_DECISIONS_REQUIRED",
    "READY_TO_APPLY",
    "MANUAL_DECISION_VALIDATION_BLOCKED",
}


def check_structure() -> list[str]:
    return [
        f"Missing file: {path.relative_to(PROJECT_ROOT)}"
        for path in REQUIRED_FILES
        if not path.is_file()
    ]


def check_cli_and_documentation() -> list[str]:
    errors: list[str] = []

    for command, module in EXPECTED_COMMANDS.items():
        spec = COMMANDS.get(command)
        if spec is None:
            errors.append(f"Missing CLI command: {command}.")
        elif spec.module != module:
            errors.append(
                f"CLI command {command} uses {spec.module}; "
                f"expected {module}."
            )

    readme = (PROJECT_ROOT / "README.md").read_text(
        encoding="utf-8-sig"
    )
    protocol = (
        PROJECT_ROOT
        / "reports"
        / "real_dataset"
        / "collection_protocol.md"
    ).read_text(encoding="utf-8-sig")

    for command in EXPECTED_COMMANDS:
        command_text = f"python -m src.project_cli {command}"
        if command_text not in readme:
            errors.append(f"README is missing: {command_text}.")
        if command_text not in protocol:
            errors.append(
                f"Collection protocol is missing: {command_text}."
            )

    return errors


def check_runtime_and_schema() -> list[str]:
    errors: list[str] = []
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(
        encoding="utf-8-sig"
    )
    if "data/real/runtime/" not in gitignore:
        errors.append("data/real/runtime/ is not Git-ignored.")

    try:
        FIRST_BATCH_REVIEW_RUNTIME_DIRECTORY.relative_to(
            PROJECT_ROOT / "data" / "real" / "runtime"
        )
    except ValueError:
        errors.append(
            "First-batch review outputs are not under the runtime boundary."
        )

    expected_columns = (
        "sequence",
        "intake_id",
        "part_group_id",
        "part_category",
        "view",
        "staging_path",
        "image_id",
        "quality_status",
        "operator_decision",
        "rejection_reason",
        "operator_notes",
        "validation_status",
    )
    if APPLICATION_PLAN_COLUMNS != expected_columns:
        errors.append(
            "Manual decision application plan columns differ from "
            "the Manual decision workflow schema."
        )

    if APPLICATION_PLAN_PATH.is_file():
        import pandas as pd

        try:
            dataframe = pd.read_csv(
                APPLICATION_PLAN_PATH,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        except Exception as error:
            errors.append(
                f"Cannot read the current application plan: {error}."
            )
        else:
            if tuple(dataframe.columns) != expected_columns:
                errors.append(
                    "Current runtime application plan has an invalid schema."
                )

    return errors


def source_contains_marker(source: str, marker: str) -> bool:
    """Find markers in source text or compiled adjacent string literals."""
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


def check_application_safeguards() -> list[str]:
    errors: list[str] = []
    validator_source = (
        PROJECT_ROOT
        / "src"
        / "validate_first_batch_manual_decisions.py"
    ).read_text(encoding="utf-8-sig")
    apply_source = (
        PROJECT_ROOT
        / "src"
        / "apply_first_batch_manual_decisions.py"
    ).read_text(encoding="utf-8-sig")

    validator_markers = (
        "canonical_plan_id",
        "queue_fingerprint",
        "workbook_fingerprint",
        "first_batch_plan_fingerprint",
        "READY_TO_APPLY",
        "All live queue rows must still be pending",
        "Rejected decisions require a rejection reason",
        "Workbook quality_status differs from the current live review",
    )
    for marker in validator_markers:
        if not source_contains_marker(validator_source, marker):
            errors.append(
                f"Validation safeguard marker is missing: {marker}."
            )

    apply_markers = (
        "snapshot_live_state",
        "restore_live_state",
        "saved validation plan is stale",
        "apply_intake",
        "Delegated approval count differs",
        "rollback_performed",
        "Applied first-batch intake IDs remain",
    )
    for marker in apply_markers:
        if not source_contains_marker(apply_source, marker):
            errors.append(
                f"Application safeguard marker is missing: {marker}."
            )

    return errors


def check_current_state() -> list[str]:
    errors: list[str] = []
    try:
        plan, report = build_manual_decision_application_plan()
    except Exception as error:
        return [f"Current manual decision validation failed: {error}."]

    if report.get("readiness") not in SAFE_READINESS_VALUES:
        errors.append(
            "Unexpected current Manual decision workflow readiness: "
            f"{report.get('readiness')}."
        )

    if tuple(plan.columns) != APPLICATION_PLAN_COLUMNS:
        errors.append("Current application plan schema is invalid.")

    if report.get("readiness") == "READY_TO_APPLY":
        if report.get("status") != "PASS":
            errors.append("READY_TO_APPLY must have PASS status.")
        if not report.get("plan_id"):
            errors.append("READY_TO_APPLY is missing a plan_id.")

    return errors


def run_check(
    name: str,
    check: Callable[[], list[str]],
) -> tuple[str, list[str]]:
    try:
        errors = check()
    except Exception as error:
        errors = [f"Unexpected verifier error: {error}."]
    return name, errors


def main() -> None:
    checks = [
        run_check("structure", check_structure),
        run_check(
            "cli_and_documentation",
            check_cli_and_documentation,
        ),
        run_check("runtime_and_schema", check_runtime_and_schema),
        run_check(
            "validation_and_application_safeguards",
            check_application_safeguards,
        ),
        run_check("current_state", check_current_state),
    ]

    print("Manual decision workflow verification")
    failed = False
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
