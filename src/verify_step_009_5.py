from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.prepare_first_batch_capture_session import (
    JSON_REPORT_PATH,
    MARKDOWN_REPORT_PATH,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_SESSION_COLUMNS,
    FIRST_BATCH_CAPTURE_SESSION_PATH,
    FIRST_BATCH_EXPECTED_GROUPS,
    FIRST_BATCH_EXPECTED_IMAGES,
    PROJECT_ROOT,
)


README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT / "reports" / "real_dataset" / "collection_protocol.md"
)
OPERATOR_GUIDE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_operator_guide.md"
)
WORKFLOW_MODULE_PATH = (
    PROJECT_ROOT / "src" / "prepare_first_batch_capture_session.py"
)
TEST_PATH = PROJECT_ROOT / "tests" / "test_first_batch_capture_session.py"


def read_columns(path: Path) -> tuple[str, ...]:
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    return tuple(frame.columns)


def validate_structure() -> list[str]:
    required = (
        OPERATOR_GUIDE_PATH,
        FIRST_BATCH_CAPTURE_SESSION_PATH,
        JSON_REPORT_PATH,
        MARKDOWN_REPORT_PATH,
        WORKFLOW_MODULE_PATH,
        TEST_PATH,
    )
    return [
        "Missing operator-session file: "
        f"{path.relative_to(PROJECT_ROOT)}."
        for path in required
        if not path.is_file()
    ]


def validate_session_schema() -> list[str]:
    if not FIRST_BATCH_CAPTURE_SESSION_PATH.is_file():
        return ["First-batch capture session CSV is missing."]
    frame = pd.read_csv(
        FIRST_BATCH_CAPTURE_SESSION_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    errors: list[str] = []
    if tuple(frame.columns) != FIRST_BATCH_CAPTURE_SESSION_COLUMNS:
        errors.append(
            "first_batch_capture_session.csv has an unexpected schema."
        )
    if len(frame) != FIRST_BATCH_EXPECTED_GROUPS:
        errors.append(
            "First-batch capture session must contain "
            f"{FIRST_BATCH_EXPECTED_GROUPS} physical-part rows."
        )
    if frame.get("front_filename", pd.Series(dtype=str)).nunique() != (
        FIRST_BATCH_EXPECTED_GROUPS
    ):
        errors.append("Capture session front filenames are not unique.")
    if frame.get("detail_filename", pd.Series(dtype=str)).nunique() != (
        FIRST_BATCH_EXPECTED_GROUPS
    ):
        errors.append("Capture session detail filenames are not unique.")
    return errors


def validate_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    expected_commands = (
        "prepare-first-real-batch-session",
        "verify-step-009-5",
    )
    for command in expected_commands:
        if command not in COMMANDS:
            errors.append(f"CLI command '{command}' is not registered.")

    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (README_PATH, PROTOCOL_PATH, OPERATOR_GUIDE_PATH)
        if path.is_file()
    )
    required_fragments = (
        "first_batch_operator_guide.md",
        "first_batch_capture_session.csv",
        "prepare-first-real-batch-session",
        "real_starter_001_front.jpg",
        "AWAITING_CAPTURE",
        "READY_FOR_LOCAL_IMPORT",
        "READY_FOR_STAGING",
        "No command in this guide approves samples automatically.",
    )
    for fragment in required_fragments:
        if fragment not in combined:
            errors.append(
                f"Operator-session documentation is missing '{fragment}'."
            )
    return errors


def validate_safeguards() -> list[str]:
    source = WORKFLOW_MODULE_PATH.read_text(encoding="utf-8-sig")
    required_fragments = (
        "protected_state",
        "validate_capture_file_map",
        "validate_session_map",
        "inspect_image",
        "Duplicate capture content",
        "live_state_unchanged",
        "CAPTURE_SESSION_BLOCKED",
        "READY_FOR_LOCAL_IMPORT",
        "READY_FOR_STAGING",
    )
    return [
        f"Capture-session workflow is missing safeguard marker: {fragment}."
        for fragment in required_fragments
        if fragment not in source
    ]


def validate_semantic_names() -> list[str]:
    forbidden = list((PROJECT_ROOT / "reports").glob("step_009_5*"))
    return [
        "Step 009.5 permanent report uses a technical step filename: "
        f"{path.relative_to(PROJECT_ROOT)}."
        for path in forbidden
    ]


def validate_current_report() -> list[str]:
    if not JSON_REPORT_PATH.is_file():
        return ["Current capture-session report is missing."]
    try:
        report = json.loads(JSON_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"Cannot read capture-session report: {error}."]

    errors: list[str] = []
    if report.get("status") != "PASS":
        errors.append("Current capture-session report is not PASS.")
    if report.get("live_state_unchanged") != "PASS":
        errors.append(
            "Current capture-session report does not prove immutability."
        )
    if report.get("readiness") not in {
        "AWAITING_CAPTURE",
        "CAPTURE_SESSION_IN_PROGRESS",
        "READY_FOR_LOCAL_IMPORT",
        "READY_FOR_STAGING",
    }:
        errors.append("Current capture-session readiness is not safe.")
    counts = report.get("counts", {})
    if counts.get("planned_files") != FIRST_BATCH_EXPECTED_IMAGES:
        errors.append("Capture-session report has the wrong planned count.")
    return errors


def build_verification_report() -> dict[str, object]:
    checks = {
        "structure": validate_structure(),
        "session_schema": validate_session_schema(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "capture_session_safeguards": validate_safeguards(),
        "semantic_filenames": validate_semantic_names(),
        "current_session_state": validate_current_report(),
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
    print("Step 009.5 verification")
    for name, status in report["checks"].items():
        print(f"- {name}: {status}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
