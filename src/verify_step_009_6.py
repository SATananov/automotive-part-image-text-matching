from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.build_first_batch_capture_dashboard import (
    DASHBOARD_GUIDE_PATH,
    DASHBOARD_HTML_PATH,
    DASHBOARD_JSON_PATH,
    DASHBOARD_MARKDOWN_PATH,
    PIPELINE_STAGES,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS,
    FIRST_BATCH_CAPTURE_PROGRESS_PATH,
    FIRST_BATCH_EXPECTED_IMAGES,
    PROJECT_ROOT,
)


README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT / "reports" / "real_dataset" / "collection_protocol.md"
)
WORKFLOW_MODULE_PATH = (
    PROJECT_ROOT / "src" / "build_first_batch_capture_dashboard.py"
)
TEST_PATH = PROJECT_ROOT / "tests" / "test_first_batch_capture_dashboard.py"


def validate_structure() -> list[str]:
    required = (
        DASHBOARD_GUIDE_PATH,
        DASHBOARD_HTML_PATH,
        DASHBOARD_JSON_PATH,
        DASHBOARD_MARKDOWN_PATH,
        FIRST_BATCH_CAPTURE_PROGRESS_PATH,
        WORKFLOW_MODULE_PATH,
        TEST_PATH,
    )
    return [
        "Missing capture-dashboard file: "
        f"{path.relative_to(PROJECT_ROOT)}."
        for path in required
        if not path.is_file()
    ]


def validate_progress_schema() -> list[str]:
    if not FIRST_BATCH_CAPTURE_PROGRESS_PATH.is_file():
        return ["First-batch capture progress CSV is missing."]
    frame = pd.read_csv(
        FIRST_BATCH_CAPTURE_PROGRESS_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    errors: list[str] = []
    if tuple(frame.columns) != FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS:
        errors.append(
            "first_batch_capture_progress.csv has an unexpected schema."
        )
    if len(frame) != FIRST_BATCH_EXPECTED_IMAGES:
        errors.append(
            "First-batch capture progress must contain "
            f"{FIRST_BATCH_EXPECTED_IMAGES} photograph rows."
        )
    unknown_stages = sorted(
        set(frame.get("pipeline_stage", pd.Series(dtype=str)))
        - set(PIPELINE_STAGES)
    )
    if unknown_stages:
        errors.append(f"Unknown dashboard pipeline stages: {unknown_stages}.")
    if frame.get("intake_id", pd.Series(dtype=str)).duplicated().any():
        errors.append("Capture progress contains duplicate intake IDs.")
    return errors


def validate_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    expected_commands = (
        "build-first-real-batch-dashboard",
        "verify-step-009-6",
    )
    for command in expected_commands:
        if command not in COMMANDS:
            errors.append(f"CLI command '{command}' is not registered.")

    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (README_PATH, PROTOCOL_PATH, DASHBOARD_GUIDE_PATH)
        if path.is_file()
    )
    required_fragments = (
        "first_batch_capture_dashboard.html",
        "first_batch_capture_progress.csv",
        "build-first-real-batch-dashboard",
        "APPROVED_DATASET",
        "live_state_unchanged",
        "self-contained",
    )
    for fragment in required_fragments:
        if fragment not in combined:
            errors.append(
                f"Capture-dashboard documentation is missing '{fragment}'."
            )
    return errors


def validate_dashboard_rendering() -> list[str]:
    if not DASHBOARD_HTML_PATH.is_file():
        return ["First-batch capture dashboard HTML is missing."]
    source = DASHBOARD_HTML_PATH.read_text(encoding="utf-8-sig")
    required_fragments = (
        "<!doctype html>",
        "First Batch Capture Dashboard",
        "Category progress",
        "Photograph pipeline",
        "<progress",
        "class='card'",
        "real_starter_001_front.jpg",
    )
    return [
        f"Capture dashboard HTML is missing '{fragment}'."
        for fragment in required_fragments
        if fragment not in source
    ]


def validate_safeguards() -> list[str]:
    source = WORKFLOW_MODULE_PATH.read_text(encoding="utf-8-sig")
    required_fragments = (
        "protected_state",
        "path_fingerprint",
        "atomic_write_text",
        "atomic_write_csv",
        "live_state_unchanged",
        "html.escape",
        "CAPTURE_DASHBOARD_BLOCKED",
        "APPROVED_DATASET",
    )
    return [
        f"Capture-dashboard workflow is missing safeguard marker: {fragment}."
        for fragment in required_fragments
        if fragment not in source
    ]


def validate_semantic_names() -> list[str]:
    forbidden = list((PROJECT_ROOT / "reports").rglob("step_009_6*"))
    return [
        "Step 009.6 permanent report uses a technical step filename: "
        f"{path.relative_to(PROJECT_ROOT)}."
        for path in forbidden
    ]


def validate_current_report() -> list[str]:
    if not DASHBOARD_JSON_PATH.is_file():
        return ["Current capture-dashboard JSON report is missing."]
    try:
        report = json.loads(DASHBOARD_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"Cannot read capture-dashboard report: {error}."]

    errors: list[str] = []
    if report.get("status") != "PASS":
        errors.append("Current capture-dashboard report is not PASS.")
    if report.get("live_state_unchanged") != "PASS":
        errors.append(
            "Current capture-dashboard report does not prove immutability."
        )
    safe_readiness = {
        "AWAITING_CAPTURE",
        "CAPTURE_SESSION_IN_PROGRESS",
        "READY_FOR_LOCAL_IMPORT",
        "READY_FOR_STAGING",
        "STAGING_IN_PROGRESS",
        "READY_FOR_MANUAL_REVIEW",
        "REVIEW_IN_PROGRESS",
        "BATCH_APPROVED",
    }
    if report.get("readiness") not in safe_readiness:
        errors.append("Current capture-dashboard readiness is not safe.")
    counts = report.get("counts", {})
    if counts.get("planned") != FIRST_BATCH_EXPECTED_IMAGES:
        errors.append("Capture-dashboard report has the wrong planned count.")
    progress = report.get("overall_progress_percent")
    if not isinstance(progress, (int, float)) or not 0 <= progress <= 100:
        errors.append("Capture-dashboard overall progress is invalid.")
    return errors


def build_verification_report() -> dict[str, object]:
    checks = {
        "structure": validate_structure(),
        "progress_schema": validate_progress_schema(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "dashboard_rendering": validate_dashboard_rendering(),
        "dashboard_safeguards": validate_safeguards(),
        "semantic_filenames": validate_semantic_names(),
        "current_dashboard_state": validate_current_report(),
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
    print("Step 009.6 verification")
    for name, status in report["checks"].items():
        print(f"- {name}: {status}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
