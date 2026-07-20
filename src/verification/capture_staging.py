from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS,
    FIRST_BATCH_CAPTURE_INVENTORY_PATH,
    FIRST_BATCH_ORIGINALS_DIRECTORY,
    FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
    PROJECT_ROOT,
    SAMPLE_INTAKE_COLUMNS,
)


README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT / "reports" / "real_dataset" / "collection_protocol.md"
)
STEP_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset/first_batch_capture_staging_readiness.md"
)
CAPTURE_JSON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_capture_readiness.json"
)
CAPTURE_MARKDOWN_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_capture_readiness.md"
)
CAPTURE_MODULE_PATH = PROJECT_ROOT / "src" / "stage_first_real_batch_capture.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_first_real_batch_capture.py"


def read_csv_columns(path: Path) -> tuple[str, ...]:
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    return tuple(frame.columns)


def validate_structure() -> list[str]:
    required = (
        CAPTURE_MODULE_PATH,
        FIRST_BATCH_CAPTURE_INVENTORY_PATH,
        FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
        CAPTURE_JSON_PATH,
        CAPTURE_MARKDOWN_PATH,
        STEP_REPORT_PATH,
        TEST_PATH,
    )
    return [
        f"Missing Capture staging file: {path.relative_to(PROJECT_ROOT)}."
        for path in required
        if not path.is_file()
    ]


def validate_schemas() -> list[str]:
    errors: list[str] = []
    if FIRST_BATCH_CAPTURE_INVENTORY_PATH.is_file():
        columns = read_csv_columns(FIRST_BATCH_CAPTURE_INVENTORY_PATH)
        if columns != FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS:
            errors.append(
                "first_batch_capture_inventory.csv uses an unexpected schema."
            )
    if FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH.is_file():
        columns = read_csv_columns(FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH)
        if columns != SAMPLE_INTAKE_COLUMNS:
            errors.append(
                "first_batch_review_queue_draft.csv uses an unexpected schema."
            )
    return errors


def validate_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    expected_commands = (
        "stage-first-real-batch-capture",
        "verify-capture-staging",
    )
    for command in expected_commands:
        if command not in COMMANDS:
            errors.append(f"CLI command '{command}' is not registered.")

    readme = README_PATH.read_text(encoding="utf-8")
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8")
    required_fragments = (
        "data/real/originals/batch_001/",
        "stage-first-real-batch-capture",
        "first_batch_capture_inventory.csv",
        "first_batch_review_queue_draft.csv",
        "pending",
    )
    for fragment in required_fragments:
        if fragment not in readme and fragment not in protocol:
            errors.append(
                f"Capture staging documentation is missing '{fragment}'."
            )
    return errors


def validate_safeguards() -> list[str]:
    source = CAPTURE_MODULE_PATH.read_text(encoding="utf-8")
    required_fragments = (
        "normalized_jpeg_bytes",
        "ImageOps.exif_transpose",
        "restore_staging",
        "atomic_write_bytes",
        "Duplicate original capture content",
        "Duplicate normalized staging content",
        "Live annotations, queue, approval log, or manifest changed",
        "plan_row_to_intake",
        "only pending decisions",
        "Staging destination already exists with different",
    )
    return [
        f"Capture workflow is missing safeguard marker: {fragment}."
        for fragment in required_fragments
        if fragment not in source
    ]


def validate_current_report() -> list[str]:
    if not CAPTURE_JSON_PATH.is_file():
        return ["Current Capture staging capture report is missing."]
    try:
        report = json.loads(CAPTURE_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"Cannot read Capture staging capture report: {error}."]

    errors: list[str] = []
    if report.get("status") != "PASS":
        errors.append("Current Capture staging capture report is not PASS.")
    if report.get("live_queue_unchanged") != "PASS":
        errors.append("Capture staging report does not prove live-queue immutability.")
    allowed_readiness = {
        "AWAITING_CAPTURE",
        "CAPTURE_IN_PROGRESS",
        "READY_FOR_MANUAL_QUEUE_IMPORT",
        "READY_FOR_CONTROLLED_INTAKE",
        "BATCH_COMPLETE",
    }
    if report.get("readiness") not in allowed_readiness:
        errors.append(
            "Current Capture staging readiness is not an accepted safe state."
        )
    return errors


def build_verification_report() -> dict[str, object]:
    checks = {
        "structure": validate_structure(),
        "schemas": validate_schemas(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "capture_safeguards": validate_safeguards(),
        "current_capture_state": validate_current_report(),
    }
    errors = [error for group in checks.values() for error in group]
    return {
        "status": "PASS" if not errors else "FAIL",
        "checks": {
            name: "PASS" if not check_errors else "FAIL"
            for name, check_errors in checks.items()
        },
        "errors": errors,
        "originals_directory": str(
            FIRST_BATCH_ORIGINALS_DIRECTORY.relative_to(PROJECT_ROOT)
        ),
    }


def main() -> None:
    report = build_verification_report()
    print("Capture staging verification")
    for name, status in report["checks"].items():
        print(f"- {name}: {status}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
