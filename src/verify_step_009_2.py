from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dry_run_first_real_batch import build_dry_run_report
from src.prepare_first_real_batch import (
    build_preparation_report,
    read_plan,
    validate_plan,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PLAN_PATH,
    FIRST_BATCH_PREVIEW_COLUMNS,
    FIRST_BATCH_PREVIEW_PATH,
    PROJECT_ROOT,
)


README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "collection_protocol.md"
)
DRY_RUN_MODULE_PATH = PROJECT_ROOT / "src" / "dry_run_first_real_batch.py"
PREPARATION_MODULE_PATH = (
    PROJECT_ROOT / "src" / "prepare_first_real_batch.py"
)

REQUIRED_FILES = (
    FIRST_BATCH_PLAN_PATH,
    FIRST_BATCH_PREVIEW_PATH,
    PREPARATION_MODULE_PATH,
    DRY_RUN_MODULE_PATH,
    PROJECT_ROOT / "src" / "verify_step_009_2.py",
    PROJECT_ROOT / "tests" / "test_first_real_batch_dry_run.py",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_preparation.json",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_preparation.md",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_dry_run.json",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_dry_run.md",
    PROJECT_ROOT
    / "reports"
    / "step_009_2_first_real_batch_preparation_dry_run.md",
)

REQUIRED_COMMANDS = (
    "prepare-first-real-batch",
    "dry-run-first-real-batch",
    "verify-step-009-2",
)


def validate_structure() -> list[str]:
    return [
        f"Missing file: {path.relative_to(PROJECT_ROOT)}"
        for path in REQUIRED_FILES
        if not path.is_file()
    ]


def validate_plan_and_preview() -> list[str]:
    plan, read_errors = read_plan()
    errors = list(read_errors)
    errors.extend(validate_plan(plan))

    if len(plan) != FIRST_BATCH_EXPECTED_IMAGES:
        errors.append(
            "The committed first batch plan does not contain the expected "
            "number of image slots."
        )

    if FIRST_BATCH_PREVIEW_PATH.is_file():
        try:
            preview = pd.read_csv(
                FIRST_BATCH_PREVIEW_PATH,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        except (OSError, UnicodeError, pd.errors.ParserError) as error:
            errors.append(f"Cannot read first batch preview: {error}")
        else:
            if tuple(preview.columns) != FIRST_BATCH_PREVIEW_COLUMNS:
                errors.append(
                    "first_batch_queue_preview.csv has an invalid schema."
                )
            if len(preview) != len(plan):
                errors.append(
                    "First batch preview row count does not match the plan."
                )

    return errors


def validate_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    readme = README_PATH.read_text(encoding="utf-8-sig")
    protocol = PROTOCOL_PATH.read_text(encoding="utf-8-sig")

    for command in REQUIRED_COMMANDS:
        if command not in COMMANDS:
            errors.append(f"CLI command is not registered: {command}")

        full_command = f"python -m src.project_cli {command}"
        if full_command not in readme:
            errors.append(f"README.md does not document: {full_command}")

    required_phrases = (
        "first_batch_plan.csv",
        "20 planned images",
        "10 physical parts",
        "front",
        "detail",
        "dry run",
        "does not approve",
    )
    combined = f"{readme}\n{protocol}".lower()
    for phrase in required_phrases:
        if phrase.lower() not in combined:
            errors.append(
                f"Step 009.2 documentation is missing phrase: {phrase}"
            )

    return errors


def validate_dry_run_safeguards() -> list[str]:
    source = DRY_RUN_MODULE_PATH.read_text(encoding="utf-8-sig")
    preparation_source = PREPARATION_MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )
    errors: list[str] = []

    required_dry_run_phrases = (
        "fingerprint_live_state",
        "tempfile.TemporaryDirectory",
        "build_apply_plan",
        "simulation_decision",
        "immutability",
    )
    for phrase in required_dry_run_phrases:
        if phrase not in source:
            errors.append(f"Dry-run safeguard is missing: {phrase}")

    required_preparation_phrases = (
        "FIRST_BATCH_PLAN_COLUMNS",
        "build_review_report",
        "READY_FOR_QUEUE_REVIEW",
        "AWAITING_CAPTURE",
    )
    for phrase in required_preparation_phrases:
        if phrase not in preparation_source:
            errors.append(f"Preparation safeguard is missing: {phrase}")

    forbidden_dry_run_phrases = (
        "atomic_write_dataframe(",
        "os.replace(",
        "apply_intake(",
    )
    for phrase in forbidden_dry_run_phrases:
        if phrase in source:
            errors.append(
                f"Dry-run module contains a live-write operation: {phrase}"
            )

    return errors


def validate_current_state() -> list[str]:
    plan, read_errors = read_plan()
    preparation, _preview = build_preparation_report(
        plan,
        initial_errors=read_errors,
    )
    dry_run = build_dry_run_report()
    errors: list[str] = []

    if preparation["status"] != "PASS":
        errors.append("Current first-batch preparation does not pass.")
        errors.extend(preparation["errors"])

    if dry_run["status"] != "PASS":
        errors.append("Current first-batch dry run does not pass.")
        errors.extend(dry_run["errors"])

    if dry_run["immutability"] != "PASS":
        errors.append("Current first-batch dry run is not immutable.")

    return errors


def build_verification_report() -> dict[str, object]:
    checks = {
        "structure": validate_structure(),
        "plan_and_preview": validate_plan_and_preview(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "dry_run_safeguards": validate_dry_run_safeguards(),
        "current_state": validate_current_state(),
    }
    errors = [
        error
        for check_errors in checks.values()
        for error in check_errors
    ]
    return {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()

    print("Step 009.2 verification")
    for check_name, check_errors in report["checks"].items():
        status = "PASS" if not check_errors else "FAIL"
        print(f"- {check_name}: {status}")
        for error in check_errors:
            print(f"  - {error}")

    print(f"Status: {report['status']}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
