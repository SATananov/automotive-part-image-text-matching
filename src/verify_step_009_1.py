from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.project_cli import COMMANDS
from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_SAMPLE_INTAKE_PATH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.review_real_sample_intake import (
    build_review_report,
    load_review_inputs,
)


README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "collection_protocol.md"
)
APPLY_MODULE_PATH = PROJECT_ROOT / "src" / "apply_real_sample_intake.py"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"

REQUIRED_FILES = (
    REAL_SAMPLE_INTAKE_PATH,
    REAL_APPROVAL_LOG_PATH,
    PROJECT_ROOT / "src" / "review_real_sample_intake.py",
    APPLY_MODULE_PATH,
    PROJECT_ROOT / "src" / "verify_step_009_1.py",
    PROJECT_ROOT
    / "tests"
    / "test_real_sample_intake_workflow.py",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_review.json",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_review.md",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_apply.json",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_apply.md",
    PROJECT_ROOT
    / "reports"
    / "real_dataset/sample_intake_approval_workflow.md",
)

REQUIRED_COMMANDS = (
    "review-real-intake",
    "apply-real-intake",
    "verify-step-009-1",
)


def validate_structure() -> list[str]:
    return [
        f"Missing file: {path.relative_to(PROJECT_ROOT)}"
        for path in REQUIRED_FILES
        if not path.is_file()
    ]


def validate_csv_schemas() -> list[str]:
    errors: list[str] = []

    for path, expected_columns in (
        (REAL_SAMPLE_INTAKE_PATH, SAMPLE_INTAKE_COLUMNS),
        (REAL_APPROVAL_LOG_PATH, APPROVAL_LOG_COLUMNS),
    ):
        if not path.is_file():
            continue

        try:
            dataframe = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        except (OSError, UnicodeError, pd.errors.ParserError) as error:
            errors.append(
                f"Cannot read {path.relative_to(PROJECT_ROOT)}: {error}"
            )
            continue

        if tuple(dataframe.columns) != expected_columns:
            errors.append(
                f"Invalid schema in {path.relative_to(PROJECT_ROOT)}."
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
            errors.append(
                f"README.md does not document: {full_command}"
            )

    required_protocol_phrases = (
        "sample_intake.csv",
        "approval_log.csv",
        "pending",
        "approved",
        "rejected",
        "transaction",
        "EXIF",
        "RGB PNG",
    )

    for phrase in required_protocol_phrases:
        if phrase not in protocol:
            errors.append(
                f"Collection protocol is missing phrase: {phrase}"
            )

    return errors


def validate_gitignore_safeguards() -> list[str]:
    lines = {
        line.strip()
        for line in GITIGNORE_PATH.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required_lines = {
        "data/real/processed/.automotive_step009_1_*/",
        "data/real/annotations/.*.tmp",
        "data/real/processed/.*.tmp",
    }
    missing = sorted(required_lines - lines)

    if missing:
        return [
            "Missing Step 009.1 temporary-file ignore rules: "
            f"{missing}."
        ]

    return []


def validate_transaction_safeguards() -> list[str]:
    source = APPLY_MODULE_PATH.read_text(encoding="utf-8-sig")
    errors: list[str] = []

    required_source_phrases = (
        "snapshot_files",
        "restore_files",
        "ImageOps.exif_transpose",
        '.convert("RGB")',
        "build_intake_validation_report",
        "atomic_write_dataframe",
    )

    for phrase in required_source_phrases:
        if phrase not in source:
            errors.append(
                f"Apply workflow is missing safeguard: {phrase}"
            )

    return errors


def validate_current_review() -> list[str]:
    intake, part_groups, images, approval_log, read_errors = (
        load_review_inputs()
    )
    report = build_review_report(
        intake,
        part_groups,
        images,
        approval_log,
        initial_errors=read_errors,
    )
    errors = list(report["errors"])

    if report["status"] != "PASS":
        errors.append("Current sample intake review does not pass.")

    return errors


def build_verification_report() -> dict[str, object]:
    checks = {
        "structure": validate_structure(),
        "csv_schemas": validate_csv_schemas(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "transaction_safeguards": validate_transaction_safeguards(),
        "gitignore_safeguards": validate_gitignore_safeguards(),
        "current_review": validate_current_review(),
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

    print("Step 009.1 verification")

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
