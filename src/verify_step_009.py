from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.project_cli import COMMANDS
from src.real_dataset_config import (
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    PROJECT_ROOT,
    REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
)
from src.validate_real_dataset import (
    build_report,
    read_annotation_csv,
)


README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "collection_protocol.md"
)
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"

REQUIRED_DIRECTORIES = (
    PROJECT_ROOT / "data" / "real" / "originals",
    PROJECT_ROOT / "data" / "real" / "staging",
    PROJECT_ROOT / "data" / "real" / "processed",
    PROJECT_ROOT / "data" / "real" / "processed" / "images",
    PROJECT_ROOT / "data" / "real" / "annotations",
    PROJECT_ROOT / "reports" / "real_dataset",
)

REQUIRED_FILES = (
    REAL_PART_GROUPS_PATH,
    REAL_IMAGES_PATH,
    REAL_IMAGE_MANIFEST_PATH,
    PROJECT_ROOT / "src" / "validate_real_dataset.py",
    PROJECT_ROOT / "src" / "verify_step_009.py",
    PROJECT_ROOT / "tests" / "test_real_dataset_intake.py",
    PROJECT_ROOT
    / "reports"
    / "step_009_real_dataset_intake_validation_foundation.md",
)

REQUIRED_COMMANDS = (
    "validate-real-data",
    "verify-step-009",
)


def validate_structure() -> list[str]:
    errors: list[str] = []

    for directory in REQUIRED_DIRECTORIES:
        if not directory.is_dir():
            errors.append(
                f"Missing directory: {directory.relative_to(PROJECT_ROOT)}"
            )

    for path in REQUIRED_FILES:
        if not path.is_file():
            errors.append(
                f"Missing file: {path.relative_to(PROJECT_ROOT)}"
            )

    return errors


def validate_csv_schemas() -> list[str]:
    errors: list[str] = []

    schema_checks = (
        (REAL_PART_GROUPS_PATH, PART_GROUP_COLUMNS),
        (REAL_IMAGES_PATH, IMAGE_MANIFEST_COLUMNS),
        (
            REAL_IMAGE_MANIFEST_PATH,
            REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
        ),
    )

    for path, expected_columns in schema_checks:
        if not path.is_file():
            continue

        try:
            dataframe = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        except Exception as error:  # pragma: no cover - verifier boundary
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
        "real_<category>_<number>",
        "real_image_manifest.csv",
        "development dataset",
        "SHA-256",
        "Cross-group",
    )

    for phrase in required_protocol_phrases:
        if phrase not in protocol:
            errors.append(
                f"Collection protocol is missing phrase: {phrase}"
            )

    return errors


def validate_gitignore_policy() -> list[str]:
    errors: list[str] = []
    lines = {
        line.strip()
        for line in GITIGNORE_PATH.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    required_lines = {
        "data/real/originals/*",
        "!data/real/originals/.gitkeep",
        "data/real/staging/*",
        "!data/real/staging/.gitkeep",
    }

    missing_lines = sorted(required_lines - lines)

    if missing_lines:
        errors.append(
            f".gitignore is missing real-data safety rules: "
            f"{missing_lines}."
        )

    forbidden_lines = {
        "data/real/processed/",
        "data/real/processed/images/",
        "data/real/annotations/",
    }

    present_forbidden = sorted(forbidden_lines & lines)

    if present_forbidden:
        errors.append(
            f".gitignore incorrectly excludes reproducible real data: "
            f"{present_forbidden}."
        )

    return errors


def validate_current_intake() -> list[str]:
    part_groups, part_group_errors = read_annotation_csv(
        REAL_PART_GROUPS_PATH,
        PART_GROUP_COLUMNS,
    )
    images, image_errors = read_annotation_csv(
        REAL_IMAGES_PATH,
        IMAGE_MANIFEST_COLUMNS,
    )
    report, manifest = build_report(
        part_groups,
        images,
        initial_errors=(*part_group_errors, *image_errors),
    )

    errors = list(report["errors"])

    if report["status"] != "PASS":
        errors.append("Current real-data intake validation does not pass.")

    if tuple(manifest.columns) != REAL_IMAGE_INTAKE_MANIFEST_COLUMNS:
        errors.append("Generated real-image manifest schema is invalid.")

    return errors


def build_verification_report() -> dict[str, object]:
    checks = {
        "structure": validate_structure(),
        "csv_schemas": validate_csv_schemas(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "gitignore_policy": validate_gitignore_policy(),
        "current_intake": validate_current_intake(),
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

    print("Step 009 verification")

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
