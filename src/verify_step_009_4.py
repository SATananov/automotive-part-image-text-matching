from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.import_first_real_batch import (
    build_inventory,
    import_first_real_batch,
    read_capture_file_map,
    read_plan,
    validate_capture_file_map,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    FIRST_BATCH_CAPTURE_FILE_MAP_PATH,
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_LOCAL_IMPORT_INVENTORY_COLUMNS,
    FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH,
    PROJECT_ROOT,
)


README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT / "reports" / "real_dataset" / "collection_protocol.md"
)
CHECKLIST_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_capture_checklist.md"
)
WORKFLOW_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_file_naming_and_local_import.md"
)
IMPORT_JSON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_local_import_readiness.json"
)
IMPORT_MARKDOWN_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_local_import_readiness.md"
)
IMPORT_MODULE_PATH = PROJECT_ROOT / "src" / "import_first_real_batch.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_first_real_batch_local_import.py"
CAPTURE_INBOX_KEEP_PATH = (
    PROJECT_ROOT / "data" / "real" / "capture_inbox" / ".gitkeep"
)


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
        FIRST_BATCH_CAPTURE_FILE_MAP_PATH,
        FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH,
        CHECKLIST_PATH,
        WORKFLOW_REPORT_PATH,
        IMPORT_JSON_PATH,
        IMPORT_MARKDOWN_PATH,
        IMPORT_MODULE_PATH,
        TEST_PATH,
        CAPTURE_INBOX_KEEP_PATH,
    )
    return [
        f"Missing first-batch naming/import file: {path.relative_to(PROJECT_ROOT)}."
        for path in required
        if not path.is_file()
    ]


def validate_schemas_and_names() -> list[str]:
    errors: list[str] = []
    if FIRST_BATCH_CAPTURE_FILE_MAP_PATH.is_file():
        if read_columns(FIRST_BATCH_CAPTURE_FILE_MAP_PATH) != (
            FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS
        ):
            errors.append(
                "first_batch_capture_file_map.csv has an unexpected schema."
            )
    if FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH.is_file():
        if read_columns(FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH) != (
            FIRST_BATCH_LOCAL_IMPORT_INVENTORY_COLUMNS
        ):
            errors.append(
                "first_batch_local_import_inventory.csv has an unexpected schema."
            )

    file_map, map_errors = read_capture_file_map()
    plan, plan_errors = read_plan()
    errors.extend(map_errors)
    errors.extend(plan_errors)
    errors.extend(validate_capture_file_map(file_map, plan))
    if len(file_map) != FIRST_BATCH_EXPECTED_IMAGES:
        errors.append("Capture file map does not contain all 20 filenames.")
    expected_examples = {
        "real_starter_001_front.jpg",
        "real_brake_disc_001_detail.jpg",
        "real_air_filter_001_detail.jpg",
    }
    actual_names = set(file_map.get("capture_filename", []))
    missing_examples = sorted(expected_examples - actual_names)
    if missing_examples:
        errors.append(
            f"Capture file map is missing descriptive filenames: {missing_examples}."
        )
    return errors


def validate_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    expected_commands = (
        "import-first-real-batch",
        "stage-first-real-batch-capture",
        "verify-step-009-4",
    )
    for command in expected_commands:
        if command not in COMMANDS:
            errors.append(f"CLI command '{command}' is not registered.")

    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (README_PATH, PROTOCOL_PATH, CHECKLIST_PATH)
        if path.is_file()
    )
    required_fragments = (
        "real_<part_category>_001_<view>.jpg",
        "real_starter_001_front.jpg",
        "data/real/capture_inbox/batch_001/",
        "first_batch_capture_file_map.csv",
        "first_batch_local_import_inventory.csv",
        "python -m src.project_cli import-first-real-batch",
        "copies original bytes",
    )
    for fragment in required_fragments:
        if fragment not in combined:
            errors.append(
                f"Naming/import documentation is missing '{fragment}'."
            )
    return errors


def validate_safeguards() -> list[str]:
    source = IMPORT_MODULE_PATH.read_text(encoding="utf-8-sig")
    required_fragments = (
        "validate_capture_file_map",
        "CAPTURE_FILENAME_PATTERN",
        "atomic_copy_bytes",
        "restore_destinations",
        "Duplicate local capture content",
        "Original destination already exists with different",
        "live_state_unchanged",
        "Unexpected image files in the first-batch capture inbox",
        "Image.open",
    )
    return [
        f"Local import workflow is missing safeguard marker: {fragment}."
        for fragment in required_fragments
        if fragment not in source
    ]


def validate_gitignore() -> list[str]:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8-sig")
    required = (
        "data/real/capture_inbox/*",
        "!data/real/capture_inbox/.gitkeep",
    )
    return [
        f".gitignore is missing local capture policy: {value}"
        for value in required
        if value not in gitignore
    ]


def validate_current_report() -> list[str]:
    if not IMPORT_JSON_PATH.is_file():
        return ["Current first-batch local import report is missing."]
    try:
        report = json.loads(IMPORT_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"Cannot read first-batch local import report: {error}."]
    errors: list[str] = []
    if report.get("status") != "PASS":
        errors.append("Current first-batch local import report is not PASS.")
    if report.get("live_state_unchanged") != "PASS":
        errors.append("Current local import report does not prove immutability.")
    if report.get("readiness") not in {
        "AWAITING_LOCAL_FILES",
        "LOCAL_IMPORT_IN_PROGRESS",
        "READY_FOR_STAGING",
    }:
        errors.append("Current local import readiness is not a safe state.")
    return errors


def build_verification_report() -> dict[str, object]:
    checks = {
        "structure": validate_structure(),
        "schemas_and_filenames": validate_schemas_and_names(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "local_import_safeguards": validate_safeguards(),
        "gitignore_policy": validate_gitignore(),
        "current_import_state": validate_current_report(),
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
    print("Step 009.4 verification")
    for name, status in report["checks"].items():
        print(f"- {name}: {status}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
