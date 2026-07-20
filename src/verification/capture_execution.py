from __future__ import annotations

import csv
import json
from pathlib import Path

from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_EXECUTION_JOURNAL_COLUMNS,
    FIRST_BATCH_EXECUTION_JOURNAL_PATH,
    FIRST_BATCH_LIVE_DASHBOARD_PATH,
    FIRST_BATCH_LIVE_PROGRESS_PATH,
    FIRST_BATCH_LIVE_STATUS_PATH,
    FIRST_BATCH_LIVE_SUMMARY_PATH,
    FIRST_BATCH_RUNTIME_DIRECTORY,
    PROJECT_ROOT,
)


README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT / "reports" / "real_dataset" / "collection_protocol.md"
)
EXECUTION_GUIDE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_capture_execution_and_live_progress.md"
)
WORKFLOW_MODULE_PATH = (
    PROJECT_ROOT / "src" / "execute_first_batch_capture_session.py"
)
REFRESH_MODULE_PATH = (
    PROJECT_ROOT / "src" / "refresh_first_batch_live_progress.py"
)
TEST_PATH = PROJECT_ROOT / "tests" / "test_first_batch_capture_execution.py"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"


def validate_structure() -> list[str]:
    required = (
        EXECUTION_GUIDE_PATH,
        WORKFLOW_MODULE_PATH,
        REFRESH_MODULE_PATH,
        TEST_PATH,
        FIRST_BATCH_LIVE_DASHBOARD_PATH,
        FIRST_BATCH_LIVE_PROGRESS_PATH,
        FIRST_BATCH_LIVE_STATUS_PATH,
        FIRST_BATCH_LIVE_SUMMARY_PATH,
        FIRST_BATCH_EXECUTION_JOURNAL_PATH,
    )
    return [
        "Missing capture-execution file: "
        f"{path.relative_to(PROJECT_ROOT)}."
        for path in required
        if not path.is_file()
    ]


def validate_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    expected_commands = (
        "run-first-real-batch-capture-session",
        "refresh-first-real-batch-live-progress",
        "verify-capture-execution",
    )
    for command in expected_commands:
        if command not in COMMANDS:
            errors.append(f"CLI command '{command}' is not registered.")

    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (README_PATH, PROTOCOL_PATH, EXECUTION_GUIDE_PATH)
        if path.is_file()
    )
    required_fragments = (
        "data/real/runtime/first_batch_capture/",
        "run-first-real-batch-capture-session",
        "refresh-first-real-batch-live-progress",
        "execution_journal.csv",
        "ROLLED_BACK",
        "live_dataset_unchanged",
        "tracked_outputs_unchanged",
        "never queue or approve",
    )
    for fragment in required_fragments:
        if fragment not in combined:
            errors.append(
                f"Capture-execution documentation is missing '{fragment}'."
            )
    return errors


def validate_runtime_isolation() -> list[str]:
    errors: list[str] = []
    if GITIGNORE_PATH.is_file():
        source = GITIGNORE_PATH.read_text(encoding="utf-8-sig")
        if "data/real/runtime/" not in source:
            errors.append("Gitignore does not isolate real-data runtime files.")
    else:
        errors.append("Project .gitignore is missing.")

    try:
        relative = FIRST_BATCH_RUNTIME_DIRECTORY.relative_to(PROJECT_ROOT)
    except ValueError:
        errors.append("Capture runtime directory is outside the project root.")
    else:
        if relative.as_posix() != "data/real/runtime/first_batch_capture":
            errors.append("Capture runtime directory has an unexpected path.")
    return errors


def validate_safeguards() -> list[str]:
    source = WORKFLOW_MODULE_PATH.read_text(encoding="utf-8-sig")
    required_fragments = (
        "runtime_output_paths",
        "directory_snapshot",
        "restore_directory",
        "LIVE_DATASET_PATHS",
        "TRACKED_OPERATIONAL_OUTPUTS",
        "live_dataset_unchanged",
        "tracked_outputs_unchanged",
        "ROLLED_BACK",
        "append_execution_journal",
        "FIRST_BATCH_RUNTIME_DIRECTORY",
    )
    return [
        f"Capture-execution workflow is missing safeguard marker: {fragment}."
        for fragment in required_fragments
        if fragment not in source
    ]


def validate_journal_schema() -> list[str]:
    if not FIRST_BATCH_EXECUTION_JOURNAL_PATH.is_file():
        return ["Capture execution journal is missing."]
    try:
        with FIRST_BATCH_EXECUTION_JOURNAL_PATH.open(
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            rows = list(reader)
    except OSError as error:
        return [f"Cannot read capture execution journal: {error}."]
    errors: list[str] = []
    if columns != FIRST_BATCH_EXECUTION_JOURNAL_COLUMNS:
        errors.append("Capture execution journal has an unexpected schema.")
    if not rows:
        errors.append("Capture execution journal has no execution cycle.")
    return errors


def validate_current_status() -> list[str]:
    if not FIRST_BATCH_LIVE_STATUS_PATH.is_file():
        return ["Current capture execution status is missing."]
    try:
        report = json.loads(
            FIRST_BATCH_LIVE_STATUS_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        return [f"Cannot read capture execution status: {error}."]

    errors: list[str] = []
    if report.get("status") != "PASS":
        errors.append("Current capture execution status is not PASS.")
    if report.get("live_dataset_unchanged") != "PASS":
        errors.append("Current execution does not prove live-data isolation.")
    if report.get("tracked_outputs_unchanged") != "PASS":
        errors.append("Current execution does not prove tracked isolation.")
    safe_results = {
        "NO_CAPTURE_FILES",
        "NO_NEW_CHANGES",
        "PROGRESS_UPDATED",
        "READY_FOR_MANUAL_REVIEW",
        "BATCH_APPROVED",
    }
    if report.get("result") not in safe_results:
        errors.append("Current capture execution result is not safe.")
    progress = report.get("overall_progress_percent")
    if not isinstance(progress, (int, float)) or not 0 <= progress <= 100:
        errors.append("Current capture execution progress is invalid.")
    return errors


def validate_semantic_names() -> list[str]:
    forbidden = list((PROJECT_ROOT / "reports").rglob("step_009_7*"))
    return [
        "Capture execution permanent report uses a technical step filename: "
        f"{path.relative_to(PROJECT_ROOT)}."
        for path in forbidden
    ]


def build_verification_report() -> dict[str, object]:
    checks = {
        "structure": validate_structure(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "runtime_isolation": validate_runtime_isolation(),
        "execution_safeguards": validate_safeguards(),
        "journal_schema": validate_journal_schema(),
        "current_execution_state": validate_current_status(),
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
    print("Capture execution verification")
    for name, status in report["checks"].items():
        print(f"- {name}: {status}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
