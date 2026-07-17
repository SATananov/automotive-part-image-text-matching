from __future__ import annotations

from pathlib import Path

from src.activate_first_batch_review_queue import (
    plan_review_queue_activation,
)
from src.prepare_first_batch_manual_decisions import (
    prepare_manual_decisions,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH,
    FIRST_BATCH_REVIEW_ACTIVATION_STATUS_PATH,
    FIRST_BATCH_REVIEW_RUNTIME_DIRECTORY,
    PROJECT_ROOT,
    REAL_SAMPLE_INTAKE_PATH,
)
from src.validate_real_dataset import sha256_file


README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT / "reports" / "real_dataset" / "collection_protocol.md"
)
WORKFLOW_GUIDE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_review_queue_activation_and_manual_decision_preparation.md"
)
ACTIVATION_MODULE_PATH = (
    PROJECT_ROOT / "src" / "activate_first_batch_review_queue.py"
)
DECISION_MODULE_PATH = (
    PROJECT_ROOT / "src" / "prepare_first_batch_manual_decisions.py"
)
TEST_PATH = (
    PROJECT_ROOT / "tests" / "test_first_batch_review_queue_activation.py"
)
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"


def fingerprint(path: Path) -> str:
    return sha256_file(path) if path.is_file() else "MISSING"


def check_structure() -> tuple[bool, list[str]]:
    required = (
        README_PATH,
        PROTOCOL_PATH,
        WORKFLOW_GUIDE_PATH,
        ACTIVATION_MODULE_PATH,
        DECISION_MODULE_PATH,
        TEST_PATH,
    )
    errors = [
        f"Missing Step 009.8 file: {path.relative_to(PROJECT_ROOT)}."
        for path in required
        if not path.is_file()
    ]
    return not errors, errors


def check_cli_and_documentation() -> tuple[bool, list[str]]:
    errors: list[str] = []
    expected_commands = {
        "activate-first-real-batch-review-queue": (
            "src.activate_first_batch_review_queue"
        ),
        "prepare-first-real-batch-manual-decisions": (
            "src.prepare_first_batch_manual_decisions"
        ),
        "verify-step-009-8": "src.verify_step_009_8",
    }
    for command, module in expected_commands.items():
        spec = COMMANDS.get(command)
        if spec is None or spec.module != module:
            errors.append(f"Missing or incorrect CLI command: {command}.")

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (README_PATH, PROTOCOL_PATH, WORKFLOW_GUIDE_PATH)
        if path.is_file()
    )
    for token in (
        "activate-first-real-batch-review-queue",
        "prepare-first-real-batch-manual-decisions",
        "Automatic approval",
        "sample_intake.csv",
        "manual_decision_workbook.csv",
    ):
        if token.lower() not in combined.lower():
            errors.append(f"Documentation is missing: {token}.")
    return not errors, errors


def check_runtime_and_schema() -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        FIRST_BATCH_REVIEW_RUNTIME_DIRECTORY.relative_to(
            PROJECT_ROOT / "data" / "real" / "runtime"
        )
    except ValueError:
        errors.append("Review runtime directory is outside data/real/runtime.")

    if len(FIRST_BATCH_MANUAL_DECISION_COLUMNS) != len(
        set(FIRST_BATCH_MANUAL_DECISION_COLUMNS)
    ):
        errors.append("Manual decision workbook columns are not unique.")

    required_columns = {
        "intake_id",
        "operator_decision",
        "rejection_reason",
        "operator_notes",
        "decision_entry_status",
        "next_action",
    }
    if not required_columns.issubset(FIRST_BATCH_MANUAL_DECISION_COLUMNS):
        errors.append("Manual decision workbook schema is incomplete.")

    gitignore = (
        GITIGNORE_PATH.read_text(encoding="utf-8")
        if GITIGNORE_PATH.is_file()
        else ""
    )
    if "data/real/runtime/" not in gitignore:
        errors.append("Runtime review outputs are not ignored by Git.")
    return not errors, errors


def check_source_safeguards() -> tuple[bool, list[str]]:
    errors: list[str] = []
    activation_source = ACTIVATION_MODULE_PATH.read_text(encoding="utf-8")
    decision_source = DECISION_MODULE_PATH.read_text(encoding="utf-8")
    for token in (
        "protected_fingerprint",
        "queue_snapshot",
        "ALREADY_ACTIVE",
        "pending",
        "Post-write queue validation failed",
    ):
        if token not in activation_source:
            errors.append(f"Activation safeguard is missing: {token}.")
    for token in (
        "PRESERVED_OPERATOR_COLUMNS",
        "live_queue_unchanged",
        "approved",
        "rejected",
        "Add a rejection reason",
    ):
        if token not in decision_source:
            errors.append(f"Decision safeguard is missing: {token}.")
    return not errors, errors


def check_current_state() -> tuple[bool, list[str]]:
    errors: list[str] = []
    queue_before = fingerprint(REAL_SAMPLE_INTAKE_PATH)
    activation = plan_review_queue_activation()
    decisions = prepare_manual_decisions()
    queue_after = fingerprint(REAL_SAMPLE_INTAKE_PATH)

    if activation["status"] != "PASS":
        errors.extend(activation["errors"])
    if decisions["status"] != "PASS":
        errors.extend(decisions["errors"])
    if queue_before != queue_after:
        errors.append("Current-state verification changed sample_intake.csv.")
    if decisions["live_queue_unchanged"] != "PASS":
        errors.append("Manual decision preparation changed the live queue.")
    if not FIRST_BATCH_REVIEW_ACTIVATION_STATUS_PATH.parent.is_dir():
        errors.append("Review runtime directory was not created.")
    if not FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH.is_file():
        errors.append("Manual decision workbook was not generated.")
    return not errors, errors


def build_verification_report() -> dict[str, object]:
    checks = {
        "structure": check_structure(),
        "cli_and_documentation": check_cli_and_documentation(),
        "runtime_and_schema": check_runtime_and_schema(),
        "activation_and_decision_safeguards": check_source_safeguards(),
        "current_review_state": check_current_state(),
    }
    errors = [error for passed, items in checks.values() for error in items]
    return {
        "status": "PASS" if not errors else "FAIL",
        "checks": {
            name: "PASS" if passed else "FAIL"
            for name, (passed, _) in checks.items()
        },
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()
    print("Step 009.8 verification")
    for name, status in report["checks"].items():
        print(f"- {name}: {status}")
    print(f"Status: {report['status']}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
