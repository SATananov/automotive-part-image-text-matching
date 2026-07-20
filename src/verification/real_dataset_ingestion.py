from __future__ import annotations

import ast
from typing import Callable

from src.finalize_first_real_dataset_ingestion import (
    STEP010_RUNTIME_DIRECTORY,
    build_ingestion_audit,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import PROJECT_ROOT

REQUIRED_FILES = (
    PROJECT_ROOT / "src" / "run_first_real_dataset_capture.py",
    PROJECT_ROOT
    / "src"
    / "finalize_first_real_dataset_ingestion.py",
    PROJECT_ROOT / "src" / "verification" / "real_dataset_ingestion.py",
    PROJECT_ROOT
    / "tests"
    / "test_first_real_dataset_capture_and_ingestion.py",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "real_dataset_capture_and_ingestion.md",
)

EXPECTED_COMMANDS = {
    "run-first-real-dataset-capture": (
        "src.run_first_real_dataset_capture"
    ),
    "finalize-first-real-dataset-ingestion": (
        "src.finalize_first_real_dataset_ingestion"
    ),
    "verify-real-dataset-ingestion": "src.verification.real_dataset_ingestion",
}

SAFE_READINESS = {
    "MANUAL_DECISIONS_REQUIRED",
    "RECAPTURE_REQUIRED",
    "FIRST_BATCH_INGESTED",
    "INGESTION_AUDIT_BLOCKED",
}


def contains_marker(source: str, marker: str) -> bool:
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


def check_structure() -> list[str]:
    return [
        f"Missing file: {path.relative_to(PROJECT_ROOT)}"
        for path in REQUIRED_FILES
        if not path.is_file()
    ]


def check_cli_and_docs() -> list[str]:
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
        text = f"python -m src.project_cli {command}"
        if text not in readme:
            errors.append(f"README is missing: {text}.")
        if text not in protocol:
            errors.append(
                f"Collection protocol is missing: {text}."
            )
    return errors


def check_runtime_boundary() -> list[str]:
    errors: list[str] = []
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(
        encoding="utf-8-sig"
    )
    if "data/real/runtime/" not in gitignore:
        errors.append("data/real/runtime/ is not Git-ignored.")
    try:
        STEP010_RUNTIME_DIRECTORY.relative_to(
            PROJECT_ROOT / "data" / "real" / "runtime"
        )
    except ValueError:
        errors.append(
            "Real dataset capture and ingestion outputs are outside the runtime boundary."
        )
    return errors


def check_safeguards() -> list[str]:
    errors: list[str] = []
    capture_source = (
        PROJECT_ROOT
        / "src"
        / "run_first_real_dataset_capture.py"
    ).read_text(encoding="utf-8-sig")
    ingestion_source = (
        PROJECT_ROOT
        / "src"
        / "finalize_first_real_dataset_ingestion.py"
    ).read_text(encoding="utf-8-sig")

    for marker in (
        "approved_dataset_fingerprint",
        "This command never creates approval or rejection decisions.",
        "to_json_compatible",
        "allow_nan=False",
        "READY_TO_APPLY",
        "CAPTURE_WORKFLOW_BLOCKED",
    ):
        if not contains_marker(capture_source, marker):
            errors.append(
                f"Capture safeguard marker is missing: {marker}."
            )

    for marker in (
        "snapshot_live_state",
        "restore_live_state",
        "READY_TO_APPLY",
        "RECAPTURE_REQUIRED",
        "FIRST_BATCH_INGESTED",
        "post-application ingestion audit failed",
        "rollback_performed",
    ):
        if not contains_marker(ingestion_source, marker):
            errors.append(
                f"Ingestion safeguard marker is missing: {marker}."
            )
    return errors


def check_current_state() -> list[str]:
    try:
        report = build_ingestion_audit()
    except Exception as error:
        return [f"Current ingestion audit failed: {error}."]

    errors: list[str] = []
    readiness = report.get("readiness")
    if readiness not in SAFE_READINESS:
        errors.append(
            f"Unexpected current Real dataset capture and ingestion readiness: {readiness}."
        )
    if (
        readiness == "FIRST_BATCH_INGESTED"
        and report.get("status") != "PASS"
    ):
        errors.append(
            "FIRST_BATCH_INGESTED must have PASS status."
        )
    return errors


def run_check(
    name: str,
    callback: Callable[[], list[str]],
) -> tuple[str, list[str]]:
    try:
        return name, callback()
    except Exception as error:
        return name, [f"Unexpected verifier error: {error}."]


def main() -> None:
    checks = [
        run_check("structure", check_structure),
        run_check("cli_and_documentation", check_cli_and_docs),
        run_check("runtime_boundary", check_runtime_boundary),
        run_check("capture_and_ingestion_safeguards", check_safeguards),
        run_check("current_ingestion_state", check_current_state),
    ]

    failed = False
    print("Real dataset capture and ingestion verification")
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
