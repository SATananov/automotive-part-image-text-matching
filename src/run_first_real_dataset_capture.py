from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.activate_first_batch_review_queue import activate_review_queue
from src.execute_first_batch_capture_session import execute_capture_cycle
from src.prepare_first_batch_manual_decisions import prepare_manual_decisions
from src.real_dataset_config import (
    FIRST_BATCH_EXPECTED_IMAGES,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_PROCESSED_IMAGES_DIRECTORY,
)
from src.validate_first_batch_manual_decisions import (
    atomic_write_text,
    file_fingerprint,
    validate_manual_decisions,
)
from src.finalize_first_real_dataset_ingestion import build_ingestion_audit

STEP010_RUNTIME_DIRECTORY = (
    PROJECT_ROOT / "data" / "real" / "runtime" / "step_010"
)
CAPTURE_STATUS_PATH = STEP010_RUNTIME_DIRECTORY / "capture_status.json"
CAPTURE_SUMMARY_PATH = STEP010_RUNTIME_DIRECTORY / "capture_summary.md"

PROTECTED_APPROVED_FILES = (
    REAL_PART_GROUPS_PATH,
    REAL_IMAGES_PATH,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
)


class FirstRealDatasetCaptureError(RuntimeError):
    pass


def approved_dataset_fingerprint() -> str:
    parts = [
        f"{path.as_posix()}:{file_fingerprint(path)}"
        for path in PROTECTED_APPROVED_FILES
    ]
    if REAL_PROCESSED_IMAGES_DIRECTORY.is_dir():
        for path in sorted(
            item
            for item in REAL_PROCESSED_IMAGES_DIRECTORY.rglob("*")
            if item.is_file()
        ):
            relative = path.relative_to(
                REAL_PROCESSED_IMAGES_DIRECTORY
            ).as_posix()
            parts.append(f"image:{relative}:{file_fingerprint(path)}")
    return "|".join(parts)


def determine_readiness(
    capture_report: dict[str, Any],
    manual_report: dict[str, Any],
    validation_report: dict[str, Any],
    audit_report: dict[str, Any],
) -> str:
    if audit_report.get("readiness") == "FIRST_BATCH_INGESTED":
        return "FIRST_BATCH_INGESTED"

    counts = capture_report.get("counts", {})
    planned = int(
        counts.get("planned", counts.get("planned_images", 0))
        or FIRST_BATCH_EXPECTED_IMAGES
    )
    captured = int(counts.get("captured", 0))
    review_ready = int(counts.get("review_ready", 0))

    if captured == 0:
        return "AWAITING_CAPTURE"
    if captured < planned or review_ready < planned:
        return "CAPTURE_IN_PROGRESS"
    if validation_report.get("readiness") == "READY_TO_APPLY":
        return "READY_TO_APPLY"
    if manual_report.get("counts", {}).get("pending_decisions", 0):
        return "MANUAL_DECISIONS_REQUIRED"
    return "MANUAL_REVIEW_REQUIRED"


def render_summary(report: dict[str, Any]) -> str:
    capture_counts = report["capture"].get("counts", {})
    manual_counts = report["manual_decisions"].get("counts", {})
    lines = [
        "# Step 010 — First Real Dataset Capture",
        "",
        f"- Status: **{report['status']}**",
        f"- Readiness: **{report['readiness']}**",
        (
            "- Approved dataset unchanged: "
            f"**{report['approved_dataset_unchanged']}**"
        ),
        (
            "- Captured: "
            f"**{capture_counts.get('captured', 0)} / "
            f"{capture_counts.get('planned', FIRST_BATCH_EXPECTED_IMAGES)}**"
        ),
        (
            "- Review-ready: "
            f"**{capture_counts.get('review_ready', 0)}**"
        ),
        (
            "- Ready decisions: "
            f"**{manual_counts.get('ready_decisions', 0)}**"
        ),
        (
            "- Pending decisions: "
            f"**{manual_counts.get('pending_decisions', 0)}**"
        ),
        "",
        "This command never creates approval or rejection decisions.",
    ]
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in report["errors"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
    return "\n".join(lines) + "\n"


def to_json_compatible(value: Any) -> Any:
    """Convert runtime report values to strict JSON-compatible data."""
    if isinstance(value, pd.DataFrame):
        return [
            to_json_compatible(record)
            for record in value.to_dict(orient="records")
        ]
    if isinstance(value, pd.Series):
        return [
            to_json_compatible(item)
            for item in value.tolist()
        ]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {
            str(key): to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, set):
        return [
            to_json_compatible(item)
            for item in sorted(value, key=str)
        ]
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            item = item_method()
        except (TypeError, ValueError):
            item = value
        if item is not value:
            return to_json_compatible(item)

    isoformat_method = getattr(value, "isoformat", None)
    if callable(isoformat_method):
        try:
            return isoformat_method()
        except (TypeError, ValueError):
            pass

    return str(value)


def write_outputs(report: dict[str, Any]) -> None:
    serializable_report = to_json_compatible(report)
    atomic_write_text(
        CAPTURE_STATUS_PATH,
        json.dumps(
            serializable_report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )
    atomic_write_text(CAPTURE_SUMMARY_PATH, render_summary(report))


def run_capture_workflow(
    *,
    capture_callable: Callable[[], dict[str, Any]] = execute_capture_cycle,
    activation_callable: Callable[[], dict[str, Any]] = activate_review_queue,
    preparation_callable: Callable[[], dict[str, Any]] = prepare_manual_decisions,
    validation_callable: Callable[..., dict[str, Any]] = validate_manual_decisions,
    audit_callable: Callable[[], dict[str, Any]] = build_ingestion_audit,
) -> dict[str, Any]:
    before = approved_dataset_fingerprint()
    errors: list[str] = []
    warnings: list[str] = []

    try:
        capture_report = capture_callable()
        activation_report = activation_callable()
        manual_report = preparation_callable()
        validation_report = validation_callable(write_outputs=True)
        audit_report = audit_callable()
    except Exception as error:
        raise FirstRealDatasetCaptureError(
            f"Step 010 capture workflow failed: {error}"
        ) from error

    for child in (
        capture_report,
        activation_report,
        manual_report,
        validation_report,
        audit_report,
    ):
        if child.get("status") != "PASS":
            errors.extend(child.get("errors", []))
        warnings.extend(child.get("warnings", []))

    after = approved_dataset_fingerprint()
    unchanged = "PASS" if before == after else "FAIL"
    if unchanged != "PASS":
        errors.append(
            "The capture workflow changed the approved dataset before "
            "controlled ingestion."
        )

    readiness = determine_readiness(
        capture_report,
        manual_report,
        validation_report,
        audit_report,
    )
    if errors:
        readiness = "CAPTURE_WORKFLOW_BLOCKED"

    report = {
        "status": "PASS" if not errors else "FAIL",
        "readiness": readiness,
        "approved_dataset_unchanged": unchanged,
        "capture": capture_report,
        "review_queue_activation": activation_report,
        "manual_decisions": manual_report,
        "decision_validation": validation_report,
        "ingestion_audit": audit_report,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }
    write_outputs(report)
    return report


def main() -> None:
    try:
        report = run_capture_workflow()
    except FirstRealDatasetCaptureError as error:
        print("Step 010 first real dataset capture")
        print("- Status: FAIL")
        print(f"- Error: {error}")
        raise SystemExit(1) from error

    counts = report["capture"].get("counts", {})
    print("Step 010 first real dataset capture")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(
        "- Captured: "
        f"{counts.get('captured', 0)} / "
        f"{counts.get('planned', FIRST_BATCH_EXPECTED_IMAGES)}"
    )
    print(
        "- Approved dataset unchanged: "
        f"{report['approved_dataset_unchanged']}"
    )
    print(
        "- Runtime report: "
        f"{CAPTURE_SUMMARY_PATH.relative_to(PROJECT_ROOT)}"
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
