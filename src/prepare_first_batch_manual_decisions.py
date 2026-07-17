from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.real_dataset_config import (
    FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    FIRST_BATCH_MANUAL_DECISION_GUIDE_PATH,
    FIRST_BATCH_MANUAL_DECISION_STATUS_PATH,
    FIRST_BATCH_MANUAL_DECISION_SUMMARY_PATH,
    FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PLAN_PATH,
    PROJECT_ROOT,
    REAL_SAMPLE_INTAKE_PATH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.review_real_sample_intake import (
    build_review_report,
    load_review_inputs,
    read_csv_exact,
)
from src.validate_real_dataset import sha256_file


OPERATOR_DECISION_VALUES = {"", "approved", "rejected"}
PRESERVED_OPERATOR_COLUMNS = (
    "operator_decision",
    "rejection_reason",
    "operator_notes",
)


class ManualDecisionPreparationError(RuntimeError):
    pass


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def atomic_write_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary, path)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def file_fingerprint(path: Path) -> str:
    return sha256_file(path) if path.is_file() else "MISSING"


def load_existing_operator_entries() -> tuple[
    dict[str, dict[str, str]],
    list[str],
]:
    if not FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH.is_file():
        return {}, []
    try:
        dataframe = pd.read_csv(
            FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        return {}, [f"Cannot read the existing manual decision workbook: {error}."]

    required = {"intake_id", *PRESERVED_OPERATOR_COLUMNS}
    if not required.issubset(dataframe.columns):
        missing = sorted(required - set(dataframe.columns))
        return {}, [
            "Existing manual decision workbook is missing operator columns: "
            f"{missing}."
        ]

    duplicate_ids = sorted(
        value
        for value, count in dataframe["intake_id"].value_counts().items()
        if value and count > 1
    )
    if duplicate_ids:
        return {}, [
            "Existing manual decision workbook contains duplicate intake IDs: "
            f"{duplicate_ids}."
        ]

    return (
        {
            str(row["intake_id"]).strip(): {
                column: str(row[column]).strip()
                for column in PRESERVED_OPERATOR_COLUMNS
            }
            for row in dataframe.to_dict(orient="records")
            if str(row["intake_id"]).strip()
        },
        [],
    )


def validate_operator_entry(entry: dict[str, str]) -> tuple[str, str]:
    decision = entry["operator_decision"].strip().lower()
    rejection_reason = entry["rejection_reason"].strip()
    if decision not in OPERATOR_DECISION_VALUES:
        return "INVALID", "Use blank, approved, or rejected."
    if decision == "rejected" and not rejection_reason:
        return "INCOMPLETE", "Add a rejection reason."
    if decision in {"approved", "rejected"}:
        return "READY", "Decision is ready for the recording step."
    return "PENDING", "Inspect the staged image and enter a decision."


def build_manual_decision_workbook() -> tuple[pd.DataFrame, dict[str, Any]]:
    intake, part_groups, images, approval_log, read_errors = load_review_inputs()
    plan, plan_errors = read_csv_exact(
        FIRST_BATCH_PLAN_PATH,
        FIRST_BATCH_PLAN_COLUMNS,
        "first_batch_plan.csv",
    )
    errors = [*read_errors, *plan_errors]
    review_report = build_review_report(
        intake,
        part_groups,
        images,
        approval_log,
        initial_errors=errors,
    )
    errors = list(review_report["errors"])

    plan_order = {
        str(row["intake_id"]).strip(): index + 1
        for index, row in enumerate(plan.to_dict(orient="records"))
    }
    first_batch_ids = set(plan_order)
    queue_rows = intake.loc[intake["intake_id"].isin(first_batch_ids)].copy()
    queue_rows = queue_rows.sort_values(
        by="intake_id",
        key=lambda series: series.map(plan_order),
    )
    review_items = {
        item["intake_id"]: item for item in review_report["items"]
    }
    preserved, workbook_errors = load_existing_operator_entries()
    errors.extend(workbook_errors)
    workbook_rows: list[dict[str, str]] = []
    ready_decisions = 0
    invalid_decisions = 0

    for row in queue_rows.to_dict(orient="records"):
        intake_id = str(row["intake_id"]).strip()
        item = review_items.get(intake_id, {})
        metrics = item.get("metrics", {})
        operator_entry = preserved.get(
            intake_id,
            {
                "operator_decision": "",
                "rejection_reason": "",
                "operator_notes": "",
            },
        )
        entry_status, next_action = validate_operator_entry(operator_entry)
        if entry_status == "READY":
            ready_decisions += 1
        elif entry_status in {"INVALID", "INCOMPLETE"}:
            invalid_decisions += 1

        workbook_rows.append(
            {
                "sequence": str(plan_order[intake_id]),
                "intake_id": intake_id,
                "part_group_id": str(row["part_group_id"]).strip(),
                "part_category": str(row["part_category"]).strip(),
                "view": str(row["view"]).strip(),
                "staging_path": str(row["staging_path"]).strip(),
                "image_id": str(item.get("image_id", "")),
                "quality_status": str(item.get("status", "NOT_REVIEWED")),
                "width": str(metrics.get("width", "")),
                "height": str(metrics.get("height", "")),
                "format": str(metrics.get("format", "")),
                "review_errors": " | ".join(item.get("errors", [])),
                "review_warnings": " | ".join(item.get("warnings", [])),
                "current_queue_decision": str(row["decision"]).strip(),
                "operator_decision": operator_entry["operator_decision"],
                "rejection_reason": operator_entry["rejection_reason"],
                "operator_notes": operator_entry["operator_notes"],
                "decision_entry_status": entry_status,
                "next_action": next_action,
            }
        )

    workbook = pd.DataFrame(
        workbook_rows,
        columns=FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    )
    queue_rows_count = int(len(workbook))
    pending_decisions = queue_rows_count - ready_decisions

    if errors:
        readiness = "MANUAL_DECISION_PREPARATION_BLOCKED"
    elif queue_rows_count == 0:
        readiness = "AWAITING_QUEUE_ACTIVATION"
    elif invalid_decisions:
        readiness = "MANUAL_DECISION_INPUT_INVALID"
    elif ready_decisions == queue_rows_count:
        readiness = "READY_TO_RECORD_DECISIONS"
    elif ready_decisions:
        readiness = "MANUAL_DECISIONS_IN_PROGRESS"
    else:
        readiness = "MANUAL_DECISION_WORKBOOK_READY"

    report = {
        "status": "PASS" if not errors and not invalid_decisions else "FAIL",
        "readiness": readiness,
        "counts": {
            "first_batch_queue_rows": queue_rows_count,
            "ready_decisions": ready_decisions,
            "pending_decisions": pending_decisions,
            "invalid_decisions": invalid_decisions,
        },
        "live_queue_unchanged": "UNKNOWN",
        "workbook": relative_path(FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH),
        "errors": sorted(set(errors)),
        "warnings": review_report.get("warnings", []),
    }
    return workbook, report


def render_manual_decision_guide() -> str:
    return "\n".join(
        [
            "# First Batch Manual Decision Guide",
            "",
            "Edit only the operator columns in the runtime workbook:",
            "",
            "- `operator_decision`: blank, `approved`, or `rejected`;",
            "- `rejection_reason`: required when rejected;",
            "- `operator_notes`: optional visual-review notes.",
            "",
            "Do not edit the live `sample_intake.csv` queue manually.",
            "This preparation step never records decisions and never applies images.",
            "Run the preparation command again to validate the workbook entries.",
        ]
    ) + "\n"


def render_manual_decision_summary(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# First Batch Manual Decision Preparation",
        "",
        f"- Status: **{report['status']}**",
        f"- Readiness: **{report['readiness']}**",
        f"- Queue rows: **{counts['first_batch_queue_rows']}**",
        f"- Ready decisions: **{counts['ready_decisions']}**",
        f"- Pending decisions: **{counts['pending_decisions']}**",
        f"- Invalid decisions: **{counts['invalid_decisions']}**",
        f"- Live queue unchanged: **{report['live_queue_unchanged']}**",
        f"- Workbook: `{report['workbook']}`",
    ]
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def prepare_manual_decisions() -> dict[str, Any]:
    queue_before = file_fingerprint(REAL_SAMPLE_INTAKE_PATH)
    workbook, report = build_manual_decision_workbook()
    if not report["errors"]:
        atomic_write_dataframe(
            workbook,
            FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH,
        )
    atomic_write_text(
        FIRST_BATCH_MANUAL_DECISION_GUIDE_PATH,
        render_manual_decision_guide(),
    )
    report["live_queue_unchanged"] = (
        "PASS"
        if queue_before == file_fingerprint(REAL_SAMPLE_INTAKE_PATH)
        else "FAIL"
    )
    if report["live_queue_unchanged"] != "PASS":
        report["status"] = "FAIL"
        report["errors"].append(
            "Manual decision preparation changed the live queue."
        )
    atomic_write_text(
        FIRST_BATCH_MANUAL_DECISION_STATUS_PATH,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        FIRST_BATCH_MANUAL_DECISION_SUMMARY_PATH,
        render_manual_decision_summary(report),
    )
    return report


def main() -> None:
    report = prepare_manual_decisions()
    counts = report["counts"]
    print("First real batch manual decision preparation")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Queue rows: {counts['first_batch_queue_rows']}")
    print(f"- Ready decisions: {counts['ready_decisions']}")
    print(f"- Pending decisions: {counts['pending_decisions']}")
    print(f"- Invalid decisions: {counts['invalid_decisions']}")
    print(f"- Live queue unchanged: {report['live_queue_unchanged']}")
    print(f"- Workbook: {report['workbook']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
