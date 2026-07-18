from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.real_dataset_config import (
    FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PLAN_PATH,
    FIRST_BATCH_REVIEW_RUNTIME_DIRECTORY,
    PROJECT_ROOT,
    REAL_SAMPLE_INTAKE_PATH,
)
from src.review_real_sample_intake import (
    build_review_report,
    load_review_inputs,
    read_csv_exact,
)

APPLICATION_PLAN_COLUMNS = (
    "sequence",
    "intake_id",
    "part_group_id",
    "part_category",
    "view",
    "staging_path",
    "image_id",
    "quality_status",
    "operator_decision",
    "rejection_reason",
    "operator_notes",
    "validation_status",
)

APPLICATION_PLAN_PATH = (
    FIRST_BATCH_REVIEW_RUNTIME_DIRECTORY
    / "manual_decision_application_plan.csv"
)
APPLICATION_VALIDATION_STATUS_PATH = (
    FIRST_BATCH_REVIEW_RUNTIME_DIRECTORY
    / "manual_decision_application_validation.json"
)
APPLICATION_VALIDATION_SUMMARY_PATH = (
    FIRST_BATCH_REVIEW_RUNTIME_DIRECTORY
    / "manual_decision_application_validation.md"
)

ALLOWED_OPERATOR_DECISIONS = {"", "approved", "rejected"}
IMMUTABLE_WORKBOOK_COLUMNS = (
    "part_group_id",
    "part_category",
    "view",
    "staging_path",
)


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def atomic_write_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(
        temporary,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def file_fingerprint(path: Path) -> str:
    if not path.is_file():
        return "MISSING"

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_plan_id(
    *,
    queue_fingerprint: str,
    workbook_fingerprint: str,
    first_batch_plan_fingerprint: str,
    application_rows: list[dict[str, str]],
) -> str:
    payload = {
        "queue_fingerprint": queue_fingerprint,
        "workbook_fingerprint": workbook_fingerprint,
        "first_batch_plan_fingerprint": first_batch_plan_fingerprint,
        "decisions": [
            {
                "intake_id": row["intake_id"],
                "operator_decision": row["operator_decision"],
                "rejection_reason": row["rejection_reason"],
                "operator_notes": row["operator_notes"],
            }
            for row in sorted(
                application_rows,
                key=lambda item: item["intake_id"],
            )
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized(value: object) -> str:
    return str(value).strip()


def build_manual_decision_application_plan(
) -> tuple[pd.DataFrame, dict[str, Any]]:
    (
        intake,
        part_groups,
        images,
        approval_log,
        read_errors,
    ) = load_review_inputs()

    first_batch_plan, plan_errors = read_csv_exact(
        FIRST_BATCH_PLAN_PATH,
        FIRST_BATCH_PLAN_COLUMNS,
        "first_batch_plan.csv",
    )

    errors = [*read_errors, *plan_errors]
    warnings: list[str] = []

    review_report = build_review_report(
        intake,
        part_groups,
        images,
        approval_log,
        initial_errors=read_errors,
    )
    errors.extend(review_report.get("errors", []))
    warnings.extend(review_report.get("warnings", []))

    plan_ids = [
        normalized(value)
        for value in first_batch_plan.get(
            "intake_id",
            pd.Series(dtype=str),
        ).tolist()
        if normalized(value)
    ]
    plan_id_set = set(plan_ids)
    plan_order = {
        intake_id: index + 1
        for index, intake_id in enumerate(plan_ids)
    }
    plan_by_id = {
        normalized(row["intake_id"]): row
        for row in first_batch_plan.to_dict(orient="records")
        if normalized(row.get("intake_id", ""))
    }

    batch_queue = intake.loc[
        intake.get(
            "intake_id",
            pd.Series(dtype=str),
        ).isin(plan_id_set)
    ].copy()

    queue_fingerprint = file_fingerprint(REAL_SAMPLE_INTAKE_PATH)
    workbook_fingerprint = file_fingerprint(
        FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH
    )
    first_batch_plan_fingerprint = file_fingerprint(
        FIRST_BATCH_PLAN_PATH
    )

    counts = {
        "first_batch_queue_rows": int(len(batch_queue)),
        "ready_decisions": 0,
        "pending_decisions": 0,
        "approved_decisions": 0,
        "rejected_decisions": 0,
        "invalid_decisions": 0,
    }

    if errors:
        report = {
            "status": "FAIL",
            "readiness": "MANUAL_DECISION_VALIDATION_BLOCKED",
            "plan_id": "",
            "counts": counts,
            "queue_fingerprint": queue_fingerprint,
            "workbook_fingerprint": workbook_fingerprint,
            "first_batch_plan_fingerprint": (
                first_batch_plan_fingerprint
            ),
            "application_plan": relative_path(APPLICATION_PLAN_PATH),
            "errors": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
        }
        return (
            pd.DataFrame(columns=APPLICATION_PLAN_COLUMNS),
            report,
        )

    if batch_queue.empty:
        report = {
            "status": "PASS",
            "readiness": "AWAITING_QUEUE_ACTIVATION",
            "plan_id": "",
            "counts": counts,
            "queue_fingerprint": queue_fingerprint,
            "workbook_fingerprint": workbook_fingerprint,
            "first_batch_plan_fingerprint": (
                first_batch_plan_fingerprint
            ),
            "application_plan": relative_path(APPLICATION_PLAN_PATH),
            "errors": [],
            "warnings": sorted(set(warnings)),
        }
        return (
            pd.DataFrame(columns=APPLICATION_PLAN_COLUMNS),
            report,
        )

    workbook, workbook_errors = read_csv_exact(
        FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH,
        FIRST_BATCH_MANUAL_DECISION_COLUMNS,
        "manual_decision_workbook.csv",
    )
    errors.extend(workbook_errors)

    if not intake.empty:
        non_pending_rows = intake.loc[
            intake["decision"].str.strip().str.lower() != "pending"
        ]
        if not non_pending_rows.empty:
            errors.append(
                "All live queue rows must still be pending before "
                "controlled manual-decision application."
            )

    queue_duplicate_ids = sorted(
        intake_id
        for intake_id, count in batch_queue[
            "intake_id"
        ].value_counts().items()
        if normalized(intake_id) and count > 1
    )
    if queue_duplicate_ids:
        errors.append(
            "The first-batch live queue contains duplicate intake "
            f"IDs: {queue_duplicate_ids}."
        )

    workbook_duplicate_ids = sorted(
        intake_id
        for intake_id, count in workbook.get(
            "intake_id",
            pd.Series(dtype=str),
        ).value_counts().items()
        if normalized(intake_id) and count > 1
    )
    if workbook_duplicate_ids:
        errors.append(
            "The manual decision workbook contains duplicate intake "
            f"IDs: {workbook_duplicate_ids}."
        )

    queue_ids = {
        normalized(value)
        for value in batch_queue.get(
            "intake_id",
            pd.Series(dtype=str),
        ).tolist()
        if normalized(value)
    }
    workbook_ids = {
        normalized(value)
        for value in workbook.get(
            "intake_id",
            pd.Series(dtype=str),
        ).tolist()
        if normalized(value)
    }

    missing_workbook_ids = sorted(queue_ids - workbook_ids)
    unexpected_workbook_ids = sorted(workbook_ids - queue_ids)
    if missing_workbook_ids:
        errors.append(
            "The manual decision workbook is missing live queue "
            f"intake IDs: {missing_workbook_ids}."
        )
    if unexpected_workbook_ids:
        errors.append(
            "The manual decision workbook contains stale or "
            f"unexpected intake IDs: {unexpected_workbook_ids}."
        )

    queue_by_id = {
        normalized(row["intake_id"]): row
        for row in batch_queue.to_dict(orient="records")
        if normalized(row.get("intake_id", ""))
    }
    workbook_by_id = {
        normalized(row["intake_id"]): row
        for row in workbook.to_dict(orient="records")
        if normalized(row.get("intake_id", ""))
    }
    review_items = {
        normalized(item.get("intake_id", "")): item
        for item in review_report.get("items", [])
        if normalized(item.get("intake_id", ""))
    }

    application_rows: list[dict[str, str]] = []

    for intake_id in sorted(
        queue_ids,
        key=lambda value: plan_order.get(value, 10**9),
    ):
        queue_row = queue_by_id[intake_id]
        workbook_row = workbook_by_id.get(intake_id)
        plan_row = plan_by_id.get(intake_id)
        review_item = review_items.get(intake_id)

        row_errors: list[str] = []
        validation_status = "READY"

        if workbook_row is None:
            continue
        if plan_row is None:
            row_errors.append(
                "Intake ID is absent from first_batch_plan.csv."
            )

        for column in IMMUTABLE_WORKBOOK_COLUMNS:
            queue_value = normalized(queue_row.get(column, ""))
            workbook_value = normalized(workbook_row.get(column, ""))
            plan_value = normalized(
                plan_row.get(column, "") if plan_row else ""
            )

            if workbook_value != queue_value:
                row_errors.append(
                    f"Workbook column '{column}' differs from the "
                    "live queue."
                )
            if plan_row is not None and queue_value != plan_value:
                row_errors.append(
                    f"Live queue column '{column}' differs from the "
                    "canonical first-batch plan."
                )

        queue_decision = normalized(
            queue_row.get("decision", "")
        ).lower()
        workbook_queue_decision = normalized(
            workbook_row.get("current_queue_decision", "")
        ).lower()
        if queue_decision != "pending":
            row_errors.append("Live queue decision is not pending.")
        if workbook_queue_decision != queue_decision:
            row_errors.append(
                "Workbook current_queue_decision is stale."
            )

        operator_decision = normalized(
            workbook_row.get("operator_decision", "")
        ).lower()
        rejection_reason = normalized(
            workbook_row.get("rejection_reason", "")
        )
        operator_notes = normalized(
            workbook_row.get("operator_notes", "")
        )
        entry_status = normalized(
            workbook_row.get("decision_entry_status", "")
        ).upper()

        if operator_decision not in ALLOWED_OPERATOR_DECISIONS:
            row_errors.append(
                "Operator decision must be blank, approved, or rejected."
            )
        elif not operator_decision:
            validation_status = "PENDING"
            counts["pending_decisions"] += 1
        else:
            if entry_status != "READY":
                row_errors.append(
                    "Run manual-decision preparation again so "
                    "decision_entry_status becomes READY."
                )
            if operator_decision == "rejected" and not rejection_reason:
                row_errors.append(
                    "Rejected decisions require a rejection reason."
                )
            if operator_decision == "approved" and rejection_reason:
                row_errors.append(
                    "Approved decisions must not contain a rejection reason."
                )

        if review_item is None:
            row_errors.append(
                "The current live review report has no matching item."
            )
            quality_status = "NOT_REVIEWED"
            image_id = normalized(workbook_row.get("image_id", ""))
        else:
            quality_status = normalized(
                review_item.get("status", "")
            ).upper()
            image_id = normalized(review_item.get("image_id", ""))

            if review_item.get("errors"):
                row_errors.append(
                    "The current live review item contains errors."
                )
            if quality_status not in {"PASS", "WARN"}:
                row_errors.append(
                    "The current image quality status is not approvable."
                )
            if normalized(
                workbook_row.get("quality_status", "")
            ).upper() != quality_status:
                row_errors.append(
                    "Workbook quality_status differs from the current "
                    "live review."
                )
            if normalized(
                workbook_row.get("image_id", "")
            ) != image_id:
                row_errors.append(
                    "Workbook image_id differs from the current "
                    "derived image ID."
                )

        if row_errors:
            validation_status = "INVALID"
            counts["invalid_decisions"] += 1
            errors.extend(
                f"{intake_id}: {message}"
                for message in row_errors
            )
        elif operator_decision:
            counts["ready_decisions"] += 1
            if operator_decision == "approved":
                counts["approved_decisions"] += 1
            else:
                counts["rejected_decisions"] += 1

        application_rows.append(
            {
                "sequence": str(plan_order.get(intake_id, "")),
                "intake_id": intake_id,
                "part_group_id": normalized(
                    queue_row.get("part_group_id", "")
                ),
                "part_category": normalized(
                    queue_row.get("part_category", "")
                ),
                "view": normalized(queue_row.get("view", "")),
                "staging_path": normalized(
                    queue_row.get("staging_path", "")
                ),
                "image_id": image_id,
                "quality_status": quality_status,
                "operator_decision": operator_decision,
                "rejection_reason": rejection_reason,
                "operator_notes": operator_notes,
                "validation_status": validation_status,
            }
        )

    plan_dataframe = pd.DataFrame(
        application_rows,
        columns=APPLICATION_PLAN_COLUMNS,
    )

    if errors:
        status = "FAIL"
        readiness = "MANUAL_DECISION_VALIDATION_BLOCKED"
        plan_id = ""
    elif counts["pending_decisions"]:
        status = "PASS"
        readiness = "MANUAL_DECISIONS_REQUIRED"
        plan_id = ""
    elif counts["ready_decisions"] != counts["first_batch_queue_rows"]:
        status = "FAIL"
        readiness = "MANUAL_DECISION_VALIDATION_BLOCKED"
        errors.append(
            "The number of ready decisions does not match the "
            "first-batch live queue."
        )
        plan_id = ""
    else:
        status = "PASS"
        readiness = "READY_TO_APPLY"
        plan_id = canonical_plan_id(
            queue_fingerprint=queue_fingerprint,
            workbook_fingerprint=workbook_fingerprint,
            first_batch_plan_fingerprint=(
                first_batch_plan_fingerprint
            ),
            application_rows=application_rows,
        )

    report = {
        "status": status,
        "readiness": readiness,
        "plan_id": plan_id,
        "counts": counts,
        "queue_fingerprint": queue_fingerprint,
        "workbook_fingerprint": workbook_fingerprint,
        "first_batch_plan_fingerprint": (
            first_batch_plan_fingerprint
        ),
        "application_plan": relative_path(APPLICATION_PLAN_PATH),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }
    return plan_dataframe, report


def render_validation_summary(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# First Batch Manual Decision Application Validation",
        "",
        f"- Status: **{report['status']}**",
        f"- Readiness: **{report['readiness']}**",
        f"- Plan ID: `{report['plan_id'] or 'NOT_CREATED'}`",
        f"- Queue rows: **{counts['first_batch_queue_rows']}**",
        f"- Ready decisions: **{counts['ready_decisions']}**",
        f"- Pending decisions: **{counts['pending_decisions']}**",
        f"- Approved decisions: **{counts['approved_decisions']}**",
        f"- Rejected decisions: **{counts['rejected_decisions']}**",
        f"- Invalid decisions: **{counts['invalid_decisions']}**",
        f"- Queue fingerprint: `{report['queue_fingerprint']}`",
        f"- Workbook fingerprint: `{report['workbook_fingerprint']}`",
        (
            "- First-batch plan fingerprint: "
            f"`{report['first_batch_plan_fingerprint']}`"
        ),
        f"- Application plan: `{report['application_plan']}`",
    ]

    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])

    return "\n".join(lines) + "\n"


def validate_manual_decisions(
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    plan, report = build_manual_decision_application_plan()

    if write_outputs:
        atomic_write_dataframe(plan, APPLICATION_PLAN_PATH)
        atomic_write_text(
            APPLICATION_VALIDATION_STATUS_PATH,
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(
            APPLICATION_VALIDATION_SUMMARY_PATH,
            render_validation_summary(report),
        )

    return report


def main() -> None:
    report = validate_manual_decisions(write_outputs=True)
    counts = report["counts"]

    print("First real batch manual decision validation")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Plan ID: {report['plan_id'] or 'NOT_CREATED'}")
    print(f"- Queue rows: {counts['first_batch_queue_rows']}")
    print(f"- Ready decisions: {counts['ready_decisions']}")
    print(f"- Pending decisions: {counts['pending_decisions']}")
    print(f"- Invalid decisions: {counts['invalid_decisions']}")
    print(f"- Plan: {report['application_plan']}")

    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
