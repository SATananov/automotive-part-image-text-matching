from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PLAN_PATH,
    FIRST_BATCH_REVIEW_ACTIVATION_STATUS_PATH,
    FIRST_BATCH_REVIEW_ACTIVATION_SUMMARY_PATH,
    FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
    FIRST_BATCH_RUNTIME_DIRECTORY,
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_PROCESSED_IMAGES_DIRECTORY,
    REAL_SAMPLE_INTAKE_PATH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.review_real_sample_intake import (
    build_review_report,
    read_csv_exact,
)
from src.validate_real_dataset import sha256_file


RUNTIME_REVIEW_QUEUE_DRAFT_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "review_queue_draft.csv"
)

PROTECTED_LIVE_PATHS = (
    REAL_PART_GROUPS_PATH,
    REAL_IMAGES_PATH,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
)


class ReviewQueueActivationError(RuntimeError):
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


def directory_fingerprint(path: Path) -> tuple[tuple[str, str], ...]:
    if not path.exists():
        return ()
    return tuple(
        (
            item.relative_to(path).as_posix(),
            sha256_file(item),
        )
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


def protected_fingerprint() -> dict[str, object]:
    return {
        relative_path(path): file_fingerprint(path)
        for path in PROTECTED_LIVE_PATHS
    } | {
        relative_path(REAL_PROCESSED_IMAGES_DIRECTORY): directory_fingerprint(
            REAL_PROCESSED_IMAGES_DIRECTORY
        )
    }


def select_draft_path() -> Path:
    if RUNTIME_REVIEW_QUEUE_DRAFT_PATH.is_file():
        return RUNTIME_REVIEW_QUEUE_DRAFT_PATH
    return FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH


def normalize_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    for column in normalized.columns:
        normalized[column] = normalized[column].astype(str).str.strip()
    return normalized


def row_signature(row: pd.Series) -> tuple[str, ...]:
    return tuple(str(row.get(column, "")).strip() for column in SAMPLE_INTAKE_COLUMNS)


def load_activation_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Path,
    list[str],
]:
    draft_path = select_draft_path()
    draft, draft_errors = read_csv_exact(
        draft_path,
        SAMPLE_INTAKE_COLUMNS,
        "first-batch review queue draft",
    )
    live_queue, queue_errors = read_csv_exact(
        REAL_SAMPLE_INTAKE_PATH,
        SAMPLE_INTAKE_COLUMNS,
        "sample_intake.csv",
    )
    plan, plan_errors = read_csv_exact(
        FIRST_BATCH_PLAN_PATH,
        FIRST_BATCH_PLAN_COLUMNS,
        "first_batch_plan.csv",
    )
    part_groups, group_errors = read_csv_exact(
        REAL_PART_GROUPS_PATH,
        PART_GROUP_COLUMNS,
        "part_groups.csv",
    )
    images, image_errors = read_csv_exact(
        REAL_IMAGES_PATH,
        IMAGE_MANIFEST_COLUMNS,
        "images.csv",
    )
    approval_log, approval_errors = read_csv_exact(
        REAL_APPROVAL_LOG_PATH,
        APPROVAL_LOG_COLUMNS,
        "approval_log.csv",
    )
    return (
        normalize_frame(draft),
        normalize_frame(live_queue),
        normalize_frame(plan),
        normalize_frame(part_groups),
        normalize_frame(images),
        normalize_frame(approval_log),
        draft_path,
        [
            *draft_errors,
            *queue_errors,
            *plan_errors,
            *group_errors,
            *image_errors,
            *approval_errors,
        ],
    )


def validate_draft_against_plan(
    draft: pd.DataFrame,
    plan: pd.DataFrame,
    approval_log: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if draft.empty:
        return errors

    for column in ("intake_id", "staging_path"):
        duplicates = sorted(
            value
            for value, count in Counter(draft[column]).items()
            if value and count > 1
        )
        if duplicates:
            errors.append(
                f"Duplicate {column} values in first-batch draft: {duplicates}."
            )

    invalid_decisions = sorted(set(draft["decision"]) - {"pending"})
    if invalid_decisions:
        errors.append(
            "First-batch draft may contain only pending decisions; found "
            f"{invalid_decisions}."
        )

    nonempty_rejections = draft.loc[
        draft["rejection_reason"].astype(bool), "intake_id"
    ].tolist()
    if nonempty_rejections:
        errors.append(
            "Pending draft rows must not contain rejection reasons: "
            f"{nonempty_rejections}."
        )

    plan_by_id = {
        row["intake_id"]: row
        for row in plan.to_dict(orient="records")
    }
    approved_ids = set(approval_log.get("intake_id", []))
    comparable_columns = tuple(
        column
        for column in SAMPLE_INTAKE_COLUMNS
        if column not in {"decision", "rejection_reason"}
    )

    for row in draft.to_dict(orient="records"):
        intake_id = row["intake_id"]
        planned = plan_by_id.get(intake_id)
        if planned is None:
            errors.append(
                f"Draft intake_id '{intake_id}' is not part of batch_001."
            )
            continue
        if intake_id in approved_ids:
            errors.append(
                f"Draft intake_id '{intake_id}' already exists in approval_log.csv."
            )

        mismatches = [
            column
            for column in comparable_columns
            if str(row.get(column, "")).strip()
            != str(planned.get(column, "")).strip()
        ]
        if mismatches:
            errors.append(
                f"Draft row '{intake_id}' conflicts with the batch plan on "
                f"fields: {mismatches}."
            )

    return errors


def merge_live_queue(
    live_queue: pd.DataFrame,
    draft: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    errors: list[str] = []
    added_ids: list[str] = []
    prospective = live_queue.copy()
    existing_by_id = {
        row["intake_id"]: row
        for row in live_queue.to_dict(orient="records")
    }

    for row in draft.to_dict(orient="records"):
        intake_id = row["intake_id"]
        existing = existing_by_id.get(intake_id)
        if existing is not None:
            existing_series = pd.Series(existing)
            row_series = pd.Series(row)
            if row_signature(existing_series) != row_signature(row_series):
                errors.append(
                    f"Live queue row '{intake_id}' conflicts with the draft."
                )
            continue
        prospective = pd.concat(
            [prospective, pd.DataFrame([row], columns=SAMPLE_INTAKE_COLUMNS)],
            ignore_index=True,
        )
        existing_by_id[intake_id] = row
        added_ids.append(intake_id)

    return prospective, added_ids, errors


def plan_review_queue_activation() -> dict[str, Any]:
    (
        draft,
        live_queue,
        plan,
        part_groups,
        images,
        approval_log,
        draft_path,
        read_errors,
    ) = load_activation_inputs()
    errors = list(read_errors)
    errors.extend(validate_draft_against_plan(draft, plan, approval_log))
    prospective, added_ids, merge_errors = merge_live_queue(live_queue, draft)
    errors.extend(merge_errors)

    review_report = build_review_report(
        prospective,
        part_groups,
        images,
        approval_log,
    )
    if review_report["status"] != "PASS":
        errors.extend(review_report["errors"])

    if errors:
        readiness = "REVIEW_QUEUE_BLOCKED"
        result = "BLOCKED"
    elif draft.empty:
        readiness = "AWAITING_REVIEW_READY_ITEMS"
        result = "NO_REVIEW_READY_ITEMS"
    elif added_ids:
        readiness = "READY_TO_ACTIVATE"
        result = "ACTIVATION_PLANNED"
    else:
        readiness = "MANUAL_REVIEW_READY"
        result = "ALREADY_ACTIVE"

    return {
        "status": "PASS" if not errors else "FAIL",
        "result": result,
        "readiness": readiness,
        "draft_path": relative_path(draft_path),
        "counts": {
            "draft_rows": int(len(draft)),
            "live_queue_rows_before": int(len(live_queue)),
            "rows_to_activate": len(added_ids),
            "already_active": int(len(draft) - len(added_ids)),
            "prospective_live_queue_rows": int(len(prospective)),
        },
        "added_intake_ids": added_ids,
        "prospective_queue": prospective,
        "errors": sorted(set(errors)),
        "warnings": review_report.get("warnings", []),
    }


def render_activation_summary(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# First Batch Review Queue Activation",
        "",
        f"- Status: **{report['status']}**",
        f"- Result: **{report['result']}**",
        f"- Readiness: **{report['readiness']}**",
        f"- Draft source: `{report['draft_path']}`",
        f"- Draft rows: **{counts['draft_rows']}**",
        f"- Newly activated: **{counts['rows_to_activate']}**",
        f"- Already active: **{counts['already_active']}**",
        f"- Live queue rows: **{counts['prospective_live_queue_rows']}**",
        "",
        "No approval or rejection decision is created by this command.",
    ]
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"


def write_activation_outputs(report: dict[str, Any]) -> None:
    serializable = {
        key: value
        for key, value in report.items()
        if key != "prospective_queue"
    }
    atomic_write_text(
        FIRST_BATCH_REVIEW_ACTIVATION_STATUS_PATH,
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        FIRST_BATCH_REVIEW_ACTIVATION_SUMMARY_PATH,
        render_activation_summary(report),
    )


def activate_review_queue() -> dict[str, Any]:
    report = plan_review_queue_activation()
    if report["status"] != "PASS":
        write_activation_outputs(report)
        raise ReviewQueueActivationError(
            "Review queue activation is blocked: "
            + " | ".join(report["errors"])
        )

    if report["result"] in {"NO_REVIEW_READY_ITEMS", "ALREADY_ACTIVE"}:
        write_activation_outputs(report)
        return report

    queue_snapshot = (
        REAL_SAMPLE_INTAKE_PATH.read_bytes()
        if REAL_SAMPLE_INTAKE_PATH.is_file()
        else None
    )
    protected_before = protected_fingerprint()

    activated_ids = list(report["added_intake_ids"])
    try:
        atomic_write_dataframe(
            report["prospective_queue"],
            REAL_SAMPLE_INTAKE_PATH,
        )
        validation = plan_review_queue_activation()
        if validation["status"] != "PASS":
            raise ReviewQueueActivationError(
                "Post-write queue validation failed: "
                + " | ".join(validation["errors"])
            )
        if protected_before != protected_fingerprint():
            raise ReviewQueueActivationError(
                "Queue activation changed approved dataset state."
            )
        report = validation
        report["result"] = "ACTIVATED"
        report["readiness"] = "MANUAL_REVIEW_READY"
        report["added_intake_ids"] = activated_ids
        report["counts"]["rows_activated"] = len(activated_ids)
        write_activation_outputs(report)
    except Exception:
        if queue_snapshot is None:
            REAL_SAMPLE_INTAKE_PATH.unlink(missing_ok=True)
        else:
            REAL_SAMPLE_INTAKE_PATH.write_bytes(queue_snapshot)
        raise

    return report


def main() -> None:
    try:
        report = activate_review_queue()
    except ReviewQueueActivationError as error:
        print("First real batch review queue activation")
        print("- Status: FAIL")
        print(f"- Error: {error}")
        raise SystemExit(1) from error

    counts = report["counts"]
    print("First real batch review queue activation")
    print(f"- Status: {report['status']}")
    print(f"- Result: {report['result']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Draft rows: {counts['draft_rows']}")
    print(f"- Rows to activate: {counts['rows_to_activate']}")
    print(
        "- Live queue rows: "
        f"{counts['prospective_live_queue_rows']}"
    )
    print("- Automatic decisions: 0")
    print(
        "- Status report: "
        f"{relative_path(FIRST_BATCH_REVIEW_ACTIVATION_STATUS_PATH)}"
    )


if __name__ == "__main__":
    main()
