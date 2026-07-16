from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.real_dataset_config import (
    ALLOWED_IMAGE_VIEWS,
    APPROVAL_LOG_COLUMNS,
    CATEGORY_TO_FAMILY,
    FIRST_BATCH_EXPECTED_GROUPS,
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_GROUP_NUMBER,
    FIRST_BATCH_ID,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PLAN_PATH,
    FIRST_BATCH_PREVIEW_COLUMNS,
    FIRST_BATCH_PREVIEW_PATH,
    FIRST_BATCH_VIEWS,
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_DATASET_CATEGORIES,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_SAMPLE_INTAKE_PATH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.review_real_sample_intake import (
    build_review_report,
    read_csv_exact,
    row_value,
)


JSON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_preparation.json"
)

MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_preparation.md"
)

BATCH_ITEM_PATTERN = re.compile(r"^batch_001_[0-9]{3}$")


def empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def read_plan() -> tuple[pd.DataFrame, list[str]]:
    return read_csv_exact(
        FIRST_BATCH_PLAN_PATH,
        FIRST_BATCH_PLAN_COLUMNS,
        "first batch plan",
    )


def plan_row_to_intake(row: pd.Series) -> dict[str, str]:
    return {
        "intake_id": row_value(row, "intake_id"),
        "staging_path": row_value(row, "staging_path"),
        "part_group_id": row_value(row, "part_group_id"),
        "part_family": row_value(row, "part_family"),
        "part_category": row_value(row, "part_category"),
        "view": row_value(row, "view"),
        "source": row_value(row, "source"),
        "match_description": row_value(row, "match_description"),
        "partial_description": row_value(row, "partial_description"),
        "mismatch_description": row_value(row, "mismatch_description"),
        "decision": "pending",
        "rejection_reason": "",
        "notes": row_value(row, "notes"),
    }


def validate_plan(plan: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    if tuple(plan.columns) != FIRST_BATCH_PLAN_COLUMNS:
        errors.append(
            "first_batch_plan.csv must use the exact configured column "
            "order."
        )
        return errors

    if len(plan) != FIRST_BATCH_EXPECTED_IMAGES:
        errors.append(
            f"First batch must contain {FIRST_BATCH_EXPECTED_IMAGES} "
            f"planned images; found {len(plan)}."
        )

    required_columns = tuple(
        column for column in FIRST_BATCH_PLAN_COLUMNS if column != "notes"
    )

    for column in required_columns:
        missing_rows = [
            str(index + 2)
            for index, value in plan[column].items()
            if not str(value).strip()
        ]
        if missing_rows:
            errors.append(
                f"first_batch_plan.csv column '{column}' contains "
                f"empty values on rows: {', '.join(missing_rows)}."
            )

    for column in (
        "batch_item_id",
        "intake_id",
        "staging_path",
    ):
        duplicates = sorted(
            value
            for value, count in Counter(plan[column]).items()
            if value and count > 1
        )
        if duplicates:
            errors.append(
                f"Duplicate {column} values in first batch plan: "
                f"{duplicates}."
            )

    if set(plan["batch_id"]) != {FIRST_BATCH_ID}:
        errors.append(
            f"All first batch rows must use batch_id '{FIRST_BATCH_ID}'."
        )

    invalid_items = sorted(
        value
        for value in plan["batch_item_id"]
        if BATCH_ITEM_PATTERN.fullmatch(value) is None
    )
    if invalid_items:
        errors.append(
            f"Invalid first batch item identifiers: {invalid_items}."
        )

    expected_categories = set(REAL_DATASET_CATEGORIES)
    actual_categories = set(plan["part_category"])
    if actual_categories != expected_categories:
        errors.append(
            "First batch categories do not match the configured real-data "
            f"categories: {sorted(actual_categories)}."
        )

    group_rows: dict[str, list[pd.Series]] = defaultdict(list)
    for _, row in plan.iterrows():
        group_rows[row_value(row, "part_group_id")].append(row)

    if len(group_rows) != FIRST_BATCH_EXPECTED_GROUPS:
        errors.append(
            f"First batch must contain {FIRST_BATCH_EXPECTED_GROUPS} "
            f"physical groups; found {len(group_rows)}."
        )

    for row_number, row in enumerate(plan.itertuples(index=False), start=2):
        category = str(row.part_category).strip()
        family = str(row.part_family).strip()
        group_id = str(row.part_group_id).strip()
        view = str(row.view).strip()
        intake_id = str(row.intake_id).strip()
        staging_path = Path(str(row.staging_path).strip())

        if category not in REAL_DATASET_CATEGORIES:
            errors.append(
                f"Plan row {row_number} has unknown category "
                f"'{category}'."
            )
            continue

        expected_family = CATEGORY_TO_FAMILY[category]
        if family != expected_family:
            errors.append(
                f"Plan row {row_number} maps '{category}' to "
                f"'{family}', expected '{expected_family}'."
            )

        expected_group = (
            f"real_{category}_{FIRST_BATCH_GROUP_NUMBER}"
        )
        if group_id != expected_group:
            errors.append(
                f"Plan row {row_number} must use group "
                f"'{expected_group}'."
            )

        if view not in ALLOWED_IMAGE_VIEWS:
            errors.append(
                f"Plan row {row_number} has invalid view '{view}'."
            )

        if staging_path.is_absolute() or ".." in staging_path.parts:
            errors.append(
                f"Plan row {row_number} has unsafe staging path "
                f"'{staging_path}'."
            )
        else:
            expected_parent = Path("data/real/staging")
            if staging_path.parent != expected_parent:
                errors.append(
                    f"Plan row {row_number} staging file must be directly "
                    "under data/real/staging/."
                )
            if staging_path.stem != intake_id:
                errors.append(
                    f"Plan row {row_number} staging filename must match "
                    f"intake_id '{intake_id}'."
                )

    for group_id, rows in group_rows.items():
        views = {row_value(row, "view") for row in rows}
        if len(rows) != len(FIRST_BATCH_VIEWS):
            errors.append(
                f"First batch group '{group_id}' must contain exactly "
                f"{len(FIRST_BATCH_VIEWS)} planned images."
            )
        if views != set(FIRST_BATCH_VIEWS):
            errors.append(
                f"First batch group '{group_id}' must use views "
                f"{FIRST_BATCH_VIEWS}; found {sorted(views)}."
            )

        metadata_fields = (
            "part_family",
            "part_category",
            "source",
            "match_description",
            "partial_description",
            "mismatch_description",
        )
        for field in metadata_fields:
            values = {row_value(row, field) for row in rows}
            if len(values) != 1:
                errors.append(
                    f"First batch group '{group_id}' has conflicting "
                    f"'{field}' values."
                )

    return errors


def load_supporting_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    part_groups, group_errors = read_csv_exact(
        REAL_PART_GROUPS_PATH,
        PART_GROUP_COLUMNS,
        "real part groups",
    )
    images, image_errors = read_csv_exact(
        REAL_IMAGES_PATH,
        IMAGE_MANIFEST_COLUMNS,
        "real images",
    )
    live_queue, queue_errors = read_csv_exact(
        REAL_SAMPLE_INTAKE_PATH,
        SAMPLE_INTAKE_COLUMNS,
        "sample intake queue",
    )
    approval_log, log_errors = read_csv_exact(
        REAL_APPROVAL_LOG_PATH,
        APPROVAL_LOG_COLUMNS,
        "approval log",
    )
    return (
        part_groups,
        images,
        live_queue,
        approval_log,
        group_errors + image_errors + queue_errors + log_errors,
    )


def compare_with_live_queue(
    planned: dict[str, str],
    live_queue: pd.DataFrame,
) -> tuple[str, list[str]]:
    if "intake_id" not in live_queue.columns:
        return "queue_unavailable", []

    matching = live_queue.loc[
        live_queue["intake_id"] == planned["intake_id"]
    ]
    if matching.empty:
        return "not_queued", []

    row = matching.iloc[0]
    fields = tuple(
        column
        for column in SAMPLE_INTAKE_COLUMNS
        if column not in {"decision", "rejection_reason"}
    )
    mismatches = [
        field
        for field in fields
        if row_value(row, field) != planned[field]
    ]
    if mismatches:
        return "queue_conflict", [
            f"Live queue row '{planned['intake_id']}' conflicts on "
            f"fields: {mismatches}."
        ]

    return row_value(row, "decision") or "queued", []


def build_preparation_report(
    plan: pd.DataFrame,
    initial_errors: list[str] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    errors = list(initial_errors or [])
    warnings: list[str] = []
    errors.extend(validate_plan(plan))

    if tuple(plan.columns) != FIRST_BATCH_PLAN_COLUMNS:
        preview = pd.DataFrame(columns=FIRST_BATCH_PREVIEW_COLUMNS)
        report = {
            "status": "FAIL",
            "readiness": "PREPARATION_BLOCKED",
            "batch_id": FIRST_BATCH_ID,
            "counts": {
                "planned_images": int(len(plan)),
                "planned_groups": 0,
                "captured_files": 0,
                "queued_rows": 0,
                "processed_rows": 0,
            },
            "captured_review": {
                "status": "NOT_RUN",
                "readiness": "INVALID_PLAN",
            },
            "errors": errors,
            "warnings": warnings,
        }
        return report, preview

    (
        part_groups,
        images,
        live_queue,
        approval_log,
        support_errors,
    ) = load_supporting_tables()
    errors.extend(support_errors)

    processed_ids = set(approval_log.get("intake_id", []))
    preview_rows: list[dict[str, str]] = []
    captured_intake_rows: list[dict[str, str]] = []
    queue_conflicts: list[str] = []

    for _, row in plan.iterrows():
        planned = plan_row_to_intake(row)
        path = PROJECT_ROOT / planned["staging_path"]
        file_present = path.is_file()
        queue_status, queue_errors = compare_with_live_queue(
            planned,
            live_queue,
        )
        queue_conflicts.extend(queue_errors)

        if planned["intake_id"] in processed_ids:
            queue_status = "processed"

        if file_present and queue_status != "processed":
            captured_intake_rows.append(planned)

        preview = {
            column: row_value(row, column)
            for column in FIRST_BATCH_PLAN_COLUMNS
        }
        preview.update(
            {
                "file_present": "yes" if file_present else "no",
                "queue_status": queue_status,
                "review_status": "NOT_REVIEWED",
                "review_errors": "",
                "review_warnings": "",
            }
        )
        preview_rows.append(preview)

    errors.extend(queue_conflicts)

    captured_review = None
    if captured_intake_rows and not errors:
        captured_frame = pd.DataFrame(
            captured_intake_rows,
            columns=SAMPLE_INTAKE_COLUMNS,
        )
        captured_review = build_review_report(
            captured_frame,
            part_groups,
            images,
            approval_log,
        )
        errors.extend(captured_review["errors"])
        warnings.extend(captured_review["warnings"])

        review_by_id = {
            item["intake_id"]: item
            for item in captured_review["items"]
        }
        for preview in preview_rows:
            item = review_by_id.get(preview["intake_id"])
            if item is None:
                continue
            preview["review_status"] = str(item["status"])
            preview["review_errors"] = " | ".join(item["errors"])
            preview["review_warnings"] = " | ".join(item["warnings"])

    preview = pd.DataFrame(
        preview_rows,
        columns=FIRST_BATCH_PREVIEW_COLUMNS,
    )
    captured = int((preview["file_present"] == "yes").sum())
    queued = int(
        preview["queue_status"].isin(
            {"pending", "approved", "rejected"}
        ).sum()
    )
    processed = int((preview["queue_status"] == "processed").sum())

    if errors:
        readiness = "PREPARATION_BLOCKED"
    elif processed == FIRST_BATCH_EXPECTED_IMAGES:
        readiness = "BATCH_COMPLETE"
    elif captured == 0:
        readiness = "AWAITING_CAPTURE"
    elif captured < FIRST_BATCH_EXPECTED_IMAGES:
        readiness = "CAPTURE_IN_PROGRESS"
    elif queued < FIRST_BATCH_EXPECTED_IMAGES:
        readiness = "READY_FOR_QUEUE_REVIEW"
    else:
        readiness = "READY_FOR_CONTROLLED_INTAKE"

    report = {
        "status": "PASS" if not errors else "FAIL",
        "readiness": readiness,
        "batch_id": FIRST_BATCH_ID,
        "counts": {
            "planned_images": int(len(plan)),
            "planned_groups": int(plan["part_group_id"].nunique()),
            "captured_files": captured,
            "queued_rows": queued,
            "processed_rows": processed,
        },
        "captured_review": {
            "status": (
                captured_review["status"]
                if captured_review is not None
                else "NOT_RUN"
            ),
            "readiness": (
                captured_review["readiness"]
                if captured_review is not None
                else "NO_CAPTURED_FILES"
            ),
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report, preview


def write_outputs(
    report: dict[str, object],
    preview: pd.DataFrame,
) -> None:
    JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    FIRST_BATCH_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    preview.to_csv(
        FIRST_BATCH_PREVIEW_PATH,
        index=False,
        encoding="utf-8",
    )

    counts = report["counts"]
    lines = [
        "# First Real Sample Batch Preparation",
        "",
        f"**Status:** {report['status']}",
        f"**Readiness:** {report['readiness']}",
        f"**Batch:** {report['batch_id']}",
        "",
        "## Counts",
        "",
        f"- Planned physical groups: {counts['planned_groups']}",
        f"- Planned images: {counts['planned_images']}",
        f"- Captured staging files: {counts['captured_files']}",
        f"- Rows already in live queue: {counts['queued_rows']}",
        f"- Rows already processed: {counts['processed_rows']}",
        "",
        "## Safety",
        "",
        "- The plan is separate from `sample_intake.csv`.",
        "- This command does not approve or process any image.",
        "- Missing staging files are an expected preparation state.",
        "- Captured files are reviewed through the Step 009.1 checks.",
        "",
        "## Errors",
        "",
    ]
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- No preparation errors found.")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No preparation warnings found.")

    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def prepare_first_batch() -> dict[str, object]:
    plan, read_errors = read_plan()
    report, preview = build_preparation_report(
        plan,
        initial_errors=read_errors,
    )
    write_outputs(report, preview)
    return report


def main() -> None:
    report = prepare_first_batch()
    counts = report["counts"]

    print("First real sample batch preparation")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Planned groups: {counts['planned_groups']}")
    print(f"- Planned images: {counts['planned_images']}")
    print(f"- Captured files: {counts['captured_files']}")
    print(f"- Queued rows: {counts['queued_rows']}")
    print(
        "- Preview: "
        f"{FIRST_BATCH_PREVIEW_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "- Report: "
        f"{MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
