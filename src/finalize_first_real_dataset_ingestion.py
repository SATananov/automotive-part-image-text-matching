from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.apply_first_batch_manual_decisions import (
    apply_manual_decisions,
    restore_live_state,
    snapshot_live_state,
)
from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    FIRST_BATCH_EXPECTED_GROUPS,
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PLAN_PATH,
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_SAMPLE_INTAKE_PATH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.validate_first_batch_manual_decisions import (
    atomic_write_text,
    relative_path,
    validate_manual_decisions,
)

STEP010_RUNTIME_DIRECTORY = (
    PROJECT_ROOT / "data" / "real" / "runtime" / "step_010"
)
INGESTION_RUNTIME_STATUS_PATH = (
    STEP010_RUNTIME_DIRECTORY / "ingestion_status.json"
)
INGESTION_RUNTIME_SUMMARY_PATH = (
    STEP010_RUNTIME_DIRECTORY / "ingestion_summary.md"
)
INGESTION_REPORT_JSON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_real_dataset_ingestion.json"
)
INGESTION_REPORT_MARKDOWN_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_real_dataset_ingestion.md"
)


class FirstRealDatasetIngestionError(RuntimeError):
    pass


class FirstRealDatasetIngestionNotReady(
    FirstRealDatasetIngestionError
):
    pass


def read_csv_exact(
    path: Path,
    columns: tuple[str, ...],
    label: str,
) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        return (
            pd.DataFrame(columns=columns),
            [f"{label} is missing: {relative_path(path)}."],
        )
    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except Exception as error:
        return (
            pd.DataFrame(columns=columns),
            [f"Cannot read {label}: {error}."],
        )
    if tuple(frame.columns) != columns:
        return (
            pd.DataFrame(columns=columns),
            [f"{label} has an invalid schema."],
        )
    return frame, []


def normalized(value: object) -> str:
    return str(value).strip()


def build_ingestion_audit() -> dict[str, Any]:
    plan, e1 = read_csv_exact(
        FIRST_BATCH_PLAN_PATH,
        FIRST_BATCH_PLAN_COLUMNS,
        "first_batch_plan.csv",
    )
    queue, e2 = read_csv_exact(
        REAL_SAMPLE_INTAKE_PATH,
        SAMPLE_INTAKE_COLUMNS,
        "sample_intake.csv",
    )
    approval, e3 = read_csv_exact(
        REAL_APPROVAL_LOG_PATH,
        APPROVAL_LOG_COLUMNS,
        "approval_log.csv",
    )
    groups, e4 = read_csv_exact(
        REAL_PART_GROUPS_PATH,
        PART_GROUP_COLUMNS,
        "part_groups.csv",
    )
    images, e5 = read_csv_exact(
        REAL_IMAGES_PATH,
        IMAGE_MANIFEST_COLUMNS,
        "images.csv",
    )
    manifest, e6 = read_csv_exact(
        REAL_IMAGE_MANIFEST_PATH,
        REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
        "real_image_manifest.csv",
    )
    errors = [*e1, *e2, *e3, *e4, *e5, *e6]

    plan_ids = set(plan.get("intake_id", pd.Series(dtype=str)))
    plan_by_id = {
        normalized(row["intake_id"]): row
        for row in plan.to_dict(orient="records")
    }
    relevant = approval.loc[
        approval.get("intake_id", pd.Series(dtype=str)).isin(plan_ids)
    ].copy()

    duplicate_ids = sorted(
        value
        for value, count in relevant.get(
            "intake_id",
            pd.Series(dtype=str),
        ).value_counts().items()
        if count > 1
    )
    if duplicate_ids:
        errors.append(
            "Duplicate first-batch approval-log decisions: "
            f"{duplicate_ids}."
        )

    log_by_id = {
        normalized(row["intake_id"]): row
        for row in relevant.to_dict(orient="records")
    }
    queue_ids = set(queue.get("intake_id", pd.Series(dtype=str)))
    remaining = sorted(plan_ids & queue_ids)
    groups_by_id = {
        normalized(row["part_group_id"]): row
        for row in groups.to_dict(orient="records")
    }
    images_by_id = {
        normalized(row["image_id"]): row
        for row in images.to_dict(orient="records")
    }
    manifest_by_id = {
        normalized(row["image_id"]): row
        for row in manifest.to_dict(orient="records")
    }

    approved = 0
    rejected = 0
    views_by_group: dict[str, set[str]] = {}
    categories: set[str] = set()

    for intake_id, log_row in log_by_id.items():
        plan_row = plan_by_id.get(intake_id)
        if plan_row is None:
            errors.append(
                f"{intake_id}: decision has no canonical plan row."
            )
            continue

        decision = normalized(log_row.get("decision", "")).lower()
        if decision == "rejected":
            rejected += 1
            if not normalized(log_row.get("rejection_reason", "")):
                errors.append(
                    f"{intake_id}: rejected decision has no reason."
                )
            continue
        if decision != "approved":
            errors.append(
                f"{intake_id}: unsupported decision '{decision}'."
            )
            continue

        approved += 1
        group_id = normalized(log_row.get("part_group_id", ""))
        image_id = normalized(log_row.get("image_id", ""))
        view = normalized(plan_row.get("view", ""))
        category = normalized(plan_row.get("part_category", ""))

        if group_id != normalized(plan_row.get("part_group_id", "")):
            errors.append(
                f"{intake_id}: approved group differs from plan."
            )

        group_row = groups_by_id.get(group_id)
        if group_row is None or normalized(
            group_row.get("approved", "")
        ).lower() != "yes":
            errors.append(
                f"{intake_id}: approved group annotation is missing."
            )

        image_row = images_by_id.get(image_id)
        if image_row is None:
            errors.append(
                f"{intake_id}: image annotation is missing."
            )
        else:
            if normalized(image_row.get("view", "")) != view:
                errors.append(
                    f"{intake_id}: image view differs from plan."
                )
            if normalized(
                image_row.get("approved", "")
            ).lower() != "yes":
                errors.append(
                    f"{intake_id}: image is not marked approved."
                )

        manifest_row = manifest_by_id.get(image_id)
        if manifest_row is None:
            errors.append(
                f"{intake_id}: manifest row is missing."
            )
        else:
            image_path = normalized(
                manifest_row.get("image_path", "")
            )
            if not image_path or not (
                PROJECT_ROOT / image_path
            ).is_file():
                errors.append(
                    f"{intake_id}: processed image is missing: "
                    f"{image_path or 'BLANK'}."
                )

        views_by_group.setdefault(group_id, set()).add(view)
        categories.add(category)

    planned_images = int(len(plan)) or FIRST_BATCH_EXPECTED_IMAGES
    planned_groups = (
        int(plan["part_group_id"].nunique())
        if not plan.empty
        else FIRST_BATCH_EXPECTED_GROUPS
    )
    complete_groups = sum(
        views.issuperset({"front", "detail"})
        for views in views_by_group.values()
    )
    decided = len(log_by_id)

    if errors:
        status = "FAIL"
        readiness = "INGESTION_AUDIT_BLOCKED"
    elif remaining or decided < planned_images:
        status = "PASS"
        readiness = "MANUAL_DECISIONS_REQUIRED"
    elif (
        approved == planned_images
        and complete_groups == planned_groups
        and len(categories) == planned_groups
    ):
        status = "PASS"
        readiness = "FIRST_BATCH_INGESTED"
    else:
        status = "PASS"
        readiness = "RECAPTURE_REQUIRED"

    return {
        "status": status,
        "readiness": readiness,
        "counts": {
            "planned_images": planned_images,
            "planned_groups": planned_groups,
            "decided": decided,
            "approved": approved,
            "rejected": rejected,
            "remaining_queue": len(remaining),
            "approved_groups": len(views_by_group),
            "complete_front_detail_groups": complete_groups,
            "approved_categories": len(categories),
        },
        "remaining_queue_ids": remaining,
        "errors": sorted(set(errors)),
        "warnings": [],
    }


def render_summary(report: dict[str, Any]) -> str:
    audit = report.get("audit", {})
    counts = audit.get("counts", {})
    lines = [
        "# Step 010 — Approved Sample Ingestion",
        "",
        f"- Status: **{report['status']}**",
        f"- Result: **{report['result']}**",
        f"- Readiness: **{report['readiness']}**",
        f"- Approved: **{counts.get('approved', 0)}**",
        f"- Rejected: **{counts.get('rejected', 0)}**",
        (
            "- Complete groups: "
            f"**{counts.get('complete_front_detail_groups', 0)} / "
            f"{counts.get('planned_groups', 0)}**"
        ),
        (
            "- Remaining queue: "
            f"**{counts.get('remaining_queue', 0)}**"
        ),
        (
            "- Rollback performed: "
            f"**{report.get('rollback_performed', 'NO')}**"
        ),
    ]
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in report["errors"])
    return "\n".join(lines) + "\n"


def write_runtime_outputs(report: dict[str, Any]) -> None:
    atomic_write_text(
        INGESTION_RUNTIME_STATUS_PATH,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        INGESTION_RUNTIME_SUMMARY_PATH,
        render_summary(report),
    )


def write_tracked_outputs(report: dict[str, Any]) -> None:
    atomic_write_text(
        INGESTION_REPORT_JSON_PATH,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        INGESTION_REPORT_MARKDOWN_PATH,
        render_summary(report),
    )


def finalize_ingestion(
    *,
    validation_callable: Callable[..., dict[str, Any]] = validate_manual_decisions,
    apply_callable: Callable[[], dict[str, Any]] = apply_manual_decisions,
    audit_callable: Callable[[], dict[str, Any]] = build_ingestion_audit,
    snapshot_callable: Callable[[], dict[str, Any]] = snapshot_live_state,
    restore_callable: Callable[[dict[str, Any]], None] = restore_live_state,
) -> dict[str, Any]:
    validation = validation_callable(write_outputs=True)
    if (
        validation.get("status") != "PASS"
        or validation.get("readiness") != "READY_TO_APPLY"
        or not validation.get("plan_id")
    ):
        blocked = {
            "status": "PASS",
            "result": "NOT_APPLIED",
            "readiness": validation.get(
                "readiness",
                "MANUAL_DECISIONS_REQUIRED",
            ),
            "plan_id": validation.get("plan_id", ""),
            "rollback_performed": "NO",
            "validation": validation,
            "audit": build_ingestion_audit(),
            "errors": [],
            "warnings": validation.get("warnings", []),
        }
        write_runtime_outputs(blocked)
        raise FirstRealDatasetIngestionNotReady(
            "Manual decisions are not READY_TO_APPLY."
        )

    snapshot = snapshot_callable()
    try:
        application = apply_callable()
        audit = audit_callable()
        errors = list(audit.get("errors", []))

        expected_approved = int(
            application.get("counts", {}).get("approved", -1)
        )
        expected_rejected = int(
            application.get("counts", {}).get("rejected", -1)
        )
        actual_approved = int(
            audit.get("counts", {}).get("approved", -2)
        )
        actual_rejected = int(
            audit.get("counts", {}).get("rejected", -2)
        )

        if actual_approved != expected_approved:
            errors.append(
                "Step 010 approved count differs from application."
            )
        if actual_rejected != expected_rejected:
            errors.append(
                "Step 010 rejected count differs from application."
            )
        if audit.get("status") != "PASS":
            errors.append(
                "Step 010 post-application ingestion audit failed."
            )
        if errors:
            raise FirstRealDatasetIngestionError(
                " | ".join(sorted(set(errors)))
            )

        readiness = audit["readiness"]
        result = (
            "FIRST_BATCH_INGESTED"
            if readiness == "FIRST_BATCH_INGESTED"
            else "APPLIED_WITH_RECAPTURE_REQUIRED"
        )
        report = {
            "status": "PASS",
            "result": result,
            "readiness": readiness,
            "plan_id": validation["plan_id"],
            "rollback_performed": "NO",
            "validation": validation,
            "application": application,
            "audit": audit,
            "errors": [],
            "warnings": sorted(
                set(
                    validation.get("warnings", [])
                    + audit.get("warnings", [])
                )
            ),
        }
        write_runtime_outputs(report)
        write_tracked_outputs(report)
        return report
    except Exception as error:
        restore_callable(snapshot)
        failure = {
            "status": "FAIL",
            "result": "ROLLED_BACK",
            "readiness": "INGESTION_AUDIT_BLOCKED",
            "plan_id": validation.get("plan_id", ""),
            "rollback_performed": "YES",
            "validation": validation,
            "errors": [str(error)],
            "warnings": validation.get("warnings", []),
        }
        write_runtime_outputs(failure)
        if isinstance(error, FirstRealDatasetIngestionError):
            raise
        raise FirstRealDatasetIngestionError(
            f"Step 010 ingestion failed and was rolled back: {error}"
        ) from error


def main() -> None:
    try:
        report = finalize_ingestion()
    except FirstRealDatasetIngestionNotReady as error:
        print("Step 010 approved sample ingestion")
        print("- Status: NOT APPLIED")
        print(f"- Reason: {error}")
        raise SystemExit(1) from error
    except FirstRealDatasetIngestionError as error:
        print("Step 010 approved sample ingestion")
        print("- Status: FAIL")
        print(f"- Error: {error}")
        raise SystemExit(1) from error

    counts = report["audit"]["counts"]
    print("Step 010 approved sample ingestion")
    print(f"- Status: {report['status']}")
    print(f"- Result: {report['result']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Approved: {counts['approved']}")
    print(f"- Rejected: {counts['rejected']}")
    print(
        "- Report: "
        f"{relative_path(INGESTION_REPORT_MARKDOWN_PATH)}"
    )


if __name__ == "__main__":
    main()
