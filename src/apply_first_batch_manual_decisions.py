from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.apply_real_sample_intake import apply_intake
from src.real_dataset_config import (
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_PROCESSED_IMAGES_DIRECTORY,
    REAL_SAMPLE_INTAKE_PATH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.review_real_sample_intake import read_csv_exact
from src.validate_first_batch_manual_decisions import (
    APPLICATION_VALIDATION_STATUS_PATH,
    atomic_write_dataframe,
    atomic_write_text,
    build_manual_decision_application_plan,
    file_fingerprint,
    relative_path,
)

APPLICATION_STATUS_PATH = (
    APPLICATION_VALIDATION_STATUS_PATH.parent
    / "manual_decision_application_status.json"
)
APPLICATION_SUMMARY_PATH = (
    APPLICATION_VALIDATION_STATUS_PATH.parent
    / "manual_decision_application_summary.md"
)

LIVE_FILE_PATHS = (
    REAL_PART_GROUPS_PATH,
    REAL_IMAGES_PATH,
    REAL_SAMPLE_INTAKE_PATH,
    REAL_APPROVAL_LOG_PATH,
    PROJECT_ROOT
    / "data"
    / "real"
    / "processed"
    / "real_image_manifest.csv",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_review.json",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_review.md",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_apply.json",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_apply.md",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "intake_validation.json",
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "intake_validation.md",
)


class ManualDecisionApplicationError(RuntimeError):
    pass


def read_validation_status() -> dict[str, Any]:
    if not APPLICATION_VALIDATION_STATUS_PATH.is_file():
        raise ManualDecisionApplicationError(
            "Validation status is missing. Run "
            "'validate-first-real-batch-manual-decisions' first."
        )

    try:
        payload = json.loads(
            APPLICATION_VALIDATION_STATUS_PATH.read_text(
                encoding="utf-8-sig"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManualDecisionApplicationError(
            f"Cannot read the validation status: {error}."
        ) from error

    if not isinstance(payload, dict):
        raise ManualDecisionApplicationError(
            "Validation status must contain a JSON object."
        )
    return payload


def snapshot_live_state() -> dict[str, Any]:
    file_snapshot = {
        path: path.read_bytes() if path.is_file() else None
        for path in LIVE_FILE_PATHS
    }

    image_snapshot: dict[Path, bytes] = {}
    if REAL_PROCESSED_IMAGES_DIRECTORY.is_dir():
        for path in REAL_PROCESSED_IMAGES_DIRECTORY.rglob("*"):
            if path.is_file():
                image_snapshot[
                    path.relative_to(REAL_PROCESSED_IMAGES_DIRECTORY)
                ] = path.read_bytes()

    return {
        "files": file_snapshot,
        "images": image_snapshot,
    }


def restore_live_state(snapshot: dict[str, Any]) -> None:
    for path, content in snapshot["files"].items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.restore.tmp")
            temporary.write_bytes(content)
            os.replace(temporary, path)

    if REAL_PROCESSED_IMAGES_DIRECTORY.exists():
        for path in sorted(
            REAL_PROCESSED_IMAGES_DIRECTORY.rglob("*"),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

    REAL_PROCESSED_IMAGES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    for relative, content in snapshot["images"].items():
        destination = REAL_PROCESSED_IMAGES_DIRECTORY / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def combined_notes(existing: object, operator_notes: object) -> str:
    parts: list[str] = []
    for value in (existing, operator_notes):
        text = str(value).strip()
        if text and text not in parts:
            parts.append(text)
    return " | ".join(parts)


def apply_decisions_to_live_queue(
    application_plan: pd.DataFrame,
) -> pd.DataFrame:
    intake, errors = read_csv_exact(
        REAL_SAMPLE_INTAKE_PATH,
        SAMPLE_INTAKE_COLUMNS,
        "sample_intake.csv",
    )
    if errors:
        raise ManualDecisionApplicationError(
            "Cannot read the live queue: " + " | ".join(errors)
        )

    decisions = {
        str(row["intake_id"]).strip(): row
        for row in application_plan.to_dict(orient="records")
    }
    found_ids: set[str] = set()
    updated = intake.copy()

    for index, row in updated.iterrows():
        intake_id = str(row["intake_id"]).strip()
        decision_row = decisions.get(intake_id)
        if decision_row is None:
            continue

        found_ids.add(intake_id)
        current_decision = str(row["decision"]).strip().lower()
        if current_decision != "pending":
            raise ManualDecisionApplicationError(
                f"{intake_id} is no longer pending."
            )

        operator_decision = str(
            decision_row["operator_decision"]
        ).strip().lower()
        if operator_decision not in {"approved", "rejected"}:
            raise ManualDecisionApplicationError(
                f"{intake_id} has no applicable operator decision."
            )

        updated.at[index, "decision"] = operator_decision
        updated.at[index, "rejection_reason"] = str(
            decision_row["rejection_reason"]
        ).strip()
        updated.at[index, "notes"] = combined_notes(
            row.get("notes", ""),
            decision_row.get("operator_notes", ""),
        )

    missing_ids = sorted(set(decisions) - found_ids)
    if missing_ids:
        raise ManualDecisionApplicationError(
            "Validated intake IDs are missing from the live queue: "
            f"{missing_ids}."
        )

    return updated


def render_application_summary(report: dict[str, Any]) -> str:
    counts = report.get("counts", {})
    lines = [
        "# First Batch Controlled Manual Decision Application",
        "",
        f"- Status: **{report['status']}**",
        f"- Result: **{report['result']}**",
        f"- Plan ID: `{report.get('plan_id', '') or 'NOT_AVAILABLE'}`",
        f"- Approved: **{counts.get('approved', 0)}**",
        f"- Rejected: **{counts.get('rejected', 0)}**",
        (
            "- Remaining pending: "
            f"**{counts.get('remaining_pending', 0)}**"
        ),
        (
            "- Live queue after application: "
            f"`{report.get('queue_fingerprint_after', 'UNKNOWN')}`"
        ),
        (
            "- Rollback performed: "
            f"**{report.get('rollback_performed', 'NO')}**"
        ),
    ]

    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])

    return "\n".join(lines) + "\n"


def write_application_outputs(report: dict[str, Any]) -> None:
    atomic_write_text(
        APPLICATION_STATUS_PATH,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        APPLICATION_SUMMARY_PATH,
        render_application_summary(report),
    )


def apply_manual_decisions(
    *,
    apply_callable: Callable[[], dict[str, object]] = apply_intake,
) -> dict[str, Any]:
    application_plan, fresh_validation = (
        build_manual_decision_application_plan()
    )

    if (
        fresh_validation["status"] != "PASS"
        or fresh_validation["readiness"] != "READY_TO_APPLY"
        or not fresh_validation["plan_id"]
    ):
        raise ManualDecisionApplicationError(
            "Manual decisions are not ready to apply. Run validation "
            "and resolve the reported readiness state."
        )

    saved_validation = read_validation_status()
    required_matches = (
        "plan_id",
        "queue_fingerprint",
        "workbook_fingerprint",
        "first_batch_plan_fingerprint",
    )
    stale_fields = [
        field
        for field in required_matches
        if str(saved_validation.get(field, ""))
        != str(fresh_validation.get(field, ""))
    ]
    if (
        saved_validation.get("status") != "PASS"
        or saved_validation.get("readiness") != "READY_TO_APPLY"
        or stale_fields
    ):
        raise ManualDecisionApplicationError(
            "The saved validation plan is stale or not ready. "
            "Run 'validate-first-real-batch-manual-decisions' again. "
            f"Mismatched fields: {stale_fields}."
        )

    snapshot = snapshot_live_state()
    expected_approved = int(
        fresh_validation["counts"]["approved_decisions"]
    )
    expected_rejected = int(
        fresh_validation["counts"]["rejected_decisions"]
    )
    planned_ids = set(application_plan["intake_id"].tolist())

    try:
        updated_queue = apply_decisions_to_live_queue(
            application_plan
        )
        atomic_write_dataframe(
            updated_queue,
            REAL_SAMPLE_INTAKE_PATH,
        )

        delegated_report = apply_callable()
        delegated_counts = delegated_report.get("counts", {})

        actual_approved = int(delegated_counts.get("approved", -1))
        actual_rejected = int(delegated_counts.get("rejected", -1))
        if actual_approved != expected_approved:
            raise ManualDecisionApplicationError(
                "Delegated approval count differs from the validated "
                f"plan: expected {expected_approved}, got "
                f"{actual_approved}."
            )
        if actual_rejected != expected_rejected:
            raise ManualDecisionApplicationError(
                "Delegated rejection count differs from the validated "
                f"plan: expected {expected_rejected}, got "
                f"{actual_rejected}."
            )

        remaining_queue, queue_errors = read_csv_exact(
            REAL_SAMPLE_INTAKE_PATH,
            SAMPLE_INTAKE_COLUMNS,
            "sample_intake.csv",
        )
        if queue_errors:
            raise ManualDecisionApplicationError(
                "Cannot verify the post-application queue: "
                + " | ".join(queue_errors)
            )

        remaining_ids = set(
            remaining_queue["intake_id"].astype(str).str.strip()
        )
        unhandled = sorted(planned_ids & remaining_ids)
        if unhandled:
            raise ManualDecisionApplicationError(
                "Applied first-batch intake IDs remain in the live "
                f"queue: {unhandled}."
            )

        report = {
            "status": "PASS",
            "result": "APPLIED",
            "plan_id": fresh_validation["plan_id"],
            "counts": {
                "approved": actual_approved,
                "rejected": actual_rejected,
                "remaining_pending": int(
                    (
                        remaining_queue["decision"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        == "pending"
                    ).sum()
                ),
            },
            "queue_fingerprint_after": file_fingerprint(
                REAL_SAMPLE_INTAKE_PATH
            ),
            "rollback_performed": "NO",
            "delegated_apply_result": delegated_report,
            "errors": [],
        }
        write_application_outputs(report)
        return report
    except Exception as error:
        restore_live_state(snapshot)
        failure_report = {
            "status": "FAIL",
            "result": "ROLLED_BACK",
            "plan_id": fresh_validation["plan_id"],
            "counts": {
                "approved": 0,
                "rejected": 0,
                "remaining_pending": int(
                    fresh_validation["counts"][
                        "first_batch_queue_rows"
                    ]
                ),
            },
            "queue_fingerprint_after": file_fingerprint(
                REAL_SAMPLE_INTAKE_PATH
            ),
            "rollback_performed": "YES",
            "errors": [str(error)],
        }
        write_application_outputs(failure_report)

        if isinstance(error, ManualDecisionApplicationError):
            raise
        raise ManualDecisionApplicationError(
            f"Controlled application failed and was rolled back: {error}"
        ) from error


def main() -> None:
    try:
        report = apply_manual_decisions()
    except ManualDecisionApplicationError as error:
        print("First real batch controlled manual decision application")
        print("- Status: FAIL")
        print(f"- Error: {error}")
        print(
            "- Validation: "
            f"{relative_path(APPLICATION_VALIDATION_STATUS_PATH)}"
        )
        raise SystemExit(1) from error

    print("First real batch controlled manual decision application")
    print(f"- Status: {report['status']}")
    print(f"- Result: {report['result']}")
    print(f"- Plan ID: {report['plan_id']}")
    print(f"- Approved: {report['counts']['approved']}")
    print(f"- Rejected: {report['counts']['rejected']}")
    print(
        "- Remaining pending: "
        f"{report['counts']['remaining_pending']}"
    )
    print(f"- Report: {relative_path(APPLICATION_SUMMARY_PATH)}")


if __name__ == "__main__":
    main()
