from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.apply_first_batch_manual_decisions as application
import src.validate_first_batch_manual_decisions as validation
from src.real_dataset_config import (
    FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    FIRST_BATCH_PLAN_COLUMNS,
    SAMPLE_INTAKE_COLUMNS,
)


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def plan_rows() -> list[dict[str, str]]:
    rows = []
    for sequence, (intake_id, view) in enumerate(
        (
            ("intake_000001", "front"),
            ("intake_000002", "detail"),
        ),
        start=1,
    ):
        rows.append(
            {
                "batch_id": "batch_001",
                "batch_item_id": f"batch_001_item_{sequence:03d}",
                "intake_id": intake_id,
                "staging_path": f"data/real/staging/{intake_id}.jpg",
                "part_group_id": "real_starter_001",
                "part_family": "starting_system",
                "part_category": "starter",
                "view": view,
                "source": "local_capture",
                "match_description": "Starter motor.",
                "partial_description": "Alternator.",
                "mismatch_description": "Brake disc.",
                "notes": "",
            }
        )
    return rows


def queue_rows() -> list[dict[str, str]]:
    return [
        {
            "intake_id": row["intake_id"],
            "staging_path": row["staging_path"],
            "part_group_id": row["part_group_id"],
            "part_family": row["part_family"],
            "part_category": row["part_category"],
            "view": row["view"],
            "source": row["source"],
            "match_description": row["match_description"],
            "partial_description": row["partial_description"],
            "mismatch_description": row["mismatch_description"],
            "decision": "pending",
            "rejection_reason": "",
            "notes": "capture note",
        }
        for row in plan_rows()
    ]


def workbook_rows(
    decisions: tuple[str, str] = ("approved", "rejected"),
) -> list[dict[str, str]]:
    rows = []
    for sequence, (plan_row, decision) in enumerate(
        zip(plan_rows(), decisions),
        start=1,
    ):
        image_id = (
            f"{plan_row['part_group_id']}_{plan_row['view']}"
        )
        rows.append(
            {
                "sequence": str(sequence),
                "intake_id": plan_row["intake_id"],
                "part_group_id": plan_row["part_group_id"],
                "part_category": plan_row["part_category"],
                "view": plan_row["view"],
                "staging_path": plan_row["staging_path"],
                "image_id": image_id,
                "quality_status": "PASS",
                "width": "800",
                "height": "600",
                "format": "JPEG",
                "review_errors": "",
                "review_warnings": "",
                "current_queue_decision": "pending",
                "operator_decision": decision,
                "rejection_reason": (
                    "Wrong view." if decision == "rejected" else ""
                ),
                "operator_notes": f"operator note {sequence}",
                "decision_entry_status": (
                    "READY" if decision else "PENDING"
                ),
                "next_action": (
                    "Decision is ready for the recording step."
                    if decision
                    else "Inspect the staged image and enter a decision."
                ),
            }
        )
    return rows


@pytest.fixture
def isolated_step0099(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runtime = (
        tmp_path
        / "data"
        / "real"
        / "runtime"
        / "first_batch_review"
    )
    plan_path = (
        tmp_path
        / "data"
        / "real"
        / "annotations"
        / "first_batch_plan.csv"
    )
    queue_path = (
        tmp_path
        / "data"
        / "real"
        / "annotations"
        / "sample_intake.csv"
    )
    workbook_path = runtime / "manual_decision_workbook.csv"

    part_groups_path = (
        tmp_path
        / "data"
        / "real"
        / "annotations"
        / "part_groups.csv"
    )
    images_path = (
        tmp_path
        / "data"
        / "real"
        / "annotations"
        / "images.csv"
    )
    approval_path = (
        tmp_path
        / "data"
        / "real"
        / "annotations"
        / "approval_log.csv"
    )
    processed_images = (
        tmp_path / "data" / "real" / "processed" / "images"
    )

    write_csv(plan_path, plan_rows(), FIRST_BATCH_PLAN_COLUMNS)
    write_csv(queue_path, queue_rows(), SAMPLE_INTAKE_COLUMNS)
    write_csv(
        workbook_path,
        workbook_rows(),
        FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    )
    part_groups_path.parent.mkdir(parents=True, exist_ok=True)
    part_groups_path.write_text(
        "baseline-groups\n",
        encoding="utf-8",
    )
    images_path.write_text(
        "baseline-images\n",
        encoding="utf-8",
    )
    approval_path.write_text(
        "baseline-approval\n",
        encoding="utf-8",
    )
    processed_images.mkdir(parents=True, exist_ok=True)
    (processed_images / "existing.png").write_bytes(
        b"existing-image"
    )

    validator_paths = {
        "PROJECT_ROOT": tmp_path,
        "FIRST_BATCH_PLAN_PATH": plan_path,
        "REAL_SAMPLE_INTAKE_PATH": queue_path,
        "FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH": workbook_path,
        "FIRST_BATCH_REVIEW_RUNTIME_DIRECTORY": runtime,
        "APPLICATION_PLAN_PATH": (
            runtime / "manual_decision_application_plan.csv"
        ),
        "APPLICATION_VALIDATION_STATUS_PATH": (
            runtime / "manual_decision_application_validation.json"
        ),
        "APPLICATION_VALIDATION_SUMMARY_PATH": (
            runtime / "manual_decision_application_validation.md"
        ),
    }
    for name, value in validator_paths.items():
        monkeypatch.setattr(validation, name, value)

    application_paths = {
        "PROJECT_ROOT": tmp_path,
        "REAL_PART_GROUPS_PATH": part_groups_path,
        "REAL_IMAGES_PATH": images_path,
        "REAL_SAMPLE_INTAKE_PATH": queue_path,
        "REAL_APPROVAL_LOG_PATH": approval_path,
        "REAL_PROCESSED_IMAGES_DIRECTORY": processed_images,
        "APPLICATION_VALIDATION_STATUS_PATH": validator_paths[
            "APPLICATION_VALIDATION_STATUS_PATH"
        ],
        "APPLICATION_STATUS_PATH": (
            runtime / "manual_decision_application_status.json"
        ),
        "APPLICATION_SUMMARY_PATH": (
            runtime / "manual_decision_application_summary.md"
        ),
    }
    for name, value in application_paths.items():
        monkeypatch.setattr(application, name, value)

    live_files = (
        part_groups_path,
        images_path,
        queue_path,
        approval_path,
        (
            tmp_path
            / "data"
            / "real"
            / "processed"
            / "real_image_manifest.csv"
        ),
        (
            tmp_path
            / "reports"
            / "real_dataset"
            / "sample_intake_review.json"
        ),
        (
            tmp_path
            / "reports"
            / "real_dataset"
            / "sample_intake_review.md"
        ),
        (
            tmp_path
            / "reports"
            / "real_dataset"
            / "sample_intake_apply.json"
        ),
        (
            tmp_path
            / "reports"
            / "real_dataset"
            / "sample_intake_apply.md"
        ),
        (
            tmp_path
            / "reports"
            / "real_dataset"
            / "intake_validation.json"
        ),
        (
            tmp_path
            / "reports"
            / "real_dataset"
            / "intake_validation.md"
        ),
    )
    monkeypatch.setattr(
        application,
        "LIVE_FILE_PATHS",
        live_files,
    )

    def fake_load_review_inputs():
        queue = pd.read_csv(
            queue_path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        return (
            queue,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            [],
        )

    def fake_build_review_report(
        intake,
        part_groups,
        images,
        approval_log,
        initial_errors=None,
    ):
        errors = list(initial_errors or [])
        items = []
        for row in intake.to_dict(orient="records"):
            items.append(
                {
                    "intake_id": row["intake_id"],
                    "decision": row["decision"],
                    "part_group_id": row["part_group_id"],
                    "image_id": (
                        f"{row['part_group_id']}_{row['view']}"
                    ),
                    "staging_path": row["staging_path"],
                    "status": "PASS",
                    "metrics": {
                        "width": 800,
                        "height": 600,
                        "format": "JPEG",
                    },
                    "errors": [],
                    "warnings": [],
                }
            )
        return {
            "status": "PASS" if not errors else "FAIL",
            "readiness": (
                "REVIEW_READY" if not errors else "REVIEW_BLOCKED"
            ),
            "counts": {
                "rows": len(intake),
                "pending": int(
                    (intake["decision"] == "pending").sum()
                ),
                "approved": int(
                    (intake["decision"] == "approved").sum()
                ),
                "rejected": int(
                    (intake["decision"] == "rejected").sum()
                ),
            },
            "items": items,
            "errors": errors,
            "warnings": [],
        }

    monkeypatch.setattr(
        validation,
        "load_review_inputs",
        fake_load_review_inputs,
    )
    monkeypatch.setattr(
        validation,
        "build_review_report",
        fake_build_review_report,
    )

    return {
        "root": tmp_path,
        "runtime": runtime,
        "plan": plan_path,
        "queue": queue_path,
        "workbook": workbook_path,
        "part_groups": part_groups_path,
        "images": images_path,
        "approval": approval_path,
        "processed_images": processed_images,
    }


def test_validation_builds_ready_plan(isolated_step0099):
    plan, report = validation.build_manual_decision_application_plan()

    assert report["status"] == "PASS"
    assert report["readiness"] == "READY_TO_APPLY"
    assert report["counts"]["approved_decisions"] == 1
    assert report["counts"]["rejected_decisions"] == 1
    assert report["plan_id"]
    assert list(plan["validation_status"]) == ["READY", "READY"]


def test_blank_decision_remains_non_applicable(isolated_step0099):
    write_csv(
        isolated_step0099["workbook"],
        workbook_rows(("approved", "")),
        FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    )

    _, report = validation.build_manual_decision_application_plan()

    assert report["status"] == "PASS"
    assert report["readiness"] == "MANUAL_DECISIONS_REQUIRED"
    assert report["counts"]["pending_decisions"] == 1
    assert report["plan_id"] == ""


def test_rejected_decision_requires_reason(isolated_step0099):
    rows = workbook_rows()
    rows[1]["rejection_reason"] = ""
    write_csv(
        isolated_step0099["workbook"],
        rows,
        FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    )

    _, report = validation.build_manual_decision_application_plan()

    assert report["status"] == "FAIL"
    assert any(
        "require a rejection reason" in error
        for error in report["errors"]
    )


def test_immutable_workbook_change_is_blocked(isolated_step0099):
    rows = workbook_rows()
    rows[0]["part_category"] = "alternator"
    write_csv(
        isolated_step0099["workbook"],
        rows,
        FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    )

    _, report = validation.build_manual_decision_application_plan()

    assert report["status"] == "FAIL"
    assert any(
        "differs from the live queue" in error
        for error in report["errors"]
    )


def test_controlled_application_delegates_validated_decisions(
    isolated_step0099,
):
    validation_report = validation.validate_manual_decisions(
        write_outputs=True
    )
    assert validation_report["readiness"] == "READY_TO_APPLY"

    def fake_apply():
        queue = pd.read_csv(
            isolated_step0099["queue"],
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        decisions = dict(
            zip(queue["intake_id"], queue["decision"])
        )
        assert decisions == {
            "intake_000001": "approved",
            "intake_000002": "rejected",
        }
        assert queue.loc[
            queue["intake_id"] == "intake_000001",
            "notes",
        ].iloc[0] == "capture note | operator note 1"

        remaining = queue.iloc[0:0].copy()
        write_csv(
            isolated_step0099["queue"],
            remaining.to_dict(orient="records"),
            SAMPLE_INTAKE_COLUMNS,
        )
        return {
            "status": "PASS",
            "result": "APPLIED",
            "counts": {
                "approved": 1,
                "rejected": 1,
                "remaining_pending": 0,
            },
            "items": [],
            "errors": [],
        }

    report = application.apply_manual_decisions(
        apply_callable=fake_apply
    )

    assert report["status"] == "PASS"
    assert report["result"] == "APPLIED"
    assert report["rollback_performed"] == "NO"
    remaining = pd.read_csv(
        isolated_step0099["queue"],
        dtype=str,
        keep_default_na=False,
    )
    assert remaining.empty


def test_stale_validation_plan_blocks_application(
    isolated_step0099,
):
    validation.validate_manual_decisions(write_outputs=True)
    rows = workbook_rows()
    rows[0]["operator_notes"] = "changed after validation"
    write_csv(
        isolated_step0099["workbook"],
        rows,
        FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    )
    before = isolated_step0099["queue"].read_bytes()

    with pytest.raises(
        application.ManualDecisionApplicationError,
        match="saved validation plan is stale",
    ):
        application.apply_manual_decisions(
            apply_callable=lambda: {}
        )

    assert isolated_step0099["queue"].read_bytes() == before


def test_downstream_failure_restores_all_live_state(
    isolated_step0099,
):
    validation.validate_manual_decisions(write_outputs=True)

    original_queue = isolated_step0099["queue"].read_bytes()
    original_groups = isolated_step0099["part_groups"].read_bytes()
    original_image = (
        isolated_step0099["processed_images"] / "existing.png"
    ).read_bytes()

    def failing_apply():
        isolated_step0099["part_groups"].write_text(
            "changed-groups\n",
            encoding="utf-8",
        )
        (
            isolated_step0099["processed_images"] / "new.png"
        ).write_bytes(b"new-image")
        raise RuntimeError("forced downstream failure")

    with pytest.raises(
        application.ManualDecisionApplicationError,
        match="rolled back",
    ):
        application.apply_manual_decisions(
            apply_callable=failing_apply
        )

    assert isolated_step0099["queue"].read_bytes() == original_queue
    assert (
        isolated_step0099["part_groups"].read_bytes()
        == original_groups
    )
    assert (
        isolated_step0099["processed_images"] / "existing.png"
    ).read_bytes() == original_image
    assert not (
        isolated_step0099["processed_images"] / "new.png"
    ).exists()

    status = json.loads(
        application.APPLICATION_STATUS_PATH.read_text(
            encoding="utf-8"
        )
    )
    assert status["result"] == "ROLLED_BACK"
    assert status["rollback_performed"] == "YES"
