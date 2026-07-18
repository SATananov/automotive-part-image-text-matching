from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.finalize_first_real_dataset_ingestion as ingestion
import src.run_first_real_dataset_capture as capture
from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    FIRST_BATCH_PLAN_COLUMNS,
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    SAMPLE_INTAKE_COLUMNS,
)


def capture_report(captured=0, ready=0, planned=20):
    return {
        "status": "PASS",
        "counts": {
            "planned": planned,
            "captured": captured,
            "review_ready": ready,
        },
        "errors": [],
        "warnings": [],
    }


def activation_report():
    return {
        "status": "PASS",
        "readiness": "MANUAL_REVIEW_READY",
        "counts": {},
        "errors": [],
        "warnings": [],
    }


def manual_report(ready=0, pending=0):
    return {
        "status": "PASS",
        "readiness": "MANUAL_DECISION_WORKBOOK_READY",
        "counts": {
            "ready_decisions": ready,
            "pending_decisions": pending,
        },
        "errors": [],
        "warnings": [],
    }


def validation_report(readiness):
    return {
        "status": "PASS",
        "readiness": readiness,
        "plan_id": "plan-010" if readiness == "READY_TO_APPLY" else "",
        "errors": [],
        "warnings": [],
    }


def audit_report(
    readiness="MANUAL_DECISIONS_REQUIRED",
    *,
    approved=0,
    rejected=0,
    status="PASS",
    errors=None,
):
    return {
        "status": status,
        "readiness": readiness,
        "counts": {
            "planned_images": 20,
            "planned_groups": 10,
            "decided": approved + rejected,
            "approved": approved,
            "rejected": rejected,
            "remaining_queue": max(0, 20 - approved - rejected),
            "approved_groups": approved // 2,
            "complete_front_detail_groups": approved // 2,
            "approved_categories": approved // 2,
        },
        "remaining_queue_ids": [],
        "errors": errors or [],
        "warnings": [],
    }


def test_capture_workflow_keeps_approved_dataset_unchanged(
    monkeypatch,
):
    fingerprints = iter(["same", "same"])
    monkeypatch.setattr(
        capture,
        "approved_dataset_fingerprint",
        lambda: next(fingerprints),
    )
    monkeypatch.setattr(capture, "write_outputs", lambda report: None)

    report = capture.run_capture_workflow(
        capture_callable=lambda: capture_report(),
        activation_callable=activation_report,
        preparation_callable=lambda: manual_report(),
        validation_callable=lambda **kwargs: validation_report(
            "AWAITING_QUEUE_ACTIVATION"
        ),
        audit_callable=lambda: audit_report(),
    )

    assert report["status"] == "PASS"
    assert report["readiness"] == "AWAITING_CAPTURE"
    assert report["approved_dataset_unchanged"] == "PASS"


def test_capture_workflow_reports_ready_to_apply(monkeypatch):
    monkeypatch.setattr(
        capture,
        "approved_dataset_fingerprint",
        lambda: "same",
    )
    monkeypatch.setattr(capture, "write_outputs", lambda report: None)

    report = capture.run_capture_workflow(
        capture_callable=lambda: capture_report(20, 20),
        activation_callable=activation_report,
        preparation_callable=lambda: manual_report(20, 0),
        validation_callable=lambda **kwargs: validation_report(
            "READY_TO_APPLY"
        ),
        audit_callable=lambda: audit_report(),
    )

    assert report["readiness"] == "READY_TO_APPLY"


def test_finalize_blocks_before_apply(monkeypatch):
    called = {"apply": False}
    monkeypatch.setattr(
        ingestion,
        "write_runtime_outputs",
        lambda report: None,
    )
    monkeypatch.setattr(
        ingestion,
        "build_ingestion_audit",
        lambda: audit_report(),
    )

    def unexpected_apply():
        called["apply"] = True
        return {}

    with pytest.raises(
        ingestion.FirstRealDatasetIngestionNotReady,
        match="READY_TO_APPLY",
    ):
        ingestion.finalize_ingestion(
            validation_callable=lambda **kwargs: validation_report(
                "MANUAL_DECISIONS_REQUIRED"
            ),
            apply_callable=unexpected_apply,
        )

    assert called["apply"] is False


def test_finalize_complete_batch(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "write_runtime_outputs",
        lambda report: None,
    )
    monkeypatch.setattr(
        ingestion,
        "write_tracked_outputs",
        lambda report: None,
    )

    report = ingestion.finalize_ingestion(
        validation_callable=lambda **kwargs: validation_report(
            "READY_TO_APPLY"
        ),
        apply_callable=lambda: {
            "status": "PASS",
            "counts": {"approved": 20, "rejected": 0},
        },
        audit_callable=lambda: audit_report(
            "FIRST_BATCH_INGESTED",
            approved=20,
        ),
        snapshot_callable=lambda: {"snapshot": True},
        restore_callable=lambda snapshot: None,
    )

    assert report["result"] == "FIRST_BATCH_INGESTED"
    assert report["rollback_performed"] == "NO"


def test_finalize_reports_recap_required(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "write_runtime_outputs",
        lambda report: None,
    )
    monkeypatch.setattr(
        ingestion,
        "write_tracked_outputs",
        lambda report: None,
    )

    report = ingestion.finalize_ingestion(
        validation_callable=lambda **kwargs: validation_report(
            "READY_TO_APPLY"
        ),
        apply_callable=lambda: {
            "status": "PASS",
            "counts": {"approved": 18, "rejected": 2},
        },
        audit_callable=lambda: audit_report(
            "RECAPTURE_REQUIRED",
            approved=18,
            rejected=2,
        ),
        snapshot_callable=lambda: {"snapshot": True},
        restore_callable=lambda snapshot: None,
    )

    assert report["result"] == "APPLIED_WITH_RECAPTURE_REQUIRED"
    assert report["readiness"] == "RECAPTURE_REQUIRED"


def test_finalize_restores_on_audit_failure(monkeypatch):
    restored = []
    monkeypatch.setattr(
        ingestion,
        "write_runtime_outputs",
        lambda report: None,
    )

    with pytest.raises(
        ingestion.FirstRealDatasetIngestionError,
        match="post-application",
    ):
        ingestion.finalize_ingestion(
            validation_callable=lambda **kwargs: validation_report(
                "READY_TO_APPLY"
            ),
            apply_callable=lambda: {
                "status": "PASS",
                "counts": {"approved": 20, "rejected": 0},
            },
            audit_callable=lambda: audit_report(
                "INGESTION_AUDIT_BLOCKED",
                approved=20,
                status="FAIL",
                errors=["forced audit failure"],
            ),
            snapshot_callable=lambda: {"snapshot": True},
            restore_callable=restored.append,
        )

    assert restored == [{"snapshot": True}]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(
        path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


@pytest.fixture
def complete_audit_state(tmp_path, monkeypatch):
    paths = {
        "FIRST_BATCH_PLAN_PATH": tmp_path / "first_batch_plan.csv",
        "REAL_SAMPLE_INTAKE_PATH": tmp_path / "sample_intake.csv",
        "REAL_APPROVAL_LOG_PATH": tmp_path / "approval_log.csv",
        "REAL_PART_GROUPS_PATH": tmp_path / "part_groups.csv",
        "REAL_IMAGES_PATH": tmp_path / "images.csv",
        "REAL_IMAGE_MANIFEST_PATH": tmp_path / "manifest.csv",
    }

    plan_rows = []
    approval_rows = []
    image_rows = []
    manifest_rows = []

    for index, view in enumerate(("front", "detail"), start=1):
        intake_id = f"intake_{index:06d}"
        image_id = f"real_starter_001_{view}"
        relative = f"processed/{image_id}.png"
        image_path = tmp_path / relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"png")

        plan_rows.append(
            {
                "batch_id": "batch_001",
                "batch_item_id": f"item_{index}",
                "intake_id": intake_id,
                "staging_path": f"staging/{intake_id}.jpg",
                "part_group_id": "real_starter_001",
                "part_family": "starting_system",
                "part_category": "starter",
                "view": view,
                "source": "local_capture",
                "match_description": "Starter.",
                "partial_description": "Alternator.",
                "mismatch_description": "Brake disc.",
                "notes": "",
            }
        )
        approval_rows.append(
            {
                "intake_id": intake_id,
                "decision": "approved",
                "part_group_id": "real_starter_001",
                "image_id": image_id,
                "processed_image_path": relative,
                "sha256": f"hash-{index}",
                "width": "800",
                "height": "600",
                "mode": "RGB",
                "format": "PNG",
                "quality_status": "PASS",
                "processed_at_utc": "2026-07-18T00:00:00+00:00",
                "rejection_reason": "",
                "notes": "",
            }
        )
        image_rows.append(
            {
                "image_id": image_id,
                "part_group_id": "real_starter_001",
                "image_path": relative,
                "view": view,
                "approved": "yes",
            }
        )
        manifest_rows.append(
            {
                "image_id": image_id,
                "part_group_id": "real_starter_001",
                "image_path": relative,
                "part_family": "starting_system",
                "part_category": "starter",
                "view": view,
                "source": "local_capture",
                "approved": "yes",
                "sha256": f"hash-{index}",
                "file_size_bytes": "3",
                "width": "800",
                "height": "600",
                "mode": "RGB",
                "format": "PNG",
            }
        )

    write_csv(
        paths["FIRST_BATCH_PLAN_PATH"],
        plan_rows,
        FIRST_BATCH_PLAN_COLUMNS,
    )
    write_csv(
        paths["REAL_SAMPLE_INTAKE_PATH"],
        [],
        SAMPLE_INTAKE_COLUMNS,
    )
    write_csv(
        paths["REAL_APPROVAL_LOG_PATH"],
        approval_rows,
        APPROVAL_LOG_COLUMNS,
    )
    write_csv(
        paths["REAL_PART_GROUPS_PATH"],
        [
            {
                "part_group_id": "real_starter_001",
                "part_family": "starting_system",
                "part_category": "starter",
                "match_description": "Starter.",
                "partial_description": "Alternator.",
                "mismatch_description": "Brake disc.",
                "source": "local_capture",
                "approved": "yes",
                "notes": "",
            }
        ],
        PART_GROUP_COLUMNS,
    )
    write_csv(
        paths["REAL_IMAGES_PATH"],
        image_rows,
        IMAGE_MANIFEST_COLUMNS,
    )
    write_csv(
        paths["REAL_IMAGE_MANIFEST_PATH"],
        manifest_rows,
        REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    )

    monkeypatch.setattr(ingestion, "PROJECT_ROOT", tmp_path)
    for name, path in paths.items():
        monkeypatch.setattr(ingestion, name, path)

    return tmp_path


def test_audit_recognizes_complete_batch(complete_audit_state):
    report = ingestion.build_ingestion_audit()
    assert report["status"] == "PASS"
    assert report["readiness"] == "FIRST_BATCH_INGESTED"
    assert report["counts"]["approved"] == 2


def test_audit_blocks_missing_processed_image(
    complete_audit_state,
):
    (
        complete_audit_state
        / "processed"
        / "real_starter_001_detail.png"
    ).unlink()

    report = ingestion.build_ingestion_audit()
    assert report["status"] == "FAIL"
    assert report["readiness"] == "INGESTION_AUDIT_BLOCKED"
    assert any(
        "processed image is missing" in item
        for item in report["errors"]
    )
