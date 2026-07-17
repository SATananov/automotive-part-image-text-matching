from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import src.build_first_batch_capture_dashboard as dashboard
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS,
    FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS,
    REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    SAMPLE_INTAKE_COLUMNS,
)
from src.verify_step_009_6 import (
    DASHBOARD_GUIDE_PATH,
    build_verification_report,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color).save(path)


def write_csv(
    path: Path,
    rows: list[dict[str, str]],
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def configure_dashboard_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    annotations = tmp_path / "data" / "real" / "annotations"
    processed = tmp_path / "data" / "real" / "processed"
    reports = tmp_path / "reports" / "real_dataset"
    inbox = tmp_path / "data" / "real" / "capture_inbox" / "batch_001"
    originals = tmp_path / "data" / "real" / "originals" / "batch_001"
    staging = tmp_path / "data" / "real" / "staging"
    for path in (annotations, processed, reports, inbox, originals, staging):
        path.mkdir(parents=True, exist_ok=True)

    file_map = annotations / "first_batch_capture_file_map.csv"
    capture_inventory = processed / "first_batch_capture_inventory.csv"
    draft = processed / "first_batch_review_queue_draft.csv"
    queue = annotations / "sample_intake.csv"
    approval = processed / "approval_log.csv"
    manifest = processed / "real_image_manifest.csv"
    groups = annotations / "part_groups.csv"
    images = annotations / "images.csv"
    progress = processed / "first_batch_capture_progress.csv"
    html_report = reports / "first_batch_capture_dashboard.html"
    json_report = reports / "first_batch_capture_dashboard.json"
    md_report = reports / "first_batch_capture_progress_summary.md"
    guide = reports / "first_batch_capture_dashboard_and_progress_tracking.md"

    mapped_rows = [
        {
            "batch_id": "batch_001",
            "batch_item_id": "batch_001_001",
            "intake_id": "intake_000001",
            "capture_filename": "real_starter_001_front.jpg",
            "part_group_id": "real_starter_001",
            "part_category": "starter",
            "view": "front",
            "staging_path": "data/real/staging/intake_000001.jpg",
        },
        {
            "batch_id": "batch_001",
            "batch_item_id": "batch_001_002",
            "intake_id": "intake_000002",
            "capture_filename": "real_starter_001_detail.jpg",
            "part_group_id": "real_starter_001",
            "part_category": "starter",
            "view": "detail",
            "staging_path": "data/real/staging/intake_000002.jpg",
        },
    ]
    inventory_rows = []
    for row in mapped_rows:
        inventory_rows.append(
            {
                "batch_id": row["batch_id"],
                "batch_item_id": row["batch_item_id"],
                "intake_id": row["intake_id"],
                "staging_path": row["staging_path"],
                "part_group_id": row["part_group_id"],
                "part_family": "electrical",
                "part_category": row["part_category"],
                "view": row["view"],
                "source": "warehouse_photo",
                "match_description": "Automotive starter.",
                "partial_description": "Automotive alternator.",
                "mismatch_description": "Automotive brake disc.",
                "notes": "Test row.",
                "capture_source_path": "",
                "capture_source_status": "missing",
                "staging_status": "missing",
                "staged_sha256": "",
                "width": "",
                "height": "",
                "mode": "",
                "format": "",
                "review_status": "NOT_REVIEWED",
                "review_errors": "",
                "review_warnings": "",
                "ready_for_queue": "no",
            }
        )

    write_csv(file_map, mapped_rows, FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS)
    write_csv(
        capture_inventory,
        inventory_rows,
        FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS,
    )
    write_csv(draft, [], SAMPLE_INTAKE_COLUMNS)
    write_csv(queue, [], SAMPLE_INTAKE_COLUMNS)
    write_csv(approval, [], APPROVAL_LOG_COLUMNS)
    write_csv(manifest, [], REAL_IMAGE_INTAKE_MANIFEST_COLUMNS)
    groups.write_text("header\n", encoding="utf-8")
    images.write_text("header\n", encoding="utf-8")
    guide.write_text("Dashboard guide.\n", encoding="utf-8")

    mapping = {
        "PROJECT_ROOT": tmp_path,
        "FIRST_BATCH_CAPTURE_FILE_MAP_PATH": file_map,
        "FIRST_BATCH_CAPTURE_INBOX_DIRECTORY": inbox,
        "FIRST_BATCH_ORIGINALS_DIRECTORY": originals,
        "REAL_STAGING_DIRECTORY": staging,
        "FIRST_BATCH_CAPTURE_INVENTORY_PATH": capture_inventory,
        "FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH": draft,
        "REAL_SAMPLE_INTAKE_PATH": queue,
        "REAL_APPROVAL_LOG_PATH": approval,
        "REAL_IMAGE_MANIFEST_PATH": manifest,
        "REAL_PART_GROUPS_PATH": groups,
        "REAL_IMAGES_PATH": images,
        "FIRST_BATCH_CAPTURE_PROGRESS_PATH": progress,
        "DASHBOARD_HTML_PATH": html_report,
        "DASHBOARD_JSON_PATH": json_report,
        "DASHBOARD_MARKDOWN_PATH": md_report,
        "DASHBOARD_GUIDE_PATH": guide,
        "FIRST_BATCH_EXPECTED_IMAGES": 2,
    }
    for name, value in mapping.items():
        monkeypatch.setattr(dashboard, name, value)
    return {
        "file_map": file_map,
        "capture_inventory": capture_inventory,
        "draft": draft,
        "queue": queue,
        "approval": approval,
        "manifest": manifest,
        "groups": groups,
        "images": images,
        "inbox": inbox,
        "originals": originals,
        "staging": staging,
        "progress": progress,
        "html": html_report,
        "json": json_report,
        "markdown": md_report,
    }


def intake_row(decision: str = "pending") -> dict[str, str]:
    return {
        "intake_id": "intake_000001",
        "staging_path": "data/real/staging/intake_000001.jpg",
        "part_group_id": "real_starter_001",
        "part_family": "electrical",
        "part_category": "starter",
        "view": "front",
        "source": "warehouse_photo",
        "match_description": "Automotive starter.",
        "partial_description": "Automotive alternator.",
        "mismatch_description": "Automotive brake disc.",
        "decision": decision,
        "rejection_reason": "",
        "notes": "Test row.",
    }


def test_empty_dashboard_is_awaiting_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_dashboard_project(monkeypatch, tmp_path)

    progress, report = dashboard.build_progress()

    assert report["status"] == "PASS"
    assert report["readiness"] == "AWAITING_CAPTURE"
    assert report["overall_progress_percent"] == 0.0
    assert tuple(progress.columns) == FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS
    assert set(progress["pipeline_stage"]) == {"AWAITING_CAPTURE"}


def test_inbox_file_advances_to_captured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    write_image(paths["inbox"] / "real_starter_001_front.jpg", (10, 20, 30))

    progress, report = dashboard.build_progress()
    row = progress.loc[progress["intake_id"] == "intake_000001"].iloc[0]

    assert report["readiness"] == "CAPTURE_SESSION_IN_PROGRESS"
    assert row["pipeline_stage"] == "CAPTURED"
    assert row["next_action"] == "Run import-first-real-batch"


def test_original_file_advances_to_imported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    write_image(
        paths["originals"] / "real_starter_001_front.jpg",
        (20, 30, 40),
    )

    progress, _ = dashboard.build_progress()

    assert progress.loc[0, "pipeline_stage"] == "IMPORTED"
    assert progress.loc[0, "progress_percent"] == "29"


def test_staging_file_advances_to_staged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    write_image(paths["staging"] / "intake_000001.jpg", (30, 40, 50))

    progress, _ = dashboard.build_progress()

    assert progress.loc[0, "pipeline_stage"] == "STAGED"
    assert progress.loc[0, "next_action"] == (
        "Review capture inventory and queue draft"
    )


def test_review_ready_inventory_advances_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    frame = pd.read_csv(paths["capture_inventory"], dtype=str)
    frame.loc[0, "review_status"] = "PASS"
    frame.loc[0, "ready_for_queue"] = "yes"
    frame.to_csv(paths["capture_inventory"], index=False)

    progress, report = dashboard.build_progress()

    assert progress.loc[0, "pipeline_stage"] == "REVIEW_READY"
    assert report["readiness"] == "READY_FOR_MANUAL_REVIEW"


def test_queue_draft_is_review_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    write_csv(paths["draft"], [intake_row()], SAMPLE_INTAKE_COLUMNS)

    progress, _ = dashboard.build_progress()

    assert progress.loc[0, "queue_status"] == "DRAFT_READY"
    assert progress.loc[0, "pipeline_stage"] == "REVIEW_READY"


def test_live_pending_queue_advances_to_decision_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    write_csv(paths["queue"], [intake_row()], SAMPLE_INTAKE_COLUMNS)

    progress, report = dashboard.build_progress()

    assert progress.loc[0, "pipeline_stage"] == "QUEUED_FOR_DECISION"
    assert progress.loc[0, "queue_status"] == "QUEUED_PENDING"
    assert report["readiness"] == "REVIEW_IN_PROGRESS"


def test_rejected_approval_log_records_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    row = {column: "" for column in APPROVAL_LOG_COLUMNS}
    row.update(
        {
            "intake_id": "intake_000001",
            "decision": "rejected",
            "part_group_id": "real_starter_001",
        }
    )
    write_csv(paths["approval"], [row], APPROVAL_LOG_COLUMNS)

    progress, _ = dashboard.build_progress()

    assert progress.loc[0, "pipeline_stage"] == "DECISION_RECORDED"
    assert progress.loc[0, "decision_status"] == "REJECTED"
    assert "replacement" in progress.loc[0, "next_action"]


def test_manifest_approval_marks_approved_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    approval_row = {column: "" for column in APPROVAL_LOG_COLUMNS}
    approval_row.update(
        {
            "intake_id": "intake_000001",
            "decision": "approved",
            "part_group_id": "real_starter_001",
            "image_id": "real_starter_001_front",
        }
    )
    manifest_row = {column: "" for column in REAL_IMAGE_INTAKE_MANIFEST_COLUMNS}
    manifest_row.update(
        {
            "image_id": "real_starter_001_front",
            "part_group_id": "real_starter_001",
            "view": "front",
            "approved": "yes",
        }
    )
    write_csv(paths["approval"], [approval_row], APPROVAL_LOG_COLUMNS)
    write_csv(
        paths["manifest"],
        [manifest_row],
        REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    )

    progress, _ = dashboard.build_progress()

    assert progress.loc[0, "pipeline_stage"] == "APPROVED_DATASET"
    assert progress.loc[0, "progress_percent"] == "100"
    assert progress.loc[0, "next_action"] == "Complete"


def test_duplicate_intake_ids_block_dashboard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    frame = pd.read_csv(paths["file_map"], dtype=str)
    frame.loc[1, "intake_id"] = frame.loc[0, "intake_id"]
    frame.to_csv(paths["file_map"], index=False)

    _, report = dashboard.build_progress()

    assert report["status"] == "FAIL"
    assert report["readiness"] == "CAPTURE_DASHBOARD_BLOCKED"
    assert any("duplicate intake IDs" in error for error in report["errors"])


def test_dashboard_scan_does_not_change_protected_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)
    protected = [
        paths["file_map"],
        paths["capture_inventory"],
        paths["draft"],
        paths["queue"],
        paths["approval"],
        paths["manifest"],
        paths["groups"],
        paths["images"],
    ]
    before = {path: path.read_bytes() for path in protected}

    report = dashboard.build_first_batch_capture_dashboard()

    assert report["live_state_unchanged"] == "PASS"
    assert {path: path.read_bytes() for path in protected} == before


def test_dashboard_outputs_are_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_dashboard_project(monkeypatch, tmp_path)

    dashboard.build_first_batch_capture_dashboard()

    assert paths["progress"].is_file()
    assert paths["html"].is_file()
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    saved = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert saved["readiness"] == "AWAITING_CAPTURE"


def test_html_renderer_escapes_operator_values() -> None:
    row = {column: "" for column in FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS}
    row.update(
        {
            "part_category": "<script>alert(1)</script>",
            "view": "front",
            "capture_filename": "<unsafe>.jpg",
            "pipeline_stage": "AWAITING_CAPTURE",
            "progress_percent": "0",
            "next_action": "Capture <unsafe>.jpg",
        }
    )
    progress = pd.DataFrame([row], columns=FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS)
    report = {
        "readiness": "AWAITING_CAPTURE",
        "overall_progress_percent": 0,
        "counts": {
            "planned": 1,
            "captured": 0,
            "imported": 0,
            "staged": 0,
            "review_ready": 0,
            "approved": 0,
        },
        "category_progress": [
            {
                "part_category": "<unsafe>",
                "progress_percent": 0,
            }
        ],
        "errors": [],
    }

    rendered = dashboard.render_html(progress, report)

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "&lt;unsafe&gt;.jpg" in rendered


def test_project_cli_registers_dashboard_commands() -> None:
    assert "build-first-real-batch-dashboard" in COMMANDS
    assert "verify-step-009-6" in COMMANDS


def test_dashboard_guide_uses_semantic_filename() -> None:
    assert DASHBOARD_GUIDE_PATH.name == (
        "first_batch_capture_dashboard_and_progress_tracking.md"
    )
    assert "step_009_6" not in DASHBOARD_GUIDE_PATH.name


def test_current_dashboard_progress_has_twenty_unique_rows() -> None:
    frame = pd.read_csv(
        dashboard.FIRST_BATCH_CAPTURE_PROGRESS_PATH,
        dtype=str,
        keep_default_na=False,
    )

    assert tuple(frame.columns) == FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS
    assert len(frame) == 20
    assert frame["intake_id"].nunique() == 20


def test_current_step_009_6_verifier_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS", report["errors"]
