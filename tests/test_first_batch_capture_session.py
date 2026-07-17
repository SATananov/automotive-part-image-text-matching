from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import src.import_first_real_batch as local_import
import src.prepare_first_batch_capture_session as session
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    FIRST_BATCH_CAPTURE_SESSION_COLUMNS,
    FIRST_BATCH_PLAN_COLUMNS,
)
from src.verify_step_009_5 import (
    OPERATOR_GUIDE_PATH,
    build_verification_report,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color).save(path)


def write_plan_and_map(root: Path) -> tuple[Path, Path]:
    annotations = root / "data" / "real" / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    plan_path = annotations / "first_batch_plan.csv"
    map_path = annotations / "first_batch_capture_file_map.csv"
    rows = [
        {
            "batch_id": "batch_001",
            "batch_item_id": "batch_001_001",
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
            "notes": "Test row.",
        },
        {
            "batch_id": "batch_001",
            "batch_item_id": "batch_001_002",
            "intake_id": "intake_000002",
            "staging_path": "data/real/staging/intake_000002.jpg",
            "part_group_id": "real_starter_001",
            "part_family": "electrical",
            "part_category": "starter",
            "view": "detail",
            "source": "warehouse_photo",
            "match_description": "Automotive starter.",
            "partial_description": "Automotive alternator.",
            "mismatch_description": "Automotive brake disc.",
            "notes": "Test row.",
        },
    ]
    pd.DataFrame(rows, columns=FIRST_BATCH_PLAN_COLUMNS).to_csv(
        plan_path,
        index=False,
    )
    mapped = [
        {
            "batch_id": row["batch_id"],
            "batch_item_id": row["batch_item_id"],
            "intake_id": row["intake_id"],
            "capture_filename": (
                f"{row['part_group_id']}_{row['view']}.jpg"
            ),
            "part_group_id": row["part_group_id"],
            "part_category": row["part_category"],
            "view": row["view"],
            "staging_path": row["staging_path"],
        }
        for row in rows
    ]
    pd.DataFrame(
        mapped,
        columns=FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    ).to_csv(map_path, index=False)
    return plan_path, map_path


def configure_session_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    plan_path, map_path = write_plan_and_map(tmp_path)
    inbox = tmp_path / "data" / "real" / "capture_inbox" / "batch_001"
    originals = tmp_path / "data" / "real" / "originals" / "batch_001"
    staging = tmp_path / "data" / "real" / "staging"
    annotations = tmp_path / "data" / "real" / "annotations"
    processed = tmp_path / "data" / "real" / "processed"
    reports = tmp_path / "reports" / "real_dataset"
    for path in (inbox, originals, staging, processed, reports):
        path.mkdir(parents=True, exist_ok=True)

    protected = {
        "queue": annotations / "sample_intake.csv",
        "approval": processed / "approval_log.csv",
        "manifest": processed / "real_image_manifest.csv",
        "groups": annotations / "part_groups.csv",
        "images": annotations / "images.csv",
    }
    for path in protected.values():
        path.write_text("header\n", encoding="utf-8")

    worksheet = processed / "first_batch_capture_session.csv"
    json_report = reports / "first_batch_capture_session_readiness.json"
    md_report = reports / "first_batch_capture_session_readiness.md"
    mapping = {
        "PROJECT_ROOT": tmp_path,
        "FIRST_BATCH_CAPTURE_FILE_MAP_PATH": map_path,
        "FIRST_BATCH_PLAN_PATH": plan_path,
        "FIRST_BATCH_CAPTURE_INBOX_DIRECTORY": inbox,
        "FIRST_BATCH_ORIGINALS_DIRECTORY": originals,
        "FIRST_BATCH_CAPTURE_SESSION_PATH": worksheet,
        "REAL_STAGING_DIRECTORY": staging,
        "REAL_SAMPLE_INTAKE_PATH": protected["queue"],
        "REAL_APPROVAL_LOG_PATH": protected["approval"],
        "REAL_IMAGE_MANIFEST_PATH": protected["manifest"],
        "REAL_PART_GROUPS_PATH": protected["groups"],
        "REAL_IMAGES_PATH": protected["images"],
        "JSON_REPORT_PATH": json_report,
        "MARKDOWN_REPORT_PATH": md_report,
        "FIRST_BATCH_EXPECTED_IMAGES": 2,
        "FIRST_BATCH_EXPECTED_GROUPS": 1,
    }
    for name, value in mapping.items():
        monkeypatch.setattr(session, name, value)
    monkeypatch.setattr(local_import, "FIRST_BATCH_EXPECTED_IMAGES", 2)
    return {
        "plan": plan_path,
        "file_map": map_path,
        "inbox": inbox,
        "originals": originals,
        "staging": staging,
        "worksheet": worksheet,
        "json_report": json_report,
        "md_report": md_report,
        **protected,
    }


def protected_bytes(paths: dict[str, Path]) -> dict[str, bytes]:
    return {
        key: paths[key].read_bytes()
        for key in ("queue", "approval", "manifest", "groups", "images")
    }


def test_empty_session_is_awaiting_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)

    report = session.prepare_first_batch_capture_session()
    worksheet = pd.read_csv(paths["worksheet"], dtype=str)

    assert report["status"] == "PASS"
    assert report["readiness"] == "AWAITING_CAPTURE"
    assert report["counts"]["missing_files"] == 2
    assert report["next_capture"] == [
        "real_starter_001_front.jpg",
        "real_starter_001_detail.jpg",
    ]
    assert tuple(worksheet.columns) == FIRST_BATCH_CAPTURE_SESSION_COLUMNS
    assert len(worksheet) == 1


def test_one_view_reports_capture_in_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)
    write_image(
        paths["inbox"] / "real_starter_001_front.jpg",
        (10, 20, 30),
    )

    report = session.prepare_first_batch_capture_session()

    assert report["readiness"] == "CAPTURE_SESSION_IN_PROGRESS"
    assert report["next_capture"] == ["real_starter_001_detail.jpg"]


def test_complete_inbox_pair_is_ready_for_local_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)
    write_image(
        paths["inbox"] / "real_starter_001_front.jpg",
        (20, 30, 40),
    )
    write_image(
        paths["inbox"] / "real_starter_001_detail.jpg",
        (50, 60, 70),
    )

    report = session.prepare_first_batch_capture_session()
    worksheet = pd.read_csv(paths["worksheet"], dtype=str)

    assert report["readiness"] == "READY_FOR_LOCAL_IMPORT"
    assert worksheet.loc[0, "pair_status"] == "READY_FOR_LOCAL_IMPORT"
    assert worksheet.loc[0, "next_action"] == "Run import-first-real-batch"


def test_complete_original_pair_is_ready_for_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)
    write_image(
        paths["originals"] / "real_starter_001_front.jpg",
        (30, 40, 50),
    )
    write_image(
        paths["originals"] / "real_starter_001_detail.jpg",
        (60, 70, 80),
    )

    report = session.prepare_first_batch_capture_session()
    worksheet = pd.read_csv(paths["worksheet"], dtype=str)

    assert report["readiness"] == "READY_FOR_STAGING"
    assert report["counts"]["originals_available"] == 2
    assert worksheet.loc[0, "pair_status"] == "READY_FOR_STAGING"


def test_unreadable_capture_blocks_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)
    bad = paths["inbox"] / "real_starter_001_front.jpg"
    bad.write_text("not an image", encoding="utf-8")

    report = session.prepare_first_batch_capture_session()

    assert report["status"] == "FAIL"
    assert report["readiness"] == "CAPTURE_SESSION_BLOCKED"
    assert any("Cannot read capture file" in error for error in report["errors"])


def test_multiple_extensions_block_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)
    write_image(
        paths["inbox"] / "real_starter_001_front.jpg",
        (40, 50, 60),
    )
    write_image(
        paths["inbox"] / "real_starter_001_front.png",
        (70, 80, 90),
    )

    report = session.prepare_first_batch_capture_session()

    assert report["status"] == "FAIL"
    assert any("Multiple inbox files" in error for error in report["errors"])


def test_duplicate_capture_content_blocks_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)
    front = paths["inbox"] / "real_starter_001_front.jpg"
    detail = paths["inbox"] / "real_starter_001_detail.jpg"
    write_image(front, (80, 90, 100))
    detail.write_bytes(front.read_bytes())

    report = session.prepare_first_batch_capture_session()

    assert report["status"] == "FAIL"
    assert any("Duplicate capture content" in error for error in report["errors"])


def test_missing_detail_mapping_blocks_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)
    frame = pd.read_csv(paths["file_map"], dtype=str)
    frame.iloc[:1].to_csv(paths["file_map"], index=False)

    report = session.prepare_first_batch_capture_session()

    assert report["status"] == "FAIL"
    assert any("must contain exactly" in error for error in report["errors"])


def test_capture_session_does_not_change_protected_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)
    write_image(
        paths["inbox"] / "real_starter_001_front.jpg",
        (90, 100, 110),
    )
    before = protected_bytes(paths)

    report = session.prepare_first_batch_capture_session()

    assert report["live_state_unchanged"] == "PASS"
    assert protected_bytes(paths) == before


def test_json_and_markdown_reports_are_written(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_session_project(monkeypatch, tmp_path)

    session.prepare_first_batch_capture_session()
    saved = json.loads(paths["json_report"].read_text(encoding="utf-8"))
    markdown = paths["md_report"].read_text(encoding="utf-8")

    assert saved["readiness"] == "AWAITING_CAPTURE"
    assert "First Batch Capture Session Readiness" in markdown
    assert "real_starter_001_front.jpg" in markdown


def test_project_cli_registers_capture_session_commands() -> None:
    assert "prepare-first-real-batch-session" in COMMANDS
    assert "verify-step-009-5" in COMMANDS


def test_operator_guide_uses_semantic_filename() -> None:
    assert OPERATOR_GUIDE_PATH.name == "first_batch_operator_guide.md"
    assert "step_009_5" not in OPERATOR_GUIDE_PATH.name


def test_operator_guide_documents_exact_capture_pair() -> None:
    guide = OPERATOR_GUIDE_PATH.read_text(encoding="utf-8")

    assert "real_starter_001_front.jpg" in guide
    assert "real_starter_001_detail.jpg" in guide
    assert "No command in this guide approves samples automatically." in guide


def test_session_columns_have_one_row_per_physical_part() -> None:
    frame = pd.read_csv(
        session.FIRST_BATCH_CAPTURE_SESSION_PATH,
        dtype=str,
        keep_default_na=False,
    )

    assert tuple(frame.columns) == FIRST_BATCH_CAPTURE_SESSION_COLUMNS
    assert frame["part_group_id"].nunique() == len(frame)


def test_current_step_009_5_verifier_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS", report["errors"]
