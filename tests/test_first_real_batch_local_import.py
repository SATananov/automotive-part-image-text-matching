from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import src.import_first_real_batch as local_import
import src.stage_first_real_batch_capture as capture
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    FIRST_BATCH_LOCAL_IMPORT_INVENTORY_COLUMNS,
)
from src.verify_step_009_4 import build_verification_report
from tests.test_first_real_batch_capture import (
    configure_capture_project,
    write_unique_image,
)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color).save(path)


def write_plan_and_map(root: Path) -> tuple[Path, Path]:
    annotations = root / "data" / "real" / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    plan_path = annotations / "first_batch_plan.csv"
    map_path = annotations / "first_batch_capture_file_map.csv"
    plan_rows = [
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
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)
    map_rows = [
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
        for row in plan_rows
    ]
    pd.DataFrame(
        map_rows,
        columns=FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    ).to_csv(map_path, index=False)
    return plan_path, map_path


def configure_import_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    plan_path, map_path = write_plan_and_map(tmp_path)
    inbox = tmp_path / "data" / "real" / "capture_inbox" / "batch_001"
    originals = tmp_path / "data" / "real" / "originals" / "batch_001"
    staging = tmp_path / "data" / "real" / "staging"
    processed = tmp_path / "data" / "real" / "processed"
    reports = tmp_path / "reports" / "real_dataset"
    annotations = tmp_path / "data" / "real" / "annotations"
    inbox.mkdir(parents=True)
    originals.mkdir(parents=True)
    staging.mkdir(parents=True)
    processed.mkdir(parents=True)
    reports.mkdir(parents=True)

    live_paths = {
        "queue": annotations / "sample_intake.csv",
        "approval": processed / "approval_log.csv",
        "manifest": processed / "real_image_manifest.csv",
        "groups": annotations / "part_groups.csv",
        "images": annotations / "images.csv",
    }
    for path in live_paths.values():
        path.write_text("header\n", encoding="utf-8")

    inventory = processed / "first_batch_local_import_inventory.csv"
    json_report = reports / "first_batch_local_import_readiness.json"
    md_report = reports / "first_batch_local_import_readiness.md"
    mapping = {
        "PROJECT_ROOT": tmp_path,
        "FIRST_BATCH_PLAN_PATH": plan_path,
        "FIRST_BATCH_CAPTURE_FILE_MAP_PATH": map_path,
        "FIRST_BATCH_CAPTURE_INBOX_DIRECTORY": inbox,
        "FIRST_BATCH_ORIGINALS_DIRECTORY": originals,
        "FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH": inventory,
        "REAL_STAGING_DIRECTORY": staging,
        "REAL_SAMPLE_INTAKE_PATH": live_paths["queue"],
        "REAL_APPROVAL_LOG_PATH": live_paths["approval"],
        "REAL_IMAGE_MANIFEST_PATH": live_paths["manifest"],
        "REAL_PART_GROUPS_PATH": live_paths["groups"],
        "REAL_IMAGES_PATH": live_paths["images"],
        "JSON_REPORT_PATH": json_report,
        "MARKDOWN_REPORT_PATH": md_report,
        "FIRST_BATCH_EXPECTED_IMAGES": 2,
    }
    for name, value in mapping.items():
        monkeypatch.setattr(local_import, name, value)
    return {
        "plan": plan_path,
        "file_map": map_path,
        "inbox": inbox,
        "originals": originals,
        "staging": staging,
        "inventory": inventory,
        "json_report": json_report,
        "md_report": md_report,
        **live_paths,
    }


def live_bytes(paths: dict[str, Path]) -> dict[str, bytes]:
    return {
        key: paths[key].read_bytes()
        for key in ("queue", "approval", "manifest", "groups", "images")
    }


def test_empty_inbox_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    before = live_bytes(paths)

    report = local_import.import_first_real_batch()
    inventory = pd.read_csv(paths["inventory"], dtype=str)

    assert report["status"] == "PASS"
    assert report["readiness"] == "AWAITING_LOCAL_FILES"
    assert tuple(inventory.columns) == FIRST_BATCH_LOCAL_IMPORT_INVENTORY_COLUMNS
    assert len(inventory) == 2
    assert live_bytes(paths) == before


def test_descriptive_jpeg_is_copied_without_pixel_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    source = paths["inbox"] / "real_starter_001_front.jpg"
    write_image(source, (20, 40, 60))
    source_before = source.read_bytes()

    report = local_import.import_first_real_batch()
    destination = paths["originals"] / source.name

    assert report["status"] == "PASS"
    assert report["readiness"] == "LOCAL_IMPORT_IN_PROGRESS"
    assert destination.read_bytes() == source_before
    assert source.read_bytes() == source_before


def test_png_extension_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    source = paths["inbox"] / "real_starter_001_front.png"
    write_image(source, (50, 70, 90))

    report = local_import.import_first_real_batch()

    assert report["status"] == "PASS"
    assert (paths["originals"] / source.name).is_file()
    assert not (paths["originals"] / "real_starter_001_front.jpg").exists()


def test_import_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    source = paths["inbox"] / "real_starter_001_front.jpg"
    write_image(source, (80, 90, 100))

    first = local_import.import_first_real_batch()
    destination = paths["originals"] / source.name
    before = destination.read_bytes()
    second = local_import.import_first_real_batch()

    assert first["status"] == "PASS"
    assert second["status"] == "PASS"
    assert second["counts"]["newly_imported"] == 0
    assert destination.read_bytes() == before


def test_existing_different_original_blocks_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    source = paths["inbox"] / "real_starter_001_front.jpg"
    destination = paths["originals"] / source.name
    write_image(source, (1, 2, 3))
    write_image(destination, (4, 5, 6))
    before = destination.read_bytes()

    report = local_import.import_first_real_batch()

    assert report["status"] == "FAIL"
    assert any("different content" in error for error in report["errors"])
    assert destination.read_bytes() == before


def test_duplicate_local_capture_content_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    first = paths["inbox"] / "real_starter_001_front.jpg"
    second = paths["inbox"] / "real_starter_001_detail.jpg"
    write_image(first, (7, 8, 9))
    second.write_bytes(first.read_bytes())

    report = local_import.import_first_real_batch()

    assert report["status"] == "FAIL"
    assert any("Duplicate local capture content" in e for e in report["errors"])
    assert not list(paths["originals"].glob("*"))


def test_multiple_extensions_for_one_capture_are_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    write_image(paths["inbox"] / "real_starter_001_front.jpg", (10, 11, 12))
    write_image(paths["inbox"] / "real_starter_001_front.png", (13, 14, 15))

    report = local_import.import_first_real_batch()

    assert report["status"] == "FAIL"
    assert any("Multiple local capture files" in e for e in report["errors"])


def test_unexpected_semantic_filename_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    write_image(paths["inbox"] / "starter_front.jpg", (16, 17, 18))

    report = local_import.import_first_real_batch()

    assert report["status"] == "FAIL"
    assert any("Unexpected image files" in e for e in report["errors"])


def test_invalid_image_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    source = paths["inbox"] / "real_starter_001_front.jpg"
    source.write_text("not an image", encoding="utf-8")

    report = local_import.import_first_real_batch()

    assert report["status"] == "FAIL"
    assert any("Cannot read local capture" in e for e in report["errors"])


def test_transaction_rolls_back_after_copy_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    write_image(paths["inbox"] / "real_starter_001_front.jpg", (19, 20, 21))
    write_image(paths["inbox"] / "real_starter_001_detail.jpg", (22, 23, 24))
    original_copy = local_import.atomic_copy_bytes
    calls = 0

    def failing_copy(destination: Path, content: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced copy failure")
        original_copy(destination, content)

    monkeypatch.setattr(local_import, "atomic_copy_bytes", failing_copy)
    report = local_import.import_first_real_batch()

    assert report["status"] == "FAIL"
    assert any("transaction failed" in e for e in report["errors"])
    assert not list(paths["originals"].glob("*"))


def test_live_project_state_is_not_changed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    write_image(paths["inbox"] / "real_starter_001_front.jpg", (25, 26, 27))
    before = live_bytes(paths)

    report = local_import.import_first_real_batch()

    assert report["live_state_unchanged"] == "PASS"
    assert live_bytes(paths) == before


def test_filename_map_must_match_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_import_project(monkeypatch, tmp_path)
    file_map = pd.read_csv(paths["file_map"], dtype=str)
    file_map.loc[0, "capture_filename"] = "starter.jpg"
    file_map.to_csv(paths["file_map"], index=False)

    report = local_import.import_first_real_batch()

    assert report["status"] == "FAIL"
    assert any("must use filename" in e for e in report["errors"])


def test_project_cli_registers_local_import_commands() -> None:
    assert "import-first-real-batch" in COMMANDS
    assert "verify-step-009-4" in COMMANDS


def test_semantic_filename_can_be_staged_by_step_009_3(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    source = paths["originals"] / "real_starter_001_front.jpg"
    write_unique_image(source, seed=91)

    report = capture.stage_first_batch_capture()

    assert report["status"] == "PASS"
    assert (paths["staging"] / "intake_000001.jpg").is_file()


def test_current_step_009_4_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
