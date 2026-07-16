from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageDraw

import src.stage_first_real_batch_capture as capture
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS,
    FIRST_BATCH_EXPECTED_IMAGES,
    SAMPLE_INTAKE_COLUMNS,
)
from src.verify_step_009_3 import build_verification_report
from tests.test_first_real_batch_dry_run import configure_batch_project


def write_unique_image(path: Path, seed: int, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "RGBA":
        image = Image.new("RGBA", (640, 480), (30, 40, 50, 255))
    else:
        image = Image.new("RGB", (640, 480), (30, 40, 50))
    draw = ImageDraw.Draw(image)
    for index in range(12):
        left = (seed * 17 + index * 31) % 560
        top = (seed * 23 + index * 19) % 400
        fill = (
            (seed * 41 + index * 13) % 255,
            (seed * 53 + index * 29) % 255,
            (seed * 67 + index * 37) % 255,
        )
        if mode == "RGBA":
            fill = (*fill, 255)
        draw.rectangle((left, top, left + 70, top + 55), fill=fill)
    image.save(path)


def hash_paths(directory: Path) -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = {}
    if not directory.exists():
        return hashes
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(str(path))
    return hashes


def configure_capture_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    paths = configure_batch_project(monkeypatch, tmp_path)
    originals = tmp_path / "data" / "real" / "originals" / "batch_001"
    inventory = paths["processed"] / "first_batch_capture_inventory.csv"
    queue_draft = paths["processed"] / "first_batch_review_queue_draft.csv"
    capture_json = paths["reports"] / "first_batch_capture_readiness.json"
    capture_md = paths["reports"] / "first_batch_capture_readiness.md"
    originals.mkdir(parents=True, exist_ok=True)

    mapping = {
        "PROJECT_ROOT": tmp_path,
        "FIRST_BATCH_PLAN_PATH": paths["plan"],
        "FIRST_BATCH_ORIGINALS_DIRECTORY": originals,
        "FIRST_BATCH_CAPTURE_INVENTORY_PATH": inventory,
        "FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH": queue_draft,
        "REAL_SAMPLE_INTAKE_PATH": paths["queue"],
        "REAL_APPROVAL_LOG_PATH": paths["approval_log"],
        "REAL_IMAGE_MANIFEST_PATH": paths["manifest"],
        "REAL_PART_GROUPS_PATH": paths["groups"],
        "REAL_IMAGES_PATH": paths["images"],
        "JSON_REPORT_PATH": capture_json,
        "MARKDOWN_REPORT_PATH": capture_md,
    }
    for name, value in mapping.items():
        monkeypatch.setattr(capture, name, value)

    monkeypatch.setattr(
        capture,
        "development_hashes",
        lambda: hash_paths(paths["development_images"]),
    )
    monkeypatch.setattr(
        capture,
        "existing_hashes_from_manifest",
        lambda: ({}, []),
    )
    return {
        **paths,
        "originals": originals,
        "inventory": inventory,
        "queue_draft": queue_draft,
        "capture_json": capture_json,
        "capture_md": capture_md,
    }


def test_empty_capture_directory_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    queue_before = paths["queue"].read_bytes()

    report = capture.stage_first_batch_capture()
    inventory = pd.read_csv(paths["inventory"], dtype=str)
    draft = pd.read_csv(paths["queue_draft"], dtype=str)

    assert report["status"] == "PASS"
    assert report["readiness"] == "AWAITING_CAPTURE"
    assert report["live_queue_unchanged"] == "PASS"
    assert tuple(inventory.columns) == FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS
    assert len(inventory) == FIRST_BATCH_EXPECTED_IMAGES
    assert tuple(draft.columns) == SAMPLE_INTAKE_COLUMNS
    assert draft.empty
    assert paths["queue"].read_bytes() == queue_before


def test_single_original_is_normalized_and_staged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    source = paths["originals"] / "intake_000001.png"
    write_unique_image(source, seed=1, mode="RGBA")
    source_before = source.read_bytes()

    report = capture.stage_first_batch_capture()
    staged = paths["staging"] / "intake_000001.jpg"
    draft = pd.read_csv(paths["queue_draft"], dtype=str)

    assert report["status"] == "PASS"
    assert report["readiness"] == "CAPTURE_IN_PROGRESS"
    assert report["counts"]["newly_staged"] == 1
    assert staged.is_file()
    with Image.open(staged) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (640, 480)
    assert source.read_bytes() == source_before
    assert list(draft["intake_id"]) == ["intake_000001"]
    assert set(draft["decision"]) == {"pending"}


def test_staging_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    write_unique_image(paths["originals"] / "intake_000001.jpg", seed=2)

    first = capture.stage_first_batch_capture()
    staged = paths["staging"] / "intake_000001.jpg"
    staged_before = staged.read_bytes()
    second = capture.stage_first_batch_capture()

    assert first["status"] == "PASS"
    assert second["status"] == "PASS"
    assert second["counts"]["newly_staged"] == 0
    assert staged.read_bytes() == staged_before


def test_multiple_source_candidates_block_without_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    write_unique_image(paths["originals"] / "intake_000001.jpg", seed=3)
    write_unique_image(paths["originals"] / "intake_000001.png", seed=4)

    report = capture.stage_first_batch_capture()

    assert report["status"] == "FAIL"
    assert any("Multiple original capture files" in error for error in report["errors"])
    assert not (paths["staging"] / "intake_000001.jpg").exists()


def test_existing_different_staging_file_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    source = paths["originals"] / "intake_000001.jpg"
    staged = paths["staging"] / "intake_000001.jpg"
    write_unique_image(source, seed=5)
    write_unique_image(staged, seed=6)
    staged_before = staged.read_bytes()

    report = capture.stage_first_batch_capture()

    assert report["status"] == "FAIL"
    assert any("already exists with different content" in error for error in report["errors"])
    assert staged.read_bytes() == staged_before


def test_duplicate_original_content_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    first = paths["originals"] / "intake_000001.jpg"
    second = paths["originals"] / "intake_000002.jpg"
    write_unique_image(first, seed=7)
    second.write_bytes(first.read_bytes())

    report = capture.stage_first_batch_capture()

    assert report["status"] == "FAIL"
    assert any("Duplicate original capture content" in error for error in report["errors"])
    assert not list(paths["staging"].glob("*.jpg"))


def test_unexpected_capture_file_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    write_unique_image(paths["originals"] / "unknown_part.jpg", seed=8)

    report = capture.stage_first_batch_capture()

    assert report["status"] == "FAIL"
    assert any("Unexpected image files" in error for error in report["errors"])


def test_development_duplicate_is_blocked_before_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    source = paths["originals"] / "intake_000001.jpg"
    development = paths["development_images"] / "reference.jpg"
    write_unique_image(source, seed=9)
    development.write_bytes(source.read_bytes())

    report = capture.stage_first_batch_capture()

    assert report["status"] == "FAIL"
    assert any("duplicates development content" in error for error in report["errors"])
    assert not (paths["staging"] / "intake_000001.jpg").exists()


def test_existing_staging_can_generate_pending_draft(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    write_unique_image(paths["staging"] / "intake_000001.jpg", seed=10)

    report = capture.stage_first_batch_capture()
    draft = pd.read_csv(paths["queue_draft"], dtype=str)

    assert report["status"] == "PASS"
    assert report["counts"]["originals_found"] == 0
    assert report["counts"]["queue_draft_rows"] == 1
    assert list(draft["decision"]) == ["pending"]


def test_live_queue_is_never_modified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    write_unique_image(paths["originals"] / "intake_000001.jpg", seed=11)
    before = paths["queue"].read_bytes()

    report = capture.stage_first_batch_capture()

    assert report["live_queue_unchanged"] == "PASS"
    assert paths["queue"].read_bytes() == before


def test_complete_capture_is_ready_for_manual_queue_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    plan = pd.read_csv(paths["plan"], dtype=str)
    for seed, intake_id in enumerate(plan["intake_id"], start=20):
        write_unique_image(paths["originals"] / f"{intake_id}.jpg", seed=seed)

    report = capture.stage_first_batch_capture()
    draft = pd.read_csv(paths["queue_draft"], dtype=str)

    assert report["status"] == "PASS"
    assert report["readiness"] == "READY_FOR_MANUAL_QUEUE_IMPORT"
    assert report["counts"]["staged_files"] == FIRST_BATCH_EXPECTED_IMAGES
    assert len(draft) == FIRST_BATCH_EXPECTED_IMAGES
    assert set(draft["decision"]) == {"pending"}


def test_failed_post_stage_preparation_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_capture_project(monkeypatch, tmp_path)
    write_unique_image(paths["originals"] / "intake_000001.jpg", seed=50)
    real_prepare = capture.batch_prepare.prepare_first_batch
    calls = {"count": 0}

    def failing_prepare() -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "status": "FAIL",
                "readiness": "PREPARATION_BLOCKED",
                "errors": ["forced preparation failure"],
                "warnings": [],
            }
        return real_prepare()

    monkeypatch.setattr(capture.batch_prepare, "prepare_first_batch", failing_prepare)
    report = capture.stage_first_batch_capture()

    assert report["status"] == "FAIL"
    assert any("forced preparation failure" in error for error in report["errors"])
    assert not (paths["staging"] / "intake_000001.jpg").exists()


def test_step_009_3_commands_are_registered() -> None:
    assert "stage-first-real-batch-capture" in COMMANDS
    assert "verify-step-009-3" in COMMANDS


def test_current_step_009_3_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
