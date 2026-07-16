from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.dry_run_first_real_batch as batch_dry
import src.prepare_first_real_batch as batch_prepare
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PREVIEW_COLUMNS,
)
from src.verify_step_009_2 import build_verification_report
from tests.test_real_sample_intake_workflow import (
    configure_temporary_project,
    write_pattern_image,
)


COMMITTED_PLAN_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "real"
    / "annotations"
    / "first_batch_plan.csv"
)


def read_committed_plan() -> pd.DataFrame:
    return pd.read_csv(
        COMMITTED_PLAN_PATH,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def configure_batch_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    plan_path = paths["annotations"] / "first_batch_plan.csv"
    preview_path = paths["processed"] / "first_batch_queue_preview.csv"
    preparation_json = paths["reports"] / "first_batch_preparation.json"
    preparation_md = paths["reports"] / "first_batch_preparation.md"
    dry_run_json = paths["reports"] / "first_batch_dry_run.json"
    dry_run_md = paths["reports"] / "first_batch_dry_run.md"

    read_committed_plan().to_csv(plan_path, index=False, encoding="utf-8")

    prepare_mapping = {
        "PROJECT_ROOT": tmp_path,
        "FIRST_BATCH_PLAN_PATH": plan_path,
        "FIRST_BATCH_PREVIEW_PATH": preview_path,
        "REAL_PART_GROUPS_PATH": paths["groups"],
        "REAL_IMAGES_PATH": paths["images"],
        "REAL_SAMPLE_INTAKE_PATH": paths["queue"],
        "REAL_APPROVAL_LOG_PATH": paths["approval_log"],
        "JSON_REPORT_PATH": preparation_json,
        "MARKDOWN_REPORT_PATH": preparation_md,
    }
    for name, value in prepare_mapping.items():
        monkeypatch.setattr(batch_prepare, name, value)

    dry_mapping = {
        "PROJECT_ROOT": tmp_path,
        "REAL_PART_GROUPS_PATH": paths["groups"],
        "REAL_IMAGES_PATH": paths["images"],
        "REAL_SAMPLE_INTAKE_PATH": paths["queue"],
        "REAL_APPROVAL_LOG_PATH": paths["approval_log"],
        "REAL_IMAGE_MANIFEST_PATH": paths["manifest"],
        "REAL_PROCESSED_IMAGES_DIRECTORY": paths["processed_images"],
        "LIVE_DATA_FILES": (
            paths["groups"],
            paths["images"],
            paths["queue"],
            paths["approval_log"],
            paths["manifest"],
        ),
        "JSON_REPORT_PATH": dry_run_json,
        "MARKDOWN_REPORT_PATH": dry_run_md,
    }
    for name, value in dry_mapping.items():
        monkeypatch.setattr(batch_dry, name, value)

    return {
        **paths,
        "plan": plan_path,
        "preview": preview_path,
        "preparation_json": preparation_json,
        "preparation_md": preparation_md,
        "dry_run_json": dry_run_json,
        "dry_run_md": dry_run_md,
    }


def live_bytes(paths: dict[str, Path]) -> dict[str, bytes]:
    return {
        name: paths[name].read_bytes()
        for name in (
            "groups",
            "images",
            "queue",
            "approval_log",
            "manifest",
        )
    }


def test_committed_plan_is_balanced() -> None:
    plan = read_committed_plan()

    assert tuple(plan.columns) == FIRST_BATCH_PLAN_COLUMNS
    assert len(plan) == FIRST_BATCH_EXPECTED_IMAGES
    assert plan["part_category"].nunique() == 10
    assert plan["part_group_id"].nunique() == 10
    assert set(plan.groupby("part_group_id")["view"].nunique()) == {2}
    assert not batch_prepare.validate_plan(plan)


def test_empty_staging_is_valid_awaiting_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    report = batch_prepare.prepare_first_batch()
    preview = pd.read_csv(paths["preview"], dtype=str)

    assert report["status"] == "PASS"
    assert report["readiness"] == "AWAITING_CAPTURE"
    assert report["counts"]["captured_files"] == 0
    assert tuple(preview.columns) == FIRST_BATCH_PREVIEW_COLUMNS
    assert len(preview) == FIRST_BATCH_EXPECTED_IMAGES
    assert set(preview["file_present"]) == {"no"}


def test_captured_file_is_reviewed_without_queue_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    queue_before = paths["queue"].read_bytes()
    write_pattern_image(paths["staging"] / "intake_000001.jpg")

    report = batch_prepare.prepare_first_batch()
    preview = pd.read_csv(paths["preview"], dtype=str)
    first = preview.loc[preview["intake_id"] == "intake_000001"].iloc[0]

    assert report["status"] == "PASS"
    assert report["readiness"] == "CAPTURE_IN_PROGRESS"
    assert report["counts"]["captured_files"] == 1
    assert first["review_status"] in {"PASS", "WARN"}
    assert paths["queue"].read_bytes() == queue_before


def test_duplicate_intake_id_blocks_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    plan = pd.read_csv(paths["plan"], dtype=str)
    plan.loc[1, "intake_id"] = plan.loc[0, "intake_id"]
    plan.to_csv(paths["plan"], index=False)

    report = batch_prepare.prepare_first_batch()

    assert report["status"] == "FAIL"
    assert any("Duplicate intake_id" in error for error in report["errors"])


def test_category_family_mismatch_blocks_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    plan = pd.read_csv(paths["plan"], dtype=str)
    plan.loc[0, "part_family"] = "braking"
    plan.to_csv(paths["plan"], index=False)

    report = batch_prepare.prepare_first_batch()

    assert report["status"] == "FAIL"
    assert any("expected 'electrical'" in error for error in report["errors"])


def test_group_view_coverage_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    plan = pd.read_csv(paths["plan"], dtype=str)
    plan.loc[1, "view"] = "front"
    plan.to_csv(paths["plan"], index=False)

    report = batch_prepare.prepare_first_batch()

    assert report["status"] == "FAIL"
    assert any("must use views" in error for error in report["errors"])


def test_live_queue_conflict_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    plan = pd.read_csv(paths["plan"], dtype=str)
    row = batch_prepare.plan_row_to_intake(plan.iloc[0])
    row["source"] = "conflicting_source"
    pd.DataFrame([row]).to_csv(paths["queue"], index=False)

    report = batch_prepare.prepare_first_batch()

    assert report["status"] == "FAIL"
    assert any("conflicts on fields" in error for error in report["errors"])


def test_empty_dry_run_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    before = live_bytes(paths)

    report = batch_dry.run_dry_run()

    assert report["status"] == "PASS"
    assert report["result"] == "AWAITING_CAPTURE"
    assert report["immutability"] == "PASS"
    assert live_bytes(paths) == before


def test_one_captured_file_simulates_approval_without_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    write_pattern_image(paths["staging"] / "intake_000001.jpg")
    before = live_bytes(paths)

    report = batch_dry.run_dry_run()

    assert report["status"] == "PASS"
    assert report["result"] == "PARTIAL_BATCH_SIMULATED"
    assert report["counts"]["simulated_approved"] == 1
    assert report["counts"]["prospective_groups"] == 1
    assert report["counts"]["prospective_images"] == 1
    assert report["immutability"] == "PASS"
    assert live_bytes(paths) == before
    assert not list(paths["processed_images"].glob("*.png"))


def test_duplicate_captured_content_blocks_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    first = paths["staging"] / "intake_000001.jpg"
    second = paths["staging"] / "intake_000003.jpg"
    write_pattern_image(first)
    second.write_bytes(first.read_bytes())

    report = batch_dry.run_dry_run()

    assert report["status"] == "FAIL"
    assert report["result"] in {"PREPARATION_BLOCKED", "REVIEW_BLOCKED"}
    assert any("Duplicate staged image content" in error for error in report["errors"])


def test_development_duplicate_blocks_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    staged = paths["staging"] / "intake_000001.jpg"
    development = paths["development_images"] / "starter_reference.jpg"
    write_pattern_image(staged)
    development.write_bytes(staged.read_bytes())

    report = batch_dry.run_dry_run()

    assert report["status"] == "FAIL"
    assert any("duplicates development content" in error for error in report["errors"])


def test_preparation_does_not_copy_plan_into_live_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_batch_project(monkeypatch, tmp_path)
    batch_prepare.prepare_first_batch()
    queue = pd.read_csv(paths["queue"], dtype=str)

    assert queue.empty


def test_step_009_2_commands_are_registered() -> None:
    assert "prepare-first-real-batch" in COMMANDS
    assert "dry-run-first-real-batch" in COMMANDS
    assert "verify-step-009-2" in COMMANDS


def test_current_step_009_2_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
