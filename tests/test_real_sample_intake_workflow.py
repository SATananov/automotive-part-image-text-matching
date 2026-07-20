from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageDraw

import src.apply_real_sample_intake as intake_apply
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    SAMPLE_INTAKE_COLUMNS,
)
import src.review_real_sample_intake as intake_review
import src.validate_real_dataset as real_validator
from src.verification.sample_intake_workflow import build_verification_report


def empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def valid_intake_row(
    intake_id: str = "intake_000001",
    group_id: str = "real_starter_101",
    view: str = "front",
    decision: str = "pending",
) -> dict[str, str]:
    return {
        "intake_id": intake_id,
        "staging_path": f"data/real/staging/{intake_id}.jpg",
        "part_group_id": group_id,
        "part_family": "electrical",
        "part_category": "starter",
        "view": view,
        "source": "warehouse_photo",
        "match_description": "Automotive starter motor.",
        "partial_description": "Automotive alternator.",
        "mismatch_description": "Automotive brake disc.",
        "decision": decision,
        "rejection_reason": "",
        "notes": "",
    }


def write_pattern_image(
    path: Path,
    size: tuple[int, int] = (640, 480),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (40, 40, 40))
    draw = ImageDraw.Draw(image)
    block = max(8, min(size) // 8)

    for y in range(0, size[1], block):
        for x in range(0, size[0], block):
            if (x // block + y // block) % 2:
                draw.rectangle(
                    (x, y, x + block - 1, y + block - 1),
                    fill=(220, 180, 80),
                )

    image.save(path, quality=92)


def configure_temporary_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    paths = {
        "staging": tmp_path / "data" / "real" / "staging",
        "processed_images": (
            tmp_path / "data" / "real" / "processed" / "images"
        ),
        "annotations": tmp_path / "data" / "real" / "annotations",
        "processed": tmp_path / "data" / "real" / "processed",
        "development_images": (
            tmp_path / "data" / "development" / "images"
        ),
        "reports": tmp_path / "reports" / "real_dataset",
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    file_paths = {
        "queue": paths["annotations"] / "sample_intake.csv",
        "groups": paths["annotations"] / "part_groups.csv",
        "images": paths["annotations"] / "images.csv",
        "approval_log": paths["processed"] / "approval_log.csv",
        "manifest": paths["processed"] / "real_image_manifest.csv",
        "development_metadata": (
            tmp_path / "data" / "development" / "metadata.csv"
        ),
        "review_json": paths["reports"] / "sample_intake_review.json",
        "review_md": paths["reports"] / "sample_intake_review.md",
        "apply_json": paths["reports"] / "sample_intake_apply.json",
        "apply_md": paths["reports"] / "sample_intake_apply.md",
        "validation_json": paths["reports"] / "intake_validation.json",
        "validation_md": paths["reports"] / "intake_validation.md",
    }

    empty_frame(SAMPLE_INTAKE_COLUMNS).to_csv(
        file_paths["queue"], index=False
    )
    empty_frame(PART_GROUP_COLUMNS).to_csv(
        file_paths["groups"], index=False
    )
    empty_frame(IMAGE_MANIFEST_COLUMNS).to_csv(
        file_paths["images"], index=False
    )
    empty_frame(APPROVAL_LOG_COLUMNS).to_csv(
        file_paths["approval_log"], index=False
    )
    empty_frame(REAL_IMAGE_INTAKE_MANIFEST_COLUMNS).to_csv(
        file_paths["manifest"], index=False
    )
    pd.DataFrame(columns=("part_group_id", "image_id")).to_csv(
        file_paths["development_metadata"], index=False
    )

    review_mapping = {
        "PROJECT_ROOT": tmp_path,
        "REAL_STAGING_DIRECTORY": paths["staging"],
        "DEVELOPMENT_IMAGES_DIRECTORY": paths["development_images"],
        "REAL_IMAGE_MANIFEST_PATH": file_paths["manifest"],
        "REAL_SAMPLE_INTAKE_PATH": file_paths["queue"],
        "REAL_PART_GROUPS_PATH": file_paths["groups"],
        "REAL_IMAGES_PATH": file_paths["images"],
        "REAL_APPROVAL_LOG_PATH": file_paths["approval_log"],
        "JSON_REPORT_PATH": file_paths["review_json"],
        "MARKDOWN_REPORT_PATH": file_paths["review_md"],
    }

    for name, value in review_mapping.items():
        monkeypatch.setattr(intake_review, name, value)

    apply_mapping = {
        "PROJECT_ROOT": tmp_path,
        "REAL_PROCESSED_IMAGES_DIRECTORY": paths["processed_images"],
        "REAL_SAMPLE_INTAKE_PATH": file_paths["queue"],
        "REAL_PART_GROUPS_PATH": file_paths["groups"],
        "REAL_IMAGES_PATH": file_paths["images"],
        "REAL_APPROVAL_LOG_PATH": file_paths["approval_log"],
        "JSON_REPORT_PATH": file_paths["apply_json"],
        "MARKDOWN_REPORT_PATH": file_paths["apply_md"],
    }

    for name, value in apply_mapping.items():
        monkeypatch.setattr(intake_apply, name, value)

    validator_mapping = {
        "PROJECT_ROOT": tmp_path,
        "REAL_PROCESSED_IMAGES_DIRECTORY": paths["processed_images"],
        "DEVELOPMENT_IMAGES_DIRECTORY": paths["development_images"],
        "REAL_IMAGE_MANIFEST_PATH": file_paths["manifest"],
        "JSON_REPORT_PATH": file_paths["validation_json"],
        "MARKDOWN_REPORT_PATH": file_paths["validation_md"],
    }

    for name, value in validator_mapping.items():
        monkeypatch.setattr(real_validator, name, value)

    return {**paths, **file_paths}


def build_review(
    rows: list[dict[str, str]],
) -> dict[str, object]:
    return intake_review.build_review_report(
        pd.DataFrame(rows, columns=SAMPLE_INTAKE_COLUMNS),
        empty_frame(PART_GROUP_COLUMNS),
        empty_frame(IMAGE_MANIFEST_COLUMNS),
        empty_frame(APPROVAL_LOG_COLUMNS),
    )


def test_empty_queue_is_valid() -> None:
    report = intake_review.build_review_report(
        empty_frame(SAMPLE_INTAKE_COLUMNS),
        empty_frame(PART_GROUP_COLUMNS),
        empty_frame(IMAGE_MANIFEST_COLUMNS),
        empty_frame(APPROVAL_LOG_COLUMNS),
    )

    assert report["status"] == "PASS"
    assert report["readiness"] == "EMPTY_QUEUE"


def test_rejected_row_requires_reason() -> None:
    row = valid_intake_row(decision="rejected")
    report = build_review([row])

    assert report["status"] == "FAIL"
    assert any("rejection_reason" in error for error in report["errors"])


def test_staging_path_must_be_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_temporary_project(monkeypatch, tmp_path)
    row = valid_intake_row()
    row["staging_path"] = "../outside.jpg"
    report = build_review([row])

    assert any("Unsafe staging path" in error for error in report["errors"])


def test_staged_filename_must_match_intake_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    row = valid_intake_row()
    row["staging_path"] = "data/real/staging/wrong_name.jpg"
    write_pattern_image(paths["staging"] / "wrong_name.jpg")
    report = build_review([row])

    assert any("filename stem" in error for error in report["errors"])


def test_valid_staged_image_collects_quality_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    row = valid_intake_row()
    write_pattern_image(paths["staging"] / "intake_000001.jpg")
    report = build_review([row])

    assert report["status"] == "PASS"
    metrics = report["items"][0]["metrics"]
    assert metrics["width"] == 640
    assert metrics["height"] == 480
    assert len(metrics["sha256"]) == 64


def test_too_small_image_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    row = valid_intake_row()
    write_pattern_image(
        paths["staging"] / "intake_000001.jpg",
        size=(96, 96),
    )
    report = build_review([row])

    assert report["status"] == "FAIL"
    assert any("below the minimum" in error for error in report["errors"])


def test_duplicate_staged_content_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    first = valid_intake_row()
    second = valid_intake_row(
        intake_id="intake_000002",
        group_id="real_starter_102",
    )
    first_path = paths["staging"] / "intake_000001.jpg"
    second_path = paths["staging"] / "intake_000002.jpg"
    write_pattern_image(first_path)
    second_path.write_bytes(first_path.read_bytes())
    report = build_review([first, second])

    assert any(
        "Duplicate staged image content" in error
        for error in report["errors"]
    )


def test_development_duplicate_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    staged = paths["staging"] / "intake_000001.jpg"
    development = paths["development_images"] / "starter_001.jpg"
    write_pattern_image(staged)
    development.write_bytes(staged.read_bytes())
    report = build_review([valid_intake_row()])

    assert any(
        "duplicates development content" in error
        for error in report["errors"]
    )


def test_same_group_rows_must_use_consistent_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    first = valid_intake_row()
    second = valid_intake_row(
        intake_id="intake_000002",
        view="rear",
    )
    second["source"] = "different_source"
    write_pattern_image(paths["staging"] / "intake_000001.jpg")
    write_pattern_image(paths["staging"] / "intake_000002.jpg")
    report = build_review([first, second])

    assert any("conflicting metadata" in error for error in report["errors"])


def test_same_group_and_view_derives_duplicate_image_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    first = valid_intake_row()
    second = valid_intake_row(intake_id="intake_000002")
    write_pattern_image(paths["staging"] / "intake_000001.jpg")
    write_pattern_image(paths["staging"] / "intake_000002.jpg")
    report = build_review([first, second])

    assert any("Duplicate derived image_id" in error for error in report["errors"])


def test_processed_intake_id_cannot_be_reused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    write_pattern_image(paths["staging"] / "intake_000001.jpg")
    approval_log = empty_frame(APPROVAL_LOG_COLUMNS)
    approval_log.loc[0] = {
        column: "" for column in APPROVAL_LOG_COLUMNS
    }
    approval_log.loc[0, "intake_id"] = "intake_000001"
    report = intake_review.build_review_report(
        pd.DataFrame([valid_intake_row()], columns=SAMPLE_INTAKE_COLUMNS),
        empty_frame(PART_GROUP_COLUMNS),
        empty_frame(IMAGE_MANIFEST_COLUMNS),
        approval_log,
    )

    assert any(
        "already exists in the approval log" in error
        for error in report["errors"]
    )


def test_apply_with_empty_queue_is_noop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    before = paths["groups"].read_bytes()
    report = intake_apply.apply_intake()

    assert report["result"] == "NO_DECISIONS"
    assert paths["groups"].read_bytes() == before


def test_approved_intake_is_normalized_and_registered(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    row = valid_intake_row(decision="approved")
    pd.DataFrame([row], columns=SAMPLE_INTAKE_COLUMNS).to_csv(
        paths["queue"], index=False
    )
    source = paths["staging"] / "intake_000001.jpg"
    write_pattern_image(source)

    report = intake_apply.apply_intake(
        timestamp_factory=lambda: "2026-07-16T00:00:00Z"
    )

    destination = (
        paths["processed_images"] / "real_starter_101_front.png"
    )
    assert report["result"] == "APPLIED"
    assert destination.is_file()
    assert source.is_file()

    with Image.open(destination) as image:
        assert image.mode == "RGB"
        assert image.format == "PNG"

    groups = pd.read_csv(paths["groups"], dtype=str)
    images = pd.read_csv(paths["images"], dtype=str)
    log = pd.read_csv(paths["approval_log"], dtype=str)
    queue = pd.read_csv(paths["queue"], dtype=str)
    manifest = pd.read_csv(paths["manifest"], dtype=str)

    assert groups["part_group_id"].tolist() == ["real_starter_101"]
    assert images["image_id"].tolist() == ["real_starter_101_front"]
    assert log["decision"].tolist() == ["approved"]
    assert log["processed_at_utc"].tolist() == ["2026-07-16T00:00:00Z"]
    assert queue.empty
    assert len(manifest) == 1


def test_rejected_intake_is_logged_without_image(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    row = valid_intake_row(decision="rejected")
    row["rejection_reason"] = "Part is obscured."
    pd.DataFrame([row], columns=SAMPLE_INTAKE_COLUMNS).to_csv(
        paths["queue"], index=False
    )

    report = intake_apply.apply_intake(
        timestamp_factory=lambda: "2026-07-16T00:00:00Z"
    )

    log = pd.read_csv(paths["approval_log"], dtype=str)
    queue = pd.read_csv(paths["queue"], dtype=str)
    images = pd.read_csv(paths["images"], dtype=str)

    assert report["counts"]["rejected"] == 1
    assert log["decision"].tolist() == ["rejected"]
    assert log["rejection_reason"].tolist() == ["Part is obscured."]
    assert queue.empty
    assert images.empty
    assert not any(paths["processed_images"].glob("*.png"))


def test_apply_rolls_back_when_final_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_temporary_project(monkeypatch, tmp_path)
    row = valid_intake_row(decision="approved")
    pd.DataFrame([row], columns=SAMPLE_INTAKE_COLUMNS).to_csv(
        paths["queue"], index=False
    )
    write_pattern_image(paths["staging"] / "intake_000001.jpg")
    original_queue = paths["queue"].read_bytes()
    original_groups = paths["groups"].read_bytes()

    monkeypatch.setattr(
        intake_apply,
        "build_intake_validation_report",
        lambda groups, images: (
            {
                "status": "FAIL",
                "readiness": "INTAKE_IN_PROGRESS",
                "annotation_counts": {
                    "part_groups": 1,
                    "images": 1,
                    "approved_part_groups": 1,
                    "approved_images": 1,
                },
                "manifest_rows": 0,
                "category_distribution": {},
                "errors": ["forced validation failure"],
                "warnings": [],
            },
            empty_frame(REAL_IMAGE_INTAKE_MANIFEST_COLUMNS),
        ),
    )

    with pytest.raises(intake_apply.IntakeApplyError):
        intake_apply.apply_intake()

    assert paths["queue"].read_bytes() == original_queue
    assert paths["groups"].read_bytes() == original_groups
    assert not (
        paths["processed_images"] / "real_starter_101_front.png"
    ).exists()


def test_step_009_1_cli_commands_are_registered() -> None:
    assert COMMANDS["review-real-intake"].module == (
        "src.review_real_sample_intake"
    )
    assert COMMANDS["apply-real-intake"].module == (
        "src.apply_real_sample_intake"
    )
    assert COMMANDS["verify-sample-intake"].module == (
        "src.verification.sample_intake_workflow"
    )


def test_step_009_1_verifier_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS"
    assert report["errors"] == []
