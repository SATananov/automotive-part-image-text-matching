from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.project_cli import COMMANDS
from src.real_dataset_config import (
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
)
import src.validate_real_dataset as real_validator
from src.verification.real_dataset_foundation import build_verification_report


def make_part_groups(
    rows: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(rows or [], columns=PART_GROUP_COLUMNS)


def make_images(
    rows: list[dict[str, str]] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(rows or [], columns=IMAGE_MANIFEST_COLUMNS)


def valid_group(
    group_id: str = "real_starter_101",
    approved: str = "yes",
) -> dict[str, str]:
    return {
        "part_group_id": group_id,
        "part_family": "electrical",
        "part_category": "starter",
        "match_description": "Automotive starter motor.",
        "partial_description": "Automotive alternator.",
        "mismatch_description": "Automotive brake disc.",
        "source": "warehouse_photo",
        "approved": approved,
        "notes": "",
    }


def valid_image(
    image_id: str = "real_starter_101_front",
    group_id: str = "real_starter_101",
    view: str = "front",
    approved: str = "yes",
    suffix: str = ".png",
) -> dict[str, str]:
    return {
        "image_id": image_id,
        "part_group_id": group_id,
        "image_path": (
            "data/real/processed/images/"
            f"{image_id}{suffix}"
        ),
        "view": view,
        "approved": approved,
    }


def configure_temporary_project(
    monkeypatch,
    tmp_path: Path,
) -> Path:
    processed_images = (
        tmp_path / "data" / "real" / "processed" / "images"
    )
    development_images = (
        tmp_path / "data" / "development" / "images"
    )
    processed_images.mkdir(parents=True)
    development_images.mkdir(parents=True)

    monkeypatch.setattr(real_validator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        real_validator,
        "REAL_PROCESSED_IMAGES_DIRECTORY",
        processed_images,
    )
    monkeypatch.setattr(
        real_validator,
        "DEVELOPMENT_IMAGES_DIRECTORY",
        development_images,
    )

    return processed_images


def write_image(path: Path, value: int = 64) -> None:
    Image.new("RGB", (32, 24), (value, value, value)).save(path)


def test_empty_real_intake_is_valid_foundation() -> None:
    report, manifest = real_validator.build_report(
        make_part_groups(),
        make_images(),
    )

    assert report["status"] == "PASS"
    assert report["readiness"] == "EMPTY_FOUNDATION"
    assert tuple(manifest.columns) == REAL_IMAGE_INTAKE_MANIFEST_COLUMNS
    assert manifest.empty


def test_real_group_requires_separate_identifier_namespace() -> None:
    invalid_group = valid_group(group_id="starter_101")
    errors = real_validator.validate_part_groups(
        make_part_groups([invalid_group])
    )

    assert any("real_<category>_<number>" in error for error in errors)
    assert any("real_" in error for error in errors)


def test_approved_group_descriptions_follow_label_semantics() -> None:
    group = valid_group()
    group["partial_description"] = "Automotive brake pad."
    group["mismatch_description"] = "Automotive alternator."

    errors = real_validator.validate_part_groups(
        make_part_groups([group])
    )

    assert any("partial_description" in error for error in errors)
    assert any("mismatch_description" in error for error in errors)


def test_image_annotation_rejects_path_outside_real_processed() -> None:
    groups = make_part_groups([valid_group()])
    image = valid_image()
    image["image_path"] = "data/development/images/starter_001_01.png"

    errors = real_validator.validate_image_annotations(
        make_images([image]),
        groups,
    )

    assert any("must be under" in error for error in errors)


def test_approved_image_requires_approved_group() -> None:
    groups = make_part_groups([valid_group(approved="no")])
    images = make_images([valid_image(approved="yes")])

    errors = real_validator.validate_image_annotations(images, groups)

    assert any("group that is not approved" in error for error in errors)


def test_manifest_records_hash_dimensions_and_format(
    monkeypatch,
    tmp_path: Path,
) -> None:
    processed_images = configure_temporary_project(
        monkeypatch,
        tmp_path,
    )
    image_path = processed_images / "real_starter_101_front.png"
    write_image(image_path)

    groups = make_part_groups([valid_group()])
    images = make_images([valid_image()])

    manifest, errors, warnings = real_validator.inspect_approved_images(
        images,
        groups,
    )

    assert errors == []
    assert len(warnings) == 1
    assert len(manifest) == 1
    assert len(manifest.loc[0, "sha256"]) == 64
    assert manifest.loc[0, "width"] == 32
    assert manifest.loc[0, "height"] == 24
    assert manifest.loc[0, "mode"] == "RGB"
    assert manifest.loc[0, "format"] == "PNG"


def test_exact_duplicate_images_are_rejected(
    monkeypatch,
    tmp_path: Path,
) -> None:
    processed_images = configure_temporary_project(
        monkeypatch,
        tmp_path,
    )
    front_path = processed_images / "real_starter_101_front.png"
    rear_path = processed_images / "real_starter_101_rear.png"
    write_image(front_path, value=90)
    rear_path.write_bytes(front_path.read_bytes())

    groups = make_part_groups([valid_group()])
    images = make_images(
        [
            valid_image(),
            valid_image(
                image_id="real_starter_101_rear",
                view="rear",
            ),
        ]
    )

    _, errors, _ = real_validator.inspect_approved_images(
        images,
        groups,
    )

    assert any("Duplicate approved image content" in error for error in errors)


def test_cross_group_duplicate_is_reported_as_leakage(
    monkeypatch,
    tmp_path: Path,
) -> None:
    processed_images = configure_temporary_project(
        monkeypatch,
        tmp_path,
    )
    first_path = processed_images / "real_starter_101_front.png"
    second_path = processed_images / "real_starter_102_front.png"
    write_image(first_path, value=110)
    second_path.write_bytes(first_path.read_bytes())

    groups = make_part_groups(
        [valid_group(), valid_group(group_id="real_starter_102")]
    )
    images = make_images(
        [
            valid_image(),
            valid_image(
                image_id="real_starter_102_front",
                group_id="real_starter_102",
            ),
        ]
    )

    _, errors, _ = real_validator.inspect_approved_images(
        images,
        groups,
    )

    assert any("Cross-group duplicate" in error for error in errors)


def test_real_image_cannot_duplicate_development_content(
    monkeypatch,
    tmp_path: Path,
) -> None:
    processed_images = configure_temporary_project(
        monkeypatch,
        tmp_path,
    )
    real_path = processed_images / "real_starter_101_front.png"
    development_path = (
        tmp_path
        / "data"
        / "development"
        / "images"
        / "starter_001_01.png"
    )
    write_image(real_path, value=130)
    development_path.write_bytes(real_path.read_bytes())

    groups = make_part_groups([valid_group()])
    images = make_images([valid_image()])

    _, errors, _ = real_validator.inspect_approved_images(
        images,
        groups,
    )

    assert any("duplicates development content" in error for error in errors)


def test_real_and_development_identifiers_must_be_disjoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(real_validator, "PROJECT_ROOT", tmp_path)
    metadata_path = tmp_path / "data" / "development" / "metadata.csv"
    metadata_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"part_group_id": "real_starter_101", "image_id": "real_starter_101_front"}]
    ).to_csv(metadata_path, index=False)

    errors = real_validator.validate_development_real_separation(
        make_part_groups([valid_group()]),
        make_images([valid_image()]),
    )

    assert any("part_group_id overlap" in error for error in errors)
    assert any("image_id overlap" in error for error in errors)


def test_step_009_cli_commands_are_registered() -> None:
    assert COMMANDS["validate-real-data"].module == (
        "src.validate_real_dataset"
    )
    assert COMMANDS["verify-real-dataset-foundation"].module == "src.verification.real_dataset_foundation"


def test_step_009_verification_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS"
    assert report["errors"] == []


def test_invalid_annotation_schema_returns_failure_without_crashing() -> None:
    malformed_groups = pd.DataFrame([{"part_group_id": "real_starter_101"}])
    malformed_images = pd.DataFrame([{"image_id": "real_starter_101_front"}])

    report, manifest = real_validator.build_report(
        malformed_groups,
        malformed_images,
        initial_errors=["Annotation schema mismatch."],
    )

    assert report["status"] == "FAIL"
    assert any("missing columns" in error for error in report["errors"])
    assert tuple(manifest.columns) == REAL_IMAGE_INTAKE_MANIFEST_COLUMNS
