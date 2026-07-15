from pathlib import Path

import pandas as pd

from src.dataset_config import METADATA_COLUMNS
from src.validate_development_dataset import (
    build_report,
    validate_identifiers,
    validate_labels_and_categories,
    validate_metadata_columns,
    validate_missing_values,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "metadata.csv"
)


def test_expected_metadata_columns_pass_validation() -> None:
    dataframe = pd.DataFrame(columns=METADATA_COLUMNS)

    assert validate_metadata_columns(dataframe) == []


def test_missing_value_is_detected() -> None:
    row = {
        column: "value"
        for column in METADATA_COLUMNS
    }
    row["description"] = None

    dataframe = pd.DataFrame([row])
    errors = validate_missing_values(dataframe)

    assert any(
        "description" in error
        and "missing values" in error
        for error in errors
    )


def test_duplicate_sample_id_is_detected() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "sample_id": "duplicate_sample",
                "image_id": "image_001",
                "label": "MATCH",
            },
            {
                "sample_id": "duplicate_sample",
                "image_id": "image_001",
                "label": "PARTIAL_MATCH",
            },
            {
                "sample_id": "unique_sample",
                "image_id": "image_001",
                "label": "MISMATCH",
            },
        ]
    )

    errors = validate_identifiers(dataframe)

    assert any(
        "duplicate sample IDs" in error
        for error in errors
    )


def test_invalid_label_and_family_pair_are_detected() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "sample_id": "sample_001",
                "label": "UNKNOWN",
                "part_category": "starter",
                "part_family": "braking",
            }
        ]
    )

    errors = validate_labels_and_categories(dataframe)

    assert any(
        "Invalid labels" in error
        for error in errors
    )
    assert any(
        "Category-family mismatch" in error
        for error in errors
    )


def test_complete_development_dataset_passes_validation() -> None:
    dataframe = pd.read_csv(METADATA_PATH)

    report = build_report(dataframe)

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["sample_count"] == 150
    assert report["image_count"] == 50
    assert report["part_group_count"] == 50