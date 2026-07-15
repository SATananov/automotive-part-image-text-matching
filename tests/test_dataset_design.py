import csv
from pathlib import Path

from src.dataset_config import (
    LABELS,
    METADATA_COLUMNS,
    PART_CATEGORIES,
    PART_FAMILIES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "metadata.csv"


def test_expected_labels_are_defined() -> None:
    assert LABELS == (
        "MATCH",
        "PARTIAL_MATCH",
        "MISMATCH",
    )


def test_each_family_contains_two_categories() -> None:
    for categories in PART_FAMILIES.values():
        assert len(categories) == 2


def test_part_categories_are_unique() -> None:
    assert len(PART_CATEGORIES) == 10
    assert len(PART_CATEGORIES) == len(set(PART_CATEGORIES))


def test_metadata_file_has_expected_header() -> None:
    with METADATA_PATH.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as metadata_file:
        header = next(csv.reader(metadata_file))

    assert tuple(header) == METADATA_COLUMNS
