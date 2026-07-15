from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from src.dataset_config import LABELS, PART_CATEGORIES, PART_FAMILIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "development" / "metadata.csv"


def read_rows() -> list[dict[str, str]]:
    with METADATA_PATH.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as metadata_file:
        return list(csv.DictReader(metadata_file))


def test_development_dataset_has_expected_size() -> None:
    rows = read_rows()

    assert len(rows) == 150
    assert len({row["image_id"] for row in rows}) == 50
    assert len({row["part_group_id"] for row in rows}) == 50


def test_development_dataset_is_balanced() -> None:
    rows = read_rows()
    label_counts = Counter(row["label"] for row in rows)

    assert label_counts == {
        "MATCH": 50,
        "PARTIAL_MATCH": 50,
        "MISMATCH": 50,
    }


def test_each_image_has_all_three_labels() -> None:
    rows = read_rows()
    labels_by_image: dict[str, set[str]] = {}

    for row in rows:
        labels_by_image.setdefault(row["image_id"], set()).add(row["label"])

    for labels in labels_by_image.values():
        assert labels == set(LABELS)


def test_all_image_files_exist() -> None:
    rows = read_rows()

    for image_path in {row["image_path"] for row in rows}:
        assert (PROJECT_ROOT / image_path).is_file()


def test_categories_and_families_are_valid() -> None:
    rows = read_rows()

    for row in rows:
        assert row["part_category"] in PART_CATEGORIES
        assert row["part_family"] in PART_FAMILIES
        assert row["part_category"] in PART_FAMILIES[row["part_family"]]


def test_sample_ids_are_unique() -> None:
    rows = read_rows()
    sample_ids = [row["sample_id"] for row in rows]

    assert len(sample_ids) == len(set(sample_ids))


def test_development_source_is_recorded() -> None:
    rows = read_rows()

    assert {
        row["source"]
        for row in rows
    } == {"generated_development"}