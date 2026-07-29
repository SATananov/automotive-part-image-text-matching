from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data import COLUMNS, DATA_DIR, PROJECT_ROOT, check_split, load_split, unique_images


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_v2_split_counts() -> dict[str, int]:
    manifest = pd.read_csv(DATA_DIR / "dataset_v2_manifest.csv")
    return {str(key): int(value) for key, value in manifest["assigned_split"].value_counts().to_dict().items()}


def test_train_and_validation_have_dataset_v2_schema_and_size() -> None:
    train = load_split("train")
    validation = load_split("validation")
    imported = dataset_v2_split_counts()
    expected_train_images = 100 + imported["train"]
    expected_validation_images = 10 + imported["validation"]
    assert tuple(train.columns) == COLUMNS
    assert tuple(validation.columns) == COLUMNS
    assert len(train) == expected_train_images * 6
    assert len(validation) == expected_validation_images * 6
    assert train["image_id"].nunique() == expected_train_images
    assert validation["image_id"].nunique() == expected_validation_images


def test_normal_loader_refuses_locked_test_split() -> None:
    with pytest.raises(ValueError, match="Only train and validation"):
        load_split("test")


def test_validation_is_real_only_and_source_counts_are_exact() -> None:
    train_images = unique_images(load_split("train"))
    validation_images = unique_images(load_split("validation"))
    imported = dataset_v2_split_counts()
    assert train_images["source"].value_counts().to_dict() == {
        "dataset_v2": imported["train"],
        "generated": 50,
        "wikimedia": 50,
    }
    assert validation_images["source"].value_counts().to_dict() == {
        "dataset_v2": imported["validation"],
        "wikimedia": 10,
    }


def test_split_identity_overlap_is_zero() -> None:
    assert check_split(load_split("train"), load_split("validation")) == {
        "part_group_overlap": 0,
        "object_group_overlap": 0,
        "image_id_overlap": 0,
        "image_path_overlap": 0,
    }


def test_every_image_has_two_rows_for_each_label() -> None:
    for split in (load_split("train"), load_split("validation")):
        counts = split.groupby(["image_id", "label"]).size().unstack(fill_value=0)
        assert (counts == 2).all().all()
        assert set(split["label"].value_counts()) == {len(split) // 3}


def test_text_categories_are_exactly_balanced_across_labels() -> None:
    for split in (load_split("train"), load_split("validation")):
        table = pd.crosstab(split["text_category"], split["label"])
        assert table.shape == (10, 3)
        assert table.nunique().max() == 1


def test_exact_descriptions_do_not_cross_splits() -> None:
    train = load_split("train")
    validation = load_split("validation")
    assert train["description"].nunique() == 40
    assert validation["description"].nunique() == 20
    assert set(train["description"]).isdisjoint(validation["description"])


def test_all_development_images_open() -> None:
    frames = [unique_images(load_split("train")), unique_images(load_split("validation"))]
    for row in pd.concat(frames, ignore_index=True).itertuples(index=False):
        with Image.open(PROJECT_ROOT / row.image_path) as image:
            image.verify()


def test_image_manifest_covers_all_unique_images() -> None:
    manifest = pd.read_csv(DATA_DIR / "image_manifest.csv")
    imported = dataset_v2_split_counts()
    expected_total = 120 + imported["train"] + imported["validation"]
    assert len(manifest) == expected_total
    assert manifest["image_id"].nunique() == expected_total
    assert manifest["sha256"].nunique() == expected_total
    assert manifest.groupby(["split", "source"]).size().to_dict() == {
        ("test", "wikimedia"): 10,
        ("train", "generated"): 50,
        ("train", "dataset_v2"): imported["train"],
        ("train", "wikimedia"): 50,
        ("validation", "dataset_v2"): imported["validation"],
        ("validation", "wikimedia"): 10,
    }


def test_wikimedia_and_dataset_v2_provenance_inventories_are_complete() -> None:
    manifest = pd.read_csv(DATA_DIR / "image_manifest.csv")
    licenses = pd.read_csv(DATA_DIR / "licenses.csv")
    dataset_v2 = pd.read_csv(DATA_DIR / "dataset_v2_manifest.csv")
    wikimedia = manifest[manifest["source"].eq("wikimedia")]
    imported = manifest[manifest["source"].eq("dataset_v2")]
    assert len(licenses) == len(wikimedia) == 70
    assert len(dataset_v2) == len(imported)
    assert not dataset_v2.empty
    assert not dataset_v2["license_short_name"].fillna("").str.strip().eq("").any()
    per_category = dataset_v2.groupby(["assigned_split", "part_category"]).size().unstack(fill_value=0)
    assert per_category.loc["train"].nunique() == 1
    assert per_category.loc["validation"].nunique() == 1
    assert sum(dataset_v2["provider"].value_counts().to_dict().values()) == len(dataset_v2)
    assert set(licenses["local_path"]) == set(wikimedia["image_path"])
    assert set(dataset_v2["local_path"]) == set(imported["image_path"])
    for frame in (licenses, dataset_v2):
        for row in frame.itertuples(index=False):
            assert sha256(PROJECT_ROOT / row.local_path) == row.sha256


def test_test_lock_hash_matches_without_using_normal_loader() -> None:
    lock = json.loads((DATA_DIR / "test_lock.json").read_text(encoding="utf-8"))
    assert lock["test_locked"] is True
    assert lock["test_evaluation_permitted"] is False
    assert lock["test_rows"] == 60
    assert lock["test_images"] == 10
    assert sha256(DATA_DIR / "test.csv") == lock["test_sha256"]
