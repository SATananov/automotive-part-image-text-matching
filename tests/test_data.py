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


def test_train_and_validation_have_expected_schema_and_size() -> None:
    train = load_split("train")
    validation = load_split("validation")
    assert tuple(train.columns) == COLUMNS
    assert tuple(validation.columns) == COLUMNS
    assert len(train) == 300
    assert len(validation) == 30
    assert train["image_id"].nunique() == 100
    assert validation["image_id"].nunique() == 10


def test_normal_loader_refuses_locked_test_split() -> None:
    with pytest.raises(ValueError, match="Only train and validation"):
        load_split("test")


def test_validation_is_real_only_and_synthetic_images_are_train_only() -> None:
    train = load_split("train")
    validation = load_split("validation")
    assert set(train["source"]) == {"generated", "wikimedia"}
    assert set(validation["source"]) == {"wikimedia"}
    train_images = unique_images(train)
    assert len(train_images[train_images["source"].eq("generated")]) == 50
    assert len(train_images[train_images["source"].eq("wikimedia")]) == 50


def test_split_identity_overlap_is_zero() -> None:
    assert check_split(load_split("train"), load_split("validation")) == {
        "part_group_overlap": 0,
        "object_group_overlap": 0,
        "image_id_overlap": 0,
        "image_path_overlap": 0,
    }


def test_every_image_has_all_three_labels() -> None:
    for split in (load_split("train"), load_split("validation")):
        counts = split.groupby("image_id")["label"].nunique()
        assert (counts == 3).all()
        assert set(split["label"].value_counts()) == {len(split) // 3}


def test_text_categories_are_balanced_across_labels() -> None:
    for split in (load_split("train"), load_split("validation")):
        table = pd.crosstab(split["text_category"], split["label"])
        assert table.shape == (10, 3)
        assert table.nunique().nunique() == 1


def test_exact_descriptions_do_not_cross_splits() -> None:
    train = load_split("train")
    validation = load_split("validation")
    assert set(train["description"]).isdisjoint(validation["description"])


def test_all_development_images_open() -> None:
    frames = [unique_images(load_split("train")), unique_images(load_split("validation"))]
    for row in pd.concat(frames, ignore_index=True).itertuples(index=False):
        with Image.open(PROJECT_ROOT / row.image_path) as image:
            image.verify()


def test_image_manifest_covers_all_120_unique_images() -> None:
    manifest = pd.read_csv(DATA_DIR / "image_manifest.csv")
    assert len(manifest) == 120
    assert manifest["image_id"].nunique() == 120
    assert manifest["sha256"].nunique() == 120
    counts = manifest.groupby(["split", "source"]).size().to_dict()
    assert counts == {
        ("test", "wikimedia"): 10,
        ("train", "generated"): 50,
        ("train", "wikimedia"): 50,
        ("validation", "wikimedia"): 10,
    }


def test_wikimedia_license_inventory_is_complete_and_unique() -> None:
    licenses = pd.read_csv(DATA_DIR / "licenses.csv")
    manifest = pd.read_csv(DATA_DIR / "image_manifest.csv")
    real = manifest[manifest["source"].eq("wikimedia")]
    assert len(licenses) == len(real) == 70
    assert not licenses["commons_title"].duplicated().any()
    assert not licenses["description_url"].duplicated().any()
    assert not licenses["sha256"].duplicated().any()
    assert set(licenses["local_path"]) == set(real["image_path"])
    for row in licenses.itertuples(index=False):
        assert sha256(PROJECT_ROOT / row.local_path) == row.sha256


def test_test_lock_hash_matches_without_using_normal_loader() -> None:
    lock = json.loads((DATA_DIR / "test_lock.json").read_text(encoding="utf-8"))
    assert lock["test_locked"] is True
    assert lock["test_evaluation_permitted"] is False
    assert sha256(DATA_DIR / "test.csv") == lock["test_sha256"]
