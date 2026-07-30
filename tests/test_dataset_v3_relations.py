from __future__ import annotations

import json

import pandas as pd
import pytest

from src.build_dataset_v3 import PAIRS_PER_LABEL, ROWS_PER_IMAGE, write_v3_development_relations
from src.data_v3 import (
    CATEGORIES,
    DEVELOPMENT_AUDIT,
    FAMILIES,
    RELATION_SUMMARY,
    check_v3_split_overlap,
    load_v3_image_manifest,
    load_v3_split,
)


def test_v3_development_loaders_refuse_locked_test() -> None:
    with pytest.raises(ValueError, match="only train and validation"):
        load_v3_image_manifest("test")
    with pytest.raises(ValueError, match="only train and validation"):
        load_v3_split("test")


def test_v3_relation_builder_is_balanced_and_development_only() -> None:
    summary = write_v3_development_relations()
    assert summary["status"] == "PASS_DATASET_V3_DEVELOPMENT_RELATIONS_READY"
    assert summary["train_images"] == 480
    assert summary["validation_images"] == 80
    assert summary["train_rows"] == 480 * ROWS_PER_IMAGE == 2880
    assert summary["validation_rows"] == 80 * ROWS_PER_IMAGE == 480
    assert summary["test_manifest_read"] is False
    assert summary["test_images_read"] is False
    assert summary["test_evaluation_executed"] is False

    for split, expected_images in (("train", 480), ("validation", 80)):
        data = load_v3_split(split)
        assert data["image_id"].nunique() == expected_images
        assert set(data["source"]) == {"dataset_v3"}
        assert set(data["part_category"]) == set(CATEGORIES)
        assert set(data["text_category"]) == set(CATEGORIES)
        assert not data["image_path"].str.replace("\\", "/", regex=False).str.startswith(
            "data/locked_test/"
        ).any()
        per_image = data.groupby(["image_id", "label"]).size().unstack(fill_value=0)
        assert (per_image == PAIRS_PER_LABEL).all().all()
        table = pd.crosstab(data["text_category"], data["label"])
        assert table.shape == (8, 3)
        assert table.nunique().max() == 1


def test_v3_relation_semantics_are_explicit() -> None:
    for split in ("train", "validation"):
        data = load_v3_split(split)
        match = data[data["label"].eq("MATCH")]
        partial = data[data["label"].eq("PARTIAL_MATCH")]
        mismatch = data[data["label"].eq("MISMATCH")]
        assert match["part_category"].eq(match["text_category"]).all()
        assert partial.apply(
            lambda row: row["part_category"] != row["text_category"]
            and FAMILIES[row["part_category"]] == FAMILIES[row["text_category"]],
            axis=1,
        ).all()
        assert mismatch.apply(
            lambda row: FAMILIES[row["part_category"]] != FAMILIES[row["text_category"]],
            axis=1,
        ).all()


def test_v3_mismatch_targets_are_diverse_and_cross_subsystem() -> None:
    for split in ("train", "validation"):
        mismatch = load_v3_split(split).query("label == 'MISMATCH'")
        diversity = mismatch.groupby("part_category")["text_category"].nunique()
        assert (diversity >= 4).all()
        assert mismatch.apply(
            lambda row: FAMILIES[row["part_category"]] != FAMILIES[row["text_category"]],
            axis=1,
        ).all()


def test_v3_development_splits_have_no_identity_or_text_overlap() -> None:
    train = load_v3_split("train")
    validation = load_v3_split("validation")
    assert check_v3_split_overlap(train, validation) == {
        "part_group_overlap": 0,
        "object_group_overlap": 0,
        "image_id_overlap": 0,
        "image_path_overlap": 0,
    }
    assert train["description"].nunique() == 32
    assert validation["description"].nunique() == 16
    assert set(train["description"]).isdisjoint(validation["description"])


def test_v3_image_manifests_remain_balanced_and_unlocked() -> None:
    train = load_v3_image_manifest("train", verify_files=True)
    validation = load_v3_image_manifest("validation", verify_files=True)
    assert train["project_category"].value_counts().to_dict() == {
        category: 60 for category in CATEGORIES
    }
    assert validation["project_category"].value_counts().to_dict() == {
        category: 10 for category in CATEGORIES
    }
    assert not train["test_locked"].astype(str).str.lower().eq("true").any()
    assert not validation["test_locked"].astype(str).str.lower().eq("true").any()


def test_v3_development_audit_passes_at_chance_shortcut_level() -> None:
    from src.audit_dataset_v3 import run_v3_development_audit

    summary = run_v3_development_audit()
    assert summary["status"] == "PASS_DATASET_V3_DEVELOPMENT_AUDIT"
    assert summary["train_images"] == 480
    assert summary["validation_images"] == 80
    assert summary["exact_cross_split_image_hash_overlap"] == 0
    assert all(value == 0 for value in summary["overlap"].values())
    assert summary["exact_description_overlap"] == 0
    assert summary["curation_exact_duplicate_groups"] == 0
    assert summary["curation_near_duplicate_review_pairs"] == 0
    assert summary["curation_current_project_overlap_pairs"] == 0
    assert summary["image_only_relation_ceiling_accuracy"] == 1 / 3
    for row in summary["shortcut_baselines"]:
        assert abs(row["accuracy"] - 1 / 3) < 1e-12
    assert summary["test_manifest_read"] is False
    assert summary["test_images_read"] is False
    assert summary["test_split_used"] is False
    assert summary["test_evaluation_executed"] is False
    assert summary["test_lock_status"] == (
        "LOCKED_NOT_AUTHORIZED_FOR_TRAINING_OR_SELECTION"
    )
    assert summary["test_lock_sha256"] == (
        "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
    )
    assert json.loads(RELATION_SUMMARY.read_text(encoding="utf-8"))["status"].startswith(
        "PASS_"
    )
    assert json.loads(DEVELOPMENT_AUDIT.read_text(encoding="utf-8"))["status"].startswith(
        "PASS_"
    )


def test_multimodal_model_supports_eight_v3_categories() -> None:
    from src.models import MultimodalRelationCNN

    model = MultimodalRelationCNN(text_dimension=16, number_of_part_categories=8)
    assert model.image_category_head.out_features == 8
    assert model.text_category_head.out_features == 8
