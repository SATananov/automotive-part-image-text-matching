from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import COLUMNS, RESULTS_DIR
from src.data_v3 import CATEGORIES
from src.models import MultimodalRelationCNN
from src.train_dataset_v3 import (
    DATASET_V3_RESULTS_DIR,
    MAIN_MODEL_SLUG,
    MODEL_SLUGS,
    assert_development_only,
)


def relation_frame(
    *,
    split: str,
    rows: int,
    images: int,
    locked: bool = False,
) -> pd.DataFrame:
    records = []
    for index in range(rows):
        image_index = index % images
        category = CATEGORIES[image_index % len(CATEGORIES)]
        records.append(
            {
                "sample_id": f"{split}_{index}",
                "image_id": f"{split}_image_{image_index}",
                "part_group_id": f"{split}_group_{image_index}",
                "object_group_id": f"{split}_group_{image_index}",
                "image_path": (
                    f"data/locked_test/dataset_v3/images/{category}/x.jpg"
                    if locked
                    else f"data/images/dataset_v3/{split}/{category}/x_{image_index}.jpg"
                ),
                "part_family": "family",
                "part_category": category,
                "text_category": category,
                "description": f"{split} description {index}",
                "label": ("MATCH", "MISMATCH", "PARTIAL_MATCH")[index % 3],
                "source": "dataset_v3",
            }
        )
    return pd.DataFrame(records, columns=COLUMNS)


def test_dataset_v3_results_are_isolated_from_dataset_v2() -> None:
    assert DATASET_V3_RESULTS_DIR == RESULTS_DIR / "dataset_v3"
    assert DATASET_V3_RESULTS_DIR != RESULTS_DIR


def test_dataset_v3_training_plan_has_seven_models() -> None:
    assert len(MODEL_SLUGS) == 7
    assert MAIN_MODEL_SLUG in MODEL_SLUGS
    assert len(set(MODEL_SLUGS)) == len(MODEL_SLUGS)


def test_multimodal_model_supports_eight_dataset_v3_categories() -> None:
    model = MultimodalRelationCNN(
        text_dimension=16,
        number_of_part_categories=len(CATEGORIES),
    )
    assert model.image_category_head.out_features == 8
    assert model.text_category_head.out_features == 8


def test_development_guard_rejects_locked_test_paths() -> None:
    train = relation_frame(split="train", rows=2880, images=480)
    validation = relation_frame(
        split="validation",
        rows=480,
        images=80,
        locked=True,
    )
    with pytest.raises(ValueError, match="Locked-test path"):
        assert_development_only(train, validation)


def test_development_guard_rejects_identity_overlap() -> None:
    train = relation_frame(split="train", rows=2880, images=480)
    validation = relation_frame(split="validation", rows=480, images=80)
    validation.loc[0, "image_id"] = train.loc[0, "image_id"]
    with pytest.raises(ValueError, match="identity overlap"):
        assert_development_only(train, validation)


def test_training_module_has_no_locked_test_loader_or_path() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "train_dataset_v3.py"
    ).read_text(encoding="utf-8")
    assert 'load_v3_split("test")' not in source
    assert "dataset_v3_test_images_LOCKED.csv" not in source
    assert "data/locked_test/dataset_v3" not in source
