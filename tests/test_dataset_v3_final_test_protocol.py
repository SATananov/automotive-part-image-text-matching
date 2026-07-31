from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data_v3 import (
    VALIDATION_IMAGE_MANIFEST,
    load_v3_split,
)
from src.final_test_protocol_v3 import (
    EXPECTED_RELATION_PROTOCOL_FINGERPRINT,
    EXPECTED_TEST_IMAGES,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEXT_DIMENSION,
    EXPECTED_VECTORIZER_FINGERPRINT,
    all_final_test_captions,
    build_final_test_relations,
    fit_frozen_text_vectorizer,
    load_frozen_multimodal_model,
    protocol_model_dry_run,
    relation_protocol_fingerprint,
    validate_final_test_relations,
)
from src.train_dataset_v3 import (
    DATASET_V3_RESULTS_DIR,
)


def test_final_test_caption_protocol_is_disjoint() -> None:
    train = load_v3_split("train")
    validation = load_v3_split("validation")
    captions = all_final_test_captions()

    assert len(captions) == 16
    assert not (
        captions
        & set(
            train["description"].astype(str)
        )
    )
    assert not (
        captions
        & set(
            validation["description"].astype(str)
        )
    )
    assert relation_protocol_fingerprint() == (
        EXPECTED_RELATION_PROTOCOL_FINGERPRINT
    )


def test_final_test_relation_protocol_dry_run() -> None:
    train = load_v3_split("train")
    validation = load_v3_split("validation")
    images = pd.read_csv(
        VALIDATION_IMAGE_MANIFEST
    ).copy()
    images["split"] = "test"
    images["test_locked"] = True

    relations = build_final_test_relations(
        images
    )
    summary = validate_final_test_relations(
        relations,
        images,
        train,
        validation,
        require_locked_paths=False,
        require_group_disjoint=False,
    )

    assert len(images) == EXPECTED_TEST_IMAGES
    assert len(relations) == EXPECTED_TEST_ROWS
    assert summary["status"] == (
        "PASS_FINAL_TEST_RELATION_PROTOCOL"
    )
    assert summary[
        "description_overlap_with_train"
    ] == 0
    assert summary[
        "description_overlap_with_validation"
    ] == 0


def test_frozen_vectorizer_and_checkpoint_interface() -> None:
    train = load_v3_split("train")
    vectorizer, fingerprint = (
        fit_frozen_text_vectorizer(
            train
        )
    )

    assert len(vectorizer.vocabulary_) == (
        EXPECTED_TEXT_DIMENSION
    )
    assert fingerprint == (
        EXPECTED_VECTORIZER_FINGERPRINT
    )

    checkpoint_path = (
        DATASET_V3_RESULTS_DIR
        / "models"
        / "torch_multimodal_dataset_v3_state.pt"
    )
    dry_run = protocol_model_dry_run(
        train,
        checkpoint_path,
    )

    assert dry_run["status"] == (
        "PASS_FINAL_TEST_PROTOCOL_MODEL_DRY_RUN"
    )
    assert dry_run["zero_vector_captions"] == 0
    assert dry_run["text_dimension"] == (
        EXPECTED_TEXT_DIMENSION
    )
    assert dry_run[
        "text_category_accuracy"
    ] == 1.0

    model, checkpoint = (
        load_frozen_multimodal_model(
            checkpoint_path
        )
    )

    with torch.no_grad():
        outputs = model(
            torch.zeros(
                (2, 3, 48, 48),
                dtype=torch.float32,
            ),
            torch.zeros(
                (
                    2,
                    EXPECTED_TEXT_DIMENSION,
                ),
                dtype=torch.float32,
            ),
        )

    assert [
        tuple(output.shape)
        for output in outputs
    ] == [
        (2, 3),
        (2, 8),
        (2, 8),
    ]
    assert checkpoint["model_slug"] == (
        "torch_multimodal_dataset_v3"
    )


def test_final_results_do_not_exist_before_execution() -> None:
    from src.evaluate_dataset_v3_final_test import (
        FINAL_RESULTS_DIR,
        TEMP_RESULTS_DIR,
    )

    assert not FINAL_RESULTS_DIR.exists()
    assert not TEMP_RESULTS_DIR.exists()
