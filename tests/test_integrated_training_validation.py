from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.integrated_training_config import (
    INTEGRATED_COMPARISON_CSV_PATH,
    INTEGRATED_METRIC_PATHS,
    INTEGRATED_RUN_STATUS_PATH,
    INTEGRATED_TEST_PATH,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
)
from src.project_cli import COMMANDS
from src.run_integrated_training_validation import (
    IntegratedTrainingError,
    build_common_metadata,
    build_text_vocabulary,
    dataset_profile,
    encode_text_sequences,
    load_integrated_datasets,
    locked_test_fingerprints,
    read_integrated_split,
    read_test_lock,
)
from src.validate_external_training_readiness import (
    sha256_canonical_csv,
)
from src.verification.integrated_training_validation import (
    build_verification_report,
)


def test_integrated_train_validation_counts_and_group_isolation() -> None:
    train_dataframe, validation_dataframe = (
        load_integrated_datasets()
    )

    assert len(train_dataframe) == 180
    assert len(validation_dataframe) == 60
    assert train_dataframe["part_group_id"].nunique() == 60
    assert validation_dataframe["part_group_id"].nunique() == 20
    assert set(train_dataframe["part_group_id"]).isdisjoint(
        set(validation_dataframe["part_group_id"])
    )


def test_training_loader_rejects_locked_test_path() -> None:
    with pytest.raises(
        IntegratedTrainingError,
        match="Locked test data",
    ):
        read_integrated_split(INTEGRATED_TEST_PATH)


def test_test_lock_authorizes_only_train_and_validation() -> None:
    lock = read_test_lock()

    assert lock["test_locked"] is True
    assert lock["test_evaluation_permitted"] is False
    assert lock["hash_normalization"] == "utf-8-lf"
    assert set(lock["training_inputs"]) == {
        "data/processed/integrated_train.csv",
        "data/processed/integrated_validation.csv",
    }
    assert "data/processed/integrated_test.csv" not in (
        lock["training_inputs"]
    )


def test_locked_test_fingerprints_match_policy() -> None:
    lock = read_test_lock()
    fingerprints = locked_test_fingerprints(lock)

    assert fingerprints[
        "data/processed/integrated_test.csv"
    ] == lock["integrated_test_sha256"]


def test_canonical_csv_hash_is_newline_independent(
    tmp_path: Path,
) -> None:
    lf_path = tmp_path / "lf.csv"
    crlf_path = tmp_path / "crlf.csv"
    lf_path.write_bytes(b"a,b\n1,2\n")
    crlf_path.write_bytes(b"a,b\r\n1,2\r\n")

    assert sha256_canonical_csv(lf_path) == (
        sha256_canonical_csv(crlf_path)
    )


def test_dataset_profile_preserves_balanced_sources() -> None:
    train_dataframe, _ = load_integrated_datasets()
    profile = dataset_profile(train_dataframe)

    assert profile["samples"] == 180
    assert profile["groups"] == 60
    assert profile["label_distribution"] == {
        "MATCH": 60,
        "PARTIAL_MATCH": 60,
        "MISMATCH": 60,
    }
    assert profile["source_distribution"] == {
        "generated_development": 90,
        "wikimedia_commons_open_license": 90,
    }


def test_common_metadata_is_validation_only() -> None:
    train_dataframe, validation_dataframe = (
        load_integrated_datasets()
    )
    metadata = build_common_metadata(
        train_dataframe,
        validation_dataframe,
    )

    assert metadata["training_input_path"] == (
        "data/processed/integrated_train.csv"
    )
    assert metadata["validation_input_path"] == (
        "data/processed/integrated_validation.csv"
    )
    assert metadata["test_split_used"] is False
    assert metadata["test_evaluation_permitted"] is False
    assert metadata["train_validation_group_overlap"] == 0


def test_text_preprocessing_is_fit_on_training_descriptions() -> None:
    train_dataframe, validation_dataframe = (
        load_integrated_datasets()
    )
    vocabulary = build_text_vocabulary(
        train_dataframe["description"]
    )
    encoded = encode_text_sequences(
        validation_dataframe["description"],
        vocabulary,
    )

    assert "automotive" in vocabulary
    assert encoded.shape == (60, 12)
    assert encoded.min() >= 0
    assert INTEGRATED_TEST_PATH != INTEGRATED_TRAIN_PATH


def test_integrated_cli_commands_are_registered() -> None:
    run_spec = COMMANDS["run-integrated-training-validation"]
    verify_spec = COMMANDS[
        "verify-integrated-training-validation"
    ]

    assert run_spec.requires_tensorflow is True
    assert verify_spec.requires_tensorflow is False


def test_generated_integrated_metrics_are_validation_only() -> None:
    for model_slug, path in INTEGRATED_METRIC_PATHS.items():
        assert path.is_file(), model_slug
        metrics = json.loads(path.read_text(encoding="utf-8"))
        assert metrics["model_slug"] == model_slug
        assert metrics["training_sample_count"] == 180
        assert metrics["sample_count"] == 60
        assert metrics["evaluation_split"] == (
            "integrated_validation"
        )
        assert metrics["test_split_used"] is False
        assert metrics["test_evaluation_permitted"] is False


def test_generated_comparison_contains_six_ranked_models() -> None:
    comparison = pd.read_csv(INTEGRATED_COMPARISON_CSV_PATH)

    assert len(comparison) == 6
    assert set(comparison["validation_rank"]) == set(range(1, 7))
    assert set(comparison["test_split_used"]) == {False}


def test_generated_status_preserves_test_lock() -> None:
    status = json.loads(
        INTEGRATED_RUN_STATUS_PATH.read_text(encoding="utf-8")
    )

    assert status["status"] == "PASS"
    assert status["model_count"] == 6
    assert status["test_split_used"] is False
    assert status["locked_test_fingerprints_unchanged"] is True


def test_current_integrated_training_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
