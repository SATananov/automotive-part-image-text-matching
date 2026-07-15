from __future__ import annotations

import json

import numpy as np

from src.dataset_config import LABELS
from src.run_baseline_models import (
    OUTPUT_DIRECTORY,
    create_image_baseline,
    create_text_baseline,
    evaluate_predictions,
    extract_image_features,
    load_datasets,
    run_image_baseline,
    run_majority_baseline,
    run_text_baseline,
)


def test_baseline_datasets_are_loaded() -> None:
    train_dataframe, validation_dataframe = load_datasets()

    assert len(train_dataframe) == 90
    assert len(validation_dataframe) == 30

    train_groups = set(train_dataframe["part_group_id"])
    validation_groups = set(
        validation_dataframe["part_group_id"]
    )

    assert train_groups.isdisjoint(validation_groups)


def test_image_features_have_expected_shape() -> None:
    train_dataframe, _ = load_datasets()

    features = extract_image_features(
        train_dataframe.head(3)
    )

    assert features.shape == (3, 32 * 32 * 3)
    assert features.dtype == np.float32
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0


def test_majority_baseline_returns_valid_predictions() -> None:
    train_dataframe, validation_dataframe = load_datasets()

    metrics, predictions = run_majority_baseline(
        train_dataframe,
        validation_dataframe,
    )

    assert metrics["evaluation_split"] == "validation"
    assert metrics["sample_count"] == 30
    assert len(predictions) == 30

    assert set(
        predictions["predicted_label"]
    ).issubset(set(LABELS))


def test_text_baseline_returns_valid_predictions() -> None:
    train_dataframe, validation_dataframe = load_datasets()

    model = create_text_baseline()

    metrics, predictions = run_text_baseline(
        train_dataframe,
        validation_dataframe,
    )

    assert model is not None
    assert metrics["evaluation_split"] == "validation"
    assert metrics["sample_count"] == 30
    assert len(predictions) == 30

    assert set(
        predictions["predicted_label"]
    ).issubset(set(LABELS))


def test_image_baseline_returns_valid_predictions() -> None:
    train_dataframe, validation_dataframe = load_datasets()

    model = create_image_baseline()

    metrics, predictions = run_image_baseline(
        train_dataframe,
        validation_dataframe,
    )

    assert model is not None
    assert metrics["evaluation_split"] == "validation"
    assert metrics["sample_count"] == 30
    assert len(predictions) == 30

    assert set(
        predictions["predicted_label"]
    ).issubset(set(LABELS))


def test_metric_report_has_expected_structure() -> None:
    _, validation_dataframe = load_datasets()

    predictions = validation_dataframe[
        "label"
    ].to_numpy()

    metrics = evaluate_predictions(
        model_name="Perfect test model",
        validation_dataframe=validation_dataframe,
        predictions=predictions,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert len(metrics["confusion_matrix"]) == 3

    assert all(
        len(row) == 3
        for row in metrics["confusion_matrix"]
    )


def test_generated_baseline_reports_use_validation() -> None:
    expected_model_slugs = (
        "majority",
        "text_tfidf_logistic_regression",
        "image_pixels_logistic_regression",
    )

    for model_slug in expected_model_slugs:
        metrics_path = (
            OUTPUT_DIRECTORY
            / f"{model_slug}_validation_metrics.json"
        )

        assert metrics_path.is_file()

        metrics = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )

        assert metrics["evaluation_split"] == "validation"
        assert metrics["sample_count"] == 30