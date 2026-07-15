from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from src.dataset_config import LABELS
from src.train_keras_image_model import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    METRICS_PATH,
    build_image_model,
    create_image_datasets,
    create_prediction_table,
    encode_labels,
    evaluate_predictions,
    load_datasets,
    load_image_tensor,
    resolve_image_paths,
)


def test_image_datasets_are_loaded_without_overlap() -> None:
    train_dataframe, validation_dataframe = load_datasets()

    assert len(train_dataframe) == 36
    assert len(validation_dataframe) == 12

    train_groups = set(
        train_dataframe["part_group_id"]
    )

    validation_groups = set(
        validation_dataframe["part_group_id"]
    )

    assert train_groups.isdisjoint(
        validation_groups
    )


def test_image_paths_are_resolved() -> None:
    train_dataframe, _ = load_datasets()

    paths = resolve_image_paths(
        train_dataframe.head(3)
    )

    assert len(paths) == 3

    for path in paths:
        assert path.endswith(".png")


def test_image_tensor_has_expected_shape() -> None:
    train_dataframe, _ = load_datasets()

    image_path = resolve_image_paths(
        train_dataframe.head(1)
    )[0]

    image, label = load_image_tensor(
        tf.constant(image_path),
        tf.constant(0),
    )

    assert image.shape == (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3,
    )

    assert image.dtype == tf.float32
    assert int(label.numpy()) == 0
    assert float(tf.reduce_min(image).numpy()) >= 0.0
    assert float(tf.reduce_max(image).numpy()) <= 255.0


def test_image_dataset_batch_has_expected_shape() -> None:
    train_dataframe, validation_dataframe = load_datasets()

    training_dataset, _ = create_image_datasets(
        train_dataframe,
        validation_dataframe,
    )

    images, labels = next(
        iter(training_dataset)
    )

    assert images.shape[1:] == (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3,
    )

    assert labels.ndim == 1
    assert images.shape[0] == labels.shape[0]


def test_image_model_has_three_outputs() -> None:
    model = build_image_model()

    sample_image = tf.zeros(
        (
            1,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        ),
        dtype=tf.float32,
    )

    probabilities = model(
        sample_image,
        training=False,
    ).numpy()

    assert probabilities.shape == (
        1,
        len(LABELS),
    )

    assert np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-5,
    )


def test_perfect_probabilities_produce_perfect_metrics() -> None:
    _, validation_dataframe = load_datasets()

    true_indices = encode_labels(
        validation_dataframe["label"]
    )

    probabilities = np.zeros(
        (
            len(validation_dataframe),
            len(LABELS),
        ),
        dtype=np.float32,
    )

    probabilities[
        np.arange(len(validation_dataframe)),
        true_indices,
    ] = 1.0

    metrics, predicted_labels = evaluate_predictions(
        validation_dataframe,
        probabilities,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0

    assert list(predicted_labels) == list(
        validation_dataframe["label"]
    )


def test_prediction_table_contains_probabilities() -> None:
    _, validation_dataframe = load_datasets()

    probabilities = np.full(
        (
            len(validation_dataframe),
            len(LABELS),
        ),
        fill_value=1.0 / len(LABELS),
        dtype=np.float32,
    )

    predicted_labels = np.asarray(
        ["MATCH"] * len(validation_dataframe),
        dtype=object,
    )

    prediction_table = create_prediction_table(
        validation_dataframe,
        predicted_labels,
        probabilities,
    )

    assert len(prediction_table) == 12
    assert "predicted_label" in prediction_table
    assert "probability_MATCH" in prediction_table
    assert "probability_PARTIAL_MATCH" in prediction_table
    assert "probability_MISMATCH" in prediction_table


def test_generated_image_metrics_use_validation_only() -> None:
    assert METRICS_PATH.is_file()

    metrics = json.loads(
        METRICS_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert metrics["model"] == (
        "Keras Image Neural Network"
    )

    assert metrics["input_modality"] == "image"
    assert metrics["evaluation_split"] == "validation"
    assert metrics["test_split_used"] is False
    assert metrics["sample_count"] == 12

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
