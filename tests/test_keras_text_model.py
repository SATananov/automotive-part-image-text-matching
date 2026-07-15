from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from src.dataset_config import LABELS
from src.train_keras_text_model import (
    METRICS_PATH,
    build_text_model,
    create_prediction_table,
    create_text_vectorizer,
    decode_label_indices,
    encode_labels,
    evaluate_predictions,
    load_datasets,
)


def test_text_datasets_are_loaded_without_overlap() -> None:
    train_dataframe, validation_dataframe = load_datasets()

    assert len(train_dataframe) == 36
    assert len(validation_dataframe) == 12

    train_groups = set(train_dataframe["part_group_id"])
    validation_groups = set(
        validation_dataframe["part_group_id"]
    )

    assert train_groups.isdisjoint(validation_groups)


def test_label_encoding_round_trip() -> None:
    train_dataframe, _ = load_datasets()

    encoded_labels = encode_labels(
        train_dataframe["label"]
    )

    decoded_labels = decode_label_indices(
        encoded_labels
    )

    assert list(decoded_labels) == list(
        train_dataframe["label"]
    )


def test_text_vectorizer_builds_vocabulary() -> None:
    train_dataframe, _ = load_datasets()

    vectorizer = create_text_vectorizer(
        train_dataframe[
            "description"
        ].astype(str).tolist()
    )

    vocabulary = vectorizer.get_vocabulary()

    assert len(vocabulary) > 2
    assert "automotive" in vocabulary


def test_text_model_has_three_outputs() -> None:
    train_dataframe, _ = load_datasets()

    vectorizer = create_text_vectorizer(
        train_dataframe[
            "description"
        ].astype(str).tolist()
    )

    model = build_text_model(vectorizer)

    probabilities = model(
        tf.constant(
            ["Automotive starter motor."]
        ),
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


def test_generated_text_metrics_use_validation_only() -> None:
    assert METRICS_PATH.is_file()

    metrics = json.loads(
        METRICS_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert metrics["model"] == "Keras Text Neural Network"
    assert metrics["input_modality"] == "text"
    assert metrics["evaluation_split"] == "validation"
    assert metrics["test_split_used"] is False
    assert metrics["sample_count"] == 12

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0