from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from src.dataset_config import LABELS
from src.train_multimodal_model import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    METRICS_PATH,
    build_multimodal_model,
    create_multimodal_datasets,
    create_prediction_table,
    create_text_vectorizer,
    encode_labels,
    evaluate_predictions,
    load_datasets,
)


def test_multimodal_datasets_have_no_group_overlap() -> None:
    train_dataframe, validation_dataframe = load_datasets()

    assert len(train_dataframe) == 90
    assert len(validation_dataframe) == 30

    train_groups = set(
        train_dataframe["part_group_id"]
    )

    validation_groups = set(
        validation_dataframe["part_group_id"]
    )

    assert train_groups.isdisjoint(
        validation_groups
    )


def test_multimodal_dataset_batch_has_both_inputs() -> None:
    train_dataframe, validation_dataframe = load_datasets()

    training_dataset, _ = create_multimodal_datasets(
        train_dataframe,
        validation_dataframe,
    )

    features, labels = next(
        iter(training_dataset)
    )

    assert set(features) == {
        "image",
        "description",
    }

    assert features["image"].shape[1:] == (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3,
    )

    assert features["description"].ndim == 1
    assert features["description"].dtype == tf.string
    assert labels.ndim == 1


def test_multimodal_vectorizer_has_vocabulary() -> None:
    train_dataframe, _ = load_datasets()

    vectorizer = create_text_vectorizer(
        train_dataframe[
            "description"
        ].astype(str).tolist()
    )

    vocabulary = vectorizer.get_vocabulary()

    assert len(vocabulary) > 2
    assert "automotive" in vocabulary


def test_multimodal_model_has_two_inputs() -> None:
    train_dataframe, _ = load_datasets()

    vectorizer = create_text_vectorizer(
        train_dataframe[
            "description"
        ].astype(str).tolist()
    )

    model = build_multimodal_model(
        vectorizer
    )

    input_names = {
        tensor.name.split(":")[0]
        for tensor in model.inputs
    }

    assert input_names == {
        "image",
        "description",
    }


def test_multimodal_model_has_three_outputs() -> None:
    train_dataframe, _ = load_datasets()

    vectorizer = create_text_vectorizer(
        train_dataframe[
            "description"
        ].astype(str).tolist()
    )

    model = build_multimodal_model(
        vectorizer
    )

    sample_inputs = {
        "image": tf.zeros(
            (
                1,
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                3,
            ),
            dtype=tf.float32,
        ),
        "description": tf.constant(
            ["Automotive starter motor."]
        ),
    }

    probabilities = model(
        sample_inputs,
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


def test_multimodal_prediction_table_has_probabilities() -> None:
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

    assert len(prediction_table) == 30
    assert "image_path" in prediction_table
    assert "description" in prediction_table
    assert "predicted_label" in prediction_table
    assert "probability_MATCH" in prediction_table
    assert "probability_PARTIAL_MATCH" in prediction_table
    assert "probability_MISMATCH" in prediction_table


def test_generated_multimodal_metrics_use_validation_only() -> None:
    assert METRICS_PATH.is_file()

    metrics = json.loads(
        METRICS_PATH.read_text(
            encoding="utf-8"
        )
    )

    assert metrics["model"] == (
        "Keras Multimodal Neural Network"
    )

    assert metrics["input_modality"] == "image_and_text"
    assert metrics["evaluation_split"] == "validation"
    assert metrics["test_split_used"] is False
    assert metrics["sample_count"] == 30

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0