from __future__ import annotations

import gc
import hashlib
import json
import random
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from src.dataset_config import LABELS, METADATA_COLUMNS
from src.integrated_training_config import (
    BATCH_SIZE,
    CLASSICAL_IMAGE_HEIGHT,
    CLASSICAL_IMAGE_WIDTH,
    DEVELOPMENT_METRIC_PATHS,
    EARLY_STOPPING_PATIENCE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INTEGRATED_ARCHITECTURE_PATHS,
    INTEGRATED_COMPARISON_CSV_PATH,
    INTEGRATED_COMPARISON_JSON_PATH,
    INTEGRATED_CONFUSION_MATRIX_PATHS,
    INTEGRATED_HISTORY_PATHS,
    INTEGRATED_METRIC_PATHS,
    INTEGRATED_PREDICTION_PATHS,
    INTEGRATED_RUN_STATUS_PATH,
    INTEGRATED_SUMMARY_PATH,
    INTEGRATED_TEST_LOCK_PATH,
    INTEGRATED_TRAINING_ROOT,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
    LOCKED_TEST_PATHS,
    MAX_EPOCHS,
    MODEL_MODALITIES,
    MODEL_TITLES,
    RANDOM_STATE,
    TEXT_EMBEDDING_DIMENSION,
    TEXT_SEQUENCE_LENGTH,
)
from src.real_dataset_config import PROJECT_ROOT
from src.validate_external_training_readiness import (
    project_relative_path,
    sha256_canonical_csv,
    validate_external_training_readiness,
)

LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}
INDEX_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_INDEX.items()
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class IntegratedTrainingError(RuntimeError):
    """Raised when integrated validation safeguards are violated."""


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            content,
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def read_integrated_split(path: Path) -> pd.DataFrame:
    resolved = path.resolve()
    locked = {locked_path.resolve() for locked_path in LOCKED_TEST_PATHS}
    if resolved in locked:
        raise IntegratedTrainingError(
            "Locked test data cannot be loaded by the integrated "
            "training workflow."
        )

    authorized = {
        INTEGRATED_TRAIN_PATH.resolve(),
        INTEGRATED_VALIDATION_PATH.resolve(),
    }
    if resolved not in authorized:
        raise IntegratedTrainingError(
            f"Unauthorized integrated training input: {path}."
        )
    if not path.is_file():
        raise IntegratedTrainingError(
            f"Integrated split is missing: {path}."
        )

    dataframe = pd.read_csv(path, dtype=str).fillna("")
    if tuple(dataframe.columns) != METADATA_COLUMNS:
        raise IntegratedTrainingError(
            f"Unexpected integrated metadata schema: {path.name}."
        )
    if dataframe.empty:
        raise IntegratedTrainingError(
            f"Integrated split is empty: {path.name}."
        )
    if not dataframe["sample_id"].is_unique:
        raise IntegratedTrainingError(
            f"Integrated split contains duplicate sample IDs: {path.name}."
        )
    if set(dataframe["label"]) != set(LABELS):
        raise IntegratedTrainingError(
            f"Integrated split has an unexpected label set: {path.name}."
        )
    return dataframe


def load_integrated_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataframe = read_integrated_split(INTEGRATED_TRAIN_PATH)
    validation_dataframe = read_integrated_split(
        INTEGRATED_VALIDATION_PATH
    )
    train_groups = set(train_dataframe["part_group_id"])
    validation_groups = set(validation_dataframe["part_group_id"])
    overlap = train_groups & validation_groups
    if overlap:
        raise IntegratedTrainingError(
            "Integrated train and validation part groups overlap."
        )
    return train_dataframe, validation_dataframe


def read_test_lock() -> dict[str, Any]:
    if not INTEGRATED_TEST_LOCK_PATH.is_file():
        raise IntegratedTrainingError(
            f"Integrated test lock is missing: "
            f"{INTEGRATED_TEST_LOCK_PATH}."
        )
    try:
        lock = json.loads(
            INTEGRATED_TEST_LOCK_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise IntegratedTrainingError(
            f"Cannot read the integrated test lock: {error}."
        ) from error
    if not isinstance(lock, dict):
        raise IntegratedTrainingError(
            "The integrated test lock is not a JSON object."
        )
    if lock.get("test_locked") is not True:
        raise IntegratedTrainingError("The integrated test lock is open.")
    if lock.get("test_evaluation_permitted") is not False:
        raise IntegratedTrainingError(
            "The integrated test lock permits evaluation."
        )
    if lock.get("hash_normalization") != "utf-8-lf":
        raise IntegratedTrainingError(
            "The integrated test lock does not use canonical hashing."
        )

    training_inputs = {str(item) for item in lock.get("training_inputs", [])}
    expected_inputs = {
        project_relative_path(INTEGRATED_TRAIN_PATH),
        project_relative_path(INTEGRATED_VALIDATION_PATH),
    }
    if training_inputs != expected_inputs:
        raise IntegratedTrainingError(
            "The integrated test lock authorizes unexpected training inputs."
        )
    forbidden_inputs = {
        project_relative_path(path)
        for path in LOCKED_TEST_PATHS
    }
    if training_inputs & forbidden_inputs:
        raise IntegratedTrainingError(
            "A locked test artifact is authorized as a training input."
        )
    return lock


def locked_test_fingerprints(lock: dict[str, Any]) -> dict[str, str]:
    lock_keys = {
        project_relative_path(LOCKED_TEST_PATHS[0]): (
            "external_test_sha256"
        ),
        project_relative_path(LOCKED_TEST_PATHS[1]): (
            "integrated_test_sha256"
        ),
    }
    fingerprints: dict[str, str] = {}
    for path, relative_path in zip(LOCKED_TEST_PATHS, lock_keys, strict=True):
        if not path.is_file():
            raise IntegratedTrainingError(
                f"Locked test artifact is missing: {relative_path}."
            )
        actual = sha256_canonical_csv(path)
        expected = str(lock.get(lock_keys[relative_path], ""))
        if actual != expected:
            raise IntegratedTrainingError(
                f"Locked test fingerprint differs: {relative_path}."
            )
        fingerprints[relative_path] = actual
    return fingerprints


def dataset_profile(dataframe: pd.DataFrame) -> dict[str, Any]:
    return {
        "samples": int(len(dataframe)),
        "images": int(dataframe["image_id"].nunique()),
        "groups": int(dataframe["part_group_id"].nunique()),
        "categories": int(dataframe["part_category"].nunique()),
        "label_distribution": {
            label: int(dataframe["label"].eq(label).sum())
            for label in LABELS
        },
        "source_distribution": {
            str(source): int(count)
            for source, count in (
                dataframe["source"].value_counts().sort_index().items()
            )
        },
    }


def build_common_metadata(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "training_input_path": project_relative_path(
            INTEGRATED_TRAIN_PATH
        ),
        "validation_input_path": project_relative_path(
            INTEGRATED_VALIDATION_PATH
        ),
        "training_sample_count": int(len(train_dataframe)),
        "validation_sample_count": int(len(validation_dataframe)),
        "training_group_count": int(
            train_dataframe["part_group_id"].nunique()
        ),
        "validation_group_count": int(
            validation_dataframe["part_group_id"].nunique()
        ),
        "train_validation_group_overlap": int(
            len(
                set(train_dataframe["part_group_id"])
                & set(validation_dataframe["part_group_id"])
            )
        ),
        "training_profile": dataset_profile(train_dataframe),
        "validation_profile": dataset_profile(validation_dataframe),
        "evaluation_split": "integrated_validation",
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "random_state": RANDOM_STATE,
    }


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def build_text_vocabulary(texts: Iterable[str]) -> dict[str, int]:
    counts = Counter(
        token
        for text in texts
        for token in tokenize(str(text))
    )
    ordered_tokens = sorted(
        counts,
        key=lambda token: (-counts[token], token),
    )
    return {
        token: index + 2
        for index, token in enumerate(ordered_tokens)
    }


def encode_text_sequences(
    texts: Iterable[str],
    vocabulary: dict[str, int],
) -> np.ndarray:
    rows: list[list[int]] = []
    for text in texts:
        tokens = tokenize(str(text))[:TEXT_SEQUENCE_LENGTH]
        encoded = [vocabulary.get(token, 1) for token in tokens]
        encoded.extend([0] * (TEXT_SEQUENCE_LENGTH - len(encoded)))
        rows.append(encoded)
    return np.asarray(rows, dtype=np.int32)


def encode_labels(labels: Sequence[str] | pd.Series) -> np.ndarray:
    values: list[int] = []
    for label in labels:
        if label not in LABEL_TO_INDEX:
            raise IntegratedTrainingError(f"Unknown label: {label}.")
        values.append(LABEL_TO_INDEX[label])
    return np.asarray(values, dtype=np.int32)


def decode_label_indices(indices: np.ndarray) -> np.ndarray:
    labels: list[str] = []
    for value in indices:
        index = int(value)
        if index not in INDEX_TO_LABEL:
            raise IntegratedTrainingError(
                f"Unknown predicted label index: {index}."
            )
        labels.append(INDEX_TO_LABEL[index])
    return np.asarray(labels, dtype=object)


def resolve_image_path(relative_path: str) -> Path:
    candidate = (PROJECT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise IntegratedTrainingError(
            f"Image path escapes the project root: {relative_path}."
        ) from error
    if not candidate.is_file():
        raise IntegratedTrainingError(
            f"Integrated image is missing: {relative_path}."
        )
    return candidate


def extract_image_arrays(
    dataframe: pd.DataFrame,
    *,
    size: tuple[int, int],
    flatten: bool,
) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for relative_path in dataframe["image_path"].astype(str):
        if relative_path not in cache:
            image_path = resolve_image_path(relative_path)
            try:
                with Image.open(image_path) as image:
                    resized = image.convert("RGB").resize(
                        size,
                        Image.Resampling.BILINEAR,
                    )
                    array = np.asarray(resized, dtype=np.float32)
            except OSError as error:
                raise IntegratedTrainingError(
                    f"Cannot read integrated image: {relative_path}."
                ) from error
            if flatten:
                array = array.reshape(-1) / 255.0
            cache[relative_path] = array
        rows.append(cache[relative_path])
    return np.stack(rows).astype(np.float32, copy=False)


def evaluate_predictions(
    *,
    model_slug: str,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    predicted_labels: np.ndarray,
    common_metadata: dict[str, Any],
    training_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    true_labels = validation_dataframe["label"].to_numpy(dtype=object)
    if len(predicted_labels) != len(true_labels):
        raise IntegratedTrainingError(
            f"Prediction count differs for {model_slug}."
        )
    invalid = set(predicted_labels) - set(LABELS)
    if invalid:
        raise IntegratedTrainingError(
            f"Unknown predictions for {model_slug}: {sorted(invalid)}."
        )

    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=LABELS,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=LABELS,
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    payload: dict[str, Any] = {
        "model_slug": model_slug,
        "model": MODEL_TITLES[model_slug],
        "input_modality": MODEL_MODALITIES[model_slug],
        "evaluation_split": "integrated_validation",
        "training_sample_count": int(len(train_dataframe)),
        "sample_count": int(len(validation_dataframe)),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "true_distribution": {
            label: int(np.count_nonzero(true_labels == label))
            for label in LABELS
        },
        "predicted_distribution": {
            label: int(np.count_nonzero(predicted_labels == label))
            for label in LABELS
        },
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(LABELS),
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "data_contract": common_metadata,
    }
    if training_metadata is not None:
        payload["training"] = training_metadata
    return payload


def prediction_table(
    validation_dataframe: pd.DataFrame,
    predicted_labels: np.ndarray,
) -> pd.DataFrame:
    table = validation_dataframe[
        [
            "sample_id",
            "part_group_id",
            "image_id",
            "part_category",
            "source",
            "description",
            "label",
        ]
    ].copy()
    table = table.rename(columns={"label": "true_label"})
    table["predicted_label"] = predicted_labels
    table["is_correct"] = (
        table["true_label"] == table["predicted_label"]
    )
    return table


def write_model_outputs(
    *,
    model_slug: str,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    history: dict[str, list[float]] | None = None,
    architecture: str | None = None,
) -> None:
    atomic_write_json(INTEGRATED_METRIC_PATHS[model_slug], metrics)
    predictions_path = INTEGRATED_PREDICTION_PATHS[model_slug]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        predictions_path,
        index=False,
        lineterminator="\n",
    )
    confusion = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"actual_{label}" for label in LABELS],
        columns=[f"predicted_{label}" for label in LABELS],
    )
    confusion.to_csv(
        INTEGRATED_CONFUSION_MATRIX_PATHS[model_slug],
        index=True,
        lineterminator="\n",
    )
    if history is not None:
        pd.DataFrame(history).to_csv(
            INTEGRATED_HISTORY_PATHS[model_slug],
            index=False,
            lineterminator="\n",
        )
    if architecture is not None:
        atomic_write_text(
            INTEGRATED_ARCHITECTURE_PATHS[model_slug],
            architecture.rstrip() + "\n",
        )


def create_text_baseline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def run_classical_models(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    common_metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    train_labels = train_dataframe["label"].to_numpy(dtype=object)

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(train_dataframe), 1)), train_labels)
    majority_predictions = majority.predict(
        np.zeros((len(validation_dataframe), 1))
    )
    majority_metrics = evaluate_predictions(
        model_slug="majority",
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        predicted_labels=majority_predictions,
        common_metadata=common_metadata,
    )
    write_model_outputs(
        model_slug="majority",
        metrics=majority_metrics,
        predictions=prediction_table(
            validation_dataframe,
            majority_predictions,
        ),
    )
    results["majority"] = majority_metrics

    text_model = create_text_baseline()
    text_model.fit(
        train_dataframe["description"].astype(str),
        train_labels,
    )
    text_predictions = text_model.predict(
        validation_dataframe["description"].astype(str)
    )
    text_metrics = evaluate_predictions(
        model_slug="text_tfidf_logistic_regression",
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        predicted_labels=text_predictions,
        common_metadata=common_metadata,
    )
    write_model_outputs(
        model_slug="text_tfidf_logistic_regression",
        metrics=text_metrics,
        predictions=prediction_table(
            validation_dataframe,
            text_predictions,
        ),
    )
    results["text_tfidf_logistic_regression"] = text_metrics

    train_image_features = extract_image_arrays(
        train_dataframe,
        size=(CLASSICAL_IMAGE_WIDTH, CLASSICAL_IMAGE_HEIGHT),
        flatten=True,
    )
    validation_image_features = extract_image_arrays(
        validation_dataframe,
        size=(CLASSICAL_IMAGE_WIDTH, CLASSICAL_IMAGE_HEIGHT),
        flatten=True,
    )
    image_model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    image_model.fit(train_image_features, train_labels)
    image_predictions = image_model.predict(validation_image_features)
    image_metrics = evaluate_predictions(
        model_slug="image_pixels_logistic_regression",
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        predicted_labels=image_predictions,
        common_metadata=common_metadata,
    )
    write_model_outputs(
        model_slug="image_pixels_logistic_regression",
        metrics=image_metrics,
        predictions=prediction_table(
            validation_dataframe,
            image_predictions,
        ),
    )
    results["image_pixels_logistic_regression"] = image_metrics
    return results


def configure_keras_runtime() -> tuple[Any, str]:
    try:
        import keras
    except ModuleNotFoundError as error:
        raise IntegratedTrainingError(
            "Keras and the configured neural backend are required. "
            "Activate the project .venv and run the workflow again."
        ) from error

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    keras.utils.set_random_seed(RANDOM_STATE)
    try:
        import tensorflow as tf

        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass
    except ModuleNotFoundError:
        pass
    return keras, str(keras.backend.backend())


def architecture_text(model: Any) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def build_text_model(keras: Any, vocabulary_size: int) -> Any:
    inputs = keras.Input(
        shape=(TEXT_SEQUENCE_LENGTH,),
        dtype="int32",
        name="description_tokens",
    )
    features = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=TEXT_EMBEDDING_DIMENSION,
        name="token_embedding",
    )(inputs)
    features = keras.layers.GlobalAveragePooling1D(
        name="text_pooling"
    )(features)
    features = keras.layers.Dense(
        32,
        activation="relu",
        name="text_features",
    )(features)
    features = keras.layers.Dropout(0.1, name="text_dropout")(features)
    outputs = keras.layers.Dense(
        len(LABELS),
        activation="softmax",
        name="class_probabilities",
    )(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="keras_text_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_image_branch(keras: Any, image_input: Any) -> Any:
    features = keras.layers.Rescaling(
        1.0 / 255.0,
        name="image_rescaling",
    )(image_input)
    features = keras.layers.Flatten(name="image_flatten")(features)
    features = keras.layers.Dense(
        64,
        activation="relu",
        name="image_dense_1",
    )(features)
    return keras.layers.Dense(
        32,
        activation="relu",
        name="image_features",
    )(features)


def build_image_model(keras: Any) -> Any:
    inputs = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype="float32",
        name="image",
    )
    features = build_image_branch(keras, inputs)
    features = keras.layers.Dropout(0.1, name="image_dropout")(features)
    outputs = keras.layers.Dense(
        len(LABELS),
        activation="softmax",
        name="class_probabilities",
    )(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="keras_image_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def build_multimodal_model(keras: Any, vocabulary_size: int) -> Any:
    text_input = keras.Input(
        shape=(TEXT_SEQUENCE_LENGTH,),
        dtype="int32",
        name="description_tokens",
    )
    text_features = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=TEXT_EMBEDDING_DIMENSION,
        name="token_embedding",
    )(text_input)
    text_features = keras.layers.GlobalAveragePooling1D(
        name="text_pooling"
    )(text_features)
    text_features = keras.layers.Dense(
        32,
        activation="relu",
        name="text_features",
    )(text_features)

    image_input = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype="float32",
        name="image",
    )
    image_features = build_image_branch(keras, image_input)
    fused = keras.layers.Concatenate(name="feature_fusion")(
        [text_features, image_features]
    )
    fused = keras.layers.Dense(
        64,
        activation="relu",
        name="fusion_dense",
    )(fused)
    fused = keras.layers.Dropout(0.15, name="fusion_dropout")(fused)
    outputs = keras.layers.Dense(
        len(LABELS),
        activation="softmax",
        name="class_probabilities",
    )(fused)
    model = keras.Model(
        inputs={
            "description_tokens": text_input,
            "image": image_input,
        },
        outputs=outputs,
        name="keras_multimodal_classifier",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def train_neural_model(
    *,
    keras: Any,
    model_slug: str,
    model: Any,
    training_inputs: Any,
    validation_inputs: Any,
    training_labels: np.ndarray,
    validation_labels: np.ndarray,
    backend: str,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    common_metadata: dict[str, Any],
    vocabulary_size: int | None = None,
) -> dict[str, Any]:
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )
    history_object = model.fit(
        training_inputs,
        training_labels,
        validation_data=(validation_inputs, validation_labels),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[callback],
        verbose=0,
    )
    probabilities = np.asarray(
        model.predict(
            validation_inputs,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
    )
    if probabilities.shape != (
        len(validation_dataframe),
        len(LABELS),
    ):
        raise IntegratedTrainingError(
            f"Unexpected probability shape for {model_slug}: "
            f"{probabilities.shape}."
        )
    predicted_labels = decode_label_indices(
        np.argmax(probabilities, axis=1)
    )
    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }
    validation_losses = history.get("val_loss", [])
    best_epoch = (
        int(np.argmin(validation_losses)) + 1
        if validation_losses
        else len(next(iter(history.values()), []))
    )
    training_metadata: dict[str, Any] = {
        "epochs_completed": int(
            len(next(iter(history.values()), []))
        ),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            float(min(validation_losses))
            if validation_losses
            else None
        ),
        "parameter_count": int(model.count_params()),
        "random_state": RANDOM_STATE,
        "keras_backend": backend,
    }
    if vocabulary_size is not None:
        training_metadata["vocabulary_size"] = vocabulary_size

    metrics = evaluate_predictions(
        model_slug=model_slug,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        predicted_labels=predicted_labels,
        common_metadata=common_metadata,
        training_metadata=training_metadata,
    )
    write_model_outputs(
        model_slug=model_slug,
        metrics=metrics,
        predictions=prediction_table(
            validation_dataframe,
            predicted_labels,
        ),
        history=history,
        architecture=architecture_text(model),
    )
    return metrics


def run_neural_models(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    common_metadata: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    keras, backend = configure_keras_runtime()
    vocabulary = build_text_vocabulary(
        train_dataframe["description"].astype(str)
    )
    vocabulary_size = max(vocabulary.values(), default=1) + 1
    train_text = encode_text_sequences(
        train_dataframe["description"].astype(str),
        vocabulary,
    )
    validation_text = encode_text_sequences(
        validation_dataframe["description"].astype(str),
        vocabulary,
    )
    train_labels = encode_labels(train_dataframe["label"])
    validation_labels = encode_labels(validation_dataframe["label"])
    train_images = extract_image_arrays(
        train_dataframe,
        size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        flatten=False,
    )
    validation_images = extract_image_arrays(
        validation_dataframe,
        size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        flatten=False,
    )

    results: dict[str, dict[str, Any]] = {}

    keras.backend.clear_session()
    model = build_text_model(keras, vocabulary_size)
    results["keras_text"] = train_neural_model(
        keras=keras,
        model_slug="keras_text",
        model=model,
        training_inputs=train_text,
        validation_inputs=validation_text,
        training_labels=train_labels,
        validation_labels=validation_labels,
        backend=backend,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        common_metadata=common_metadata,
        vocabulary_size=vocabulary_size,
    )
    del model
    keras.backend.clear_session()
    gc.collect()

    model = build_image_model(keras)
    results["keras_image"] = train_neural_model(
        keras=keras,
        model_slug="keras_image",
        model=model,
        training_inputs=train_images,
        validation_inputs=validation_images,
        training_labels=train_labels,
        validation_labels=validation_labels,
        backend=backend,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        common_metadata=common_metadata,
    )
    del model
    keras.backend.clear_session()
    gc.collect()

    model = build_multimodal_model(keras, vocabulary_size)
    results["keras_multimodal"] = train_neural_model(
        keras=keras,
        model_slug="keras_multimodal",
        model=model,
        training_inputs={
            "description_tokens": train_text,
            "image": train_images,
        },
        validation_inputs={
            "description_tokens": validation_text,
            "image": validation_images,
        },
        training_labels=train_labels,
        validation_labels=validation_labels,
        backend=backend,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        common_metadata=common_metadata,
        vocabulary_size=vocabulary_size,
    )
    del model
    keras.backend.clear_session()
    gc.collect()
    return results


def read_development_reference_metrics() -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for model_slug, path in DEVELOPMENT_METRIC_PATHS.items():
        if not path.is_file():
            raise IntegratedTrainingError(
                f"Development metric artifact is missing: "
                f"{project_relative_path(path)}."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise IntegratedTrainingError(
                f"Development metric artifact is invalid: "
                f"{project_relative_path(path)}."
            )
        metrics[model_slug] = payload
    return metrics


def build_comparison(
    integrated_metrics: dict[str, dict[str, Any]],
    development_metrics: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_order, model_slug in enumerate(MODEL_TITLES, start=1):
        current = integrated_metrics[model_slug]
        reference = development_metrics[model_slug]
        rows.append(
            {
                "model_slug": model_slug,
                "model": MODEL_TITLES[model_slug],
                "input_modality": MODEL_MODALITIES[model_slug],
                "integrated_validation_accuracy": float(
                    current["accuracy"]
                ),
                "integrated_validation_macro_f1": float(
                    current["macro_f1"]
                ),
                "development_validation_accuracy": float(
                    reference["accuracy"]
                ),
                "development_validation_macro_f1": float(
                    reference["macro_f1"]
                ),
                "accuracy_change": float(
                    current["accuracy"] - reference["accuracy"]
                ),
                "macro_f1_change": float(
                    current["macro_f1"] - reference["macro_f1"]
                ),
                "test_split_used": False,
                "_model_order": model_order,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        by=[
            "integrated_validation_macro_f1",
            "integrated_validation_accuracy",
            "_model_order",
        ],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    comparison.insert(
        0,
        "validation_rank",
        range(1, len(comparison) + 1),
    )
    return comparison.drop(columns=["_model_order"])


def write_comparison_outputs(
    comparison: pd.DataFrame,
    common_metadata: dict[str, Any],
) -> None:
    INTEGRATED_COMPARISON_CSV_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    comparison.to_csv(
        INTEGRATED_COMPARISON_CSV_PATH,
        index=False,
        lineterminator="\n",
    )
    models = comparison.to_dict(orient="records")
    payload = {
        "status": "PASS",
        "readiness": "VALIDATION_COMPARISON_COMPLETE",
        "model_count": int(len(comparison)),
        "selection_metric": "integrated_validation_macro_f1",
        "best_model_slug": str(comparison.iloc[0]["model_slug"]),
        "best_model": str(comparison.iloc[0]["model"]),
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "data_contract": common_metadata,
        "models": models,
    }
    atomic_write_json(INTEGRATED_COMPARISON_JSON_PATH, payload)


def render_summary(
    comparison: pd.DataFrame,
    common_metadata: dict[str, Any],
    backend: str,
) -> str:
    lines = [
        "# Integrated Training Baselines and Validation Comparison",
        "",
        "- Status: **PASS**",
        "- Readiness: **VALIDATION_COMPARISON_COMPLETE**",
        (
            "- Training input: "
            "`data/processed/integrated_train.csv` "
            f"({common_metadata['training_sample_count']} samples, "
            f"{common_metadata['training_group_count']} groups)"
        ),
        (
            "- Validation input: "
            "`data/processed/integrated_validation.csv` "
            f"({common_metadata['validation_sample_count']} samples, "
            f"{common_metadata['validation_group_count']} groups)"
        ),
        f"- Keras backend recorded for this run: `{backend}`",
        "- The locked test split was not loaded as a dataset or evaluated.",
        "- Test evaluation remains prohibited by the committed lock.",
        "",
        "## Validation comparison",
        "",
        "| Rank | Model | Accuracy | Macro F1 | Development macro F1 | Change |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict(orient="records"):
        lines.append(
            f"| {int(row['validation_rank'])} "
            f"| {row['model']} "
            f"| {float(row['integrated_validation_accuracy']):.4f} "
            f"| {float(row['integrated_validation_macro_f1']):.4f} "
            f"| {float(row['development_validation_macro_f1']):.4f} "
            f"| {float(row['macro_f1_change']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The six models were trained only on the integrated training "
            "split and compared only on the integrated validation split.",
            "",
            "The comparison is a model-selection checkpoint. It is not a "
            "final test result, and it does not unlock either test CSV.",
            "",
            "The integrated data combines the generated development samples "
            "with approved open-license images while preserving physical-part "
            "group isolation.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_integrated_training_validation() -> dict[str, Any]:
    readiness = validate_external_training_readiness()
    if readiness.get("status") != "PASS" or readiness.get(
        "readiness"
    ) != "READY_FOR_TRAINING":
        raise IntegratedTrainingError(
            "External dataset integration is not READY_FOR_TRAINING."
        )

    lock = read_test_lock()
    before_fingerprints = locked_test_fingerprints(lock)
    train_dataframe, validation_dataframe = load_integrated_datasets()
    common_metadata = build_common_metadata(
        train_dataframe,
        validation_dataframe,
    )
    if common_metadata["train_validation_group_overlap"] != 0:
        raise IntegratedTrainingError(
            "Integrated train and validation groups overlap."
        )

    INTEGRATED_TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
    integrated_metrics = run_classical_models(
        train_dataframe,
        validation_dataframe,
        common_metadata,
    )
    neural_metrics = run_neural_models(
        train_dataframe,
        validation_dataframe,
        common_metadata,
    )
    integrated_metrics.update(neural_metrics)

    if set(integrated_metrics) != set(MODEL_TITLES):
        raise IntegratedTrainingError(
            "The integrated comparison does not contain all six models."
        )

    after_fingerprints = locked_test_fingerprints(lock)
    if after_fingerprints != before_fingerprints:
        raise IntegratedTrainingError(
            "A locked test artifact changed during training."
        )

    development_metrics = read_development_reference_metrics()
    comparison = build_comparison(
        integrated_metrics,
        development_metrics,
    )
    write_comparison_outputs(comparison, common_metadata)
    backend = str(
        integrated_metrics["keras_text"]["training"]["keras_backend"]
    )
    atomic_write_text(
        INTEGRATED_SUMMARY_PATH,
        render_summary(comparison, common_metadata, backend),
    )
    status = {
        "status": "PASS",
        "readiness": "VALIDATION_COMPARISON_COMPLETE",
        "model_count": len(MODEL_TITLES),
        "best_model_slug": str(comparison.iloc[0]["model_slug"]),
        "best_model": str(comparison.iloc[0]["model"]),
        "best_validation_accuracy": float(
            comparison.iloc[0]["integrated_validation_accuracy"]
        ),
        "best_validation_macro_f1": float(
            comparison.iloc[0]["integrated_validation_macro_f1"]
        ),
        "keras_backend": backend,
        "training_sample_count": int(len(train_dataframe)),
        "validation_sample_count": int(len(validation_dataframe)),
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "locked_test_fingerprints_unchanged": True,
        "locked_test_fingerprints": after_fingerprints,
        "comparison_csv": project_relative_path(
            INTEGRATED_COMPARISON_CSV_PATH
        ),
        "summary": project_relative_path(INTEGRATED_SUMMARY_PATH),
    }
    atomic_write_json(INTEGRATED_RUN_STATUS_PATH, status)
    return status


def main() -> None:
    status = run_integrated_training_validation()
    print("Integrated training baselines and validation comparison")
    print(f"- Status: {status['status']}")
    print(f"- Readiness: {status['readiness']}")
    print(f"- Models compared: {status['model_count']}")
    print(
        "- Best validation model: "
        f"{status['best_model']} "
        f"(accuracy={status['best_validation_accuracy']:.4f}, "
        f"macro_f1={status['best_validation_macro_f1']:.4f})"
    )
    print(f"- Keras backend: {status['keras_backend']}")
    print("- Locked test split used: no")


if __name__ == "__main__":
    main()
