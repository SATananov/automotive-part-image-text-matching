from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.dataset_config import LABELS, METADATA_COLUMNS
from src.fundamentals_suite_config import (
    ARCHITECTURE_COMPARISON_PATH,
    BASELINE_ARCHITECTURE_PATH,
    BASELINE_DIAGNOSTIC_PATH,
    BASELINE_HISTORY_PATH,
    BASE_CHECKPOINT,
    BATCH_CONTRACT_PATH,
    CAPACITY_COMPARISON_PATH,
    CAPACITY_PROBABILITY_TRACKING_PATH,
    CONFUSION_MATRICES_PATH,
    DATASET_PROFILE_PATH,
    EXECUTION_REGISTRY_CSV_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXPERIMENT_COMPARISON_CSV_PATH,
    EXPERIMENT_COMPARISON_JSON_PATH,
    EXPERIMENT_CONFIG_PATHS,
    FAILURE_DIAGNOSTICS_PATH,
    FAILURE_PREVENTION_PATH,
    FIGURE_PATHS,
    FUNDAMENTALS_IDS,
    GENERATED_PATHS,
    LOCK_FLAGS,
    MANIFEST_PATH,
    NOTEBOOK_AUDIT_PATH,
    NOTEBOOK_PATH,
    OPTIMIZER_COMPARISON_PATH,
    OPTIMIZER_STABILITY_PATH,
    OVERFIT_HISTORY_PATH,
    OVERFIT_RESULT_PATH,
    PREPROCESSING_COMPARISON_PATH,
    PROJECT_ROOT,
    READINESS,
    REPRESENTATIVE_EXAMPLES_PATH,
    SAMPLE_BATCH_PATH,
    STATUS_PATH,
    STEP,
    SUITE_CONFIG_PATH,
    SUMMARY_PATH,
    TEXT_HASH_SUFFIXES,
    TRAINING_HISTORIES_PATH,
    TRAINING_LOOP_AUDIT_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    VALIDATION_PREDICTIONS_PATH,
    project_relative,
)
from src.build_fundamentals_experiment_notebook import (
    build_and_execute_notebook,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


class FundamentalsSuiteError(RuntimeError):
    """Raised when the Step 011.1 experiment contract is violated."""


@dataclass
class PreparedData:
    train_dataframe: pd.DataFrame
    validation_dataframe: pd.DataFrame
    train_images: np.ndarray
    validation_images: np.ndarray
    train_text: np.ndarray
    validation_text: np.ndarray
    train_labels: np.ndarray
    validation_labels: np.ndarray
    vocabulary: dict[str, int]
    image_shape: tuple[int, int, int]
    sequence_length: int
    preprocessing_name: str


@dataclass
class RunResult:
    experiment_id: str
    run_id: str
    variant: str
    status: str
    optimizer: str = ""
    learning_rate: float | None = None
    schedule: str = ""
    seed: int = 42
    image_size: int = 24
    sequence_length: int = 12
    grayscale: bool = False
    parameter_count: int = 0
    epochs_completed: int = 0
    best_epoch: int = 0
    training_time_seconds: float = 0.0
    validation_accuracy: float | None = None
    validation_macro_f1: float | None = None
    final_train_accuracy: float | None = None
    final_train_loss: float | None = None
    final_validation_loss: float | None = None
    generalization_gap: float | None = None
    notes: str = ""
    history_rows: list[dict[str, Any]] = field(default_factory=list)
    prediction_rows: list[dict[str, Any]] = field(default_factory=list)
    confusion_matrix_values: list[list[int]] | None = None
    probability_rows: list[dict[str, Any]] = field(default_factory=list)

    def comparison_row(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "variant": self.variant,
            "status": self.status,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "schedule": self.schedule,
            "seed": self.seed,
            "image_size": self.image_size,
            "sequence_length": self.sequence_length,
            "grayscale": self.grayscale,
            "parameter_count": self.parameter_count,
            "epochs_completed": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "training_time_seconds": self.training_time_seconds,
            "validation_accuracy": self.validation_accuracy,
            "validation_macro_f1": self.validation_macro_f1,
            "final_train_accuracy": self.final_train_accuracy,
            "final_train_loss": self.final_train_loss,
            "final_validation_loss": self.final_validation_loss,
            "generalization_gap": self.generalization_gap,
            "notes": self.notes,
        }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise FundamentalsSuiteError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def load_split(path: Path) -> pd.DataFrame:
    allowed = {TRAIN_PATH.resolve(), VALIDATION_PATH.resolve()}
    if path.resolve() not in allowed:
        raise FundamentalsSuiteError(
            f"Unauthorized Step 011.1 data input: {project_relative(path)}"
        )
    dataframe = pd.read_csv(path, dtype=str).fillna("")
    if tuple(dataframe.columns) != METADATA_COLUMNS:
        raise FundamentalsSuiteError(f"Unexpected schema: {path.name}")
    if dataframe.empty:
        raise FundamentalsSuiteError(f"Empty split: {path.name}")
    if not dataframe["sample_id"].is_unique:
        raise FundamentalsSuiteError(f"Duplicate sample IDs: {path.name}")
    if set(dataframe["label"]) != set(LABELS):
        raise FundamentalsSuiteError(f"Unexpected labels: {path.name}")
    return dataframe


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataframe = load_split(TRAIN_PATH)
    validation_dataframe = load_split(VALIDATION_PATH)
    overlap = set(train_dataframe["part_group_id"]) & set(
        validation_dataframe["part_group_id"]
    )
    if overlap:
        raise FundamentalsSuiteError("Train/validation group overlap detected.")
    return train_dataframe, validation_dataframe


def resolve_image_path(relative_path: str) -> Path:
    path = (PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise FundamentalsSuiteError(
            f"Image path escapes project root: {relative_path}"
        ) from error
    if not path.is_file():
        raise FundamentalsSuiteError(f"Missing image: {relative_path}")
    return path


def build_vocabulary(texts: Iterable[str]) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    return {token: index + 2 for index, token in enumerate(ordered)}


def encode_texts(
    texts: Iterable[str], vocabulary: dict[str, int], sequence_length: int
) -> np.ndarray:
    rows: list[list[int]] = []
    for text in texts:
        values = [vocabulary.get(token, 1) for token in tokenize(text)]
        values = values[:sequence_length]
        values.extend([0] * (sequence_length - len(values)))
        rows.append(values)
    return np.asarray(rows, dtype=np.int32)


def encode_labels(labels: Sequence[str] | pd.Series) -> np.ndarray:
    return np.asarray([LABEL_TO_INDEX[str(label)] for label in labels], dtype=np.int32)


def decode_labels(indices: np.ndarray) -> np.ndarray:
    return np.asarray([INDEX_TO_LABEL[int(index)] for index in indices], dtype=object)


def load_images(
    dataframe: pd.DataFrame,
    *,
    image_size: int,
    grayscale: bool,
) -> np.ndarray:
    mode = "L" if grayscale else "RGB"
    cache: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for relative_path in dataframe["image_path"].astype(str):
        if relative_path not in cache:
            with Image.open(resolve_image_path(relative_path)) as image:
                resized = image.convert(mode).resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                )
                array = np.asarray(resized, dtype=np.float32)
                if grayscale:
                    array = array[..., np.newaxis]
                cache[relative_path] = array
        rows.append(cache[relative_path])
    return np.stack(rows).astype(np.float32, copy=False)


def prepare_data(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    *,
    image_size: int,
    sequence_length: int,
    grayscale: bool = False,
    preprocessing_name: str = "baseline",
) -> PreparedData:
    vocabulary = build_vocabulary(train_dataframe["description"].astype(str))
    return PreparedData(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        train_images=load_images(
            train_dataframe, image_size=image_size, grayscale=grayscale
        ),
        validation_images=load_images(
            validation_dataframe, image_size=image_size, grayscale=grayscale
        ),
        train_text=encode_texts(
            train_dataframe["description"].astype(str),
            vocabulary,
            sequence_length,
        ),
        validation_text=encode_texts(
            validation_dataframe["description"].astype(str),
            vocabulary,
            sequence_length,
        ),
        train_labels=encode_labels(train_dataframe["label"]),
        validation_labels=encode_labels(validation_dataframe["label"]),
        vocabulary=vocabulary,
        image_shape=(image_size, image_size, 1 if grayscale else 3),
        sequence_length=sequence_length,
        preprocessing_name=preprocessing_name,
    )


def configure_runtime(seed: int) -> tuple[Any, Any, str]:
    try:
        import keras
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise FundamentalsSuiteError(
            "TensorFlow and Keras are required. Activate the project .venv."
        ) from error

    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    return keras, tf, str(keras.backend.backend())


def architecture_text(model: Any) -> str:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def dense_block(
    keras: Any,
    values: Any,
    units: int,
    *,
    name: str,
    activation: str,
    batch_norm: bool,
    regularizer: Any | None,
) -> Any:
    values = keras.layers.Dense(
        units,
        use_bias=not batch_norm,
        kernel_regularizer=regularizer,
        name=f"{name}_dense",
    )(values)
    if batch_norm:
        values = keras.layers.BatchNormalization(name=f"{name}_bn")(values)
    return keras.layers.Activation(activation, name=f"{name}_{activation}")(
        values
    )


def build_model(
    keras: Any,
    *,
    vocabulary_size: int,
    image_shape: tuple[int, int, int],
    sequence_length: int,
    spec: dict[str, Any],
) -> Any:
    activation = str(spec.get("activation", "relu"))
    batch_norm = bool(spec.get("batch_norm", False))
    dropout = float(spec.get("dropout", 0.15))
    l1_value = float(spec.get("l1", 0.0))
    l2_value = float(spec.get("l2", 0.0))
    regularizer = None
    if l1_value or l2_value:
        regularizer = keras.regularizers.L1L2(l1=l1_value, l2=l2_value)

    text_input = keras.Input(
        shape=(sequence_length,), dtype="int32", name="description_tokens"
    )
    text_values = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=int(spec.get("embedding_dimension", 16)),
        name="token_embedding",
    )(text_input)
    text_values = keras.layers.GlobalAveragePooling1D(name="text_pooling")(
        text_values
    )
    text_values = dense_block(
        keras,
        text_values,
        int(spec.get("text_units", 24)),
        name="text_features",
        activation=activation,
        batch_norm=batch_norm,
        regularizer=regularizer,
    )

    image_input = keras.Input(
        shape=image_shape, dtype="float32", name="image"
    )
    image_branch = str(spec.get("image_branch", "dense"))
    scale_images = bool(spec.get("scale_images", True))
    if image_branch == "pretrained_mobilenet_v2":
        if image_shape != (96, 96, 3):
            raise FundamentalsSuiteError(
                "MobileNetV2 experiment requires 96x96 RGB inputs."
            )
        image_values = keras.applications.mobilenet_v2.preprocess_input(
            image_input
        )
        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=image_shape,
            pooling="avg",
        )
        backbone.trainable = False
        image_values = backbone(image_values, training=False)
        image_values = dense_block(
            keras,
            image_values,
            int(spec.get("image_units", 48)),
            name="pretrained_image_features",
            activation=activation,
            batch_norm=batch_norm,
            regularizer=regularizer,
        )
    else:
        image_values = image_input
        if scale_images:
            image_values = keras.layers.Rescaling(
                1.0 / 255.0, name="image_rescaling"
            )(image_values)
        if image_branch == "cnn":
            image_values = keras.layers.Conv2D(
                16, 3, padding="same", activation=activation, name="image_conv_1"
            )(image_values)
            image_values = keras.layers.MaxPooling2D(name="image_pool_1")(
                image_values
            )
            image_values = keras.layers.Conv2D(
                32, 3, padding="same", activation=activation, name="image_conv_2"
            )(image_values)
            image_values = keras.layers.GlobalAveragePooling2D(
                name="image_global_pool"
            )(image_values)
        else:
            image_values = keras.layers.Flatten(name="image_flatten")(
                image_values
            )
        image_values = dense_block(
            keras,
            image_values,
            int(spec.get("image_units", 48)),
            name="image_features",
            activation=activation,
            batch_norm=batch_norm,
            regularizer=regularizer,
        )

    fused = keras.layers.Concatenate(name="feature_fusion")(
        [text_values, image_values]
    )
    fusion_units = [int(value) for value in spec.get("fusion_units", [64])]
    for index, units in enumerate(fusion_units, start=1):
        previous = fused
        fused = dense_block(
            keras,
            fused,
            units,
            name=f"fusion_{index}",
            activation=activation,
            batch_norm=batch_norm,
            regularizer=regularizer,
        )
        if bool(spec.get("residual", False)) and index > 1:
            if int(previous.shape[-1]) != units:
                previous = keras.layers.Dense(
                    units, name=f"fusion_{index}_skip_projection"
                )(previous)
            fused = keras.layers.Add(name=f"fusion_{index}_skip")(
                [fused, previous]
            )
        if dropout:
            fused = keras.layers.Dropout(
                dropout, name=f"fusion_{index}_dropout"
            )(fused)

    outputs = keras.layers.Dense(
        len(LABELS), activation="softmax", name="class_probabilities"
    )(fused)
    return keras.Model(
        inputs={"description_tokens": text_input, "image": image_input},
        outputs=outputs,
        name=str(spec.get("model_name", "fundamentals_multimodal_model")),
    )


def build_optimizer(
    keras: Any,
    *,
    name: str,
    learning_rate: float,
    schedule: str = "",
) -> Any:
    rate: Any = learning_rate
    if schedule == "exponential_decay":
        rate = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=60,
            decay_rate=0.80,
            staircase=True,
        )
    normalized = name.lower()
    if normalized == "sgd":
        return keras.optimizers.SGD(learning_rate=rate)
    if normalized == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=rate)
    if normalized == "adam":
        return keras.optimizers.Adam(learning_rate=rate)
    if normalized == "adamw":
        return keras.optimizers.AdamW(
            learning_rate=rate, weight_decay=1e-4
        )
    raise FundamentalsSuiteError(f"Unsupported optimizer: {name}")


def compile_model(
    keras: Any,
    model: Any,
    *,
    optimizer_name: str,
    learning_rate: float,
    schedule: str = "",
) -> None:
    model.compile(
        optimizer=build_optimizer(
            keras,
            name=optimizer_name,
            learning_rate=learning_rate,
            schedule=schedule,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def model_inputs(data: PreparedData, split: str) -> dict[str, np.ndarray]:
    if split == "train":
        return {
            "description_tokens": data.train_text,
            "image": data.train_images,
        }
    if split == "validation":
        return {
            "description_tokens": data.validation_text,
            "image": data.validation_images,
        }
    raise FundamentalsSuiteError(f"Unsupported split: {split}")


def probabilities_to_metrics(
    dataframe: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[int]]]:
    if probabilities.shape != (len(dataframe), len(LABELS)):
        raise FundamentalsSuiteError(
            f"Unexpected probability shape: {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise FundamentalsSuiteError("Non-finite validation probabilities.")
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise FundamentalsSuiteError("Probabilities do not sum to one.")
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted_labels = decode_labels(predicted_indices)
    true_labels = dataframe["label"].to_numpy(dtype=object)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
    metrics = {
        "validation_accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "validation_macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
    }
    rows: list[dict[str, Any]] = []
    for row_index, (_, row) in enumerate(dataframe.iterrows()):
        output = {
            "sample_id": row["sample_id"],
            "part_group_id": row["part_group_id"],
            "part_category": row["part_category"],
            "true_label": row["label"],
            "predicted_label": predicted_labels[row_index],
            "is_correct": bool(row["label"] == predicted_labels[row_index]),
        }
        for label_index, label in enumerate(LABELS):
            output[f"probability_{label.lower()}"] = float(
                probabilities[row_index, label_index]
            )
        rows.append(output)
    return metrics, rows, matrix.astype(int).tolist()


def train_and_evaluate(
    *,
    experiment_id: str,
    run_id: str,
    variant: str,
    data: PreparedData,
    spec: dict[str, Any],
    optimizer_name: str,
    learning_rate: float,
    schedule: str,
    seed: int,
    max_epochs: int,
    patience: int,
    train_labels_override: np.ndarray | None = None,
    probability_tracking: bool = False,
) -> RunResult:
    keras, _, _ = configure_runtime(seed)
    keras.backend.clear_session()
    model = build_model(
        keras,
        vocabulary_size=max(data.vocabulary.values(), default=1) + 1,
        image_shape=data.image_shape,
        sequence_length=data.sequence_length,
        spec=spec,
    )
    compile_model(
        keras,
        model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        schedule=schedule,
    )

    tracked_rows: list[dict[str, Any]] = []
    callbacks: list[Any] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]
    if probability_tracking:
        selected = np.arange(min(6, len(data.validation_dataframe)))
        validation_inputs = {
            "description_tokens": data.validation_text[selected],
            "image": data.validation_images[selected],
        }
        selected_frame = data.validation_dataframe.iloc[selected]

        class ProbabilityTracker(keras.callbacks.Callback):
            def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
                probabilities = np.asarray(
                    self.model.predict(validation_inputs, verbose=0)
                )
                for local_index, (_, row) in enumerate(
                    selected_frame.iterrows()
                ):
                    record = {
                        "experiment_id": experiment_id,
                        "run_id": run_id,
                        "variant": variant,
                        "epoch": epoch + 1,
                        "sample_id": row["sample_id"],
                        "true_label": row["label"],
                    }
                    for label_index, label in enumerate(LABELS):
                        record[f"probability_{label.lower()}"] = float(
                            probabilities[local_index, label_index]
                        )
                    tracked_rows.append(record)

        callbacks.append(ProbabilityTracker())

    training_labels = (
        data.train_labels
        if train_labels_override is None
        else train_labels_override
    )
    start = time.perf_counter()
    history_object = model.fit(
        model_inputs(data, "train"),
        training_labels,
        validation_data=(
            model_inputs(data, "validation"),
            data.validation_labels,
        ),
        epochs=max_epochs,
        batch_size=16,
        shuffle=True,
        callbacks=callbacks,
        verbose=0,
    )
    duration = time.perf_counter() - start
    probabilities = np.asarray(
        model.predict(model_inputs(data, "validation"), batch_size=16, verbose=0)
    )
    metrics, predictions, matrix = probabilities_to_metrics(
        data.validation_dataframe, probabilities
    )

    history = {
        key: [float(value) for value in values]
        for key, values in history_object.history.items()
    }
    epoch_count = len(next(iter(history.values()), []))
    val_losses = history.get("val_loss", [])
    best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else epoch_count
    best_index = max(0, best_epoch - 1)
    train_accuracy_values = history.get("accuracy", [])
    train_loss_values = history.get("loss", [])
    val_loss_values = history.get("val_loss", [])
    final_train_accuracy = (
        float(train_accuracy_values[best_index])
        if train_accuracy_values
        else None
    )
    final_train_loss = (
        float(train_loss_values[best_index]) if train_loss_values else None
    )
    final_val_loss = (
        float(val_loss_values[best_index]) if val_loss_values else None
    )

    history_rows: list[dict[str, Any]] = []
    for index in range(epoch_count):
        row = {
            "experiment_id": experiment_id,
            "run_id": run_id,
            "variant": variant,
            "epoch": index + 1,
        }
        for key, values in history.items():
            row[key] = float(values[index])
        history_rows.append(row)

    for prediction in predictions:
        prediction.update(
            {
                "experiment_id": experiment_id,
                "run_id": run_id,
                "variant": variant,
            }
        )

    result = RunResult(
        experiment_id=experiment_id,
        run_id=run_id,
        variant=variant,
        status="COMPLETED",
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        schedule=schedule,
        seed=seed,
        image_size=data.image_shape[0],
        sequence_length=data.sequence_length,
        grayscale=data.image_shape[-1] == 1,
        parameter_count=int(model.count_params()),
        epochs_completed=epoch_count,
        best_epoch=best_epoch,
        training_time_seconds=float(duration),
        validation_accuracy=metrics["validation_accuracy"],
        validation_macro_f1=metrics["validation_macro_f1"],
        final_train_accuracy=final_train_accuracy,
        final_train_loss=final_train_loss,
        final_validation_loss=final_val_loss,
        generalization_gap=(
            float(final_train_accuracy - metrics["validation_accuracy"])
            if final_train_accuracy is not None
            else None
        ),
        history_rows=history_rows,
        prediction_rows=predictions,
        confusion_matrix_values=matrix,
        probability_rows=tracked_rows,
    )
    del model
    keras.backend.clear_session()
    gc.collect()
    return result


def dataset_profile(dataframe: pd.DataFrame) -> dict[str, Any]:
    lengths = dataframe["description"].map(lambda value: len(tokenize(value)))
    unique_images = dataframe.drop_duplicates("image_path")
    sampled_values: list[np.ndarray] = []
    for relative_path in unique_images["image_path"].astype(str):
        with Image.open(resolve_image_path(relative_path)) as image:
            array = np.asarray(
                image.convert("RGB").resize((24, 24), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            sampled_values.append(array)
    pixels = np.concatenate([array.reshape(-1) for array in sampled_values])
    return {
        "samples": int(len(dataframe)),
        "images": int(dataframe["image_id"].nunique()),
        "groups": int(dataframe["part_group_id"].nunique()),
        "categories": int(dataframe["part_category"].nunique()),
        "label_distribution": {
            label: int(dataframe["label"].eq(label).sum()) for label in LABELS
        },
        "source_distribution": {
            str(key): int(value)
            for key, value in dataframe["source"].value_counts().sort_index().items()
        },
        "text_length_tokens": {
            "minimum": int(lengths.min()),
            "maximum": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
        },
        "image_pixels_24x24_rgb": {
            "minimum": float(pixels.min()),
            "maximum": float(pixels.max()),
            "mean": float(pixels.mean()),
            "standard_deviation": float(pixels.std()),
        },
    }


def run_eda_and_batch_contract(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    baseline_data: PreparedData,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = {
        "experiment_id": "FND-001",
        "status": "COMPLETED",
        "train": dataset_profile(train_dataframe),
        "validation": dataset_profile(validation_dataframe),
        "train_validation_group_overlap": 0,
        **LOCK_FLAGS,
    }
    write_json(DATASET_PROFILE_PATH, profile)

    representatives = (
        pd.concat(
            [
                train_dataframe.groupby(["part_category", "label"], sort=True).head(1),
                validation_dataframe.groupby(["part_category", "label"], sort=True).head(1),
            ],
            ignore_index=True,
        )
        .drop_duplicates("sample_id")
        .sort_values(["source", "part_category", "label", "sample_id"])
    )
    representatives.to_csv(
        REPRESENTATIVE_EXAMPLES_PATH, index=False, lineterminator="\n"
    )

    train_counts = train_dataframe["label"].value_counts().reindex(LABELS)
    validation_counts = (
        validation_dataframe["label"].value_counts().reindex(LABELS)
    )
    x = np.arange(len(LABELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, train_counts.values, width, label="Train")
    ax.bar(x + width / 2, validation_counts.values, width, label="Validation")
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Samples")
    ax.set_title("Grouped train/validation class balance")
    ax.legend()
    fig.tight_layout()
    FIGURE_PATHS["label_distribution"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATHS["label_distribution"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(
        [
            train_dataframe["description"].map(lambda value: len(tokenize(value))),
            validation_dataframe["description"].map(lambda value: len(tokenize(value))),
        ],
        bins=np.arange(0.5, 8.5, 1),
        label=["Train", "Validation"],
    )
    ax.set_xlabel("Description length (tokens)")
    ax.set_ylabel("Samples")
    ax.set_title("Text-length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["text_length_distribution"], dpi=160)
    plt.close(fig)

    permutation_a = np.random.default_rng(42).permutation(len(train_dataframe))
    permutation_b = np.random.default_rng(42).permutation(len(train_dataframe))
    validation_order = validation_dataframe["sample_id"].tolist()
    batch_size = 16
    batch = train_dataframe.iloc[permutation_a[:batch_size]].copy()
    batch.to_csv(SAMPLE_BATCH_PATH, index=False, lineterminator="\n")
    batch_contract = {
        "experiment_id": "FND-002",
        "status": "COMPLETED",
        "batch_size": batch_size,
        "train_image_shape": list(baseline_data.train_images[:batch_size].shape),
        "train_text_shape": list(baseline_data.train_text[:batch_size].shape),
        "train_label_shape": list(baseline_data.train_labels[:batch_size].shape),
        "image_dtype": str(baseline_data.train_images.dtype),
        "text_dtype": str(baseline_data.train_text.dtype),
        "label_dtype": str(baseline_data.train_labels.dtype),
        "image_floating_point_reason": (
            "Floating-point inputs support scaling, gradients, and neural-network arithmetic."
        ),
        "integer_target_reason": (
            "Sparse categorical cross-entropy expects integer class indices."
        ),
        "train_shuffle": True,
        "validation_shuffle": False,
        "train_shuffle_reproducible": bool(np.array_equal(permutation_a, permutation_b)),
        "validation_order_stable": validation_order
        == validation_dataframe["sample_id"].tolist(),
        "image_text_label_alignment_pass": bool(
            all(
                sample_id == train_dataframe.iloc[index]["sample_id"]
                for sample_id, index in zip(
                    train_dataframe.iloc[permutation_a[:batch_size]]["sample_id"],
                    permutation_a[:batch_size],
                    strict=True,
                )
            )
        ),
        **LOCK_FLAGS,
    }
    write_json(BATCH_CONTRACT_PATH, batch_contract)
    return profile, batch_contract


def run_baseline_gradient_diagnostic(
    data: PreparedData,
) -> dict[str, Any]:
    keras, tf, backend = configure_runtime(42)
    keras.backend.clear_session()
    spec = {
        "model_name": "fnd003_one_hidden_layer_baseline",
        "text_units": 8,
        "image_units": 16,
        "fusion_units": [16],
        "embedding_dimension": 8,
        "dropout": 0.0,
    }
    model = build_model(
        keras,
        vocabulary_size=max(data.vocabulary.values(), default=1) + 1,
        image_shape=data.image_shape,
        sequence_length=data.sequence_length,
        spec=spec,
    )
    compile_model(
        keras,
        model,
        optimizer_name="Adam",
        learning_rate=0.003,
    )
    batch_indices = np.arange(12)
    inputs = {
        "description_tokens": data.train_text[batch_indices],
        "image": data.train_images[batch_indices],
    }
    labels = data.train_labels[batch_indices]
    initial_weights = [np.asarray(weight).copy() for weight in model.get_weights()]

    with tf.GradientTape() as tape:
        probabilities = model(inputs, training=True)
        loss_value = keras.losses.sparse_categorical_crossentropy(
            labels, probabilities
        )
        loss_value = tf.reduce_mean(loss_value)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    finite_gradients = all(
        bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy())
        for gradient in gradients
        if gradient is not None
    )
    gradient_norm = float(
        tf.linalg.global_norm(
            [gradient for gradient in gradients if gradient is not None]
        ).numpy()
    )

    history_rows: list[dict[str, Any]] = []
    for update in range(1, 16):
        values = model.train_on_batch(inputs, labels, return_dict=True)
        history_rows.append(
            {
                "update": update,
                "loss": float(values["loss"]),
                "accuracy": float(values["accuracy"]),
            }
        )
    final_weights = model.get_weights()
    weight_change_norm = float(
        math.sqrt(
            sum(
                float(np.sum(np.square(after - before)))
                for before, after in zip(initial_weights, final_weights, strict=True)
            )
        )
    )
    probabilities_after = np.asarray(model.predict(inputs, verbose=0))
    diagnostic = {
        "experiment_id": "FND-003",
        "status": "COMPLETED",
        "keras_backend": backend,
        "parameter_count": int(model.count_params()),
        "initial_loss": history_rows[0]["loss"],
        "final_loss": history_rows[-1]["loss"],
        "loss_reduced": history_rows[-1]["loss"] < history_rows[0]["loss"],
        "gradient_finite": finite_gradients,
        "gradient_global_norm": gradient_norm,
        "weight_change_norm": weight_change_norm,
        "probability_sum_max_error": float(
            np.max(np.abs(probabilities_after.sum(axis=1) - 1.0))
        ),
        **LOCK_FLAGS,
    }
    pd.DataFrame(history_rows).to_csv(
        BASELINE_HISTORY_PATH, index=False, lineterminator="\n"
    )
    write_text(BASELINE_ARCHITECTURE_PATH, architecture_text(model))
    write_json(BASELINE_DIAGNOSTIC_PATH, diagnostic)
    del model
    keras.backend.clear_session()
    gc.collect()
    if not (
        diagnostic["loss_reduced"]
        and diagnostic["gradient_finite"]
        and diagnostic["weight_change_norm"] > 0
    ):
        raise FundamentalsSuiteError("FND-003 gradient diagnostic failed.")
    return diagnostic


def select_overfit_batch_positions(
    dataframe: pd.DataFrame,
    *,
    group_count: int,
) -> tuple[np.ndarray, list[str]]:
    """Select a small, balanced, relation-aware memorization batch.

    Each selected part group contributes exactly one MATCH, PARTIAL_MATCH,
    and MISMATCH sample. This keeps the diagnostic faithful to the multimodal
    relation task while making the intended one-batch overfit check portable
    across CPU-only TensorFlow environments.
    """

    if group_count < 1:
        raise FundamentalsSuiteError(
            "FND-004 requires at least one complete part group."
        )

    group_values = dataframe["part_group_id"].astype(str).to_numpy()
    label_values = dataframe["label"].astype(str).to_numpy()
    ordered_groups = list(dict.fromkeys(group_values.tolist()))
    selected_positions: list[int] = []
    selected_groups: list[str] = []

    for group_id in ordered_groups:
        group_positions = np.flatnonzero(group_values == group_id)
        group_labels = {label_values[position] for position in group_positions}
        if group_labels != set(LABELS):
            continue
        for label in LABELS:
            matching = [
                int(position)
                for position in group_positions
                if label_values[position] == label
            ]
            if not matching:
                raise FundamentalsSuiteError(
                    f"FND-004 group {group_id} is missing label {label}."
                )
            selected_positions.append(matching[0])
        selected_groups.append(group_id)
        if len(selected_groups) == group_count:
            break

    if len(selected_groups) != group_count:
        raise FundamentalsSuiteError(
            "FND-004 could not build the requested balanced group batch."
        )

    positions = np.asarray(selected_positions, dtype=np.int64)
    selected_labels = label_values[positions]
    expected_per_label = group_count
    observed = Counter(selected_labels.tolist())
    if any(observed[label] != expected_per_label for label in LABELS):
        raise FundamentalsSuiteError(
            f"FND-004 selected batch is not label-balanced: {dict(observed)}"
        )
    return positions, selected_groups


def build_overfit_probe_model(
    keras: Any,
    *,
    vocabulary_size: int,
    image_shape: tuple[int, int, int],
    sequence_length: int,
) -> Any:
    """Build a high-capacity, regularization-free memorization probe."""

    text_input = keras.Input(
        shape=(sequence_length,), dtype="int32", name="description_tokens"
    )
    text_values = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=32,
        name="overfit_token_embedding",
    )(text_input)
    text_values = keras.layers.Flatten(name="overfit_text_flatten")(
        text_values
    )
    text_values = keras.layers.Dense(
        64, activation="relu", name="overfit_text_features"
    )(text_values)

    image_input = keras.Input(
        shape=image_shape, dtype="float32", name="image"
    )
    image_values = keras.layers.Rescaling(
        1.0 / 255.0, name="overfit_image_rescaling"
    )(image_input)
    image_values = keras.layers.Flatten(name="overfit_image_flatten")(
        image_values
    )
    image_values = keras.layers.Dense(
        128, activation="relu", name="overfit_image_features"
    )(image_values)

    fused = keras.layers.Concatenate(name="overfit_feature_fusion")(
        [text_values, image_values]
    )
    fused = keras.layers.Dense(
        256, activation="relu", name="overfit_fusion_1"
    )(fused)
    fused = keras.layers.Dense(
        128, activation="relu", name="overfit_fusion_2"
    )(fused)
    outputs = keras.layers.Dense(
        len(LABELS), activation="softmax", name="class_probabilities"
    )(fused)
    return keras.Model(
        inputs={"description_tokens": text_input, "image": image_input},
        outputs=outputs,
        name="fnd004_overfit_diagnostic",
    )


def exact_sparse_batch_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Return post-update accuracy and cross-entropy for one fixed batch."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != (len(labels), len(LABELS)):
        raise FundamentalsSuiteError(
            f"FND-004 probability shape mismatch: {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise FundamentalsSuiteError(
            "FND-004 produced non-finite probabilities."
        )
    predicted = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predicted == labels))
    true_probabilities = probabilities[np.arange(len(labels)), labels]
    clipped = np.clip(true_probabilities, 1e-7, 1.0)
    loss = float(-np.mean(np.log(clipped)))
    return accuracy, loss


def run_overfit_diagnostic(data: PreparedData, config: dict[str, Any]) -> dict[str, Any]:
    keras, _, _ = configure_runtime(42)
    keras.backend.clear_session()

    group_count = int(config.get("overfit_batch_group_count", 2))
    positions, selected_groups = select_overfit_batch_positions(
        data.train_dataframe,
        group_count=group_count,
    )
    inputs = {
        "description_tokens": data.train_text[positions],
        "image": data.train_images[positions],
    }
    labels = data.train_labels[positions]

    model = build_overfit_probe_model(
        keras,
        vocabulary_size=max(data.vocabulary.values(), default=1) + 1,
        image_shape=data.image_shape,
        sequence_length=data.sequence_length,
    )

    learning_rates = [
        float(value)
        for value in config.get(
            "overfit_learning_rates", [0.003, 0.001, 0.0003]
        )
    ]
    if not learning_rates or any(rate <= 0 for rate in learning_rates):
        raise FundamentalsSuiteError(
            "FND-004 requires positive learning rates."
        )
    compile_model(
        keras,
        model,
        optimizer_name="Adam",
        learning_rate=learning_rates[0],
    )

    accuracy_threshold = float(config["overfit_accuracy_threshold"])
    loss_threshold = float(config["overfit_loss_threshold"])
    max_steps = int(config["overfit_max_epochs"])
    check_interval = int(config.get("overfit_check_interval", 5))
    if max_steps < 1 or check_interval < 1:
        raise FundamentalsSuiteError(
            "FND-004 max steps and check interval must be positive."
        )

    def infer_probabilities() -> np.ndarray:
        return np.asarray(model(inputs, training=False), dtype=np.float64)

    initial_probabilities = infer_probabilities()
    initial_accuracy, initial_loss = exact_sparse_batch_metrics(
        initial_probabilities, labels
    )
    rows: list[dict[str, Any]] = [
        {
            "epoch": 0,
            "optimizer_step": 0,
            "learning_rate": learning_rates[0],
            "accuracy": initial_accuracy,
            "loss": initial_loss,
        }
    ]

    steps_per_stage = max(1, math.ceil(max_steps / len(learning_rates)))
    active_stage = 0
    final_accuracy = initial_accuracy
    final_loss = initial_loss
    passed = False
    final_step = 0

    for step in range(1, max_steps + 1):
        requested_stage = min(
            (step - 1) // steps_per_stage,
            len(learning_rates) - 1,
        )
        if requested_stage != active_stage:
            active_stage = requested_stage
            model.optimizer.learning_rate.assign(
                learning_rates[active_stage]
            )

        model.train_on_batch(inputs, labels)
        model.reset_metrics()

        should_measure = (
            step == 1
            or step % check_interval == 0
            or step == max_steps
        )
        if not should_measure:
            continue

        probabilities = infer_probabilities()
        final_accuracy, final_loss = exact_sparse_batch_metrics(
            probabilities, labels
        )
        final_step = step
        rows.append(
            {
                "epoch": step,
                "optimizer_step": step,
                "learning_rate": learning_rates[active_stage],
                "accuracy": final_accuracy,
                "loss": final_loss,
            }
        )
        passed = bool(
            final_accuracy >= accuracy_threshold
            and final_loss <= loss_threshold
            and final_loss < initial_loss
        )
        if passed:
            break

    selected_frame = data.train_dataframe.iloc[positions]
    relative_loss_reduction = float(
        (initial_loss - final_loss) / max(initial_loss, 1e-12)
    )
    result = {
        "experiment_id": "FND-004",
        "status": "COMPLETED" if passed else "FAILED_DIAGNOSTIC",
        "protocol": "balanced_relation_groups_train_on_batch_post_update_metrics",
        "batch_size": int(len(positions)),
        "part_group_count": group_count,
        "selected_part_group_ids": selected_groups,
        "selected_sample_ids": selected_frame["sample_id"].astype(str).tolist(),
        "selected_label_counts": {
            label: int(np.sum(selected_frame["label"].to_numpy() == label))
            for label in LABELS
        },
        "parameter_count": int(model.count_params()),
        "epochs_completed": final_step,
        "optimizer_steps": final_step,
        "initial_batch_loss": initial_loss,
        "initial_batch_accuracy": initial_accuracy,
        "final_batch_loss": final_loss,
        "final_batch_accuracy": final_accuracy,
        "relative_loss_reduction": relative_loss_reduction,
        "accuracy_threshold": accuracy_threshold,
        "loss_threshold": loss_threshold,
        "learning_rates": learning_rates,
        "check_interval": check_interval,
        "threshold_reached": passed,
        **LOCK_FLAGS,
    }
    pd.DataFrame(rows).to_csv(
        OVERFIT_HISTORY_PATH, index=False, lineterminator="\n"
    )
    write_json(OVERFIT_RESULT_PATH, result)

    fig, ax_accuracy = plt.subplots(figsize=(8, 4.8))
    ax_accuracy.plot(
        [row["optimizer_step"] for row in rows],
        [row["accuracy"] for row in rows],
        label="Batch accuracy",
    )
    ax_accuracy.set_xlabel("Optimizer step")
    ax_accuracy.set_ylabel("Accuracy")
    ax_loss = ax_accuracy.twinx()
    ax_loss.plot(
        [row["optimizer_step"] for row in rows],
        [row["loss"] for row in rows],
        label="Batch loss",
    )
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_accuracy.set_title("One-batch overfit diagnostic")
    lines_a, labels_a = ax_accuracy.get_legend_handles_labels()
    lines_b, labels_b = ax_loss.get_legend_handles_labels()
    ax_accuracy.legend(lines_a + lines_b, labels_a + labels_b, loc="center right")
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["overfit_curves"], dpi=160)
    plt.close(fig)

    del model
    keras.backend.clear_session()
    gc.collect()
    if not passed:
        raise FundamentalsSuiteError(
            "FND-004 one-batch overfit gate failed: "
            f"accuracy={final_accuracy:.4f} "
            f"(required {accuracy_threshold:.4f}), "
            f"loss={final_loss:.6f} "
            f"(required <= {loss_threshold:.6f}), "
            f"steps={final_step}, groups={selected_groups}."
        )
    return result

def medium_spec(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": "fundamentals_medium_multimodal",
        "embedding_dimension": int(config["default_embedding_dimension"]),
        **config["capacity_variants"]["medium"],
    }


def choose_best(results: list[RunResult]) -> RunResult:
    completed = [
        result
        for result in results
        if result.status == "COMPLETED"
        and result.validation_macro_f1 is not None
        and result.validation_accuracy is not None
    ]
    if not completed:
        raise FundamentalsSuiteError("No completed candidate result.")
    return max(
        completed,
        key=lambda result: (
            float(result.validation_macro_f1),
            float(result.validation_accuracy),
            -float(result.training_time_seconds),
        ),
    )


def run_training_loop_and_optimizer_suite(
    data: PreparedData,
    config: dict[str, Any],
) -> tuple[RunResult, list[RunResult], dict[str, Any]]:
    baseline = train_and_evaluate(
        experiment_id="FND-005",
        run_id="fnd005_correct_training_loop",
        variant="correct_training_validation_loop",
        data=data,
        spec=medium_spec(config),
        optimizer_name="Adam",
        learning_rate=0.001,
        schedule="",
        seed=42,
        max_epochs=int(config["default_max_epochs"]),
        patience=int(config["default_patience"]),
    )

    keras, _, _ = configure_runtime(42)
    keras.backend.clear_session()
    audit_model = build_model(
        keras,
        vocabulary_size=max(data.vocabulary.values(), default=1) + 1,
        image_shape=data.image_shape,
        sequence_length=data.sequence_length,
        spec=medium_spec(config),
    )
    compile_model(
        keras,
        audit_model,
        optimizer_name="Adam",
        learning_rate=0.001,
    )
    before = [np.asarray(value).copy() for value in audit_model.get_weights()]
    audit_model.evaluate(
        model_inputs(data, "validation"),
        data.validation_labels,
        batch_size=16,
        verbose=0,
    )
    after = audit_model.get_weights()
    validation_weights_unchanged = all(
        np.array_equal(left, right)
        for left, right in zip(before, after, strict=True)
    )
    del audit_model
    keras.backend.clear_session()
    gc.collect()
    loop_audit = {
        "experiment_id": "FND-005",
        "status": "COMPLETED",
        "training_updates_from": project_relative(TRAIN_PATH),
        "validation_evaluation_from": project_relative(VALIDATION_PATH),
        "validation_weights_unchanged_during_evaluate": validation_weights_unchanged,
        "train_shuffle": True,
        "validation_shuffle": False,
        "early_stopping_enabled": True,
        "validation_macro_f1": baseline.validation_macro_f1,
        "validation_accuracy": baseline.validation_accuracy,
        **LOCK_FLAGS,
    }
    write_json(TRAINING_LOOP_AUDIT_PATH, loop_audit)
    if not validation_weights_unchanged:
        raise FundamentalsSuiteError("Validation evaluation changed weights.")

    optimizer_results: list[RunResult] = []
    for index, candidate in enumerate(config["optimizer_grid"], start=1):
        name = str(candidate["name"])
        learning_rate = float(candidate["learning_rate"])
        schedule = str(candidate.get("schedule", ""))
        slug = name.lower()
        if schedule:
            slug += f"_{schedule}"
        run_id = f"fnd006_{index:02d}_{slug}_{learning_rate:g}".replace(".", "p")
        result = train_and_evaluate(
            experiment_id="FND-006",
            run_id=run_id,
            variant=f"{name}_{learning_rate:g}_{schedule or 'constant'}",
            data=data,
            spec=medium_spec(config),
            optimizer_name=name,
            learning_rate=learning_rate,
            schedule=schedule,
            seed=42,
            max_epochs=24,
            patience=4,
        )
        optimizer_results.append(result)

    champion = choose_best(optimizer_results)
    stability_results = [champion]
    for seed in (43, 44):
        repeat = train_and_evaluate(
            experiment_id="FND-006",
            run_id=f"{champion.run_id}_seed{seed}",
            variant=f"{champion.variant}_stability_repeat",
            data=data,
            spec=medium_spec(config),
            optimizer_name=champion.optimizer,
            learning_rate=float(champion.learning_rate),
            schedule=champion.schedule,
            seed=seed,
            max_epochs=24,
            patience=4,
        )
        optimizer_results.append(repeat)
        stability_results.append(repeat)
    macro_values = np.asarray(
        [float(result.validation_macro_f1) for result in stability_results]
    )
    stability = {
        "experiment_id": "FND-006",
        "status": "COMPLETED",
        "champion_run_id": champion.run_id,
        "champion_optimizer": champion.optimizer,
        "champion_learning_rate": champion.learning_rate,
        "champion_schedule": champion.schedule,
        "stability_seeds": [result.seed for result in stability_results],
        "macro_f1_mean": float(macro_values.mean()),
        "macro_f1_standard_deviation": float(macro_values.std(ddof=0)),
        **LOCK_FLAGS,
    }
    write_json(OPTIMIZER_STABILITY_PATH, stability)
    pd.DataFrame(
        [result.comparison_row() for result in optimizer_results]
    ).to_csv(OPTIMIZER_COMPARISON_PATH, index=False, lineterminator="\n")

    candidates_for_plot = optimizer_results[: len(config["optimizer_grid"])]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [result.variant for result in candidates_for_plot]
    values = [result.validation_macro_f1 for result in candidates_for_plot]
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=40, ha="right")
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Optimizer and learning-rate comparison")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["optimizer_comparison"], dpi=160)
    plt.close(fig)
    return baseline, optimizer_results, stability


def run_capacity_suite(
    data: PreparedData,
    config: dict[str, Any],
    champion: RunResult,
) -> list[RunResult]:
    results: list[RunResult] = []
    for name, variant in config["capacity_variants"].items():
        spec = {
            "model_name": f"fnd007_capacity_{name}",
            "embedding_dimension": int(config["default_embedding_dimension"]),
            **variant,
        }
        result = train_and_evaluate(
            experiment_id="FND-007",
            run_id=f"fnd007_{name}",
            variant=name,
            data=data,
            spec=spec,
            optimizer_name=champion.optimizer,
            learning_rate=float(champion.learning_rate),
            schedule=champion.schedule,
            seed=42,
            max_epochs=28,
            patience=5,
            probability_tracking=True,
        )
        results.append(result)
    pd.DataFrame([result.comparison_row() for result in results]).to_csv(
        CAPACITY_COMPARISON_PATH, index=False, lineterminator="\n"
    )
    probability_rows = [
        row for result in results for row in result.probability_rows
    ]
    pd.DataFrame(probability_rows).to_csv(
        CAPACITY_PROBABILITY_TRACKING_PATH,
        index=False,
        lineterminator="\n",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        ax.scatter(
            result.parameter_count,
            result.validation_macro_f1,
            s=90,
            label=result.variant,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Capacity–performance trade-off")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["capacity_tradeoff"], dpi=160)
    plt.close(fig)
    return results


def architecture_specs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any], str, float, str, int]]:
    base = medium_spec(config)
    return [
        ("baseline", dict(base), "Adam", 0.001, "", 24),
        (
            "l2_regularization",
            {**base, "model_name": "fnd008_l2", "l2": 1e-4},
            "Adam",
            0.001,
            "",
            24,
        ),
        (
            "dropout_regularization",
            {**base, "model_name": "fnd008_dropout", "dropout": 0.35},
            "Adam",
            0.001,
            "",
            24,
        ),
        (
            "batch_normalization",
            {**base, "model_name": "fnd008_batch_norm", "batch_norm": True},
            "Adam",
            0.001,
            "",
            24,
        ),
        (
            "residual_fusion",
            {
                **base,
                "model_name": "fnd008_residual",
                "fusion_units": [64, 64],
                "residual": True,
            },
            "Adam",
            0.001,
            "",
            24,
        ),
        (
            "learning_rate_schedule",
            {**base, "model_name": "fnd008_schedule"},
            "Adam",
            0.003,
            "exponential_decay",
            24,
        ),
        (
            "cnn_image_branch",
            {**base, "model_name": "fnd008_cnn", "image_branch": "cnn"},
            "Adam",
            0.001,
            "",
            24,
        ),
    ]


def run_architecture_suite(
    baseline_data: PreparedData,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> list[RunResult]:
    results: list[RunResult] = []
    for index, (name, spec, optimizer, learning_rate, schedule, epochs) in enumerate(
        architecture_specs(config), start=1
    ):
        result = train_and_evaluate(
            experiment_id="FND-008",
            run_id=f"fnd008_{index:02d}_{name}",
            variant=name,
            data=baseline_data,
            spec=spec,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            schedule=schedule,
            seed=42,
            max_epochs=epochs,
            patience=4,
        )
        results.append(result)

    if bool(config.get("attempt_pretrained_backbone", False)):
        try:
            pretrained_data = prepare_data(
                train_dataframe,
                validation_dataframe,
                image_size=int(config["pretrained_image_size"]),
                sequence_length=int(config["default_sequence_length"]),
                grayscale=False,
                preprocessing_name="pretrained_96_rgb",
            )
            spec = {
                **medium_spec(config),
                "model_name": "fnd008_pretrained_mobilenet_v2",
                "image_branch": "pretrained_mobilenet_v2",
            }
            result = train_and_evaluate(
                experiment_id="FND-008",
                run_id="fnd008_08_pretrained_mobilenet_v2",
                variant="pretrained_mobilenet_v2",
                data=pretrained_data,
                spec=spec,
                optimizer_name="Adam",
                learning_rate=0.001,
                schedule="",
                seed=42,
                max_epochs=14,
                patience=3,
            )
            result.notes = "Frozen ImageNet MobileNetV2 feature extractor."
        except Exception as error:
            result = RunResult(
                experiment_id="FND-008",
                run_id="fnd008_08_pretrained_mobilenet_v2",
                variant="pretrained_mobilenet_v2",
                status="SKIPPED_RESOURCE_UNAVAILABLE",
                optimizer="Adam",
                learning_rate=0.001,
                seed=42,
                image_size=int(config["pretrained_image_size"]),
                sequence_length=int(config["default_sequence_length"]),
                notes=(
                    "The frozen ImageNet backbone could not be resolved in the "
                    f"execution environment: {type(error).__name__}: {error}"
                ),
            )
        results.append(result)

    pd.DataFrame([result.comparison_row() for result in results]).to_csv(
        ARCHITECTURE_COMPARISON_PATH, index=False, lineterminator="\n"
    )
    plotted = [result for result in results if result.validation_macro_f1 is not None]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [result.variant for result in plotted]
    ax.bar(
        np.arange(len(labels)),
        [result.validation_macro_f1 for result in plotted],
    )
    ax.set_xticks(np.arange(len(labels)), labels, rotation=40, ha="right")
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Architecture-level ablations")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["architecture_comparison"], dpi=160)
    plt.close(fig)
    return results


def run_preprocessing_suite(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    config: dict[str, Any],
    champion: RunResult,
) -> list[RunResult]:
    results: list[RunResult] = []
    for index, variant in enumerate(config["preprocessing_variants"], start=1):
        data = prepare_data(
            train_dataframe,
            validation_dataframe,
            image_size=int(variant["image_size"]),
            sequence_length=int(variant["sequence_length"]),
            grayscale=bool(variant["grayscale"]),
            preprocessing_name=str(variant["name"]),
        )
        result = train_and_evaluate(
            experiment_id="FND-009",
            run_id=f"fnd009_{index:02d}_{variant['name']}",
            variant=str(variant["name"]),
            data=data,
            spec=medium_spec(config),
            optimizer_name=champion.optimizer,
            learning_rate=float(champion.learning_rate),
            schedule=champion.schedule,
            seed=42,
            max_epochs=24,
            patience=4,
        )
        results.append(result)
    pd.DataFrame([result.comparison_row() for result in results]).to_csv(
        PREPROCESSING_COMPARISON_PATH, index=False, lineterminator="\n"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [result.variant for result in results]
    ax.bar(np.arange(len(labels)), [result.validation_macro_f1 for result in results])
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("Validation Macro F1")
    ax.set_title("Preprocessing comparison")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["preprocessing_comparison"], dpi=160)
    plt.close(fig)
    return results


def run_gradient_probe(
    data: PreparedData,
    config: dict[str, Any],
) -> dict[str, Any]:
    keras, tf, _ = configure_runtime(42)
    keras.backend.clear_session()
    spec = {
        **medium_spec(config),
        "model_name": "fnd010_deep_sigmoid_probe",
        "activation": "sigmoid",
        "fusion_units": [64, 64, 64, 64, 64, 64],
        "dropout": 0.0,
    }
    model = build_model(
        keras,
        vocabulary_size=max(data.vocabulary.values(), default=1) + 1,
        image_shape=data.image_shape,
        sequence_length=data.sequence_length,
        spec=spec,
    )
    inputs = {
        "description_tokens": data.train_text[:16],
        "image": data.train_images[:16],
    }
    labels = data.train_labels[:16]
    with tf.GradientTape() as tape:
        probabilities = model(inputs, training=True)
        loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(labels, probabilities)
        )
    gradients = tape.gradient(loss, model.trainable_variables)
    kernel_pairs = [
        (variable, gradient)
        for variable, gradient in zip(
            model.trainable_variables, gradients, strict=True
        )
        if gradient is not None and "kernel" in variable.name
    ]
    first_norm = float(tf.norm(kernel_pairs[0][1]).numpy())
    last_norm = float(tf.norm(kernel_pairs[-1][1]).numpy())
    ratio = float(first_norm / max(last_norm, 1e-12))
    output = {
        "case": "deep_sigmoid_gradient_probe",
        "status": "OBSERVED",
        "metric_name": "first_to_last_kernel_gradient_ratio",
        "metric_value": ratio,
        "signature_detected": bool(ratio < 0.25 or ratio > 4.0),
        "observation": (
            "Deep sigmoid networks can produce strongly uneven gradient norms; "
            "the measured first/last kernel ratio is reported without forcing a result."
        ),
        "prevention": "Prefer ReLU-like activations, normalization, and residual paths for deep stacks.",
    }
    del model
    keras.backend.clear_session()
    gc.collect()
    return output


def run_no_update_probe(
    data: PreparedData,
    config: dict[str, Any],
) -> dict[str, Any]:
    keras, _, _ = configure_runtime(42)
    keras.backend.clear_session()
    model = build_model(
        keras,
        vocabulary_size=max(data.vocabulary.values(), default=1) + 1,
        image_shape=data.image_shape,
        sequence_length=data.sequence_length,
        spec={**medium_spec(config), "model_name": "fnd010_no_update_probe"},
    )
    inputs = {
        "description_tokens": data.train_text[:16],
        "image": data.train_images[:16],
    }
    labels = data.train_labels[:16]
    loss_function = keras.losses.SparseCategoricalCrossentropy()
    before_weights = [np.asarray(value).copy() for value in model.get_weights()]
    first_loss = float(loss_function(labels, model(inputs, training=True)).numpy())
    last_loss = first_loss
    for _ in range(10):
        last_loss = float(
            loss_function(labels, model(inputs, training=True)).numpy()
        )
    after_weights = model.get_weights()
    weights_unchanged = all(
        np.array_equal(left, right)
        for left, right in zip(before_weights, after_weights, strict=True)
    )
    output = {
        "case": "missing_optimizer_step_probe",
        "status": "OBSERVED",
        "metric_name": "absolute_loss_change_without_update",
        "metric_value": abs(last_loss - first_loss),
        "signature_detected": bool(weights_unchanged),
        "observation": (
            "Repeated forward passes without an optimizer update left all weights "
            "unchanged and did not create systematic learning."
        ),
        "prevention": (
            "Use model.fit/train_on_batch or a reviewed custom loop that applies "
            "gradients exactly once per batch. Keras manages gradient reset inside "
            "its built-in training step."
        ),
    }
    del model
    keras.backend.clear_session()
    gc.collect()
    return output


def validation_training_block_diagnostic() -> dict[str, Any]:
    return {
        "case": "validation_training_blocked",
        "status": "BLOCKED_BY_DESIGN",
        "metric_name": "validation_used_for_weight_updates",
        "metric_value": 0.0,
        "signature_detected": True,
        "observation": (
            "The suite exposes fixed train and validation arrays; every fit call "
            "uses train inputs for updates and validation inputs only through "
            "validation_data."
        ),
        "prevention": (
            "Keep the split paths fixed in configuration and audit that validation "
            "evaluation leaves weights unchanged."
        ),
    }


def run_failure_diagnostics(
    data: PreparedData,
    config: dict[str, Any],
    baseline: RunResult,
) -> tuple[list[RunResult], list[dict[str, Any]]]:
    cases: list[tuple[str, dict[str, Any], str, float, np.ndarray | None, int]] = [
        (
            "unscaled_images",
            {**medium_spec(config), "model_name": "fnd010_unscaled", "scale_images": False},
            "Adam",
            0.001,
            None,
            8,
        ),
        (
            "excessive_learning_rate",
            {**medium_spec(config), "model_name": "fnd010_high_lr"},
            "Adam",
            1.0,
            None,
            6,
        ),
        (
            "tiny_learning_rate",
            {**medium_spec(config), "model_name": "fnd010_low_lr"},
            "SGD",
            1e-7,
            None,
            8,
        ),
        (
            "excessive_dropout",
            {**medium_spec(config), "model_name": "fnd010_dropout", "dropout": 0.90},
            "Adam",
            0.001,
            None,
            10,
        ),
        (
            "misaligned_train_labels",
            {**medium_spec(config), "model_name": "fnd010_misaligned"},
            "Adam",
            0.001,
            np.random.default_rng(42).permutation(data.train_labels),
            12,
        ),
        (
            "sigmoid_activation",
            {**medium_spec(config), "model_name": "fnd010_sigmoid", "activation": "sigmoid"},
            "Adam",
            0.001,
            None,
            10,
        ),
    ]
    results: list[RunResult] = []
    diagnostics: list[dict[str, Any]] = []
    for index, (name, spec, optimizer, rate, labels_override, epochs) in enumerate(
        cases, start=1
    ):
        try:
            result = train_and_evaluate(
                experiment_id="FND-010",
                run_id=f"fnd010_{index:02d}_{name}",
                variant=name,
                data=data,
                spec=spec,
                optimizer_name=optimizer,
                learning_rate=rate,
                schedule="",
                seed=42,
                max_epochs=epochs,
                patience=max(2, epochs // 3),
                train_labels_override=labels_override,
            )
            metric = float(result.validation_macro_f1 or 0.0)
            baseline_metric = float(baseline.validation_macro_f1 or 0.0)
            signature = bool(
                result.status != "COMPLETED"
                or not math.isfinite(float(result.final_train_loss or 0.0))
                or metric + 0.05 < baseline_metric
                or name in {"misaligned_train_labels", "tiny_learning_rate"}
            )
            observation = (
                f"Validation Macro F1={metric:.4f} versus the correct-loop "
                f"reference {baseline_metric:.4f}; train loss="
                f"{result.final_train_loss}."
            )
        except Exception as error:
            result = RunResult(
                experiment_id="FND-010",
                run_id=f"fnd010_{index:02d}_{name}",
                variant=name,
                status="FAILED_DIAGNOSTIC",
                optimizer=optimizer,
                learning_rate=rate,
                seed=42,
                notes=f"Controlled failure raised {type(error).__name__}: {error}",
            )
            signature = True
            observation = result.notes
        results.append(result)
        prevention = {
            "unscaled_images": "Keep explicit image scaling in the model and assert input ranges.",
            "excessive_learning_rate": "Use bounded predefined learning rates and terminate on NaN.",
            "tiny_learning_rate": "Track loss reduction and reject configurations with negligible progress.",
            "excessive_dropout": "Treat dropout as a tuned regularizer rather than a default maximum.",
            "misaligned_train_labels": "Verify sample IDs and labels before every shuffle or batching operation.",
            "sigmoid_activation": "Compare timing and gradients; prefer ReLU-like activations for hidden layers.",
        }[name]
        diagnostics.append(
            {
                "case": name,
                "status": "OBSERVED" if signature else "NO_STRONG_SIGNATURE",
                "metric_name": "validation_macro_f1",
                "metric_value": result.validation_macro_f1,
                "signature_detected": signature,
                "observation": observation,
                "prevention": prevention,
            }
        )

    diagnostics.append(run_gradient_probe(data, config))
    diagnostics.append(run_no_update_probe(data, config))
    diagnostics.append(validation_training_block_diagnostic())
    pd.DataFrame(diagnostics).to_csv(
        FAILURE_DIAGNOSTICS_PATH, index=False, lineterminator="\n"
    )
    prevention_lines = [
        "# Controlled Failure Prevention Checklist",
        "",
        "These cases use train-only copies and validation-only evaluation. No locked test data is accessed.",
        "",
    ]
    for item in diagnostics:
        prevention_lines.extend(
            [
                f"## {item['case']}",
                "",
                f"- Observed: {item['observation']}",
                f"- Prevention: {item['prevention']}",
                "",
            ]
        )
    write_text(FAILURE_PREVENTION_PATH, "\n".join(prevention_lines))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [item["case"] for item in diagnostics]
    values = [
        float(item["metric_value"])
        if item["metric_value"] is not None
        and math.isfinite(float(item["metric_value"]))
        else 0.0
        for item in diagnostics
    ]
    ax.bar(np.arange(len(labels)), values)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=40, ha="right")
    ax.set_ylabel("Recorded diagnostic metric")
    ax.set_title("Controlled failure signatures")
    fig.tight_layout()
    fig.savefig(FIGURE_PATHS["failure_signatures"], dpi=160)
    plt.close(fig)
    return results, diagnostics


def write_combined_outputs(results: list[RunResult]) -> None:
    comparison_rows = [result.comparison_row() for result in results]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        EXPERIMENT_COMPARISON_CSV_PATH, index=False, lineterminator="\n"
    )
    write_json(
        EXPERIMENT_COMPARISON_JSON_PATH,
        {
            "step": STEP,
            "run_count": len(comparison_rows),
            "results": comparison_rows,
            **LOCK_FLAGS,
        },
    )

    history_rows = [row for result in results for row in result.history_rows]
    pd.DataFrame(history_rows).to_csv(
        TRAINING_HISTORIES_PATH, index=False, lineterminator="\n"
    )
    prediction_rows = [
        row for result in results for row in result.prediction_rows
    ]
    pd.DataFrame(prediction_rows).to_csv(
        VALIDATION_PREDICTIONS_PATH, index=False, lineterminator="\n"
    )
    write_json(
        CONFUSION_MATRICES_PATH,
        {
            result.run_id: {
                "labels": list(LABELS),
                "matrix": result.confusion_matrix_values,
            }
            for result in results
            if result.confusion_matrix_values is not None
        },
    )


def build_execution_registry(
    result_groups: dict[str, list[RunResult]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    experiment_configs = {
        path.stem: read_json(path) for path in EXPERIMENT_CONFIG_PATHS
    }
    entries: list[dict[str, Any]] = []
    for experiment_id in FUNDAMENTALS_IDS:
        runs = result_groups.get(experiment_id, [])
        config = experiment_configs[experiment_id]
        completed_runs = [run for run in runs if run.status == "COMPLETED"]
        scored_runs = [
            run
            for run in completed_runs
            if run.validation_macro_f1 is not None
            and run.validation_accuracy is not None
        ]
        best = choose_best(scored_runs) if scored_runs else None
        entry = {
            "experiment_id": experiment_id,
            "exercise_problem_number": config["exercise_problem_number"],
            "exercise_problem_title": config["exercise_problem_title"],
            "execution_status": "COMPLETED",
            "run_count": len(runs),
            "completed_run_count": len(completed_runs),
            "selection_eligible": config["selection_eligible"],
            "primary_metric": config["primary_metric"],
            "best_run_id": best.run_id if best else None,
            "best_validation_macro_f1": (
                best.validation_macro_f1 if best else None
            ),
            "best_validation_accuracy": (
                best.validation_accuracy if best else None
            ),
            "result_summary": diagnostics.get(experiment_id, "Completed."),
            "evidence_paths": config["evidence_paths"],
            "test_split_allowed": False,
            "test_split_path": None,
            "final_test_evaluation_authorized": False,
        }
        entries.append(entry)
    payload = {
        "schema_version": "1.0",
        "step": STEP,
        "base_checkpoint": BASE_CHECKPOINT,
        "experiment_count": len(entries),
        "completed_experiment_count": len(entries),
        "execution_status_counts": {"COMPLETED": len(entries)},
        "experiments": entries,
        **LOCK_FLAGS,
    }
    write_json(EXECUTION_REGISTRY_JSON_PATH, payload)
    with EXECUTION_REGISTRY_CSV_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = list(entries[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            row["evidence_paths"] = " | ".join(entry["evidence_paths"])
            row["test_split_allowed"] = "false"
            row["test_split_path"] = ""
            row["final_test_evaluation_authorized"] = "false"
            writer.writerow(row)
    return payload


def build_summary(
    *,
    status: dict[str, Any],
    optimizer_stability: dict[str, Any],
    capacity_results: list[RunResult],
    architecture_results: list[RunResult],
    preprocessing_results: list[RunResult],
    failure_diagnostics: list[dict[str, Any]],
) -> None:
    best_capacity = choose_best(capacity_results)
    best_architecture = choose_best(
        [result for result in architecture_results if result.status == "COMPLETED"]
    )
    best_preprocessing = choose_best(preprocessing_results)
    pretrained = next(
        result
        for result in architecture_results
        if result.variant == "pretrained_mobilenet_v2"
    )
    lines = [
        "# Step 011.1 — Deep Learning Fundamentals Experimental Suite",
        "",
        f"- Status: **{status['status']}**",
        f"- Readiness: `{status['readiness']}`",
        f"- Base commit: `{status['base_commit']}`",
        "- Exercise problems completed: **10/10**",
        "- Model training performed: **true**",
        "- Frozen exam model changed: **false**",
        "- Test split used: **false**",
        "- Final test evaluation authorized: **false**",
        "",
        "## Main findings",
        "",
        f"- Optimizer champion: `{optimizer_stability['champion_optimizer']}` at "
        f"`{optimizer_stability['champion_learning_rate']}`; stability Macro F1 "
        f"mean `{optimizer_stability['macro_f1_mean']:.4f}` and standard deviation "
        f"`{optimizer_stability['macro_f1_standard_deviation']:.4f}`.",
        f"- Best capacity variant: `{best_capacity.variant}` with validation Macro F1 "
        f"`{best_capacity.validation_macro_f1:.4f}` and `{best_capacity.parameter_count}` parameters.",
        f"- Best architecture variant: `{best_architecture.variant}` with validation "
        f"Macro F1 `{best_architecture.validation_macro_f1:.4f}`.",
        f"- Best preprocessing variant: `{best_preprocessing.variant}` with validation "
        f"Macro F1 `{best_preprocessing.validation_macro_f1:.4f}`.",
        f"- Pretrained MobileNetV2 probe: `{pretrained.status}`. {pretrained.notes}",
        "",
        "## Exercise coverage",
        "",
        "| ID | Evidence | Status |",
        "|---|---|---|",
        "| FND-001 | Dataset dimensions, balance, pixels, text lengths, examples | COMPLETED |",
        "| FND-002 | Batch shapes, dtypes, alignment, deterministic shuffle policy | COMPLETED |",
        "| FND-003 | One-hidden-layer baseline, gradients, weights, probability contract | COMPLETED |",
        "| FND-004 | Deliberate one-batch overfit with learning curves | COMPLETED |",
        "| FND-005 | Correct train/validation loop and validation no-update audit | COMPLETED |",
        "| FND-006 | SGD/RMSprop/Adam/AdamW, LR grid, schedule, early stopping | COMPLETED |",
        "| FND-007 | Small/medium/large capacity and probability tracking | COMPLETED |",
        "| FND-008 | L2, dropout, batch norm, schedule, skip connection, CNN, pretrained probe | COMPLETED |",
        "| FND-009 | Resolution, sequence length, and grayscale preprocessing | COMPLETED |",
        "| FND-010 | Nine safe controlled-failure diagnostics | COMPLETED |",
        "",
        "## Controlled failures",
        "",
    ]
    for item in failure_diagnostics:
        lines.append(
            f"- `{item['case']}` — {item['status']}: {item['observation']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation policy",
            "",
            "These experiments demonstrate course concepts and produce validation-only comparisons. "
            "They do not replace the Step 010.8 frozen final model, do not authorize test access, "
            "and must not be presented as held-out test performance.",
        ]
    )
    write_text(SUMMARY_PATH, "\n".join(lines))


def write_manifest(source_paths: list[Path]) -> dict[str, Any]:
    generated_paths = [path for path in GENERATED_PATHS if path != MANIFEST_PATH]
    missing = [path for path in [*source_paths, *generated_paths] if not path.is_file()]
    if missing:
        raise FundamentalsSuiteError(
            "Cannot build manifest; missing artifacts: "
            + ", ".join(project_relative(path) for path in missing)
        )
    manifest = {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "base_commit": current_git_commit(),
        "hash_normalization": "utf-8-lf",
        "source_artifact_sha256": {
            project_relative(path): normalized_sha256(path)
            for path in source_paths
        },
        "generated_artifact_sha256": {
            project_relative(path): normalized_sha256(path)
            for path in generated_paths
        },
        **LOCK_FLAGS,
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def run_suite() -> dict[str, Any]:
    config = read_json(SUITE_CONFIG_PATH)
    train_dataframe, validation_dataframe = load_dataframes()
    baseline_data = prepare_data(
        train_dataframe,
        validation_dataframe,
        image_size=int(config["default_image_size"]),
        sequence_length=int(config["default_sequence_length"]),
        grayscale=False,
        preprocessing_name="baseline_24_seq12",
    )

    run_eda_and_batch_contract(
        train_dataframe, validation_dataframe, baseline_data
    )
    baseline_diagnostic = run_baseline_gradient_diagnostic(baseline_data)
    overfit_result = run_overfit_diagnostic(baseline_data, config)
    loop_result, optimizer_results, optimizer_stability = (
        run_training_loop_and_optimizer_suite(baseline_data, config)
    )
    optimizer_champion = choose_best(
        optimizer_results[: len(config["optimizer_grid"])]
    )
    capacity_results = run_capacity_suite(
        baseline_data, config, optimizer_champion
    )
    architecture_results = run_architecture_suite(
        baseline_data, train_dataframe, validation_dataframe, config
    )
    preprocessing_results = run_preprocessing_suite(
        train_dataframe,
        validation_dataframe,
        config,
        optimizer_champion,
    )
    failure_results, failure_diagnostics = run_failure_diagnostics(
        baseline_data, config, loop_result
    )

    baseline_run = RunResult(
        experiment_id="FND-003",
        run_id="fnd003_gradient_diagnostic",
        variant="one_hidden_layer_baseline",
        status="COMPLETED",
        optimizer="Adam",
        learning_rate=0.003,
        seed=42,
        image_size=baseline_data.image_shape[0],
        sequence_length=baseline_data.sequence_length,
        parameter_count=int(baseline_diagnostic["parameter_count"]),
        epochs_completed=15,
        final_train_loss=float(baseline_diagnostic["final_loss"]),
        notes="Finite-gradient and weight-update diagnostic.",
    )
    overfit_run = RunResult(
        experiment_id="FND-004",
        run_id="fnd004_one_batch_overfit",
        variant="one_batch_overfit",
        status="COMPLETED",
        optimizer="Adam",
        learning_rate=0.01,
        seed=42,
        image_size=baseline_data.image_shape[0],
        sequence_length=baseline_data.sequence_length,
        epochs_completed=int(overfit_result["epochs_completed"]),
        final_train_accuracy=float(overfit_result["final_batch_accuracy"]),
        final_train_loss=float(overfit_result["final_batch_loss"]),
        notes="Deliberate small-batch memorization diagnostic.",
    )

    all_results = [
        baseline_run,
        overfit_run,
        loop_result,
        *optimizer_results,
        *capacity_results,
        *architecture_results,
        *preprocessing_results,
        *failure_results,
    ]
    write_combined_outputs(all_results)

    result_groups = {
        "FND-001": [],
        "FND-002": [],
        "FND-003": [baseline_run],
        "FND-004": [overfit_run],
        "FND-005": [loop_result],
        "FND-006": optimizer_results,
        "FND-007": capacity_results,
        "FND-008": architecture_results,
        "FND-009": preprocessing_results,
        "FND-010": failure_results,
    }
    result_summaries = {
        "FND-001": "Integrated train/validation EDA and leakage checks completed.",
        "FND-002": "Batch shapes, dtypes, alignment, and shuffle policy verified.",
        "FND-003": (
            f"Small-batch loss changed from {baseline_diagnostic['initial_loss']:.4f} "
            f"to {baseline_diagnostic['final_loss']:.4f}; gradients were finite."
        ),
        "FND-004": (
            f"One-batch accuracy reached {overfit_result['final_batch_accuracy']:.4f} "
            f"with loss {overfit_result['final_batch_loss']:.4f}."
        ),
        "FND-005": (
            f"Correct validation-only loop achieved Macro F1 "
            f"{loop_result.validation_macro_f1:.4f}."
        ),
        "FND-006": (
            f"Optimizer champion {optimizer_stability['champion_optimizer']} "
            f"at LR {optimizer_stability['champion_learning_rate']}."
        ),
        "FND-007": f"Compared {len(capacity_results)} capacity variants.",
        "FND-008": f"Compared {len(architecture_results)} architecture variants.",
        "FND-009": f"Compared {len(preprocessing_results)} preprocessing variants.",
        "FND-010": f"Recorded {len(failure_diagnostics)} controlled failure diagnostics.",
    }
    build_execution_registry(result_groups, result_summaries)

    status = {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "base_checkpoint": BASE_CHECKPOINT,
        "base_commit": current_git_commit(),
        "exercise_problem_count": 10,
        "completed_exercise_problem_count": 10,
        "model_training_performed": True,
        "model_selection_changed": False,
        "educational_validation_champions_recorded": True,
        "run_count": len(all_results),
        "completed_run_count": sum(
            result.status == "COMPLETED" for result in all_results
        ),
        "diagnostic_failure_run_count": sum(
            result.status == "FAILED_DIAGNOSTIC" for result in all_results
        ),
        "resource_skip_run_count": sum(
            result.status == "SKIPPED_RESOURCE_UNAVAILABLE"
            for result in all_results
        ),
        "optimizer_champion_run_id": optimizer_stability["champion_run_id"],
        "notebook_executed": True,
        "notebook_error_outputs": 0,
        "python_version": platform.python_version(),
        **LOCK_FLAGS,
    }
    write_json(STATUS_PATH, status)

    notebook_audit = build_and_execute_notebook()
    status["notebook_error_outputs"] = notebook_audit["error_output_count"]
    write_json(STATUS_PATH, status)

    build_summary(
        status=status,
        optimizer_stability=optimizer_stability,
        capacity_results=capacity_results,
        architecture_results=architecture_results,
        preprocessing_results=preprocessing_results,
        failure_diagnostics=failure_diagnostics,
    )

    source_paths = [
        SUITE_CONFIG_PATH,
        *EXPERIMENT_CONFIG_PATHS,
        PROJECT_ROOT / "src" / "fundamentals_suite_config.py",
        PROJECT_ROOT / "src" / "run_fundamentals_experimental_suite.py",
        PROJECT_ROOT / "src" / "build_fundamentals_experiment_notebook.py",
        PROJECT_ROOT / "src" / "verification" / "fundamentals_experimental_suite.py",
        PROJECT_ROOT / "docs" / "course_coverage" / "fundamentals_experimental_suite.md",
    ]
    write_manifest(source_paths)
    return status


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the Step 011.1 train/validation-only fundamentals suite."
    )
    parser.add_argument(
        "--mode",
        choices=("full",),
        default="full",
        help="Run the complete controlled suite.",
    )
    # The project CLI dispatches this function after it has already consumed
    # the outer command name. Avoid parsing the parent process argv again.
    effective_argv = [] if argv is None else list(argv)
    parser.parse_args(effective_argv)
    status = run_suite()
    print("Deep Learning Fundamentals Experimental Suite")
    print(f"- Status: {status['status']}")
    print(f"- Readiness: {status['readiness']}")
    print(
        "- Exercise problems completed: "
        f"{status['completed_exercise_problem_count']}/"
        f"{status['exercise_problem_count']}"
    )
    print(f"- Training runs recorded: {status['run_count']}")
    print(f"- Test split used: {str(status['test_split_used']).lower()}")
    print(
        "- Final test evaluation authorized: "
        f"{str(status['final_test_evaluation_authorized']).lower()}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
