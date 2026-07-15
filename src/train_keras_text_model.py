from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.dataset_config import LABELS, METADATA_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "development_train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "development_validation.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "keras_text_model.keras"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "keras_text"
)

METRICS_PATH = (
    OUTPUT_DIRECTORY
    / "validation_metrics.json"
)

PREDICTIONS_PATH = (
    OUTPUT_DIRECTORY
    / "validation_predictions.csv"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIRECTORY
    / "validation_confusion_matrix.csv"
)

TRAINING_HISTORY_PATH = (
    OUTPUT_DIRECTORY
    / "training_history.csv"
)

MODEL_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "model_architecture.txt"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "keras_text_model_summary.md"
)

RANDOM_STATE = 42
MAX_TOKENS = 1000
SEQUENCE_LENGTH = 12
EMBEDDING_DIMENSION = 32
DENSE_UNITS = 32
DROPOUT_RATE = 0.20
BATCH_SIZE = 8
MAX_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 8

LABEL_TO_INDEX = {
    label: index
    for index, label in enumerate(LABELS)
}

INDEX_TO_LABEL = {
    index: label
    for label, index in LABEL_TO_INDEX.items()
}


def set_reproducibility() -> None:
    keras.utils.set_random_seed(RANDOM_STATE)

    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass


def load_split(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)

    if tuple(dataframe.columns) != METADATA_COLUMNS:
        raise ValueError(
            f"Unexpected metadata schema in {path.name}."
        )

    if dataframe.empty:
        raise ValueError(
            f"The split file is empty: {path.name}"
        )

    invalid_labels = (
        set(dataframe["label"])
        - set(LABELS)
    )

    if invalid_labels:
        raise ValueError(
            f"Invalid labels in {path.name}: "
            f"{sorted(invalid_labels)}"
        )

    return dataframe


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataframe = load_split(TRAIN_PATH)
    validation_dataframe = load_split(VALIDATION_PATH)

    train_groups = set(
        train_dataframe["part_group_id"]
    )

    validation_groups = set(
        validation_dataframe["part_group_id"]
    )

    if not train_groups.isdisjoint(validation_groups):
        raise ValueError(
            "Train and validation part groups overlap."
        )

    return train_dataframe, validation_dataframe


def encode_labels(
    labels: pd.Series,
) -> np.ndarray:
    invalid_labels = set(labels) - set(LABELS)

    if invalid_labels:
        raise ValueError(
            f"Cannot encode labels: "
            f"{sorted(invalid_labels)}"
        )

    return np.asarray(
        [
            LABEL_TO_INDEX[label]
            for label in labels
        ],
        dtype=np.int32,
    )


def decode_label_indices(
    indices: np.ndarray,
) -> np.ndarray:
    decoded_labels: list[str] = []

    for index in indices:
        integer_index = int(index)

        if integer_index not in INDEX_TO_LABEL:
            raise ValueError(
                f"Invalid label index: {integer_index}"
            )

        decoded_labels.append(
            INDEX_TO_LABEL[integer_index]
        )

    return np.asarray(
        decoded_labels,
        dtype=object,
    )


def create_text_vectorizer(
    training_texts: list[str],
) -> keras.layers.TextVectorization:
    vectorizer = keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        name="text_vectorization",
    )

    text_dataset = (
        tf.data.Dataset
        .from_tensor_slices(training_texts)
        .batch(BATCH_SIZE)
    )

    vectorizer.adapt(text_dataset)

    return vectorizer


def create_training_datasets(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
]:
    training_texts = train_dataframe[
        "description"
    ].astype(str).tolist()

    validation_texts = validation_dataframe[
        "description"
    ].astype(str).tolist()

    training_labels = encode_labels(
        train_dataframe["label"]
    )

    validation_labels = encode_labels(
        validation_dataframe["label"]
    )

    training_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                training_texts,
                training_labels,
            )
        )
        .shuffle(
            buffer_size=len(train_dataframe),
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True,
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    validation_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                validation_texts,
                validation_labels,
            )
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return training_dataset, validation_dataset

def build_text_model(
    vectorizer: keras.layers.TextVectorization,
) -> keras.Model:
    vocabulary_size = len(
        vectorizer.get_vocabulary()
    )

    inputs = keras.Input(
        shape=(),
        dtype=tf.string,
        name="description",
    )

    x = vectorizer(inputs)

    x = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=EMBEDDING_DIMENSION,
        mask_zero=True,
        name="token_embedding",
    )(x)

    x = keras.layers.GlobalAveragePooling1D(
        name="text_pooling",
    )(x)

    x = keras.layers.Dense(
        DENSE_UNITS,
        activation="relu",
        name="text_dense",
    )(x)

    x = keras.layers.Dropout(
        DROPOUT_RATE,
        name="text_dropout",
    )(x)

    outputs = keras.layers.Dense(
        len(LABELS),
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="keras_text_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001,
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy",
            )
        ],
    )

    return model


def train_model(
    model: keras.Model,
    training_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
) -> keras.callbacks.History:
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    return model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        verbose=0,
    )


def evaluate_predictions(
    validation_dataframe: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    if probabilities.shape != (
        len(validation_dataframe),
        len(LABELS),
    ):
        raise ValueError(
            "The prediction probability shape is invalid."
        )

    predicted_indices = np.argmax(
        probabilities,
        axis=1,
    )

    predicted_labels = decode_label_indices(
        predicted_indices
    )

    true_labels = validation_dataframe[
        "label"
    ].to_numpy()

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

    per_class = {}

    for index, label in enumerate(LABELS):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }

    metrics = {
        "model": "Keras Text Neural Network",
        "input_modality": "text",
        "evaluation_split": "validation",
        "test_split_used": False,
        "sample_count": int(
            len(validation_dataframe)
        ),
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
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
            label: int(
                np.count_nonzero(
                    true_labels == label
                )
            )
            for label in LABELS
        },
        "predicted_distribution": {
            label: int(
                np.count_nonzero(
                    predicted_labels == label
                )
            )
            for label in LABELS
        },
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(LABELS),
    }

    return metrics, predicted_labels


def create_prediction_table(
    validation_dataframe: pd.DataFrame,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    prediction_table = validation_dataframe[
        [
            "sample_id",
            "part_group_id",
            "image_id",
            "part_category",
            "description",
            "label",
        ]
    ].copy()

    prediction_table = prediction_table.rename(
        columns={
            "label": "true_label",
        }
    )

    prediction_table["predicted_label"] = (
        predicted_labels
    )

    prediction_table["is_correct"] = (
        prediction_table["true_label"]
        == prediction_table["predicted_label"]
    )

    for index, label in enumerate(LABELS):
        prediction_table[
            f"probability_{label}"
        ] = probabilities[:, index]

    return prediction_table


def create_history_table(
    history: keras.callbacks.History,
) -> pd.DataFrame:
    history_table = pd.DataFrame(
        history.history
    )

    history_table.insert(
        loc=0,
        column="epoch",
        value=np.arange(
            1,
            len(history_table) + 1,
        ),
    )

    return history_table


def write_model_architecture(
    model: keras.Model,
) -> None:
    summary_lines: list[str] = []

    model.summary(
        print_fn=summary_lines.append
    )

    MODEL_SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def write_outputs(
    model: keras.Model,
    metrics: dict[str, object],
    prediction_table: pd.DataFrame,
    history_table: pd.DataFrame,
) -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(MODEL_PATH)

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    prediction_table.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    history_table.to_csv(
        TRAINING_HISTORY_PATH,
        index=False,
    )

    confusion_dataframe = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[
            f"actual_{label}"
            for label in LABELS
        ],
        columns=[
            f"predicted_{label}"
            for label in LABELS
        ],
    )

    confusion_dataframe.to_csv(
        CONFUSION_MATRIX_PATH,
        index=True,
    )

    write_model_architecture(model)


def write_summary(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    model: keras.Model,
    vectorizer: keras.layers.TextVectorization,
    metrics: dict[str, object],
    history_table: pd.DataFrame,
) -> None:
    best_epoch = int(
        history_table.loc[
            history_table["val_loss"].idxmin(),
            "epoch",
        ]
    )

    best_validation_loss = float(
        history_table["val_loss"].min()
    )

    summary_lines = [
        "# Keras Text Neural Network",
        "",
        "The model uses only the text description.",
        "",
        "The image input and test split were not used.",
        "",
        "## Dataset",
        "",
        f"- Training samples: {len(train_dataframe)}",
        (
            "- Validation samples: "
            f"{len(validation_dataframe)}"
        ),
        "",
        "## Architecture",
        "",
        "- Text vectorization",
        (
            "- Token embedding: "
            f"{EMBEDDING_DIMENSION} dimensions"
        ),
        "- Global average pooling",
        f"- Dense layer: {DENSE_UNITS} units",
        f"- Dropout: {DROPOUT_RATE}",
        f"- Output classes: {len(LABELS)}",
        "",
        "## Training",
        "",
        (
            "- Vocabulary size: "
            f"{len(vectorizer.get_vocabulary())}"
        ),
        f"- Maximum epochs: {MAX_EPOCHS}",
        (
            "- Completed epochs: "
            f"{len(history_table)}"
        ),
        f"- Best epoch: {best_epoch}",
        (
            "- Best validation loss: "
            f"{best_validation_loss:.4f}"
        ),
        f"- Trainable parameters: {model.count_params()}",
        "",
        "## Validation results",
        "",
        (
            "- Accuracy: "
            f"{metrics['accuracy']:.4f}"
        ),
        (
            "- Macro F1: "
            f"{metrics['macro_f1']:.4f}"
        ),
        "",
        "## Notes",
        "",
        "The label describes the relationship between an image "
        "and a text description.",
        "",
        "A text-only model cannot directly observe that "
        "relationship because it does not receive the image.",
        "",
        "The generated development dataset is used only to "
        "check the training and evaluation pipeline.",
    ]

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    set_reproducibility()

    (
        train_dataframe,
        validation_dataframe,
    ) = load_datasets()

    training_texts = train_dataframe[
        "description"
    ].astype(str).tolist()

    vectorizer = create_text_vectorizer(
        training_texts
    )

    (
        training_dataset,
        validation_dataset,
    ) = create_training_datasets(
        train_dataframe,
        validation_dataframe,
    )

    model = build_text_model(
        vectorizer
    )

    history = train_model(
        model,
        training_dataset,
        validation_dataset,
    )

    validation_texts = tf.constant(
        validation_dataframe[
            "description"
        ].astype(str).tolist()
    )

    probabilities = model.predict(
        validation_texts,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    metrics, predicted_labels = (
        evaluate_predictions(
            validation_dataframe,
            probabilities,
        )
    )

    prediction_table = create_prediction_table(
        validation_dataframe,
        predicted_labels,
        probabilities,
    )

    history_table = create_history_table(
        history
    )

    metrics["training"] = {
        "epochs_completed": int(
            len(history_table)
        ),
        "best_epoch": int(
            history_table.loc[
                history_table["val_loss"].idxmin(),
                "epoch",
            ]
        ),
        "best_validation_loss": float(
            history_table["val_loss"].min()
        ),
        "vocabulary_size": int(
            len(vectorizer.get_vocabulary())
        ),
        "parameter_count": int(
            model.count_params()
        ),
        "random_state": RANDOM_STATE,
    }

    write_outputs(
        model=model,
        metrics=metrics,
        prediction_table=prediction_table,
        history_table=history_table,
    )

    write_summary(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        model=model,
        vectorizer=vectorizer,
        metrics=metrics,
        history_table=history_table,
    )

    print("Keras text model training completed.")
    print(
        f"Epochs completed: "
        f"{metrics['training']['epochs_completed']}"
    )
    print(
        f"Validation accuracy: "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Validation macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )
    print(f"Model: {MODEL_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Test split used: no")


if __name__ == "__main__":
    main()