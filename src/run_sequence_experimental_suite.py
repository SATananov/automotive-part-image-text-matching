from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)

from src.build_sequence_experiment_notebook import build_and_execute_notebook
from src.dataset_config import LABELS, METADATA_COLUMNS
from src.real_dataset_config import PROJECT_ROOT
from src.sequence_suite_config import (
    ATTENTION_EVIDENCE_PATH,
    ATTENTION_TOKEN_SUMMARY_PATH,
    BASE_CHECKPOINT,
    COMPLETED_SEQUENCE_IDS,
    CONFUSION_MATRICES_PATH,
    DEFERRED_SEQUENCE_IDS,
    DOCUMENTATION_PATH,
    ERROR_ANALYSIS_PATH,
    EXECUTION_REGISTRY_CSV_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXPERIMENT_CONFIG_PATHS,
    FIGURE_PATHS,
    FIGURE_ROOT,
    GENERATED_PATHS,
    LOADER_CONTRACT_PATH,
    LOCK_FLAGS,
    MANIFEST_PATH,
    MODEL_COMPARISON_CSV_PATH,
    MODEL_COMPARISON_JSON_PATH,
    NOTEBOOK_AUDIT_PATH,
    NOTEBOOK_PATH,
    PRETRAINED_GATE_PATH,
    READINESS,
    REPRESENTATIVE_EXAMPLES_PATH,
    ROC_CURVES_PATH,
    SAMPLE_BATCH_PATH,
    SEQUENCE_IDS,
    STATUS_PATH,
    STEP,
    SUITE_CONFIG_PATH,
    SUMMARY_PATH,
    TEXT_HASH_SUFFIXES,
    TEXT_PROFILE_PATH,
    TOKENIZATION_EXAMPLES_PATH,
    TOKENIZATION_SUMMARY_PATH,
    TRAINING_HISTORIES_PATH,
    TRAINING_RUNS_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    VALIDATION_PREDICTIONS_PATH,
    VOCABULARY_PATH,
    project_relative,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


class SequenceSuiteError(RuntimeError):
    """Raised when the Step 011.2 experiment contract is violated."""


@dataclass
class RunResult:
    experiment_id: str
    run_id: str
    family: str
    variant: str
    seed: int
    status: str = "COMPLETED"
    parameter_count: int = 0
    epochs_completed: int = 0
    best_epoch: int = 0
    training_time_seconds: float = 0.0
    validation_accuracy: float = 0.0
    validation_macro_f1: float = 0.0
    final_train_accuracy: float | None = None
    final_train_loss: float | None = None
    final_validation_loss: float | None = None
    generalization_gap: float | None = None
    notes: str = ""
    history_rows: list[dict[str, Any]] = field(default_factory=list)
    prediction_rows: list[dict[str, Any]] = field(default_factory=list)
    confusion_matrix_values: list[list[int]] = field(default_factory=list)
    probabilities: np.ndarray | None = None
    predictions: np.ndarray | None = None

    def row(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "family": self.family,
            "variant": self.variant,
            "seed": self.seed,
            "status": self.status,
            "parameter_count": self.parameter_count,
            "epochs_completed": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "training_time_seconds": round(self.training_time_seconds, 6),
            "validation_accuracy": round(self.validation_accuracy, 10),
            "validation_macro_f1": round(self.validation_macro_f1, 10),
            "final_train_accuracy": self.final_train_accuracy,
            "final_train_loss": self.final_train_loss,
            "final_validation_loss": self.final_validation_loss,
            "generalization_gap": self.generalization_gap,
            "notes": self.notes,
        }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SequenceSuiteError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
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
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
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
        return BASE_CHECKPOINT
    return result.stdout.strip() or BASE_CHECKPOINT


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def load_split(path: Path) -> pd.DataFrame:
    allowed = {TRAIN_PATH.resolve(), VALIDATION_PATH.resolve()}
    if path.resolve() not in allowed:
        raise SequenceSuiteError(f"Unauthorized Step 011.2 data input: {path}")
    dataframe = pd.read_csv(path, dtype=str).fillna("")
    if tuple(dataframe.columns) != METADATA_COLUMNS:
        raise SequenceSuiteError(f"Unexpected schema: {path.name}")
    if dataframe.empty or not dataframe["sample_id"].is_unique:
        raise SequenceSuiteError(f"Invalid split: {path.name}")
    if set(dataframe["label"]) != set(LABELS):
        raise SequenceSuiteError(f"Unexpected labels: {path.name}")
    return dataframe


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)
    overlap = set(train["part_group_id"]) & set(validation["part_group_id"])
    if overlap:
        raise SequenceSuiteError("Train/validation group overlap detected.")
    return train, validation


def build_vocabulary(texts: Iterable[str], max_vocabulary_size: int) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(text))
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    for token in ordered[: max(0, max_vocabulary_size - 2)]:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], sequence_length: int) -> list[int]:
    encoded = [vocabulary.get(token, 1) for token in tokenize(text)][:sequence_length]
    return encoded + [0] * (sequence_length - len(encoded))


def prepare_text_evidence(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    suite_config: dict[str, Any],
) -> tuple[dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequence_length = int(suite_config["sequence_length"])
    max_vocabulary_size = int(suite_config["max_vocabulary_size"])
    vocabulary = build_vocabulary(train["description"], max_vocabulary_size)

    train_sequences = np.asarray(
        [encode_text(text, vocabulary, sequence_length) for text in train["description"]],
        dtype="int32",
    )
    validation_sequences = np.asarray(
        [encode_text(text, vocabulary, sequence_length) for text in validation["description"]],
        dtype="int32",
    )
    y_train = train["label"].map(LABEL_TO_INDEX).to_numpy(dtype="int32")
    y_validation = validation["label"].map(LABEL_TO_INDEX).to_numpy(dtype="int32")

    def split_profile(frame: pd.DataFrame) -> dict[str, Any]:
        lengths = frame["description"].map(lambda value: len(tokenize(value)))
        unique_tokens = sorted({token for text in frame["description"] for token in tokenize(text)})
        return {
            "rows": int(len(frame)),
            "groups": int(frame["part_group_id"].nunique()),
            "unique_descriptions": int(frame["description"].nunique()),
            "label_counts": {label: int((frame["label"] == label).sum()) for label in LABELS},
            "token_length": {
                "minimum": int(lengths.min()),
                "median": float(lengths.median()),
                "mean": float(lengths.mean()),
                "maximum": int(lengths.max()),
            },
            "unique_token_count": len(unique_tokens),
        }

    profile = {
        "step": STEP,
        "status": "PASS",
        "train": split_profile(train),
        "validation": split_profile(validation),
        "group_overlap": 0,
        "label_order": list(LABELS),
        "text_column": "description",
        **LOCK_FLAGS,
    }
    write_json(TEXT_PROFILE_PATH, profile)

    representative_rows: list[pd.DataFrame] = []
    for split_name, frame in (("train", train), ("validation", validation)):
        for label in LABELS:
            selected = frame[frame["label"] == label].sort_values("sample_id").head(2).copy()
            selected.insert(0, "split", split_name)
            representative_rows.append(selected[["split", "sample_id", "part_category", "description", "label"]])
    pd.concat(representative_rows, ignore_index=True).to_csv(REPRESENTATIVE_EXAMPLES_PATH, index=False)

    loader_contract = {
        "status": "PASS",
        "authorized_inputs": [project_relative(TRAIN_PATH), project_relative(VALIDATION_PATH)],
        "forbidden_input_patterns": ["*test*.csv", "data/processed/integrated_test.csv"],
        "shuffle_train": True,
        "shuffle_validation": False,
        "shuffle_seed": int(suite_config["random_seeds"][0]),
        "batch_size": int(suite_config["batch_size"]),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "group_overlap": 0,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }
    write_json(LOADER_CONTRACT_PATH, loader_contract)

    batch = train.sample(n=min(int(suite_config["batch_size"]), len(train)), random_state=int(suite_config["random_seeds"][0])).copy()
    batch["token_count"] = batch["description"].map(lambda value: len(tokenize(value)))
    batch["encoded_sequence"] = [json.dumps(encode_text(value, vocabulary, sequence_length)) for value in batch["description"]]
    batch[["sample_id", "description", "label", "token_count", "encoded_sequence"]].to_csv(SAMPLE_BATCH_PATH, index=False)

    inverse = {index: token for token, index in vocabulary.items()}
    example_rows: list[dict[str, Any]] = []
    for _, row in pd.concat([train.head(6), validation.head(6)], ignore_index=True).iterrows():
        encoded = encode_text(row["description"], vocabulary, sequence_length)
        example_rows.append(
            {
                "sample_id": row["sample_id"],
                "description": row["description"],
                "tokens": json.dumps(tokenize(row["description"])),
                "encoded_sequence": json.dumps(encoded),
                "decoded_non_padding": json.dumps([inverse[index] for index in encoded if index != 0]),
                "label": row["label"],
            }
        )
    pd.DataFrame(example_rows).to_csv(TOKENIZATION_EXAMPLES_PATH, index=False)
    write_json(VOCABULARY_PATH, {"size": len(vocabulary), "token_to_index": vocabulary})
    write_json(
        TOKENIZATION_SUMMARY_PATH,
        {
            "status": "PASS",
            "token_pattern": TOKEN_PATTERN.pattern,
            "lowercase": True,
            "vocabulary_fit_split": "train_only",
            "vocabulary_size": len(vocabulary),
            "maximum_vocabulary_size": max_vocabulary_size,
            "sequence_length": sequence_length,
            "padding": "post",
            "truncation": "post",
            "padding_index": 0,
            "unknown_index": 1,
            "train_shape": list(train_sequences.shape),
            "validation_shape": list(validation_sequences.shape),
            "test_split_used": False,
        },
    )
    return vocabulary, train_sequences, validation_sequences, y_train, y_validation


def save_eda_figures(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(
        [train["description"].map(lambda value: len(tokenize(value))), validation["description"].map(lambda value: len(tokenize(value)))],
        bins=np.arange(1.5, 6.5, 1),
        label=["train", "validation"],
        alpha=0.75,
    )
    plt.xlabel("Tokens per description")
    plt.ylabel("Samples")
    plt.title("Automotive-part text length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["text_length_distribution"], dpi=150)
    plt.close()

    counts = pd.DataFrame(
        {
            "train": [int((train["label"] == label).sum()) for label in LABELS],
            "validation": [int((validation["label"] == label).sum()) for label in LABELS],
        },
        index=LABELS,
    )
    ax = counts.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Samples")
    ax.set_title("Train/validation label distribution")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["label_distribution"], dpi=150)
    plt.close()


def import_keras() -> Any:
    if "KERAS_BACKEND" not in os.environ:
        try:
            import tensorflow  # noqa: F401
        except ModuleNotFoundError:
            os.environ["KERAS_BACKEND"] = "torch"
    import keras

    return keras


def set_seed(keras: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_neural_model(
    keras: Any,
    family: str,
    vocabulary_size: int,
    sequence_length: int,
) -> tuple[Any, Any | None]:
    layers = keras.layers
    inputs = keras.Input(shape=(sequence_length,), dtype="int32", name="tokens")
    embedded = layers.Embedding(vocabulary_size, 24, name="token_embedding")(inputs)
    attention_model = None

    if family == "embedding_average":
        x = layers.GlobalAveragePooling1D()(embedded)
        x = layers.Dense(24, activation="relu")(x)
    elif family == "textcnn":
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(embedded)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dense(24, activation="relu")(x)
    elif family == "gru":
        x = layers.GRU(28)(embedded)
    elif family == "lstm":
        x = layers.LSTM(28)(embedded)
    elif family == "transformer":
        attention_layer = layers.MultiHeadAttention(num_heads=2, key_dim=12, name="multi_head_attention")
        attended, scores = attention_layer(embedded, embedded, return_attention_scores=True)
        x = layers.LayerNormalization()(embedded + attended)
        feed_forward = layers.Dense(48, activation="relu")(x)
        feed_forward = layers.Dense(24)(feed_forward)
        x = layers.LayerNormalization()(x + feed_forward)
        x = layers.GlobalAveragePooling1D()(x)
        attention_model = keras.Model(inputs, scores, name="transformer_attention_scores")
    else:
        raise SequenceSuiteError(f"Unknown neural family: {family}")

    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(len(LABELS), activation="softmax", name="classification")(x)
    model = keras.Model(inputs, outputs, name=f"sequence_{family}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, attention_model


def prediction_rows(
    validation: pd.DataFrame,
    run_id: str,
    family: str,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (_, sample) in enumerate(validation.iterrows()):
        row: dict[str, Any] = {
            "run_id": run_id,
            "family": family,
            "sample_id": sample["sample_id"],
            "description": sample["description"],
            "true_label": sample["label"],
            "predicted_label": INDEX_TO_LABEL[int(predictions[index])],
            "correct": bool(LABEL_TO_INDEX[sample["label"]] == int(predictions[index])),
        }
        for label_index, label in enumerate(LABELS):
            row[f"probability_{label.lower()}"] = float(probabilities[index, label_index])
        rows.append(row)
    return rows


def run_neural_experiment(
    keras: Any,
    family: str,
    experiment_id: str,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    validation: pd.DataFrame,
    suite_config: dict[str, Any],
) -> tuple[RunResult, Any | None, Any]:
    set_seed(keras, seed)
    model, attention_model = build_neural_model(
        keras,
        family,
        int(np.max(x_train)) + 2,
        int(suite_config["sequence_length"]),
    )
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(suite_config["early_stopping_patience"]),
        restore_best_weights=True,
    )
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=int(suite_config["max_epochs"]),
        batch_size=int(suite_config["batch_size"]),
        verbose=0,
        shuffle=True,
        callbacks=[callback],
    ).history
    elapsed = time.perf_counter() - started
    probabilities = np.asarray(model.predict(x_validation, verbose=0))
    predictions = probabilities.argmax(axis=1)
    accuracy = float(accuracy_score(y_validation, predictions))
    macro_f1 = float(f1_score(y_validation, predictions, average="macro", zero_division=0))
    val_losses = list(map(float, history.get("val_loss", [])))
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else len(history.get("loss", []))
    run_id = f"{family}-seed-{seed}"
    history_rows: list[dict[str, Any]] = []
    epochs_completed = len(history.get("loss", []))
    for epoch in range(epochs_completed):
        history_rows.append(
            {
                "run_id": run_id,
                "family": family,
                "seed": seed,
                "epoch": epoch + 1,
                "loss": float(history["loss"][epoch]),
                "accuracy": float(history["accuracy"][epoch]),
                "val_loss": float(history["val_loss"][epoch]),
                "val_accuracy": float(history["val_accuracy"][epoch]),
            }
        )
    final_train_accuracy = float(history["accuracy"][-1])
    result = RunResult(
        experiment_id=experiment_id,
        run_id=run_id,
        family=family,
        variant=family,
        seed=seed,
        parameter_count=int(model.count_params()),
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        training_time_seconds=elapsed,
        validation_accuracy=accuracy,
        validation_macro_f1=macro_f1,
        final_train_accuracy=final_train_accuracy,
        final_train_loss=float(history["loss"][-1]),
        final_validation_loss=float(history["val_loss"][-1]),
        generalization_gap=float(final_train_accuracy - accuracy),
        notes="Keras sequence model; train/validation only.",
        history_rows=history_rows,
        prediction_rows=prediction_rows(validation, run_id, family, probabilities, predictions),
        confusion_matrix_values=confusion_matrix(y_validation, predictions, labels=list(range(len(LABELS)))).tolist(),
        probabilities=probabilities,
        predictions=predictions,
    )
    return result, attention_model, model


def run_tfidf_experiments(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seeds: Sequence[int],
) -> list[RunResult]:
    results: list[RunResult] = []
    variants = [(1, 1, 0.1), (1, 1, 1.0), (1, 1, 10.0), (1, 2, 0.1), (1, 2, 1.0), (1, 2, 10.0)]
    seed = int(seeds[0])
    for min_ngram, max_ngram, c_value in variants:
        variant = f"ngram-{min_ngram}-{max_ngram}-c-{c_value:g}"
        run_id = f"tfidf-{variant}"
        vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-z0-9]+\b", ngram_range=(min_ngram, max_ngram))
        started = time.perf_counter()
        x_train = vectorizer.fit_transform(train["description"])
        x_validation = vectorizer.transform(validation["description"])
        model = LogisticRegression(C=c_value, max_iter=1000, random_state=seed)
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        raw_probabilities = model.predict_proba(x_validation)
        probabilities = np.zeros((len(validation), len(LABELS)), dtype=float)
        for source_index, label_index in enumerate(model.classes_):
            probabilities[:, int(label_index)] = raw_probabilities[:, source_index]
        predictions = probabilities.argmax(axis=1)
        accuracy = float(accuracy_score(y_validation, predictions))
        macro_f1 = float(f1_score(y_validation, predictions, average="macro", zero_division=0))
        train_predictions = model.predict(x_train)
        train_accuracy = float(accuracy_score(y_train, train_predictions))
        results.append(
            RunResult(
                experiment_id="SEQ-005",
                run_id=run_id,
                family="tfidf_logistic",
                variant=variant,
                seed=seed,
                parameter_count=int(model.coef_.size + model.intercept_.size),
                epochs_completed=1,
                best_epoch=1,
                training_time_seconds=elapsed,
                validation_accuracy=accuracy,
                validation_macro_f1=macro_f1,
                final_train_accuracy=train_accuracy,
                generalization_gap=train_accuracy - accuracy,
                notes=f"TF-IDF vocabulary={len(vectorizer.vocabulary_)}; classical baseline.",
                prediction_rows=prediction_rows(validation, run_id, "tfidf_logistic", probabilities, predictions),
                confusion_matrix_values=confusion_matrix(y_validation, predictions, labels=list(range(len(LABELS)))).tolist(),
                probabilities=probabilities,
                predictions=predictions,
            )
        )
    return results


def select_champions(results: Sequence[RunResult]) -> dict[str, RunResult]:
    champions: dict[str, RunResult] = {}
    for result in results:
        current = champions.get(result.family)
        score = (result.validation_macro_f1, result.validation_accuracy, -result.parameter_count, -result.training_time_seconds)
        current_score = (-1.0, -1.0, float("-inf"), float("-inf")) if current is None else (
            current.validation_macro_f1,
            current.validation_accuracy,
            -current.parameter_count,
            -current.training_time_seconds,
        )
        if current is None or score > current_score:
            champions[result.family] = result
    return champions


def write_model_evidence(results: list[RunResult], champions: dict[str, RunResult]) -> None:
    run_rows = [result.row() for result in results]
    pd.DataFrame(run_rows).to_csv(TRAINING_RUNS_PATH, index=False)
    history_rows = [row for result in results for row in result.history_rows]
    pd.DataFrame(history_rows, columns=["run_id", "family", "seed", "epoch", "loss", "accuracy", "val_loss", "val_accuracy"]).to_csv(TRAINING_HISTORIES_PATH, index=False)
    prediction_table = pd.DataFrame([row for result in results for row in result.prediction_rows])
    prediction_table.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    write_json(
        CONFUSION_MATRICES_PATH,
        {
            result.run_id: {
                "family": result.family,
                "label_order": list(LABELS),
                "matrix": result.confusion_matrix_values,
            }
            for result in results
        },
    )

    comparison_rows: list[dict[str, Any]] = []
    for family in sorted(champions):
        champion = champions[family]
        family_results = [result for result in results if result.family == family]
        precision, recall, f1_values, support = precision_recall_fscore_support(
            [LABEL_TO_INDEX[row["true_label"]] for row in champion.prediction_rows],
            champion.predictions,
            labels=list(range(len(LABELS))),
            zero_division=0,
        )
        row = champion.row()
        row.update(
            {
                "selected_run_id": champion.run_id,
                "run_count": len(family_results),
                "mean_macro_f1": float(np.mean([item.validation_macro_f1 for item in family_results])),
                "std_macro_f1": float(np.std([item.validation_macro_f1 for item in family_results])),
                "per_class_f1": json.dumps({LABELS[index]: float(f1_values[index]) for index in range(len(LABELS))}),
                "per_class_support": json.dumps({LABELS[index]: int(support[index]) for index in range(len(LABELS))}),
            }
        )
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).sort_values(["validation_macro_f1", "validation_accuracy"], ascending=False)
    comparison.to_csv(MODEL_COMPARISON_CSV_PATH, index=False)
    write_json(MODEL_COMPARISON_JSON_PATH, {"status": "PASS", "label_order": list(LABELS), "models": comparison.to_dict(orient="records")})

    ax = comparison.set_index("family")["validation_macro_f1"].sort_values().plot(kind="barh", figsize=(9, 5))
    ax.set_xlabel("Validation Macro F1")
    ax.set_title("Selected model per sequence family")
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["model_macro_f1"], dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(comparison["parameter_count"].clip(lower=1), comparison["validation_macro_f1"])
    for _, row in comparison.iterrows():
        plt.annotate(row["family"], (max(1, row["parameter_count"]), row["validation_macro_f1"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.xscale("log")
    plt.xlabel("Parameter / coefficient count (log scale)")
    plt.ylabel("Validation Macro F1")
    plt.title("Model complexity versus validation quality")
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["complexity_tradeoff"], dpi=150)
    plt.close()


def write_roc_evidence(champions: dict[str, RunResult], y_validation: np.ndarray) -> None:
    payload: dict[str, Any] = {"status": "PASS", "label_order": list(LABELS), "models": {}}
    plt.figure(figsize=(9, 6))
    for family, champion in sorted(champions.items()):
        model_payload: dict[str, Any] = {"run_id": champion.run_id, "classes": {}}
        family_aucs: list[float] = []
        for label_index, label in enumerate(LABELS):
            binary = (y_validation == label_index).astype(int)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(binary, champion.probabilities[:, label_index])
            area = float(auc(false_positive_rate, true_positive_rate))
            family_aucs.append(area)
            model_payload["classes"][label] = {
                "auc": area,
                "false_positive_rate": false_positive_rate.tolist(),
                "true_positive_rate": true_positive_rate.tolist(),
                "thresholds": thresholds.tolist(),
            }
        model_payload["macro_auc"] = float(np.mean(family_aucs))
        payload["models"][family] = model_payload
        micro_binary = np.eye(len(LABELS))[y_validation].ravel()
        micro_scores = champion.probabilities.ravel()
        fpr, tpr, _ = roc_curve(micro_binary, micro_scores)
        plt.plot(fpr, tpr, label=f"{family} (micro AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Validation one-vs-rest ROC comparison")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["roc_curves"], dpi=150)
    plt.close()
    write_json(ROC_CURVES_PATH, payload)


def write_error_analysis(champion: RunResult, validation: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for index, (_, sample) in enumerate(validation.iterrows()):
        predicted_index = int(champion.predictions[index])
        true_index = LABEL_TO_INDEX[sample["label"]]
        if predicted_index == true_index:
            continue
        ordered = np.argsort(champion.probabilities[index])[::-1]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "description": sample["description"],
                "part_category": sample["part_category"],
                "true_label": sample["label"],
                "predicted_label": INDEX_TO_LABEL[predicted_index],
                "predicted_probability": float(champion.probabilities[index, predicted_index]),
                "true_label_probability": float(champion.probabilities[index, true_index]),
                "probability_margin": float(champion.probabilities[index, ordered[0]] - champion.probabilities[index, ordered[1]]),
                "error_type": f"{sample['label']}→{INDEX_TO_LABEL[predicted_index]}",
                "selected_model_family": champion.family,
                "selected_run_id": champion.run_id,
            }
        )
    pd.DataFrame(rows, columns=[
        "sample_id", "description", "part_category", "true_label", "predicted_label",
        "predicted_probability", "true_label_probability", "probability_margin", "error_type",
        "selected_model_family", "selected_run_id",
    ]).to_csv(ERROR_ANALYSIS_PATH, index=False)


def to_numpy(keras: Any, value: Any) -> np.ndarray:
    try:
        return np.asarray(keras.ops.convert_to_numpy(value))
    except Exception:
        return np.asarray(value)


def write_attention_evidence(
    keras: Any,
    transformer_result: RunResult,
    transformer_attention_model: Any,
    x_validation: np.ndarray,
    validation: pd.DataFrame,
    vocabulary: dict[str, int],
) -> None:
    if transformer_attention_model is None:
        raise SequenceSuiteError("Transformer attention model is unavailable.")
    inverse = {index: token for token, index in vocabulary.items()}
    correctness = transformer_result.predictions == validation["label"].map(LABEL_TO_INDEX).to_numpy()
    correct_indices = np.flatnonzero(correctness)
    incorrect_indices = np.flatnonzero(~correctness)
    correct_index = int(correct_indices[0]) if len(correct_indices) else 0
    incorrect_index = int(incorrect_indices[0]) if len(incorrect_indices) else min(1, len(validation) - 1)
    selections = {"correct": correct_index, "incorrect": incorrect_index}
    evidence: dict[str, Any] = {
        "status": "PASS",
        "transformer_run_id": transformer_result.run_id,
        "head_count": 2,
        "selection": {},
        "test_split_used": False,
    }
    token_summary_rows: list[dict[str, Any]] = []
    for selection_name, row_index in selections.items():
        sequence = x_validation[row_index : row_index + 1]
        scores = to_numpy(keras, transformer_attention_model(sequence, training=False))[0]
        tokens = [inverse.get(int(index), "<UNK>") for index in sequence[0] if int(index) != 0]
        token_count = max(1, len(tokens))
        scores = scores[:, :token_count, :token_count]
        sample = validation.iloc[row_index]
        predicted_label = INDEX_TO_LABEL[int(transformer_result.predictions[row_index])]
        selection_payload = {
            "sample_id": sample["sample_id"],
            "description": sample["description"],
            "tokens": tokens,
            "true_label": sample["label"],
            "predicted_label": predicted_label,
            "correct": bool(selection_name == "correct" and correctness[row_index]) if len(correct_indices) else bool(correctness[row_index]),
            "heads": [],
        }
        for head_index in range(min(2, scores.shape[0])):
            matrix = scores[head_index]
            mean_received = matrix.mean(axis=0)
            ranked_indices = np.argsort(mean_received)[::-1]
            head_payload = {
                "head": head_index + 1,
                "matrix": matrix.tolist(),
                "top_attended_tokens": [tokens[int(index)] for index in ranked_indices[: min(3, token_count)]],
            }
            selection_payload["heads"].append(head_payload)
            for token_index, token in enumerate(tokens):
                token_summary_rows.append(
                    {
                        "selection": selection_name,
                        "sample_id": sample["sample_id"],
                        "head": head_index + 1,
                        "token_position": token_index,
                        "token": token,
                        "mean_attention_received": float(mean_received[token_index]),
                    }
                )
            figure_path = FIGURE_PATHS[f"attention_{selection_name}_head_{head_index + 1}"]
            plt.figure(figsize=(6, 5))
            plt.imshow(matrix, aspect="auto")
            plt.xticks(range(token_count), tokens, rotation=45, ha="right")
            plt.yticks(range(token_count), tokens)
            plt.xlabel("Key token")
            plt.ylabel("Query token")
            plt.title(f"{selection_name.title()} example — attention head {head_index + 1}")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(figure_path, dpi=150)
            plt.close()
        evidence["selection"][selection_name] = selection_payload
    write_json(ATTENTION_EVIDENCE_PATH, evidence)
    pd.DataFrame(token_summary_rows).to_csv(ATTENTION_TOKEN_SUMMARY_PATH, index=False)


def write_pretrained_gate() -> None:
    write_json(
        PRETRAINED_GATE_PATH,
        {
            "experiment_id": "SEQ-010",
            "status": "DEFERRED_EXPLICIT_APPROVAL_REQUIRED",
            "reason": "Downloading pretrained transformer weights requires explicit user authorization and license/revision recording.",
            "approval_received": False,
            "network_download_attempted": False,
            "pretrained_weights_downloaded": False,
            "pretrained_model_loaded": False,
            "model_identifier": None,
            "model_revision": None,
            "model_license": None,
            "test_split_used": False,
            "final_test_evaluation_authorized": False,
        },
    )


def write_registry(results: Sequence[RunResult]) -> None:
    run_counts = Counter(result.experiment_id for result in results)
    records: list[dict[str, Any]] = []
    for sequence_id in SEQUENCE_IDS:
        if sequence_id == "SEQ-010":
            status = "DEFERRED_EXPLICIT_APPROVAL_REQUIRED"
            runs = 0
            evidence = [project_relative(PRETRAINED_GATE_PATH)]
        else:
            status = "COMPLETED"
            runs = int(run_counts.get(sequence_id, 0))
            evidence = [project_relative(MODEL_COMPARISON_CSV_PATH)]
            if sequence_id in {"SEQ-001", "SEQ-002"}:
                evidence = [project_relative(TEXT_PROFILE_PATH), project_relative(LOADER_CONTRACT_PATH)]
            elif sequence_id == "SEQ-003":
                evidence = [project_relative(TOKENIZATION_SUMMARY_PATH), project_relative(VOCABULARY_PATH)]
            elif sequence_id == "SEQ-008":
                evidence = [project_relative(ROC_CURVES_PATH), project_relative(ERROR_ANALYSIS_PATH)]
            elif sequence_id == "SEQ-009":
                evidence = [project_relative(ATTENTION_EVIDENCE_PATH), project_relative(ATTENTION_TOKEN_SUMMARY_PATH)]
        records.append(
            {
                "experiment_id": sequence_id,
                "step": STEP,
                "status": status,
                "training_runs": runs,
                "evidence": evidence,
                "test_split_allowed": False,
                "final_test_evaluation_authorized": False,
                "production_final_model_changed": False,
                "pretrained_weights_downloaded": False,
            }
        )
    write_json(
        EXECUTION_REGISTRY_JSON_PATH,
        {
            "step": STEP,
            "status": "PASS",
            "readiness": READINESS,
            "experiments": records,
            "total_training_runs": len(results),
            **LOCK_FLAGS,
        },
    )
    csv_rows = []
    for record in records:
        row = dict(record)
        row["evidence"] = ";".join(record["evidence"])
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(EXECUTION_REGISTRY_CSV_PATH, index=False)


def write_summary_and_status(results: Sequence[RunResult], champions: dict[str, RunResult]) -> None:
    best = max(champions.values(), key=lambda result: (result.validation_macro_f1, result.validation_accuracy))
    status = {
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "base_checkpoint": BASE_CHECKPOINT,
        "sequence_exercise_problems": "9/9 core completed; 1/1 pretrained extension gated",
        "completed_problem_ids": list(COMPLETED_SEQUENCE_IDS),
        "deferred_problem_ids": list(DEFERRED_SEQUENCE_IDS),
        "required_core_problem_count": 9,
        "completed_core_problem_count": 9,
        "training_runs_recorded": len(results),
        "model_families_compared": sorted(champions),
        "selected_educational_validation_run": best.run_id,
        "selected_educational_validation_family": best.family,
        "selected_validation_accuracy": best.validation_accuracy,
        "selected_validation_macro_f1": best.validation_macro_f1,
        "selection_scope": "educational_validation_only",
        "production_final_model_changed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "pretrained_weights_downloaded": False,
        "pretrained_extension_status": "DEFERRED_EXPLICIT_APPROVAL_REQUIRED",
    }
    write_json(STATUS_PATH, status)
    write_text(
        SUMMARY_PATH,
        f"""
# Step 011.2 — Transformers & Sequence Modelling Experimental Suite

## Result

**Status:** PASS  
**Readiness:** `{READINESS}`

The core suite completes SEQ-001 through SEQ-009 with deterministic text
loading, train-only vocabulary construction, padded integer sequences, a dense
embedding baseline, TF-IDF + logistic regression, TextCNN, GRU, LSTM, and a
small Transformer encoder with two inspected attention heads.

## Experimental evidence

- Core exercise problems completed: **9 / 9**
- Optional pretrained extension: **gated, not downloaded**
- Training runs recorded: **{len(results)}**
- Model families compared: **{len(champions)}**
- Best educational validation family: **{best.family}**
- Best validation accuracy: **{best.validation_accuracy:.4f}**
- Best validation Macro F1: **{best.validation_macro_f1:.4f}**

The short text descriptions alone do not uniquely determine an image–text
relationship label. Results are therefore interpreted as an educational
sequence-modelling comparison, not as a replacement for the multimodal final
model.

## Safety boundary

- Production/final model changed: **false**
- Locked test CSV files opened: **false**
- Test split used: **false**
- Final test evaluation authorized: **false**
- Pretrained weights downloaded: **false**

SEQ-010 remains `DEFERRED_EXPLICIT_APPROVAL_REQUIRED`. A future pretrained
run must first record explicit approval, model identifier, fixed revision, and
license evidence.
""",
    )


def write_documentation() -> None:
    write_text(
        DOCUMENTATION_PATH,
        f"""
# Step 011.2 — Transformers & Sequence Modelling Experimental Suite

This suite maps the ten sequence-modelling exercise slots to the
`automotive-part-image-text-matching` project.

| ID | Project implementation | State |
|---|---|---|
| SEQ-001 | Inspect committed automotive-part descriptions | Complete |
| SEQ-002 | Deterministic train/validation text loader | Complete |
| SEQ-003 | Train-only vocabulary, tokenization, padding | Complete |
| SEQ-004 | Dense embedding baseline | Complete |
| SEQ-005 | TF-IDF + logistic-regression baseline | Complete |
| SEQ-006 | TextCNN, GRU, and LSTM comparison | Complete |
| SEQ-007 | Small Transformer encoder | Complete |
| SEQ-008 | Metrics, confusion matrices, ROC, and error analysis | Complete |
| SEQ-009 | Two-head attention inspection | Complete |
| SEQ-010 | Pretrained transformer | Deferred — explicit approval required |

## Reproducibility

The suite uses seeds 42, 43, and 44 for neural comparisons. Vocabulary is fit
on the training descriptions only. Reports, figures, an executed notebook,
and an integrity manifest are committed; model weights are not committed.

## Locked evaluation boundary

The only authorized inputs are `data/processed/integrated_train.csv` and
`data/processed/integrated_validation.csv`. No test CSV is loaded or scored.
The frozen final model remains unchanged. Readiness is
`{READINESS}`.
""",
    )


def write_manifest() -> None:
    static_paths = [SUITE_CONFIG_PATH, *EXPERIMENT_CONFIG_PATHS, DOCUMENTATION_PATH]
    artifacts = []
    for path in [*static_paths, *GENERATED_PATHS]:
        if path == MANIFEST_PATH:
            continue
        if not path.is_file():
            raise SequenceSuiteError(f"Manifest input missing: {project_relative(path)}")
        artifacts.append(
            {
                "path": project_relative(path),
                "sha256": normalized_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(
        MANIFEST_PATH,
        {
            "step": STEP,
            "status": "PASS",
            "readiness": READINESS,
            "base_checkpoint": BASE_CHECKPOINT,
            "source_commit": current_git_commit(),
            "artifact_count": len(artifacts),
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
            "training_runs_recorded": len(pd.read_csv(TRAINING_RUNS_PATH)),
            "completed_core_problems": len(COMPLETED_SEQUENCE_IDS),
            "deferred_pretrained_problems": len(DEFERRED_SEQUENCE_IDS),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "keras_backend": os.environ.get("KERAS_BACKEND", "tensorflow_default"),
            },
            **LOCK_FLAGS,
        },
    )


def validate_configs() -> dict[str, Any]:
    suite_config = read_json(SUITE_CONFIG_PATH)
    if suite_config.get("step") != STEP or suite_config.get("readiness") != READINESS:
        raise SequenceSuiteError("Sequence suite configuration is invalid.")
    configs = [read_json(path) for path in EXPERIMENT_CONFIG_PATHS]
    if [payload.get("experiment_id") for payload in configs] != list(SEQUENCE_IDS):
        raise SequenceSuiteError("Sequence experiment configuration IDs are invalid.")
    if any(payload.get("test_split_allowed") is not False for payload in configs):
        raise SequenceSuiteError("A sequence configuration allows the test split.")
    return suite_config


def run_suite() -> dict[str, Any]:
    suite_config = validate_configs()
    train, validation = load_dataframes()
    save_eda_figures(train, validation)
    vocabulary, x_train, x_validation, y_train, y_validation = prepare_text_evidence(train, validation, suite_config)

    keras = import_keras()
    results: list[RunResult] = []
    results.extend(run_tfidf_experiments(train, validation, y_train, y_validation, suite_config["random_seeds"]))

    family_to_experiment = {
        "embedding_average": "SEQ-004",
        "textcnn": "SEQ-006",
        "gru": "SEQ-006",
        "lstm": "SEQ-006",
        "transformer": "SEQ-007",
    }
    best_transformer: RunResult | None = None
    best_transformer_attention: Any | None = None
    best_transformer_model: Any | None = None
    for family, experiment_id in family_to_experiment.items():
        for seed in suite_config["random_seeds"]:
            result, attention_model, model = run_neural_experiment(
                keras,
                family,
                experiment_id,
                int(seed),
                x_train,
                y_train,
                x_validation,
                y_validation,
                validation,
                suite_config,
            )
            results.append(result)
            if family == "transformer":
                if best_transformer is None or (result.validation_macro_f1, result.validation_accuracy) > (
                    best_transformer.validation_macro_f1,
                    best_transformer.validation_accuracy,
                ):
                    best_transformer = result
                    best_transformer_attention = attention_model
                    best_transformer_model = model

    if len(results) != 21:
        raise SequenceSuiteError(f"Expected 21 training runs, found {len(results)}.")
    champions = select_champions(results)
    write_model_evidence(results, champions)
    write_roc_evidence(champions, y_validation)
    overall_champion = max(champions.values(), key=lambda result: (result.validation_macro_f1, result.validation_accuracy))
    write_error_analysis(overall_champion, validation)
    if best_transformer is None or best_transformer_attention is None or best_transformer_model is None:
        raise SequenceSuiteError("Transformer evidence was not produced.")
    write_attention_evidence(keras, best_transformer, best_transformer_attention, x_validation, validation, vocabulary)
    write_pretrained_gate()
    write_registry(results)
    write_summary_and_status(results, champions)
    write_documentation()
    notebook_audit = build_and_execute_notebook()
    write_manifest()
    return {
        "status": "PASS",
        "readiness": READINESS,
        "training_runs": len(results),
        "model_families": len(champions),
        "notebook_status": notebook_audit["status"],
        "test_split_used": False,
        "pretrained_weights_downloaded": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 011.2 sequence experiments.")
    parser.add_argument("--skip-notebook", action="store_true", help="Reserved compatibility flag; notebook remains required.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    effective_argv = argv
    if effective_argv is None:
        effective_argv = [] if Path(sys.argv[0]).stem == "project_cli" else sys.argv[1:]
    build_parser().parse_args(effective_argv)
    result = run_suite()
    print("Step 011.2 sequence experimental suite")
    for key, value in result.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
