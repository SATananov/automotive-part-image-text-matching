from __future__ import annotations

import io
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data import LABELS, PROJECT_ROOT, check_split, encode_labels, load_images, load_split
from src.models import build_image_model, build_multimodal_model, build_text_model

RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_STATE = 42
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 32


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocabulary(texts: pd.Series) -> dict[str, int]:
    counts = Counter(token for text in texts for token in tokenize(str(text)))
    return {token: index + 2 for index, token in enumerate(sorted(counts))}


def encode_text(texts: pd.Series, vocabulary: dict[str, int], length: int = 12) -> np.ndarray:
    rows = []
    for text in texts:
        row = [vocabulary.get(token, 1) for token in tokenize(str(text))[:length]]
        rows.append(row + [0] * (length - len(row)))
    return np.asarray(rows, dtype=np.int32)


def score(name: str, modality: str, true: pd.Series, predicted: np.ndarray) -> dict[str, object]:
    return {
        "model": name,
        "modality": modality,
        "full_validation_accuracy": accuracy_score(true, predicted),
        "full_validation_macro_f1": f1_score(true, predicted, average="macro", zero_division=0),
    }


def add_real_scores(row: dict[str, object], validation: pd.DataFrame, predicted: np.ndarray) -> None:
    mask = validation["source"].eq("wikimedia").to_numpy()
    row["real_image_accuracy"] = accuracy_score(validation.loc[mask, "label"], predicted[mask])
    row["real_image_macro_f1"] = f1_score(
        validation.loc[mask, "label"], predicted[mask], average="macro", zero_division=0
    )


def prediction_rows(
    model: str,
    model_slug: str,
    validation: pd.DataFrame,
    predicted: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "part_group_id",
        "image_id",
        "part_category",
        "source",
        "description",
        "label",
        "image_path",
    ]
    out = validation[columns].copy().rename(columns={"label": "true_label"})
    out["predicted_label"] = predicted
    out["is_correct"] = out["true_label"].eq(out["predicted_label"])
    out["model"] = model
    out["model_slug"] = model_slug
    ordered = [
        "sample_id",
        "part_group_id",
        "image_id",
        "part_category",
        "source",
        "description",
        "true_label",
        "predicted_label",
        "is_correct",
        "image_path",
        "model",
        "model_slug",
    ]
    return out[ordered]


def fit_keras_model(keras, model, x_train, y_train, x_val, y_val):
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[callback],
        verbose=0,
    )
    predicted = model.predict(x_val, batch_size=BATCH_SIZE, verbose=0).argmax(axis=1)
    return predicted, pd.DataFrame(history.history)


def save_model_summary(model, path: Path) -> None:
    buffer = io.StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    with path.open("w", encoding="utf-8", newline="\\n") as handle:
        handle.write(buffer.getvalue().rstrip() + "\\n")


def save_multimodal_tables(all_predictions: pd.DataFrame) -> None:
    multimodal = all_predictions[all_predictions["model_slug"].eq("keras_multimodal")].copy()
    if len(multimodal) == 0:
        raise ValueError("The Keras multimodal predictions are missing.")
    multimodal.to_csv(
        RESULTS_DIR / "multimodal_validation_predictions.csv",
        index=False,
        lineterminator="\n",
    )

    rows = []
    for category, group in multimodal.groupby("part_category", sort=True):
        rows.append({
            "part_category": category,
            "samples": len(group),
            "accuracy": accuracy_score(group["true_label"], group["predicted_label"]),
            "macro_f1": f1_score(
                group["true_label"], group["predicted_label"], average="macro", zero_division=0
            ),
        })
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "multimodal_per_category.csv",
        index=False,
        lineterminator="\n",
    )


def main() -> None:
    try:
        import keras
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise SystemExit("TensorFlow is required. Install the packages from requirements.txt.") from error

    RESULTS_DIR.mkdir(exist_ok=True)
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    keras.utils.set_random_seed(RANDOM_STATE)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    train = load_split("train")
    validation = load_split("validation")
    if any(check_split(train, validation).values()):
        raise SystemExit("Train and validation overlap.")

    true_labels = validation["label"]
    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    def record(name: str, slug: str, modality: str, predicted: np.ndarray) -> None:
        row = score(name, modality, true_labels, predicted)
        add_real_scores(row, validation, predicted)
        results.append(row)
        predictions.append(prediction_rows(name, slug, validation, predicted))

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(train), 1)), train["label"])
    record(
        "Majority baseline",
        "majority",
        "none",
        majority.predict(np.zeros((len(validation), 1))),
    )

    text_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    train_tfidf = text_vectorizer.fit_transform(train["description"])
    validation_tfidf = text_vectorizer.transform(validation["description"])
    text_baseline = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    text_baseline.fit(train_tfidf, train["label"])
    record(
        "TF-IDF + Logistic Regression",
        "tfidf_logistic_regression",
        "text",
        text_baseline.predict(validation_tfidf),
    )

    flat_train = load_images(train, (32, 32)).reshape(len(train), -1) / 255.0
    flat_validation = load_images(validation, (32, 32)).reshape(len(validation), -1) / 255.0
    image_baseline = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    image_baseline.fit(flat_train, train["label"])
    record(
        "Image pixels + Logistic Regression",
        "image_logistic_regression",
        "image",
        image_baseline.predict(flat_validation),
    )

    combined_baseline = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    combined_baseline.fit(
        np.hstack([flat_train, train_tfidf.toarray()]),
        train["label"],
    )
    record(
        "Image + text Logistic Regression",
        "image_text_logistic_regression",
        "image + text",
        combined_baseline.predict(
            np.hstack([flat_validation, validation_tfidf.toarray()])
        ),
    )

    vocabulary = build_vocabulary(train["description"])
    train_text = encode_text(train["description"], vocabulary)
    validation_text = encode_text(validation["description"], vocabulary)
    train_images = load_images(train)
    validation_images = load_images(validation)
    y_train = encode_labels(train["label"])
    y_validation = encode_labels(validation["label"])
    index_to_label = np.asarray(LABELS)

    jobs = [
        (
            "Keras text model",
            "keras_text",
            "text",
            build_text_model(keras, len(vocabulary) + 2),
            train_text,
            validation_text,
        ),
        (
            "Keras image model",
            "keras_image",
            "image",
            build_image_model(keras),
            train_images,
            validation_images,
        ),
        (
            "Keras multimodal model",
            "keras_multimodal",
            "image + text",
            build_multimodal_model(keras, len(vocabulary) + 2),
            {"image": train_images, "text": train_text},
            {"image": validation_images, "text": validation_text},
        ),
    ]
    for name, slug, modality, model, x_train, x_validation in jobs:
        save_model_summary(model, RESULTS_DIR / f"{slug}_architecture.txt")
        pred_index, history = fit_keras_model(
            keras,
            model,
            x_train,
            y_train,
            x_validation,
            y_validation,
        )
        history.to_csv(
            RESULTS_DIR / f"{slug}_training_history.csv",
            index=False,
            lineterminator="\n",
        )
        record(name, slug, modality, index_to_label[pred_index])
        keras.backend.clear_session()

    result_table = pd.DataFrame(results).sort_values(
        ["real_image_macro_f1", "full_validation_macro_f1"],
        ascending=False,
        kind="stable",
    )
    all_predictions = pd.concat(predictions, ignore_index=True)
    result_table.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False,
        lineterminator="\n",
    )
    all_predictions.to_csv(
        RESULTS_DIR / "validation_predictions.csv",
        index=False,
        lineterminator="\n",
    )
    save_multimodal_tables(all_predictions)

    run_info = {
        "result_source": "Generated by python -m src.train from the current project files.",
        "random_state": RANDOM_STATE,
        "training_split": "data/train.csv",
        "model_selection_split": "data/validation.csv",
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "saved_model_count": len(result_table),
    }
    with (RESULTS_DIR / "run_info.json").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(json.dumps(run_info, indent=2) + "\n")
    print(result_table.to_string(index=False))


if __name__ == "__main__":
    main()
