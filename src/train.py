from __future__ import annotations

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
from sklearn.pipeline import make_pipeline

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


def prediction_rows(model: str, validation: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    out = validation[
        ["sample_id", "part_group_id", "image_id", "image_path", "part_category", "source", "description", "label"]
    ].copy()
    out = out.rename(columns={"label": "true_label"})
    out["predicted_label"] = predicted
    out["is_correct"] = out["true_label"].eq(out["predicted_label"])
    out["model"] = model
    return out


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


def main() -> None:
    try:
        import keras
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise SystemExit("TensorFlow is required. Install the packages from requirements.txt.") from error

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

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(train), 1)), train["label"])
    pred = majority.predict(np.zeros((len(validation), 1)))
    row = score("Majority baseline", "none", true_labels, pred)
    add_real_scores(row, validation, pred)
    results.append(row)
    predictions.append(prediction_rows("Majority baseline", validation, pred))

    text_baseline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2)),
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    )
    text_baseline.fit(train["description"], train["label"])
    pred = text_baseline.predict(validation["description"])
    row = score("TF-IDF + Logistic Regression", "text", true_labels, pred)
    add_real_scores(row, validation, pred)
    results.append(row)
    predictions.append(prediction_rows("TF-IDF + Logistic Regression", validation, pred))

    flat_train = load_images(train, (32, 32)).reshape(len(train), -1) / 255.0
    flat_val = load_images(validation, (32, 32)).reshape(len(validation), -1) / 255.0
    image_baseline = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    image_baseline.fit(flat_train, train["label"])
    pred = image_baseline.predict(flat_val)
    row = score("Image pixels + Logistic Regression", "image", true_labels, pred)
    add_real_scores(row, validation, pred)
    results.append(row)
    predictions.append(prediction_rows("Image pixels + Logistic Regression", validation, pred))

    vocabulary = build_vocabulary(train["description"])
    train_text = encode_text(train["description"], vocabulary)
    val_text = encode_text(validation["description"], vocabulary)
    train_images = load_images(train)
    val_images = load_images(validation)
    y_train = encode_labels(train["label"])
    y_val = encode_labels(validation["label"])
    index_to_label = np.asarray(LABELS)

    jobs = [
        ("Keras text model", "text", build_text_model(keras, len(vocabulary) + 2), train_text, val_text),
        ("Keras image model", "image", build_image_model(keras), train_images, val_images),
        (
            "Keras multimodal model",
            "image + text",
            build_multimodal_model(keras, len(vocabulary) + 2),
            {"image": train_images, "text": train_text},
            {"image": val_images, "text": val_text},
        ),
    ]
    for name, modality, model, x_train, x_val in jobs:
        pred_index, history = fit_keras_model(keras, model, x_train, y_train, x_val, y_val)
        pred = index_to_label[pred_index]
        row = score(name, modality, true_labels, pred)
        add_real_scores(row, validation, pred)
        results.append(row)
        predictions.append(prediction_rows(name, validation, pred))
        slug = name.lower().replace(" ", "_").replace("+", "")
        history.to_csv(RESULTS_DIR / f"{slug}_training_history.csv", index=False)
        keras.backend.clear_session()

    pd.DataFrame(results).sort_values("real_image_macro_f1", ascending=False).to_csv(
        RESULTS_DIR / "model_comparison.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        RESULTS_DIR / "validation_predictions.csv", index=False
    )
    run_info = {
        "random_state": RANDOM_STATE,
        "test_split_used": False,
        "model_selection_split": "validation",
    }
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
