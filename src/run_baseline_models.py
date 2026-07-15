from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

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

OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "baselines"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "baseline_models_summary.md"

RANDOM_STATE = 42
IMAGE_FEATURE_SIZE = (32, 32)


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

    return dataframe


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_dataframe = load_split(TRAIN_PATH)
    validation_dataframe = load_split(VALIDATION_PATH)

    train_groups = set(train_dataframe["part_group_id"])
    validation_groups = set(
        validation_dataframe["part_group_id"]
    )

    if not train_groups.isdisjoint(validation_groups):
        raise ValueError(
            "Train and validation part groups overlap."
        )

    return train_dataframe, validation_dataframe


def extract_image_features(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    feature_cache: dict[str, np.ndarray] = {}
    feature_rows: list[np.ndarray] = []

    for relative_path in dataframe["image_path"]:
        if relative_path not in feature_cache:
            image_path = PROJECT_ROOT / relative_path

            with Image.open(image_path) as image:
                resized_image = (
                    image
                    .convert("RGB")
                    .resize(
                        IMAGE_FEATURE_SIZE,
                        Image.Resampling.BILINEAR,
                    )
                )

                feature_cache[relative_path] = (
                    np.asarray(
                        resized_image,
                        dtype=np.float32,
                    ).reshape(-1)
                    / 255.0
                )

        feature_rows.append(feature_cache[relative_path])

    return np.vstack(feature_rows)


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


def create_image_baseline() -> LogisticRegression:
    return LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )


def evaluate_predictions(
    model_name: str,
    validation_dataframe: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, object]:
    true_labels = validation_dataframe["label"].to_numpy()

    if len(predictions) != len(true_labels):
        raise ValueError(
            "Prediction count does not match validation rows."
        )

    invalid_predictions = set(predictions) - set(LABELS)

    if invalid_predictions:
        raise ValueError(
            f"Unknown predictions: {sorted(invalid_predictions)}"
        )

    precision, recall, class_f1, support = (
        precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=LABELS,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
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

    return {
        "model": model_name,
        "evaluation_split": "validation",
        "sample_count": int(len(true_labels)),
        "accuracy": float(
            accuracy_score(true_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "true_distribution": {
            label: int(
                np.count_nonzero(true_labels == label)
            )
            for label in LABELS
        },
        "predicted_distribution": {
            label: int(
                np.count_nonzero(predictions == label)
            )
            for label in LABELS
        },
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(LABELS),
    }


def create_prediction_table(
    validation_dataframe: pd.DataFrame,
    predictions: np.ndarray,
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
        columns={"label": "true_label"}
    )

    prediction_table["predicted_label"] = predictions

    prediction_table["is_correct"] = (
        prediction_table["true_label"]
        == prediction_table["predicted_label"]
    )

    return prediction_table


def write_model_outputs(
    model_slug: str,
    metrics: dict[str, object],
    predictions: pd.DataFrame,
) -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        OUTPUT_DIRECTORY
        / f"{model_slug}_validation_metrics.json"
    )

    predictions_path = (
        OUTPUT_DIRECTORY
        / f"{model_slug}_validation_predictions.csv"
    )

    confusion_matrix_path = (
        OUTPUT_DIRECTORY
        / f"{model_slug}_validation_confusion_matrix.csv"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions.to_csv(
        predictions_path,
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
        confusion_matrix_path,
        index=True,
    )


def write_summary(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    results: dict[str, dict[str, object]],
) -> None:
    training_distribution = Counter(
        train_dataframe["label"]
    )

    summary_lines = [
        "# Baseline Models",
        "",
        "The baseline models were trained on the development "
        "training split and evaluated on the validation split.",
        "",
        "The test split was not used.",
        "",
        "## Dataset",
        "",
        f"- Training samples: {len(train_dataframe)}",
        f"- Validation samples: {len(validation_dataframe)}",
        (
            "- Training label distribution: "
            f"{dict(training_distribution)}"
        ),
        "",
        "## Results",
        "",
        "| Model | Accuracy | Macro F1 |",
        "|---|---:|---:|",
    ]

    model_titles = {
        "majority": "Majority baseline",
        "text_tfidf_logistic_regression": (
            "TF-IDF + Logistic Regression"
        ),
        "image_pixels_logistic_regression": (
            "Image pixels + Logistic Regression"
        ),
    }

    for model_slug, metrics in results.items():
        summary_lines.append(
            f"| {model_titles[model_slug]} "
            f"| {metrics['accuracy']:.4f} "
            f"| {metrics['macro_f1']:.4f} |"
        )

    summary_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The majority model provides the minimum reference "
            "performance for the balanced three-class task.",
            "",
            "The text model measures how much label information "
            "can be learned from the description alone.",
            "",
            "The image model measures whether image features alone "
            "can predict the relationship label.",
            "",
            "Because the same image is paired with all three labels, "
            "an image-only model should not reliably solve the task.",
            "",
            "These results are based on a small generated development "
            "dataset and are used only to validate the pipeline.",
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def run_majority_baseline(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    training_features = np.zeros(
        shape=(len(train_dataframe), 1),
        dtype=np.float32,
    )

    validation_features = np.zeros(
        shape=(len(validation_dataframe), 1),
        dtype=np.float32,
    )

    model = DummyClassifier(
        strategy="most_frequent",
    )

    model.fit(
        training_features,
        train_dataframe["label"],
    )

    predictions = model.predict(
        validation_features
    )

    metrics = evaluate_predictions(
        model_name="Majority baseline",
        validation_dataframe=validation_dataframe,
        predictions=predictions,
    )

    prediction_table = create_prediction_table(
        validation_dataframe,
        predictions,
    )

    return metrics, prediction_table


def run_text_baseline(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    model = create_text_baseline()

    model.fit(
        train_dataframe["description"],
        train_dataframe["label"],
    )

    predictions = model.predict(
        validation_dataframe["description"]
    )

    metrics = evaluate_predictions(
        model_name="TF-IDF + Logistic Regression",
        validation_dataframe=validation_dataframe,
        predictions=predictions,
    )

    prediction_table = create_prediction_table(
        validation_dataframe,
        predictions,
    )

    return metrics, prediction_table


def run_image_baseline(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    training_features = extract_image_features(
        train_dataframe
    )

    validation_features = extract_image_features(
        validation_dataframe
    )

    model = create_image_baseline()

    model.fit(
        training_features,
        train_dataframe["label"],
    )

    predictions = model.predict(
        validation_features
    )

    metrics = evaluate_predictions(
        model_name="Image pixels + Logistic Regression",
        validation_dataframe=validation_dataframe,
        predictions=predictions,
    )

    prediction_table = create_prediction_table(
        validation_dataframe,
        predictions,
    )

    return metrics, prediction_table


def main() -> None:
    (
        train_dataframe,
        validation_dataframe,
    ) = load_datasets()

    baseline_runners = {
        "majority": run_majority_baseline,
        "text_tfidf_logistic_regression": run_text_baseline,
        "image_pixels_logistic_regression": run_image_baseline,
    }

    results: dict[str, dict[str, object]] = {}

    for model_slug, baseline_runner in baseline_runners.items():
        metrics, prediction_table = baseline_runner(
            train_dataframe,
            validation_dataframe,
        )

        results[model_slug] = metrics

        write_model_outputs(
            model_slug=model_slug,
            metrics=metrics,
            predictions=prediction_table,
        )

        print(
            f"{metrics['model']}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    write_summary(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        results=results,
    )

    print(f"Summary: {SUMMARY_PATH}")
    print("Test split used: no")


if __name__ == "__main__":
    main()