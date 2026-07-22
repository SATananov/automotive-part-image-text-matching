from __future__ import annotations

from pathlib import Path

from src.external_dataset_integration_config import (
    EXTERNAL_TEST_PATH,
    INTEGRATED_TEST_LOCK_PATH,
    INTEGRATED_TEST_PATH,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
)
from src.real_dataset_config import PROJECT_ROOT

INTEGRATED_TRAINING_ROOT = (
    PROJECT_ROOT / "reports" / "integrated_training"
)

INTEGRATED_COMPARISON_CSV_PATH = (
    INTEGRATED_TRAINING_ROOT / "validation_comparison.csv"
)
INTEGRATED_COMPARISON_JSON_PATH = (
    INTEGRATED_TRAINING_ROOT / "validation_comparison.json"
)
INTEGRATED_SUMMARY_PATH = (
    INTEGRATED_TRAINING_ROOT
    / "integrated_training_validation_summary.md"
)
INTEGRATED_RUN_STATUS_PATH = (
    INTEGRATED_TRAINING_ROOT / "integrated_training_run_status.json"
)

MODEL_TITLES = {
    "majority": "Majority baseline",
    "text_tfidf_logistic_regression": (
        "TF-IDF + Logistic Regression"
    ),
    "image_pixels_logistic_regression": (
        "Image pixels + Logistic Regression"
    ),
    "keras_text": "Keras Text Neural Network",
    "keras_image": "Keras Image Neural Network",
    "keras_multimodal": "Keras Multimodal Neural Network",
}

MODEL_MODALITIES = {
    "majority": "none",
    "text_tfidf_logistic_regression": "text",
    "image_pixels_logistic_regression": "image",
    "keras_text": "text",
    "keras_image": "image",
    "keras_multimodal": "image_and_text",
}

INTEGRATED_MODEL_DIRECTORIES = {
    model_slug: INTEGRATED_TRAINING_ROOT / model_slug
    for model_slug in MODEL_TITLES
}

INTEGRATED_METRIC_PATHS = {
    model_slug: directory / "validation_metrics.json"
    for model_slug, directory in INTEGRATED_MODEL_DIRECTORIES.items()
}

INTEGRATED_PREDICTION_PATHS = {
    model_slug: directory / "validation_predictions.csv"
    for model_slug, directory in INTEGRATED_MODEL_DIRECTORIES.items()
}

INTEGRATED_CONFUSION_MATRIX_PATHS = {
    model_slug: directory / "validation_confusion_matrix.csv"
    for model_slug, directory in INTEGRATED_MODEL_DIRECTORIES.items()
}

INTEGRATED_HISTORY_PATHS = {
    model_slug: INTEGRATED_MODEL_DIRECTORIES[model_slug]
    / "training_history.csv"
    for model_slug in (
        "keras_text",
        "keras_image",
        "keras_multimodal",
    )
}

INTEGRATED_ARCHITECTURE_PATHS = {
    model_slug: INTEGRATED_MODEL_DIRECTORIES[model_slug]
    / "model_architecture.txt"
    for model_slug in (
        "keras_text",
        "keras_image",
        "keras_multimodal",
    )
}

DEVELOPMENT_METRIC_PATHS = {
    "majority": (
        PROJECT_ROOT
        / "reports"
        / "baselines"
        / "majority_validation_metrics.json"
    ),
    "text_tfidf_logistic_regression": (
        PROJECT_ROOT
        / "reports"
        / "baselines"
        / "text_tfidf_logistic_regression_validation_metrics.json"
    ),
    "image_pixels_logistic_regression": (
        PROJECT_ROOT
        / "reports"
        / "baselines"
        / "image_pixels_logistic_regression_validation_metrics.json"
    ),
    "keras_text": (
        PROJECT_ROOT
        / "reports"
        / "keras_text"
        / "validation_metrics.json"
    ),
    "keras_image": (
        PROJECT_ROOT
        / "reports"
        / "keras_image"
        / "validation_metrics.json"
    ),
    "keras_multimodal": (
        PROJECT_ROOT
        / "reports"
        / "keras_multimodal"
        / "validation_metrics.json"
    ),
}

LOCKED_TEST_PATHS = (
    EXTERNAL_TEST_PATH,
    INTEGRATED_TEST_PATH,
)

RANDOM_STATE = 42
TEXT_SEQUENCE_LENGTH = 12
TEXT_EMBEDDING_DIMENSION = 16
IMAGE_HEIGHT = 24
IMAGE_WIDTH = 24
CLASSICAL_IMAGE_HEIGHT = 32
CLASSICAL_IMAGE_WIDTH = 32
BATCH_SIZE = 32
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
