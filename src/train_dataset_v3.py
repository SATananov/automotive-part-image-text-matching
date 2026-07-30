from __future__ import annotations

import copy
import json
import platform
import shutil
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from src.data import LABELS, PROJECT_ROOT, encode_labels
from src.data_v3 import CATEGORIES, check_v3_split_overlap, load_v3_split
from src.evaluation import grouped_paired_randomization
from src.models import ImageRelationCNN, MultimodalRelationCNN, TextRelationMLP
from src.train import (
    AUXILIARY_LOSS_WEIGHT,
    CANONICAL_TORCH_VERSION,
    RANDOM_STATE,
    fit_multimodal_model,
    fit_single_input_model,
    prediction_rows,
    require_canonical_torch_version,
    save_model_summary,
    score_model,
    set_seed,
    simple_image_features,
    torch_images,
)

RESULTS_ROOT = PROJECT_ROOT / "results"
DATASET_V3_RESULTS_DIR = RESULTS_ROOT / "dataset_v3"
DATASET_V3_TEMP_RESULTS_DIR = RESULTS_ROOT / ".dataset_v3_training_tmp"
DATASET_VERSION = "3.0-development"
MAIN_MODEL_SLUG = "torch_multimodal_dataset_v3"
EXPECTED_TEST_LOCK_SHA256 = (
    "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
)
MODEL_SLUGS = (
    "majority",
    "tfidf_logistic_regression",
    "image_logistic_regression",
    "image_text_logistic_regression",
    "torch_text_dataset_v3",
    "torch_cnn_image_dataset_v3",
    MAIN_MODEL_SLUG,
)


def assert_development_only(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Reject overlap, non-V3 sources, and any path that enters the locked area."""
    overlap = check_v3_split_overlap(train, validation)
    if any(overlap.values()):
        raise ValueError(f"Dataset V3 development identity overlap: {overlap}")

    for split, data, expected_rows, expected_images in (
        ("train", train, 2880, 480),
        ("validation", validation, 480, 80),
    ):
        if len(data) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} Dataset V3 {split} rows, found {len(data)}."
            )
        if data["image_id"].nunique() != expected_images:
            raise ValueError(
                f"Expected {expected_images} Dataset V3 {split} images."
            )
        if set(data["source"].astype(str)) != {"dataset_v3"}:
            raise ValueError(f"Unexpected source in Dataset V3 {split}.")
        normalized = data["image_path"].astype(str).str.replace("\\", "/", regex=False)
        if normalized.str.startswith("data/locked_test/").any():
            raise ValueError(f"Locked-test path found in Dataset V3 {split}.")
        if normalized.str.contains("/test/", case=False, regex=False).any():
            raise ValueError(f"Test path found in Dataset V3 {split}.")

    if set(train["description"]) & set(validation["description"]):
        raise ValueError("Exact descriptions overlap across Dataset V3 development splits.")


def prepare_result_directory() -> Path:
    """Create an isolated temporary directory for an atomic first training run."""
    if DATASET_V3_RESULTS_DIR.exists():
        raise FileExistsError(
            "results/dataset_v3 already exists. Preserve it and verify before any rerun."
        )
    if DATASET_V3_TEMP_RESULTS_DIR.exists():
        shutil.rmtree(DATASET_V3_TEMP_RESULTS_DIR)
    DATASET_V3_TEMP_RESULTS_DIR.mkdir(parents=True)
    (DATASET_V3_TEMP_RESULTS_DIR / "models").mkdir()
    return DATASET_V3_TEMP_RESULTS_DIR


def save_model_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    model_slug: str,
    text_dimension: int | None = None,
    categories: tuple[str, ...] = CATEGORIES,
) -> None:
    payload = {
        "model_slug": model_slug,
        "dataset_version": DATASET_VERSION,
        "labels": list(LABELS),
        "categories": list(categories),
        "text_dimension": text_dimension,
        "state_dict": copy.deepcopy(model.state_dict()),
    }
    torch.save(payload, path)


def write_environment_lock(directory: Path) -> dict[str, object]:
    distributions = (
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "pillow",
        "jupyter",
        "nbformat",
        "nbclient",
        "pytest",
    )
    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"

    python_version = platform.python_version()
    text = "\n".join(
        [f"python=={python_version}"]
        + [f"{name}=={version}" for name, version in sorted(versions.items())]
    ) + "\n"
    text_path = directory / "environment_lock.txt"
    text_path.write_text(text, encoding="utf-8", newline="\n")

    import hashlib

    summary = {
        "python_version": python_version,
        "packages": versions,
        "text_lock_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
    }
    (directory / "environment_lock.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def save_main_model_tables(
    all_predictions: pd.DataFrame,
    directory: Path,
) -> None:
    main = all_predictions[
        all_predictions["model_slug"].eq(MAIN_MODEL_SLUG)
    ].copy()
    if main.empty:
        raise ValueError("Dataset V3 main multimodal predictions are missing.")
    main.to_csv(
        directory / "multimodal_validation_predictions.csv",
        index=False,
        lineterminator="\n",
    )

    rows: list[dict[str, object]] = []
    for category, group in main.groupby("part_category", sort=True):
        rows.append(
            {
                "part_category": category,
                "samples": len(group),
                "independent_images": int(group["image_id"].nunique()),
                "correct": int(group["is_correct"].sum()),
                "accuracy": float(
                    accuracy_score(group["true_label"], group["predicted_label"])
                ),
                "macro_f1": float(
                    f1_score(
                        group["true_label"],
                        group["predicted_label"],
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        directory / "multimodal_per_category.csv",
        index=False,
        lineterminator="\n",
    )


def build_training_summary(
    result_table: pd.DataFrame,
    paired: pd.DataFrame,
    environment: dict[str, object],
) -> dict[str, object]:
    indexed = result_table.set_index("model_slug")
    main = indexed.loc[MAIN_MODEL_SLUG]
    baseline = indexed.loc["image_text_logistic_regression"]
    paired_row = paired[
        paired["right_model_slug"].eq("image_text_logistic_regression")
    ].iloc[0]
    return {
        "status": "PASS_DATASET_V3_DEVELOPMENT_TRAINING_ARTIFACTS_READY",
        "dataset_version": DATASET_VERSION,
        "models": len(result_table),
        "main_model_slug": MAIN_MODEL_SLUG,
        "main_validation_accuracy": float(main["validation_accuracy"]),
        "main_validation_macro_f1": float(main["validation_macro_f1"]),
        "main_correct_predictions": int(main["correct_predictions"]),
        "main_total_predictions": int(main["total_predictions"]),
        "main_accuracy_ci": [
            float(main["accuracy_ci_low"]),
            float(main["accuracy_ci_high"]),
        ],
        "baseline_validation_accuracy": float(baseline["validation_accuracy"]),
        "baseline_validation_macro_f1": float(baseline["validation_macro_f1"]),
        "paired_comparison_method": str(paired_row["method"]),
        "paired_independent_groups": int(paired_row["independent_groups"]),
        "paired_two_sided_p_value": float(
            paired_row["grouped_two_sided_p_value"]
        ),
        "training_rows": 2880,
        "training_images": 480,
        "validation_rows": 480,
        "validation_images": 80,
        "categories": list(CATEGORIES),
        "test_lock_status": "LOCKED_NOT_AUTHORIZED_FOR_TRAINING_OR_SELECTION",
        "test_lock_sha256": EXPECTED_TEST_LOCK_SHA256,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "test_evaluation_executed": False,
        "environment_lock_sha256": environment["text_lock_sha256"],
    }


def main() -> None:
    require_canonical_torch_version()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    set_seed(RANDOM_STATE)

    directory = prepare_result_directory()
    try:
        train = load_v3_split("train")
        validation = load_v3_split("validation")
        assert_development_only(train, validation)

        results: list[dict[str, object]] = []
        predictions: list[pd.DataFrame] = []

        def record(
            name: str,
            slug: str,
            modality: str,
            training_data: str,
            predicted: np.ndarray,
            image_category_accuracy: float | None = None,
            text_category_accuracy: float | None = None,
        ) -> None:
            predicted = np.asarray(predicted)
            results.append(
                score_model(
                    name,
                    slug,
                    modality,
                    training_data,
                    validation,
                    predicted,
                    image_category_accuracy,
                    text_category_accuracy,
                )
            )
            predictions.append(
                prediction_rows(name, slug, validation, predicted)
            )

        majority = DummyClassifier(strategy="most_frequent")
        majority.fit(np.zeros((len(train), 1)), train["label"])
        record(
            "Majority baseline",
            "majority",
            "none",
            "Dataset V3 train relations",
            majority.predict(np.zeros((len(validation), 1))),
        )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )
        train_tfidf_sparse = vectorizer.fit_transform(train["description"])
        validation_tfidf_sparse = vectorizer.transform(validation["description"])

        text_baseline = LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        text_baseline.fit(train_tfidf_sparse, train["label"])
        record(
            "TF-IDF + Logistic Regression",
            "tfidf_logistic_regression",
            "text",
            "Dataset V3 train relations",
            text_baseline.predict(validation_tfidf_sparse),
        )

        flat_train = simple_image_features(train)
        flat_validation = simple_image_features(validation)
        image_baseline = LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        image_baseline.fit(flat_train, train["label"])
        record(
            "Image pixels + Logistic Regression",
            "image_logistic_regression",
            "image",
            "Dataset V3 train relations",
            image_baseline.predict(flat_validation),
        )

        combined_train = sparse.hstack(
            [sparse.csr_matrix(flat_train), train_tfidf_sparse],
            format="csr",
        )
        combined_validation = sparse.hstack(
            [sparse.csr_matrix(flat_validation), validation_tfidf_sparse],
            format="csr",
        )
        combined_baseline = LogisticRegression(
            max_iter=4000,
            random_state=RANDOM_STATE,
        )
        combined_baseline.fit(combined_train, train["label"])
        record(
            "Image + text Logistic Regression",
            "image_text_logistic_regression",
            "image + text",
            "Dataset V3 train relations",
            combined_baseline.predict(combined_validation),
        )

        train_text = train_tfidf_sparse.toarray().astype(np.float32)
        validation_text = validation_tfidf_sparse.toarray().astype(np.float32)
        train_images = torch_images(train)
        validation_images = torch_images(validation)
        y_train = encode_labels(train["label"])
        y_validation = encode_labels(validation["label"])
        index_to_label = np.asarray(LABELS)
        text_dimension = train_text.shape[1]

        category_names = tuple(sorted(CATEGORIES))
        category_to_index = {
            category: index for index, category in enumerate(category_names)
        }
        train_image_categories = train["part_category"].map(
            category_to_index
        ).to_numpy(np.int64)
        train_text_categories = train["text_category"].map(
            category_to_index
        ).to_numpy(np.int64)
        validation_image_categories = validation["part_category"].map(
            category_to_index
        ).to_numpy(np.int64)
        validation_text_categories = validation["text_category"].map(
            category_to_index
        ).to_numpy(np.int64)

        if any(
            np.isnan(values).any()
            for values in (
                train_image_categories,
                train_text_categories,
                validation_image_categories,
                validation_text_categories,
            )
        ):
            raise ValueError("Dataset V3 category encoding failed.")

        set_seed(RANDOM_STATE)
        text_model = TextRelationMLP(text_dimension)
        text_slug = "torch_text_dataset_v3"
        save_model_summary(text_model, directory / f"{text_slug}_architecture.txt")
        predicted_index, history = fit_single_input_model(
            text_model,
            train_text,
            y_train,
            validation_text,
            y_validation,
            augment_image_batches=False,
        )
        history.to_csv(
            directory / f"{text_slug}_training_history.csv",
            index=False,
            lineterminator="\n",
        )
        save_model_checkpoint(
            text_model,
            directory / "models" / f"{text_slug}_state.pt",
            model_slug=text_slug,
            text_dimension=text_dimension,
            categories=category_names,
        )
        record(
            "PyTorch neural text model",
            text_slug,
            "text",
            "Dataset V3 train relations",
            index_to_label[predicted_index],
        )

        set_seed(RANDOM_STATE + 1)
        image_model = ImageRelationCNN()
        image_slug = "torch_cnn_image_dataset_v3"
        save_model_summary(image_model, directory / f"{image_slug}_architecture.txt")
        predicted_index, history = fit_single_input_model(
            image_model,
            train_images,
            y_train,
            validation_images,
            y_validation,
            augment_image_batches=True,
        )
        history.to_csv(
            directory / f"{image_slug}_training_history.csv",
            index=False,
            lineterminator="\n",
        )
        save_model_checkpoint(
            image_model,
            directory / "models" / f"{image_slug}_state.pt",
            model_slug=image_slug,
            categories=category_names,
        )
        record(
            "PyTorch CNN image model",
            image_slug,
            "image",
            "Dataset V3 train relations",
            index_to_label[predicted_index],
        )

        set_seed(RANDOM_STATE + 2)
        multimodal_model = MultimodalRelationCNN(
            text_dimension,
            number_of_part_categories=len(category_names),
        )
        save_model_summary(
            multimodal_model,
            directory / f"{MAIN_MODEL_SLUG}_architecture.txt",
        )
        predicted_index, history, auxiliary_metrics = fit_multimodal_model(
            multimodal_model,
            train_images,
            train_text,
            y_train,
            train_image_categories,
            train_text_categories,
            validation_images,
            validation_text,
            y_validation,
            validation_image_categories,
            validation_text_categories,
        )
        history.to_csv(
            directory / f"{MAIN_MODEL_SLUG}_training_history.csv",
            index=False,
            lineterminator="\n",
        )
        save_model_checkpoint(
            multimodal_model,
            directory / "models" / f"{MAIN_MODEL_SLUG}_state.pt",
            model_slug=MAIN_MODEL_SLUG,
            text_dimension=text_dimension,
            categories=category_names,
        )
        record(
            "PyTorch multimodal CNN (Dataset V3)",
            MAIN_MODEL_SLUG,
            "image + text",
            "480 curated Dataset V3 train images",
            index_to_label[predicted_index],
            auxiliary_metrics["image_category_accuracy"],
            auxiliary_metrics["text_category_accuracy"],
        )

        result_table = pd.DataFrame(results).sort_values(
            ["validation_macro_f1", "validation_accuracy", "model"],
            ascending=[False, False, True],
            kind="stable",
        ).reset_index(drop=True)
        if tuple(sorted(result_table["model_slug"])) != tuple(sorted(MODEL_SLUGS)):
            raise ValueError("Unexpected Dataset V3 model inventory.")

        all_predictions = pd.concat(predictions, ignore_index=True)
        result_table.to_csv(
            directory / "model_comparison.csv",
            index=False,
            lineterminator="\n",
        )
        all_predictions.to_csv(
            directory / "validation_predictions.csv",
            index=False,
            lineterminator="\n",
        )
        save_main_model_tables(all_predictions, directory)

        comparisons = [
            grouped_paired_randomization(
                all_predictions,
                MAIN_MODEL_SLUG,
                "image_text_logistic_regression",
            ),
            grouped_paired_randomization(
                all_predictions,
                MAIN_MODEL_SLUG,
                "majority",
            ),
        ]
        paired = pd.DataFrame(comparisons)
        paired.to_csv(
            directory / "paired_comparisons.csv",
            index=False,
            lineterminator="\n",
        )

        environment = write_environment_lock(directory)
        run_info = {
            "result_source": (
                "Generated by python -m src.train_dataset_v3 from Dataset V3 "
                "development train and validation relations."
            ),
            "dataset_version": DATASET_VERSION,
            "random_state": RANDOM_STATE,
            "deep_learning_framework": "PyTorch",
            "torch_version": torch.__version__,
            "canonical_torch_version": CANONICAL_TORCH_VERSION,
            "python_version": platform.python_version(),
            "training_split": (
                "data/manifests/dataset_v3/dataset_v3_train_relations.csv"
            ),
            "model_selection_split": (
                "data/manifests/dataset_v3/"
                "dataset_v3_validation_relations.csv "
                "(80 independent curated images)"
            ),
            "training_rows": len(train),
            "training_images": int(train["image_id"].nunique()),
            "validation_rows": len(validation),
            "validation_images": int(validation["image_id"].nunique()),
            "validation_sources": sorted(validation["source"].unique().tolist()),
            "categories": list(category_names),
            "saved_model_count": len(result_table),
            "main_model_slug": MAIN_MODEL_SLUG,
            "auxiliary_loss_weight": AUXILIARY_LOSS_WEIGHT,
            "test_lock_status": (
                "LOCKED_NOT_AUTHORIZED_FOR_TRAINING_OR_SELECTION"
            ),
            "test_lock_sha256": EXPECTED_TEST_LOCK_SHA256,
            "test_manifest_read": False,
            "test_images_read": False,
            "test_split_used": False,
            "test_evaluation_permitted": False,
            "test_evaluation_executed": False,
        }
        (directory / "run_info.json").write_text(
            json.dumps(run_info, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        training_summary = build_training_summary(
            result_table,
            paired,
            environment,
        )
        (directory / "training_summary.json").write_text(
            json.dumps(training_summary, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        from src.verify_dataset_v3_training import verify_training_artifacts

        verification = verify_training_artifacts(directory)
        (directory / "verification_summary.json").write_text(
            json.dumps(verification, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        directory.replace(DATASET_V3_RESULTS_DIR)
        print(result_table.to_string(index=False))
        print()
        print(json.dumps(training_summary, indent=2))
    except Exception:
        if DATASET_V3_TEMP_RESULTS_DIR.exists():
            shutil.rmtree(DATASET_V3_TEMP_RESULTS_DIR)
        raise


if __name__ == "__main__":
    main()
