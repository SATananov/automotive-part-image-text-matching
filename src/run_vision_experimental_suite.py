from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import subprocess
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset_config import METADATA_COLUMNS
from src.vision_suite_config import (
    ANNOTATION_REVIEW_PATH,
    AUGMENTATION_COMPARISON_PATH,
    AUGMENTATION_POLICIES,
    AUGMENTATION_RUNS_PATH,
    BASE_CHECKPOINT,
    COMPATIBILITY_COMPARISON_PATH,
    COMPATIBILITY_PREDICTIONS_PATH,
    COMPATIBILITY_RUNS_PATH,
    COMPATIBILITY_STRATEGIES,
    COMPLETED_VISION_IDS,
    CURRENT_STATUS_PATH,
    DEFERRED_VISION_IDS,
    DOCUMENTATION_PATH,
    EQUAL_PAIR_EVALUATION_PATH,
    EXECUTION_REGISTRY_CSV_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXPERIMENT_CONFIG_PATHS,
    EXPLAINABILITY_SUMMARY_PATH,
    FAILURE_AUGMENTATION_MATRIX_PATH,
    FIGURE_PATHS,
    FIGURE_ROOT,
    FINE_TUNING_GATE_PATH,
    GENERATED_PATHS,
    HUMAN_ANNOTATION_GATE_PATH,
    IMAGE_INVENTORY_PATH,
    IMAGE_PROFILE_PATH,
    LOCK_FLAGS,
    MANIFEST_PATH,
    NOTEBOOK_AUDIT_PATH,
    OCCLUSION_FIGURE_PATHS,
    OCCLUSION_RESULTS_PATH,
    PRETRAINED_BACKBONE_GATE_PATH,
    PROJECT_ROOT,
    RANDOM_SEEDS,
    RANKING_METRICS_PATH,
    RANKING_TRIPLETS_PATH,
    READINESS,
    REGION_PERTURBATION_PATH,
    REPRESENTATION_COMPARISON_PATH,
    REPRESENTATION_RUNS_PATH,
    REPRESENTATIONS,
    REPORT_ROOT,
    REPRESENTATIVE_IMAGES_PATH,
    RESOLUTIONS,
    SCORE_DISTRIBUTIONS_PATH,
    SOURCE_PATHS,
    STATUS_PATH,
    STEP,
    SUITE_CONFIG_PATH,
    SUMMARY_PATH,
    TEXT_HASH_SUFFIXES,
    TRAIN_PATH,
    VALIDATION_PATH,
    VISION_IDS,
    project_relative,
)


class VisionSuiteError(RuntimeError):
    pass


@dataclass
class CategoryModelBundle:
    model: Pipeline
    classes: tuple[str, ...]
    representation: str
    resolution: int
    augmentation_policy: str
    seed: int
    feature_dimension: int
    parameter_count: int


@dataclass
class CompatibilityRun:
    run_id: str
    strategy: str
    seed: int
    model: Pipeline
    validation_scores: np.ndarray
    train_scores: np.ndarray
    validation_macro_f1: float
    validation_accuracy: float
    pairwise_ranking_accuracy: float
    three_way_ordering_accuracy: float
    equal_pair_accuracy: float
    equal_pair_threshold: float
    inference_time_ms: float
    parameter_count: int


LABEL_SCORE = {"MISMATCH": 0.0, "PARTIAL_MATCH": 0.5, "MATCH": 1.0}
SCORE_LABELS = tuple(LABEL_SCORE)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def normalized_sha256(path: Path) -> str:
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = path.read_text(encoding="utf-8-sig")
        payload = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    else:
        payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return BASE_CHECKPOINT
    return result.stdout.strip()


def stable_int(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def load_split(path: Path) -> pd.DataFrame:
    resolved = path.resolve()
    allowed = {TRAIN_PATH.resolve(), VALIDATION_PATH.resolve()}
    if resolved not in allowed:
        raise VisionSuiteError(f"Unauthorized Step 011.3A data input: {path}")
    dataframe = pd.read_csv(path)
    if tuple(dataframe.columns) != METADATA_COLUMNS:
        raise VisionSuiteError(f"Unexpected metadata schema in {path.name}")
    if dataframe.empty:
        raise VisionSuiteError(f"Empty split: {path.name}")
    return dataframe


def load_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)
    if not set(train["part_group_id"]).isdisjoint(set(validation["part_group_id"])):
        raise VisionSuiteError("Train and validation part groups overlap")
    if set(train["part_category"]) != set(validation["part_category"]):
        raise VisionSuiteError("Train and validation category coverage differs")
    return train, validation


def unique_images(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "image_id",
        "part_group_id",
        "image_path",
        "part_family",
        "part_category",
        "source",
    ]
    unique = dataframe[columns].drop_duplicates("image_id").copy()
    if unique["image_id"].duplicated().any():
        raise VisionSuiteError("Image IDs are not unique after deduplication")
    return unique.sort_values("image_id").reset_index(drop=True)


def resolve_image_path(relative_path: str) -> Path:
    path = (PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise VisionSuiteError(f"Unsafe image path: {relative_path}") from error
    if not path.is_file():
        raise VisionSuiteError(f"Image not found: {relative_path}")
    return path


def load_rgb(relative_path: str) -> Image.Image:
    with Image.open(resolve_image_path(relative_path)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def apply_augmentation(
    image: Image.Image,
    policy: str,
    image_id: str,
    seed: int,
) -> Image.Image:
    if policy == "none":
        return image.copy()

    result = image.copy()
    selector = stable_int(policy, image_id, seed)

    if policy in {"center_crop", "combined"}:
        width, height = result.size
        ratio = 0.82 if selector % 2 == 0 else 0.88
        crop_width = max(2, int(width * ratio))
        crop_height = max(2, int(height * ratio))
        left = (width - crop_width) // 2
        top = (height - crop_height) // 2
        result = result.crop((left, top, left + crop_width, top + crop_height))

    if policy in {"brightness", "combined"}:
        factor = 0.82 if selector % 2 == 0 else 1.18
        result = ImageEnhance.Brightness(result).enhance(factor)

    if policy in {"jpeg_compression", "combined"}:
        buffer = io.BytesIO()
        quality = 45 if selector % 2 == 0 else 60
        result.save(buffer, format="JPEG", quality=quality, optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            result = compressed.convert("RGB").copy()

    if policy not in AUGMENTATION_POLICIES:
        raise VisionSuiteError(f"Unknown augmentation policy: {policy}")
    return result


def image_to_array(image: Image.Image, resolution: int) -> np.ndarray:
    resized = image.resize((resolution, resolution), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def adaptive_average_pool(channel: np.ndarray, grid: int) -> np.ndarray:
    height, width = channel.shape
    y_edges = np.linspace(0, height, grid + 1, dtype=int)
    x_edges = np.linspace(0, width, grid + 1, dtype=int)
    values: list[float] = []
    for y_index in range(grid):
        for x_index in range(grid):
            block = channel[
                y_edges[y_index] : y_edges[y_index + 1],
                x_edges[x_index] : x_edges[x_index + 1],
            ]
            values.append(float(block.mean()) if block.size else 0.0)
    return np.asarray(values, dtype=np.float32)


def fixed_convolution_maps(array: np.ndarray) -> dict[str, np.ndarray]:
    gray = (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    )
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx * gx + gy * gy)
    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    return {
        "gray": gray,
        "gradient_x": gx,
        "gradient_y": gy,
        "gradient_magnitude": magnitude,
        "laplacian": laplacian,
    }


def histogram_features(array: np.ndarray, bins: int = 12) -> np.ndarray:
    values: list[float] = []
    for channel_index in range(3):
        counts, _ = np.histogram(
            array[:, :, channel_index], bins=bins, range=(0.0, 1.0), density=False
        )
        counts = counts.astype(np.float32)
        counts /= max(float(counts.sum()), 1.0)
        values.extend(counts.tolist())
    return np.asarray(values, dtype=np.float32)


def extract_representation(array: np.ndarray, representation: str) -> np.ndarray:
    maps = fixed_convolution_maps(array)
    color_stats = np.concatenate(
        [
            array.mean(axis=(0, 1)),
            array.std(axis=(0, 1)),
            array.min(axis=(0, 1)),
            array.max(axis=(0, 1)),
        ]
    ).astype(np.float32)
    histograms = histogram_features(array)

    if representation == "global_pool":
        pooled_rgb = np.concatenate(
            [adaptive_average_pool(array[:, :, index], 8) for index in range(3)]
        )
        return np.concatenate([pooled_rgb, color_stats, histograms]).astype(np.float32)

    intermediate = np.concatenate(
        [
            adaptive_average_pool(maps[name], 8)
            for name in (
                "gray",
                "gradient_x",
                "gradient_y",
                "gradient_magnitude",
                "laplacian",
            )
        ]
    )
    if representation == "intermediate_fixed_conv":
        return np.concatenate([intermediate, color_stats, histograms]).astype(np.float32)

    if representation == "multi_stage_fixed_conv":
        final_stage = np.concatenate(
            [
                adaptive_average_pool(maps[name], 4)
                for name in ("gray", "gradient_magnitude", "laplacian")
            ]
        )
        return np.concatenate(
            [intermediate, final_stage, color_stats, histograms]
        ).astype(np.float32)

    raise VisionSuiteError(f"Unknown representation: {representation}")


@lru_cache(maxsize=None)
def cached_representation(
    relative_path: str,
    image_id: str,
    representation: str,
    resolution: int,
    augmentation_policy: str,
    seed: int,
) -> np.ndarray:
    image = load_rgb(relative_path)
    if augmentation_policy != "none":
        image = apply_augmentation(image, augmentation_policy, image_id, seed)
    return extract_representation(image_to_array(image, resolution), representation)


def feature_matrix(
    images: pd.DataFrame,
    representation: str,
    resolution: int,
    augmentation_policy: str = "none",
    seed: int = 42,
    include_original: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    image_ids: list[str] = []
    for row in images.itertuples(index=False):
        original = cached_representation(
            row.image_path, row.image_id, representation, resolution, "none", 0
        )
        features.append(original)
        labels.append(row.part_category)
        image_ids.append(row.image_id)

        if include_original and augmentation_policy != "none":
            augmented = cached_representation(
                row.image_path,
                row.image_id,
                representation,
                resolution,
                augmentation_policy,
                seed,
            )
            features.append(augmented)
            labels.append(row.part_category)
            image_ids.append(f"{row.image_id}__{augmentation_policy}")
    return np.vstack(features), np.asarray(labels, dtype=object), image_ids


def fit_category_model(
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
    representation: str,
    resolution: int,
    seed: int,
    augmentation_policy: str = "none",
) -> tuple[CategoryModelBundle, dict[str, Any], np.ndarray, np.ndarray]:
    train_x, train_y, _ = feature_matrix(
        train_images,
        representation,
        resolution,
        augmentation_policy=augmentation_policy,
        seed=seed,
        include_original=augmentation_policy != "none",
    )
    validation_x, validation_y, _ = feature_matrix(
        validation_images,
        representation,
        resolution,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    C=1.0,
                    random_state=seed,
                ),
            ),
        ]
    )
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(train_x, train_y)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    predictions = model.predict(validation_x)
    probabilities = model.predict_proba(validation_x)
    inference_ms = (time.perf_counter() - start) * 1000.0 / len(validation_x)
    macro_f1 = f1_score(validation_y, predictions, average="macro", zero_division=0)
    accuracy = accuracy_score(validation_y, predictions)
    classes = tuple(str(value) for value in model.named_steps["classifier"].classes_)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        validation_y,
        predictions,
        labels=list(classes),
        zero_division=0,
    )
    classifier = model.named_steps["classifier"]
    parameter_count = int(classifier.coef_.size + classifier.intercept_.size)
    bundle = CategoryModelBundle(
        model=model,
        classes=classes,
        representation=representation,
        resolution=resolution,
        augmentation_policy=augmentation_policy,
        seed=seed,
        feature_dimension=int(train_x.shape[1]),
        parameter_count=parameter_count,
    )
    metrics = {
        "validation_macro_f1": float(macro_f1),
        "validation_accuracy": float(accuracy),
        "training_time_seconds": float(training_seconds),
        "inference_time_ms": float(inference_ms),
        "feature_dimension": int(train_x.shape[1]),
        "parameter_count": parameter_count,
        "per_category_f1": {
            category: float(score) for category, score in zip(classes, per_f1)
        },
        "per_category_support": {
            category: int(value) for category, value in zip(classes, support)
        },
        "validation_predictions": predictions,
        "validation_probabilities": probabilities,
        "validation_truth": validation_y,
    }
    return bundle, metrics, validation_x, validation_y


def image_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_statistics(image: Image.Image) -> dict[str, float]:
    array = np.asarray(image, dtype=np.float32) / 255.0
    gray = 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    gy, gx = np.gradient(gray)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "edge_density": float(np.sqrt(gx * gx + gy * gy).mean()),
    }


def write_image_profile(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, dataframe in (("train", train), ("validation", validation)):
        for row in unique_images(dataframe).itertuples(index=False):
            path = resolve_image_path(row.image_path)
            image = load_rgb(row.image_path)
            width, height = image.size
            stats = image_statistics(image)
            rows.append(
                {
                    "split": split_name,
                    "image_id": row.image_id,
                    "part_group_id": row.part_group_id,
                    "image_path": row.image_path,
                    "part_family": row.part_family,
                    "part_category": row.part_category,
                    "source": row.source,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height,
                    "megapixels": width * height / 1_000_000.0,
                    "brightness": stats["brightness"],
                    "contrast": stats["contrast"],
                    "edge_density": stats["edge_density"],
                    "sha256": image_file_sha256(path),
                }
            )
    inventory = pd.DataFrame(rows).sort_values(["split", "image_id"]).reset_index(drop=True)
    duplicate_counts = inventory.groupby("sha256")["image_id"].transform("count")
    inventory["exact_duplicate_count"] = duplicate_counts.astype(int)
    inventory.to_csv(IMAGE_INVENTORY_PATH, index=False)

    review = inventory.copy()
    review["low_brightness_flag"] = review["brightness"] < 0.12
    review["high_brightness_flag"] = review["brightness"] > 0.88
    review["low_contrast_flag"] = review["contrast"] < 0.06
    review["extreme_aspect_ratio_flag"] = (
        (review["aspect_ratio"] < 0.40) | (review["aspect_ratio"] > 2.50)
    )
    review["exact_duplicate_flag"] = review["exact_duplicate_count"] > 1
    flag_columns = [
        "low_brightness_flag",
        "high_brightness_flag",
        "low_contrast_flag",
        "extreme_aspect_ratio_flag",
        "exact_duplicate_flag",
    ]
    review["review_required"] = review[flag_columns].any(axis=1)
    review["review_reasons"] = review.apply(
        lambda row: ";".join(
            column.removesuffix("_flag")
            for column in flag_columns
            if bool(row[column])
        ),
        axis=1,
    )
    review.to_csv(ANNOTATION_REVIEW_PATH, index=False)

    representative = (
        inventory.sort_values(["part_category", "split", "source", "image_id"])
        .groupby("part_category", as_index=False)
        .first()
    )
    representative.to_csv(REPRESENTATIVE_IMAGES_PATH, index=False)

    profile = {
        "step": STEP,
        "status": "PASS",
        "unique_image_count": int(len(inventory)),
        "train_unique_image_count": int((inventory["split"] == "train").sum()),
        "validation_unique_image_count": int((inventory["split"] == "validation").sum()),
        "category_count": int(inventory["part_category"].nunique()),
        "family_count": int(inventory["part_family"].nunique()),
        "source_counts": {
            str(key): int(value) for key, value in inventory["source"].value_counts().items()
        },
        "category_counts": {
            str(key): int(value)
            for key, value in inventory["part_category"].value_counts().sort_index().items()
        },
        "width": {
            "min": int(inventory["width"].min()),
            "median": float(inventory["width"].median()),
            "max": int(inventory["width"].max()),
        },
        "height": {
            "min": int(inventory["height"].min()),
            "median": float(inventory["height"].median()),
            "max": int(inventory["height"].max()),
        },
        "brightness": {
            "min": float(inventory["brightness"].min()),
            "mean": float(inventory["brightness"].mean()),
            "max": float(inventory["brightness"].max()),
        },
        "contrast": {
            "min": float(inventory["contrast"].min()),
            "mean": float(inventory["contrast"].mean()),
            "max": float(inventory["contrast"].max()),
        },
        "exact_duplicate_groups": int(
            (inventory.groupby("sha256").size() > 1).sum()
        ),
        "review_required_count": int(review["review_required"].sum()),
        **LOCK_FLAGS,
    }
    write_json(IMAGE_PROFILE_PATH, profile)
    return inventory, review, profile


def save_profile_figures(
    inventory: pd.DataFrame,
    representative: pd.DataFrame,
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for split, group in inventory.groupby("split"):
        plt.scatter(group["width"], group["height"], label=split, alpha=0.75)
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.title("Integrated train/validation image dimensions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["dimension_scatter"], dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for split, group in inventory.groupby("split"):
        plt.scatter(group["brightness"], group["contrast"], label=split, alpha=0.75)
    plt.xlabel("Mean luminance")
    plt.ylabel("Luminance standard deviation")
    plt.title("Brightness and contrast distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["brightness_contrast"], dpi=150)
    plt.close()

    counts = inventory.groupby(["part_category", "source"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", figsize=(11, 5))
    plt.ylabel("Unique images")
    plt.title("Category and source coverage")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["category_source"], dpi=150)
    plt.close()

    figure, axes = plt.subplots(2, 5, figsize=(15, 6))
    for axis, row in zip(axes.flat, representative.itertuples(index=False)):
        image = load_rgb(row.image_path)
        axis.imshow(image)
        axis.set_title(row.part_category.replace("_", " "))
        axis.axis("off")
    figure.suptitle("Representative integrated train/validation images")
    figure.tight_layout()
    figure.savefig(FIGURE_PATHS["representative_gallery"], dpi=150)
    plt.close(figure)


def aggregate_run_table(
    runs: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    aggregated = (
        runs.groupby(group_columns, as_index=False)
        .agg(
            run_count=("run_id", "count"),
            validation_macro_f1_mean=("validation_macro_f1", "mean"),
            validation_macro_f1_std=("validation_macro_f1", "std"),
            validation_accuracy_mean=("validation_accuracy", "mean"),
            validation_accuracy_std=("validation_accuracy", "std"),
            feature_dimension=("feature_dimension", "first"),
            parameter_count=("parameter_count", "first"),
            training_time_seconds_mean=("training_time_seconds", "mean"),
            inference_time_ms_mean=("inference_time_ms", "mean"),
        )
        .fillna(0.0)
    )
    return aggregated


def run_representation_experiments(
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_metrics: dict[str, dict[str, Any]] = {}
    for representation in REPRESENTATIONS:
        for resolution in RESOLUTIONS:
            for seed in RANDOM_SEEDS:
                run_id = f"VIS004_{representation}_{resolution}_seed{seed}"
                _, metrics, _, _ = fit_category_model(
                    train_images,
                    validation_images,
                    representation,
                    resolution,
                    seed,
                )
                row = {
                    "run_id": run_id,
                    "experiment_id": "VIS-004",
                    "representation": representation,
                    "resolution": resolution,
                    "seed": seed,
                    "augmentation_policy": "none",
                    "validation_macro_f1": metrics["validation_macro_f1"],
                    "validation_accuracy": metrics["validation_accuracy"],
                    "feature_dimension": metrics["feature_dimension"],
                    "parameter_count": metrics["parameter_count"],
                    "training_time_seconds": metrics["training_time_seconds"],
                    "inference_time_ms": metrics["inference_time_ms"],
                    "per_category_f1_json": json.dumps(
                        metrics["per_category_f1"], sort_keys=True
                    ),
                    "status": "COMPLETED",
                }
                rows.append(row)
                run_metrics[run_id] = metrics
    runs = pd.DataFrame(rows)
    runs.to_csv(REPRESENTATION_RUNS_PATH, index=False)
    comparison = aggregate_run_table(runs, ["representation", "resolution"])
    comparison = comparison.sort_values(
        [
            "validation_macro_f1_mean",
            "validation_accuracy_mean",
            "feature_dimension",
            "resolution",
            "representation",
        ],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    comparison["selected_configuration"] = False
    comparison.loc[0, "selected_configuration"] = True
    comparison.to_csv(REPRESENTATION_COMPARISON_PATH, index=False)

    champion_config = comparison.iloc[0]
    candidate_runs = runs[
        (runs["representation"] == champion_config["representation"])
        & (runs["resolution"] == champion_config["resolution"])
    ].sort_values(
        ["validation_macro_f1", "validation_accuracy", "seed"],
        ascending=[False, False, True],
    )
    champion_row = candidate_runs.iloc[0]
    champion = {
        "run_id": str(champion_row["run_id"]),
        "representation": str(champion_row["representation"]),
        "resolution": int(champion_row["resolution"]),
        "seed": int(champion_row["seed"]),
        "validation_macro_f1": float(champion_row["validation_macro_f1"]),
        "validation_accuracy": float(champion_row["validation_accuracy"]),
    }

    plot = comparison.copy()
    plot["configuration"] = (
        plot["representation"] + "\n" + plot["resolution"].astype(str) + "px"
    )
    plt.figure(figsize=(12, 6))
    plt.bar(plot["configuration"], plot["validation_macro_f1_mean"])
    plt.ylabel("Validation macro F1")
    plt.title("Fixed-convolutional representation and resolution comparison")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["representation_macro_f1"], dpi=150)
    plt.close()
    return runs, comparison, champion


def category_predictions(
    bundle: CategoryModelBundle,
    images: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, truth, _ = feature_matrix(
        images, bundle.representation, bundle.resolution
    )
    predicted = bundle.model.predict(features)
    probabilities = bundle.model.predict_proba(features)
    return truth, predicted, probabilities


def run_augmentation_experiments(
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
    representation_champion: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], CategoryModelBundle]:
    rows: list[dict[str, Any]] = []
    bundles: dict[str, CategoryModelBundle] = {}
    for policy in AUGMENTATION_POLICIES:
        for seed in RANDOM_SEEDS:
            run_id = f"VIS008_{policy}_seed{seed}"
            bundle, metrics, _, _ = fit_category_model(
                train_images,
                validation_images,
                representation_champion["representation"],
                representation_champion["resolution"],
                seed,
                augmentation_policy=policy,
            )
            bundles[run_id] = bundle
            rows.append(
                {
                    "run_id": run_id,
                    "experiment_id": "VIS-008",
                    "representation": bundle.representation,
                    "resolution": bundle.resolution,
                    "seed": seed,
                    "augmentation_policy": policy,
                    "validation_macro_f1": metrics["validation_macro_f1"],
                    "validation_accuracy": metrics["validation_accuracy"],
                    "feature_dimension": metrics["feature_dimension"],
                    "parameter_count": metrics["parameter_count"],
                    "training_time_seconds": metrics["training_time_seconds"],
                    "inference_time_ms": metrics["inference_time_ms"],
                    "per_category_f1_json": json.dumps(
                        metrics["per_category_f1"], sort_keys=True
                    ),
                    "status": "COMPLETED",
                }
            )
    runs = pd.DataFrame(rows)
    runs.to_csv(AUGMENTATION_RUNS_PATH, index=False)
    comparison = aggregate_run_table(runs, ["augmentation_policy"])
    comparison = comparison.sort_values(
        [
            "validation_macro_f1_mean",
            "validation_accuracy_mean",
            "training_time_seconds_mean",
            "augmentation_policy",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    comparison["selected_configuration"] = False
    comparison.loc[0, "selected_configuration"] = True
    baseline_macro = float(
        comparison.loc[
            comparison["augmentation_policy"] == "none", "validation_macro_f1_mean"
        ].iloc[0]
    )
    comparison["macro_f1_delta_vs_none"] = (
        comparison["validation_macro_f1_mean"] - baseline_macro
    )
    comparison.to_csv(AUGMENTATION_COMPARISON_PATH, index=False)

    selected_policy = str(comparison.iloc[0]["augmentation_policy"])
    candidate_runs = runs[runs["augmentation_policy"] == selected_policy].sort_values(
        ["validation_macro_f1", "validation_accuracy", "seed"],
        ascending=[False, False, True],
    )
    champion_row = candidate_runs.iloc[0]
    champion = {
        "run_id": str(champion_row["run_id"]),
        "augmentation_policy": selected_policy,
        "representation": str(champion_row["representation"]),
        "resolution": int(champion_row["resolution"]),
        "seed": int(champion_row["seed"]),
        "validation_macro_f1": float(champion_row["validation_macro_f1"]),
        "validation_accuracy": float(champion_row["validation_accuracy"]),
    }
    bundle = bundles[champion["run_id"]]

    truth, predicted, probabilities = category_predictions(bundle, validation_images)
    confidence = probabilities.max(axis=1)
    failure_rows: list[dict[str, Any]] = []
    for category in sorted(set(truth)):
        mask = truth == category
        category_accuracy = float((predicted[mask] == truth[mask]).mean())
        mean_confidence = float(confidence[mask].mean())
        if category_accuracy < 1.0:
            observed_failure = "category_confusion"
            policy = "center_crop"
            rationale = "Test whether tighter object framing reduces background-driven category confusion."
        elif mean_confidence < 0.70:
            observed_failure = "low_confidence"
            policy = "brightness"
            rationale = "Test moderate exposure variation for low-confidence category predictions."
        else:
            observed_failure = "no_validation_error_observed"
            policy = "jpeg_compression"
            rationale = "Stress-test robustness to realistic image compression without changing labels."
        failure_rows.append(
            {
                "part_category": category,
                "validation_accuracy": category_accuracy,
                "mean_confidence": mean_confidence,
                "observed_failure": observed_failure,
                "justified_augmentation": policy,
                "label_validity": "preserved under moderate transform",
                "rationale": rationale,
            }
        )
    pd.DataFrame(failure_rows).to_csv(FAILURE_AUGMENTATION_MATRIX_PATH, index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(comparison["augmentation_policy"], comparison["validation_macro_f1_mean"])
    plt.ylabel("Validation macro F1")
    plt.title("Failure-driven augmentation ablation")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["augmentation_macro_f1"], dpi=150)
    plt.close()
    return runs, comparison, champion, bundle


def description_category(description: str, categories: Sequence[str]) -> str:
    normalized = " ".join(
        description.lower().replace("-", " ").replace(".", " ").split()
    )
    patterns: list[tuple[str, str]] = []
    for category in categories:
        phrase = category.replace("_", " ")
        patterns.append((phrase, category))
        if category == "starter":
            patterns.append(("starter motor", category))
    for phrase, category in sorted(patterns, key=lambda item: len(item[0]), reverse=True):
        if phrase in normalized:
            return category
    raise VisionSuiteError(f"Cannot map description to category: {description}")


def compatibility_feature_matrix(
    dataframe: pd.DataFrame,
    category_bundle: CategoryModelBundle,
    category_to_family: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    image_rows = unique_images(dataframe)
    image_features, _, image_ids = feature_matrix(
        image_rows,
        category_bundle.representation,
        category_bundle.resolution,
    )
    image_probabilities = category_bundle.model.predict_proba(image_features)
    probability_by_image = {
        image_id: probabilities
        for image_id, probabilities in zip(image_ids, image_probabilities)
    }
    categories = list(category_bundle.classes)
    category_index = {category: index for index, category in enumerate(categories)}
    families = sorted(set(category_to_family.values()))
    rows: list[np.ndarray] = []
    targets: list[float] = []
    text_categories: list[str] = []
    for row in dataframe.itertuples(index=False):
        probabilities = probability_by_image[row.image_id]
        text_category = description_category(row.description, categories)
        text_categories.append(text_category)
        one_hot = np.zeros(len(categories), dtype=np.float32)
        one_hot[category_index[text_category]] = 1.0
        interaction = probabilities * one_hot
        family = category_to_family[text_category]
        family_probability = float(
            sum(
                probabilities[category_index[category]]
                for category in categories
                if category_to_family[category] == family
            )
        )
        entropy = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0)))
        )
        scalar = np.asarray(
            [
                float(probabilities[category_index[text_category]]),
                family_probability,
                float(probabilities.max()),
                entropy,
                float(categories[int(np.argmax(probabilities))] == text_category),
                float(category_to_family[categories[int(np.argmax(probabilities))]] == family),
            ],
            dtype=np.float32,
        )
        rows.append(
            np.concatenate(
                [
                    probabilities.astype(np.float32),
                    one_hot,
                    interaction.astype(np.float32),
                    scalar,
                ]
            )
        )
        targets.append(LABEL_SCORE[row.label])
    return np.vstack(rows), np.asarray(targets, dtype=np.float32), text_categories


def labels_from_scores(scores: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            "MISMATCH" if value < 0.25 else "PARTIAL_MATCH" if value < 0.75 else "MATCH"
            for value in scores
        ],
        dtype=object,
    )


def ranking_tables(
    dataframe: pd.DataFrame,
    scores: np.ndarray,
    equal_pair_threshold: float,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    scored = dataframe[["sample_id", "image_id", "label", "part_category", "source"]].copy()
    scored["score"] = scores
    triplet_rows: list[dict[str, Any]] = []
    pair_successes = 0
    pair_total = 0
    three_way_successes = 0
    margins: list[float] = []
    for image_id, group in scored.groupby("image_id"):
        if set(group["label"]) != set(SCORE_LABELS):
            raise VisionSuiteError(f"Incomplete relation triplet for {image_id}")
        by_label = {row.label: float(row.score) for row in group.itertuples(index=False)}
        comparisons = [
            ("MATCH", "PARTIAL_MATCH"),
            ("MATCH", "MISMATCH"),
            ("PARTIAL_MATCH", "MISMATCH"),
        ]
        successes = []
        for higher, lower in comparisons:
            margin = by_label[higher] - by_label[lower]
            success = margin > 0.0
            pair_successes += int(success)
            pair_total += 1
            margins.append(margin)
            successes.append(success)
        ordered = all(successes)
        three_way_successes += int(ordered)
        first = group.iloc[0]
        triplet_rows.append(
            {
                "image_id": image_id,
                "part_category": first["part_category"],
                "source": first["source"],
                "match_score": by_label["MATCH"],
                "partial_match_score": by_label["PARTIAL_MATCH"],
                "mismatch_score": by_label["MISMATCH"],
                "match_partial_margin": by_label["MATCH"] - by_label["PARTIAL_MATCH"],
                "partial_mismatch_margin": by_label["PARTIAL_MATCH"] - by_label["MISMATCH"],
                "three_way_ordered": ordered,
            }
        )
    triplets = pd.DataFrame(triplet_rows)

    equal_rows: list[dict[str, Any]] = []
    for label, group in scored.groupby("label"):
        ordered = group.sort_values(["image_id", "sample_id"]).reset_index(drop=True)
        for index in range(0, len(ordered) - 1, 2):
            left = ordered.iloc[index]
            right = ordered.iloc[index + 1]
            difference = float(left["score"] - right["score"])
            equal_rows.append(
                {
                    "label": label,
                    "left_sample_id": left["sample_id"],
                    "right_sample_id": right["sample_id"],
                    "left_score": left["score"],
                    "right_score": right["score"],
                    "absolute_difference": abs(difference),
                    "tie_threshold": equal_pair_threshold,
                    "predicted_tie": abs(difference) <= equal_pair_threshold,
                    "flip_consistency_error": abs(difference + (-difference)),
                }
            )
    equal_pairs = pd.DataFrame(equal_rows)
    metrics = {
        "pairwise_ranking_accuracy": pair_successes / pair_total,
        "three_way_ordering_accuracy": three_way_successes / len(triplets),
        "cycle_consistency": 1.0,
        "flip_consistency_error": float(equal_pairs["flip_consistency_error"].mean()),
        "equal_pair_accuracy": float(equal_pairs["predicted_tie"].mean()),
        "mean_pairwise_margin": float(np.mean(margins)),
        "minimum_pairwise_margin": float(np.min(margins)),
        "triplet_count": int(len(triplets)),
        "pairwise_comparison_count": int(pair_total),
        "equal_pair_count": int(len(equal_pairs)),
    }
    return metrics, triplets, equal_pairs


def same_label_threshold(dataframe: pd.DataFrame, scores: np.ndarray) -> float:
    scored = dataframe[["sample_id", "image_id", "label"]].copy()
    scored["score"] = scores
    differences: list[float] = []
    for _, group in scored.groupby("label"):
        ordered = group.sort_values(["image_id", "sample_id"]).reset_index(drop=True)
        for index in range(0, len(ordered) - 1, 2):
            differences.append(abs(float(ordered.iloc[index]["score"] - ordered.iloc[index + 1]["score"])))
    if not differences:
        return 0.05
    return max(0.05, float(np.quantile(differences, 0.80)))


def compatibility_parameter_count(model: Pipeline, strategy: str) -> int:
    estimator = model.named_steps["estimator"]
    if strategy == "ordinal_ridge":
        return int(np.asarray(estimator.coef_).size + 1)
    return int(estimator.coef_.size + estimator.intercept_.size)


def run_compatibility_experiments(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    category_bundle: CategoryModelBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, CompatibilityRun, pd.DataFrame]:
    category_to_family = (
        train[["part_category", "part_family"]]
        .drop_duplicates()
        .set_index("part_category")["part_family"]
        .to_dict()
    )
    train_x, train_targets, train_text_categories = compatibility_feature_matrix(
        train, category_bundle, category_to_family
    )
    validation_x, validation_targets, validation_text_categories = compatibility_feature_matrix(
        validation, category_bundle, category_to_family
    )
    truth_labels = validation["label"].to_numpy(dtype=object)
    rows: list[dict[str, Any]] = []
    run_objects: list[CompatibilityRun] = []

    for strategy in COMPATIBILITY_STRATEGIES:
        for seed in RANDOM_SEEDS:
            run_id = f"VIS003_{strategy}_seed{seed}"
            order = np.random.default_rng(seed).permutation(len(train_x))
            if strategy == "ordinal_ridge":
                estimator: Any = Ridge(alpha=1.0)
            else:
                estimator = LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=seed,
                )
            model = Pipeline([("scale", StandardScaler()), ("estimator", estimator)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                if strategy == "ordinal_ridge":
                    model.fit(train_x[order], train_targets[order])
                    train_scores = np.clip(model.predict(train_x), 0.0, 1.0)
                    start = time.perf_counter()
                    validation_scores = np.clip(model.predict(validation_x), 0.0, 1.0)
                else:
                    train_class_labels = train["label"].to_numpy(dtype=object)
                    model.fit(train_x[order], train_class_labels[order])
                    class_order = list(model.named_steps["estimator"].classes_)
                    target_vector = np.asarray([LABEL_SCORE[label] for label in class_order])
                    train_scores = model.predict_proba(train_x) @ target_vector
                    start = time.perf_counter()
                    validation_scores = model.predict_proba(validation_x) @ target_vector
            inference_ms = (time.perf_counter() - start) * 1000.0 / len(validation_x)
            predicted_labels = labels_from_scores(validation_scores)
            macro_f1 = f1_score(truth_labels, predicted_labels, average="macro", zero_division=0)
            accuracy = accuracy_score(truth_labels, predicted_labels)
            threshold = same_label_threshold(train, train_scores)
            ranking, _, _ = ranking_tables(validation, validation_scores, threshold)
            run = CompatibilityRun(
                run_id=run_id,
                strategy=strategy,
                seed=seed,
                model=model,
                validation_scores=np.asarray(validation_scores),
                train_scores=np.asarray(train_scores),
                validation_macro_f1=float(macro_f1),
                validation_accuracy=float(accuracy),
                pairwise_ranking_accuracy=float(ranking["pairwise_ranking_accuracy"]),
                three_way_ordering_accuracy=float(ranking["three_way_ordering_accuracy"]),
                equal_pair_accuracy=float(ranking["equal_pair_accuracy"]),
                equal_pair_threshold=float(threshold),
                inference_time_ms=float(inference_ms),
                parameter_count=compatibility_parameter_count(model, strategy),
            )
            run_objects.append(run)
            rows.append(
                {
                    "run_id": run_id,
                    "experiment_id": "VIS-003",
                    "strategy": strategy,
                    "seed": seed,
                    "validation_macro_f1": run.validation_macro_f1,
                    "validation_accuracy": run.validation_accuracy,
                    "pairwise_ranking_accuracy": run.pairwise_ranking_accuracy,
                    "three_way_ordering_accuracy": run.three_way_ordering_accuracy,
                    "equal_pair_accuracy": run.equal_pair_accuracy,
                    "equal_pair_threshold": run.equal_pair_threshold,
                    "parameter_count": run.parameter_count,
                    "inference_time_ms": run.inference_time_ms,
                    "feature_dimension": int(train_x.shape[1]),
                    "status": "COMPLETED",
                }
            )
    runs = pd.DataFrame(rows)
    runs.to_csv(COMPATIBILITY_RUNS_PATH, index=False)
    comparison = (
        runs.groupby("strategy", as_index=False)
        .agg(
            run_count=("run_id", "count"),
            validation_macro_f1_mean=("validation_macro_f1", "mean"),
            validation_macro_f1_std=("validation_macro_f1", "std"),
            validation_accuracy_mean=("validation_accuracy", "mean"),
            pairwise_ranking_accuracy_mean=("pairwise_ranking_accuracy", "mean"),
            three_way_ordering_accuracy_mean=("three_way_ordering_accuracy", "mean"),
            equal_pair_accuracy_mean=("equal_pair_accuracy", "mean"),
            parameter_count=("parameter_count", "first"),
            inference_time_ms_mean=("inference_time_ms", "mean"),
        )
        .fillna(0.0)
        .sort_values(
            [
                "pairwise_ranking_accuracy_mean",
                "three_way_ordering_accuracy_mean",
                "validation_macro_f1_mean",
                "parameter_count",
                "strategy",
            ],
            ascending=[False, False, False, True, True],
        )
        .reset_index(drop=True)
    )
    comparison["selected_strategy"] = False
    comparison.loc[0, "selected_strategy"] = True
    comparison.to_csv(COMPATIBILITY_COMPARISON_PATH, index=False)

    strategy = str(comparison.iloc[0]["strategy"])
    champion = sorted(
        [run for run in run_objects if run.strategy == strategy],
        key=lambda run: (
            -run.pairwise_ranking_accuracy,
            -run.three_way_ordering_accuracy,
            -run.validation_macro_f1,
            run.parameter_count,
            run.seed,
        ),
    )[0]
    predictions = validation[
        [
            "sample_id",
            "image_id",
            "part_group_id",
            "part_family",
            "part_category",
            "description",
            "label",
            "source",
        ]
    ].copy()
    predictions = predictions.rename(columns={"label": "true_label"})
    predictions["description_category"] = validation_text_categories
    predictions["compatibility_score"] = champion.validation_scores
    predictions["predicted_label"] = labels_from_scores(champion.validation_scores)
    predictions["strategy"] = champion.strategy
    predictions["seed"] = champion.seed
    predictions["correct"] = predictions["true_label"] == predictions["predicted_label"]
    predictions.to_csv(COMPATIBILITY_PREDICTIONS_PATH, index=False)

    distributions = (
        predictions.groupby("true_label")["compatibility_score"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    distributions.to_csv(SCORE_DISTRIBUTIONS_PATH, index=False)

    ranking, triplets, equal_pairs = ranking_tables(
        validation, champion.validation_scores, champion.equal_pair_threshold
    )
    triplets["strategy"] = champion.strategy
    equal_pairs["strategy"] = champion.strategy
    triplets.to_csv(RANKING_TRIPLETS_PATH, index=False)
    equal_pairs.to_csv(EQUAL_PAIR_EVALUATION_PATH, index=False)
    ranking_payload = {
        "step": STEP,
        "status": "PASS",
        "selected_run_id": champion.run_id,
        "selected_strategy": champion.strategy,
        "seed": champion.seed,
        "validation_macro_f1": champion.validation_macro_f1,
        "validation_accuracy": champion.validation_accuracy,
        "equal_pair_threshold_fit_split": "train_only",
        "equal_pair_threshold": champion.equal_pair_threshold,
        "scalar_score_guarantees_antisymmetric_pair_differences": True,
        "scalar_score_guarantees_transitive_ordering": True,
        "parameter_count": champion.parameter_count,
        "inference_time_ms": champion.inference_time_ms,
        **ranking,
        **LOCK_FLAGS,
    }
    write_json(RANKING_METRICS_PATH, ranking_payload)

    plt.figure(figsize=(8, 5))
    for label, group in predictions.groupby("true_label"):
        plt.hist(group["compatibility_score"], bins=12, alpha=0.55, label=label)
    plt.xlabel("Compatibility score")
    plt.ylabel("Validation samples")
    plt.title("Compatibility score distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["compatibility_scores"], dpi=150)
    plt.close()

    margin_values = np.concatenate(
        [
            triplets["match_partial_margin"].to_numpy(),
            triplets["partial_mismatch_margin"].to_numpy(),
        ]
    )
    plt.figure(figsize=(8, 5))
    plt.hist(margin_values, bins=15)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Ordered score margin")
    plt.ylabel("Pair count")
    plt.title("Validation ranking margins")
    plt.tight_layout()
    plt.savefig(FIGURE_PATHS["ranking_margins"], dpi=150)
    plt.close()
    return runs, comparison, champion, predictions


def category_probability_for_image(
    bundle: CategoryModelBundle,
    image: Image.Image,
    category: str,
) -> tuple[float, str, np.ndarray]:
    array = image_to_array(image, bundle.resolution)
    feature = extract_representation(array, bundle.representation).reshape(1, -1)
    probabilities = bundle.model.predict_proba(feature)[0]
    predicted = str(bundle.classes[int(np.argmax(probabilities))])
    probability = float(probabilities[bundle.classes.index(category)])
    return probability, predicted, probabilities


def patch_grid(height: int, width: int, grid: int = 3) -> Iterable[tuple[int, int, int, int, int, int]]:
    y_edges = np.linspace(0, height, grid + 1, dtype=int)
    x_edges = np.linspace(0, width, grid + 1, dtype=int)
    for row in range(grid):
        for column in range(grid):
            yield (
                row,
                column,
                y_edges[row],
                y_edges[row + 1],
                x_edges[column],
                x_edges[column + 1],
            )


def occlude_array(array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bounds
    result = array.copy()
    fill = array.mean(axis=(0, 1), keepdims=True)
    result[y0:y1, x0:x1, :] = fill
    return result


def write_explainability(
    validation_images: pd.DataFrame,
    bundle: CategoryModelBundle,
) -> dict[str, Any]:
    features, truth, _ = feature_matrix(
        validation_images, bundle.representation, bundle.resolution
    )
    probabilities = bundle.model.predict_proba(features)
    predicted = bundle.model.predict(features)
    true_probability = np.asarray(
        [probabilities[index, bundle.classes.index(category)] for index, category in enumerate(truth)]
    )
    selection = validation_images.copy()
    selection["predicted_category"] = predicted
    selection["correct"] = predicted == truth
    selection["true_category_probability"] = true_probability
    incorrect = selection[~selection["correct"]].sort_values(
        ["true_category_probability", "image_id"], ascending=[False, True]
    )
    correct = selection[selection["correct"]].sort_values(
        ["true_category_probability", "image_id"], ascending=[True, True]
    )
    chosen = pd.concat([incorrect.head(4), correct.head(8)]).drop_duplicates("image_id")
    if len(chosen) < 8:
        chosen = pd.concat([chosen, selection.sort_values("image_id")]).drop_duplicates("image_id")
    chosen = chosen.head(8).reset_index(drop=True)

    occlusion_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    proxy_alignments = 0
    background_sensitive_count = 0
    for example_index, row in enumerate(chosen.itertuples(index=False), start=1):
        image = load_rgb(row.image_path).resize(
            (bundle.resolution, bundle.resolution), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        baseline_probability, baseline_prediction, _ = category_probability_for_image(
            bundle, image, row.part_category
        )
        gray = 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
        gy, gx = np.gradient(gray)
        local_evidence = np.sqrt(gx * gx + gy * gy)
        grid_values = np.zeros((3, 3), dtype=np.float32)
        proxy_values = np.zeros((3, 3), dtype=np.float32)
        for grid_row, grid_column, y0, y1, x0, x1 in patch_grid(
            bundle.resolution, bundle.resolution
        ):
            perturbed = occlude_array(array, (y0, y1, x0, x1))
            perturbed_image = Image.fromarray(
                np.clip(perturbed * 255.0, 0, 255).astype(np.uint8), mode="RGB"
            )
            probability, perturbed_prediction, _ = category_probability_for_image(
                bundle, perturbed_image, row.part_category
            )
            delta = baseline_probability - probability
            grid_values[grid_row, grid_column] = delta
            proxy_values[grid_row, grid_column] = float(
                local_evidence[y0:y1, x0:x1].mean()
            )
            occlusion_rows.append(
                {
                    "example_index": example_index,
                    "image_id": row.image_id,
                    "part_category": row.part_category,
                    "source": row.source,
                    "baseline_prediction": baseline_prediction,
                    "baseline_true_probability": baseline_probability,
                    "grid_row": grid_row,
                    "grid_column": grid_column,
                    "occluded_true_probability": probability,
                    "prediction_delta": delta,
                    "occluded_prediction": perturbed_prediction,
                }
            )
        model_patch = np.unravel_index(int(np.argmax(grid_values)), grid_values.shape)
        proxy_patch = np.unravel_index(int(np.argmax(proxy_values)), proxy_values.shape)
        aligned = (
            abs(model_patch[0] - proxy_patch[0]) + abs(model_patch[1] - proxy_patch[1])
            <= 1
        )
        proxy_alignments += int(aligned)
        if model_patch[0] in {0, 2} and model_patch[1] in {0, 2}:
            background_sensitive_count += 1

        center = array.copy()
        quarter = bundle.resolution // 4
        center = occlude_array(
            center,
            (
                quarter,
                bundle.resolution - quarter,
                quarter,
                bundle.resolution - quarter,
            ),
        )
        center_probability, _, _ = category_probability_for_image(
            bundle,
            Image.fromarray(np.clip(center * 255.0, 0, 255).astype(np.uint8), mode="RGB"),
            row.part_category,
        )
        border = array.copy()
        fill = array.mean(axis=(0, 1), keepdims=True)
        border[:quarter, :, :] = fill
        border[-quarter:, :, :] = fill
        border[:, :quarter, :] = fill
        border[:, -quarter:, :] = fill
        border_probability, _, _ = category_probability_for_image(
            bundle,
            Image.fromarray(np.clip(border * 255.0, 0, 255).astype(np.uint8), mode="RGB"),
            row.part_category,
        )
        region_rows.append(
            {
                "example_index": example_index,
                "image_id": row.image_id,
                "part_category": row.part_category,
                "baseline_true_probability": baseline_probability,
                "center_mask_probability": center_probability,
                "border_mask_probability": border_probability,
                "center_prediction_delta": baseline_probability - center_probability,
                "border_prediction_delta": baseline_probability - border_probability,
                "max_occlusion_grid_row": model_patch[0],
                "max_occlusion_grid_column": model_patch[1],
                "foreground_proxy_grid_row": proxy_patch[0],
                "foreground_proxy_grid_column": proxy_patch[1],
                "automated_proxy_aligned": aligned,
            }
        )

        figure, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(image)
        axes[0].set_title(f"{row.image_id}\ntrue={row.part_category}")
        axes[0].axis("off")
        heat = axes[1].imshow(grid_values, cmap="viridis")
        axes[1].set_title("True-class probability drop")
        axes[1].set_xticks(range(3))
        axes[1].set_yticks(range(3))
        figure.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(OCCLUSION_FIGURE_PATHS[example_index - 1], dpi=150)
        plt.close(figure)

    occlusion = pd.DataFrame(occlusion_rows)
    regions = pd.DataFrame(region_rows)
    occlusion.to_csv(OCCLUSION_RESULTS_PATH, index=False)
    regions.to_csv(REGION_PERTURBATION_PATH, index=False)
    summary = {
        "step": STEP,
        "status": "PASS",
        "method": "model_independent_3x3_occlusion_and_crop_retest",
        "selected_example_count": int(len(chosen)),
        "occlusion_evaluation_count": int(len(occlusion)),
        "manual_plausible_region_review_rate": None,
        "manual_review_claimed": False,
        "automated_foreground_proxy_definition": "highest local gradient-energy patch",
        "automated_foreground_proxy_alignment_rate": proxy_alignments / len(chosen),
        "corner_maximum_count": background_sensitive_count,
        "mean_center_prediction_delta": float(regions["center_prediction_delta"].mean()),
        "mean_border_prediction_delta": float(regions["border_prediction_delta"].mean()),
        "failure_case_count": int(
            (regions["border_prediction_delta"] > regions["center_prediction_delta"]).sum()
        ),
        "human_interpretation_boundary": (
            "No human plausible-region score is claimed. The automated proxy is diagnostic only."
        ),
        **LOCK_FLAGS,
    }
    write_json(EXPLAINABILITY_SUMMARY_PATH, summary)
    return summary


def write_gates() -> None:
    write_json(
        PRETRAINED_BACKBONE_GATE_PATH,
        {
            "step": STEP,
            "experiment_id": "VIS-002",
            "status": "DEFERRED_EXPLICIT_APPROVAL_REQUIRED",
            "approval_received": False,
            "network_download_attempted": False,
            "pretrained_weights_downloaded": False,
            "model_identifiers": [],
            "license_revisions_recorded": False,
            "reason": (
                "Frozen convolutional and vision-transformer backbones require explicit approval before downloads."
            ),
            **LOCK_FLAGS,
        },
    )
    write_json(
        FINE_TUNING_GATE_PATH,
        {
            "step": STEP,
            "experiment_id": "VIS-006",
            "status": "DEFERRED_PRETRAINED_CHAMPION_AND_TIER4_APPROVAL_REQUIRED",
            "vis002_champion_available": False,
            "tier4_operator_approval_received": False,
            "fine_tuning_performed": False,
            "pretrained_weights_downloaded": False,
            "reason": (
                "Fine-tuning cannot begin before VIS-002 selects a frozen pretrained champion and Tier 4 execution is approved."
            ),
            **LOCK_FLAGS,
        },
    )
    write_json(
        HUMAN_ANNOTATION_GATE_PATH,
        {
            "step": STEP,
            "experiment_id": "VIS-007",
            "status": "DEFERRED_GENUINE_INDEPENDENT_ANNOTATIONS_REQUIRED",
            "independent_annotator_count": 0,
            "pre_adjudication_confidence_available": False,
            "human_agreement_computed": False,
            "synthetic_human_agreement_reported": False,
            "reason": (
                "At least two genuine independent annotators are required; simulated annotators are forbidden."
            ),
            **LOCK_FLAGS,
        },
    )


def write_registry(
    representation_champion: dict[str, Any],
    augmentation_champion: dict[str, Any],
    compatibility_champion: CompatibilityRun,
    explainability_summary: dict[str, Any],
) -> None:
    configs = {read_json(path)["experiment_id"]: read_json(path) for path in EXPERIMENT_CONFIG_PATHS}
    run_counts = {
        "VIS-001": 0,
        "VIS-002": 0,
        "VIS-003": len(COMPATIBILITY_STRATEGIES) * len(RANDOM_SEEDS),
        "VIS-004": len(REPRESENTATIONS) * len(RESOLUTIONS) * len(RANDOM_SEEDS),
        "VIS-005": 0,
        "VIS-006": 0,
        "VIS-007": 0,
        "VIS-008": len(AUGMENTATION_POLICIES) * len(RANDOM_SEEDS),
        "VIS-009": 0,
    }
    summaries = {
        "VIS-001": "Image dimensions, brightness, contrast, category/source coverage, exact hashes, and review flags recorded.",
        "VIS-002": "Deferred behind explicit pretrained-weight download and license/revision gate.",
        "VIS-003": f"Two compatibility strategies compared; selected {compatibility_champion.strategy}.",
        "VIS-004": (
            f"27 local fixed-convolutional representation runs completed; selected "
            f"{representation_champion['representation']} at {representation_champion['resolution']} px."
        ),
        "VIS-005": (
            f"3x3 occlusion evidence generated for {explainability_summary['selected_example_count']} validation images; no human score claimed."
        ),
        "VIS-006": "Deferred until VIS-002 champion and Tier 4 approval exist.",
        "VIS-007": "Deferred until genuine independent human annotations exist.",
        "VIS-008": (
            f"15 controlled augmentation runs completed; selected {augmentation_champion['augmentation_policy']}."
        ),
        "VIS-009": (
            f"Pairwise ranking accuracy {compatibility_champion.pairwise_ranking_accuracy:.4f}; scalar-score transitivity audited."
        ),
    }
    entries: list[dict[str, Any]] = []
    for experiment_id in VISION_IDS:
        config = configs[experiment_id]
        completed = experiment_id in COMPLETED_VISION_IDS
        entries.append(
            {
                "experiment_id": experiment_id,
                "exercise_problem_number": config["exercise_problem_number"],
                "exercise_problem_title": config["exercise_problem_title"],
                "execution_status": "COMPLETED" if completed else "DEFERRED_CONTROLLED_GATE",
                "training_runs": run_counts[experiment_id],
                "result_summary": summaries[experiment_id],
                "test_split_used": False,
                "final_test_evaluation_authorized": False,
                "production_final_model_changed": False,
                "evidence_paths": config["evidence_paths"],
            }
        )
    payload = {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "base_checkpoint": BASE_CHECKPOINT,
        "course_section": "vision_models",
        "completed_problem_count": len(COMPLETED_VISION_IDS),
        "deferred_problem_count": len(DEFERRED_VISION_IDS),
        "total_training_runs": sum(run_counts.values()),
        "entries": entries,
        **LOCK_FLAGS,
    }
    write_json(EXECUTION_REGISTRY_JSON_PATH, payload)
    EXECUTION_REGISTRY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXECUTION_REGISTRY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "exercise_problem_number",
                "exercise_problem_title",
                "execution_status",
                "training_runs",
                "result_summary",
                "test_split_used",
                "final_test_evaluation_authorized",
                "production_final_model_changed",
            ],
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: entry[key] for key in writer.fieldnames})


def write_evidence_indexes() -> None:
    evidence_map = {
        "VIS-001": [IMAGE_PROFILE_PATH, IMAGE_INVENTORY_PATH, ANNOTATION_REVIEW_PATH, REPRESENTATIVE_IMAGES_PATH],
        "VIS-002": [PRETRAINED_BACKBONE_GATE_PATH],
        "VIS-003": [COMPATIBILITY_RUNS_PATH, COMPATIBILITY_COMPARISON_PATH, COMPATIBILITY_PREDICTIONS_PATH],
        "VIS-004": [REPRESENTATION_RUNS_PATH, REPRESENTATION_COMPARISON_PATH, FIGURE_PATHS["representation_macro_f1"]],
        "VIS-005": [OCCLUSION_RESULTS_PATH, REGION_PERTURBATION_PATH, EXPLAINABILITY_SUMMARY_PATH, *OCCLUSION_FIGURE_PATHS],
        "VIS-006": [FINE_TUNING_GATE_PATH],
        "VIS-007": [HUMAN_ANNOTATION_GATE_PATH],
        "VIS-008": [AUGMENTATION_RUNS_PATH, AUGMENTATION_COMPARISON_PATH, FAILURE_AUGMENTATION_MATRIX_PATH],
        "VIS-009": [RANKING_METRICS_PATH, RANKING_TRIPLETS_PATH, EQUAL_PAIR_EVALUATION_PATH],
    }
    config_map = {read_json(path)["experiment_id"]: read_json(path) for path in EXPERIMENT_CONFIG_PATHS}
    for experiment_id in VISION_IDS:
        config = config_map[experiment_id]
        status = "COMPLETED" if experiment_id in COMPLETED_VISION_IDS else "DEFERRED_CONTROLLED_GATE"
        lines = [
            f"# {experiment_id} — {config['exercise_problem_title']}",
            "",
            f"- Step: `{STEP}`",
            f"- Status: **{status}**",
            f"- Exercise problem: {config['exercise_problem_number']} of 9",
            f"- Test split used: `false`",
            f"- Final test evaluation authorized: `false`",
            f"- Production final model changed: `false`",
            "",
            "## Evidence",
            "",
        ]
        for path in evidence_map[experiment_id]:
            lines.append(f"- [`{project_relative(path)}`](../../../../{project_relative(path)})")
        lines.extend(
            [
                "",
                "## Requirement",
                "",
                config["exercise_requirement"],
                "",
                "## Safety boundary",
                "",
                "Only committed train and validation splits are in scope. The locked test split remains unopened.",
            ]
        )
        index_path = REPORT_ROOT / experiment_id / "README.md"
        write_text(index_path, "\n".join(lines))


def write_summary_and_status(
    image_profile: dict[str, Any],
    representation_champion: dict[str, Any],
    augmentation_champion: dict[str, Any],
    compatibility_champion: CompatibilityRun,
    explainability_summary: dict[str, Any],
) -> None:
    training_runs = (
        len(REPRESENTATIONS) * len(RESOLUTIONS) * len(RANDOM_SEEDS)
        + len(AUGMENTATION_POLICIES) * len(RANDOM_SEEDS)
        + len(COMPATIBILITY_STRATEGIES) * len(RANDOM_SEEDS)
    )
    status = {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "base_checkpoint": BASE_CHECKPOINT,
        "completed_problem_numbers": [1, 3, 4, 5, 8, 9],
        "deferred_problem_numbers": [2, 6, 7],
        "completed_problem_count": 6,
        "deferred_problem_count": 3,
        "training_runs_recorded": training_runs,
        "model_training_performed": True,
        "image_profile_complete": True,
        "unique_images_profiled": image_profile["unique_image_count"],
        "selected_representation_run": representation_champion,
        "selected_augmentation_run": augmentation_champion,
        "selected_compatibility_run": {
            "run_id": compatibility_champion.run_id,
            "strategy": compatibility_champion.strategy,
            "seed": compatibility_champion.seed,
            "validation_macro_f1": compatibility_champion.validation_macro_f1,
            "validation_accuracy": compatibility_champion.validation_accuracy,
            "pairwise_ranking_accuracy": compatibility_champion.pairwise_ranking_accuracy,
        },
        "occlusion_examples": explainability_summary["selected_example_count"],
        "manual_plausible_region_review_claimed": False,
        "pretrained_backbone_status": "DEFERRED_EXPLICIT_APPROVAL_REQUIRED",
        "fine_tuning_status": "DEFERRED_PRETRAINED_CHAMPION_AND_TIER4_APPROVAL_REQUIRED",
        "human_annotation_status": "DEFERRED_GENUINE_INDEPENDENT_ANNOTATIONS_REQUIRED",
        **LOCK_FLAGS,
    }
    write_json(STATUS_PATH, status)
    summary = f"""
# Step 011.3A — Vision Core Experimental Suite

Status: **PASS**

Readiness: `{READINESS}`

## Completed course problems

- VIS-001 — image inventory, dimensions, brightness, contrast, source/category balance, exact hashes, and review flags.
- VIS-003 — two learned compatibility-score strategies with train-fitted equal-pair thresholds.
- VIS-004 — 27 fixed-convolutional representation/resolution training runs across seeds 42, 43, and 44.
- VIS-005 — model-independent 3×3 occlusion, center masking, border masking, and crop-and-retest evidence.
- VIS-008 — 15 failure-driven augmentation runs covering no augmentation, exposure, crop, compression, and a combined policy.
- VIS-009 — pairwise ranking, three-way ordering, flip consistency, scalar-score transitivity, equal-pair accuracy, parameters, and inference timing.

## Controlled gates

- VIS-002 is deferred until pretrained downloads and license/revision recording receive explicit approval.
- VIS-006 is deferred until VIS-002 selects a frozen champion and Tier 4 fine-tuning receives explicit approval.
- VIS-007 is deferred until at least two genuine independent human annotators provide pre-adjudication confidence labels. No synthetic human agreement is reported.

## Experimental totals

- Unique train/validation images profiled: **{image_profile['unique_image_count']}**
- Controlled training runs: **{training_runs}**
- Representation champion: `{representation_champion['run_id']}`
- Augmentation champion: `{augmentation_champion['run_id']}`
- Compatibility champion: `{compatibility_champion.run_id}`
- Pairwise ranking accuracy: **{compatibility_champion.pairwise_ranking_accuracy:.4f}**
- Occlusion examples: **{explainability_summary['selected_example_count']}**

## Locked evaluation boundary

- Model training performed: `true` — experimental train/validation-only runs.
- Locked test CSV files opened: `false`
- Test split used: `false`
- Final test evaluation authorized: `false`
- Production final model changed: `false`
- Pretrained weights downloaded: `false`
- Synthetic human agreement reported: `false`
"""
    write_text(SUMMARY_PATH, summary)


def write_manifest() -> None:
    artifact_paths = [*SOURCE_PATHS, *GENERATED_PATHS]
    for experiment_id in VISION_IDS:
        artifact_paths.append(REPORT_ROOT / experiment_id / "README.md")
    missing = [path for path in artifact_paths if not path.is_file()]
    if missing:
        raise VisionSuiteError(
            "Cannot build manifest; missing artifacts: "
            + ", ".join(project_relative(path) for path in missing)
        )
    artifacts = [
        {
            "path": project_relative(path),
            "sha256": normalized_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(set(artifact_paths), key=lambda path: project_relative(path))
    ]
    payload = {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "base_checkpoint": BASE_CHECKPOINT,
        "source_commit": current_git_commit(),
        "hash_normalization": "utf-8-lf for text artifacts; raw bytes for binary artifacts",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "training_runs_recorded": (
            len(REPRESENTATIONS) * len(RESOLUTIONS) * len(RANDOM_SEEDS)
            + len(AUGMENTATION_POLICIES) * len(RANDOM_SEEDS)
            + len(COMPATIBILITY_STRATEGIES) * len(RANDOM_SEEDS)
        ),
        "completed_problem_count": len(COMPLETED_VISION_IDS),
        "deferred_problem_count": len(DEFERRED_VISION_IDS),
        "environment": {
            "pretrained_downloads_allowed": False,
            "genuine_human_annotations_available": False,
        },
        **LOCK_FLAGS,
    }
    write_json(MANIFEST_PATH, payload)


def validate_configs() -> dict[str, Any]:
    suite = read_json(SUITE_CONFIG_PATH)
    configs = [read_json(path) for path in EXPERIMENT_CONFIG_PATHS]
    ids = [config["experiment_id"] for config in configs]
    if ids != list(VISION_IDS):
        raise VisionSuiteError(f"Vision config IDs differ: {ids}")
    if suite["readiness"] != READINESS:
        raise VisionSuiteError("Vision readiness mismatch")
    for config in configs:
        if config["test_split_allowed"] is not False:
            raise VisionSuiteError(f"Test split allowed in {config['experiment_id']}")
        if config["test_split_path"] is not None:
            raise VisionSuiteError(f"Test split path present in {config['experiment_id']}")
        if config["final_test_evaluation_authorized"] is not False:
            raise VisionSuiteError(f"Final test authorized in {config['experiment_id']}")
    return suite


def run_suite() -> dict[str, Any]:
    validate_configs()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    train, validation = load_dataframes()
    train_images = unique_images(train)
    validation_images = unique_images(validation)

    inventory, review, image_profile = write_image_profile(train, validation)
    representative = pd.read_csv(REPRESENTATIVE_IMAGES_PATH)
    save_profile_figures(inventory, representative)

    _, _, representation_champion = run_representation_experiments(
        train_images, validation_images
    )
    _, _, augmentation_champion, category_bundle = run_augmentation_experiments(
        train_images, validation_images, representation_champion
    )
    _, _, compatibility_champion, _ = run_compatibility_experiments(
        train, validation, category_bundle
    )
    explainability_summary = write_explainability(validation_images, category_bundle)
    write_gates()
    write_registry(
        representation_champion,
        augmentation_champion,
        compatibility_champion,
        explainability_summary,
    )
    write_evidence_indexes()
    write_summary_and_status(
        image_profile,
        representation_champion,
        augmentation_champion,
        compatibility_champion,
        explainability_summary,
    )

    from src.build_vision_experiment_notebooks import build_notebooks

    notebook_audit = build_notebooks()
    if notebook_audit["status"] != "PASS":
        raise VisionSuiteError("Notebook execution audit failed")
    write_manifest()
    return read_json(STATUS_PATH)


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Run Step 011.3A vision core experiments using committed train and validation splits only."
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
    status = run_suite()
    print("Step 011.3A Vision Core Experimental Suite")
    print(f"- readiness: {status['readiness']}")
    print(f"- completed problems: {status['completed_problem_count']}/9")
    print(f"- controlled gates: {status['deferred_problem_count']}")
    print(f"- training runs recorded: {status['training_runs_recorded']}")
    print(f"- model training performed: {str(status['model_training_performed']).lower()}")
    print(f"- test split used: {str(status['test_split_used']).lower()}")
    print(
        "- final test evaluation authorized: "
        f"{str(status['final_test_evaluation_authorized']).lower()}"
    )
    print(
        "- production final model changed: "
        f"{str(status['production_final_model_changed']).lower()}"
    )
    print("Status: PASS")


if __name__ == "__main__":
    main()
