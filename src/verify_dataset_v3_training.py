from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from sklearn.metrics import accuracy_score, f1_score

from src.data import PROJECT_ROOT
from src.data_v3 import CATEGORIES, check_v3_split_overlap, load_v3_split
from src.evaluation import grouped_paired_randomization
from src.train import CANONICAL_TORCH_VERSION

RESULTS_DIR = PROJECT_ROOT / "results" / "dataset_v3"
DATASET_VERSION = "3.0-development"
MAIN_MODEL_SLUG = "torch_multimodal_dataset_v3"
EXPECTED_TEST_LOCK_SHA256 = (
    "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
)
EXPECTED_MODEL_SLUGS = {
    "majority",
    "tfidf_logistic_regression",
    "image_logistic_regression",
    "image_text_logistic_regression",
    "torch_text_dataset_v3",
    "torch_cnn_image_dataset_v3",
    MAIN_MODEL_SLUG,
}


def base_torch_version(version: str) -> str:
    return version.split("+", 1)[0]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_training_artifacts(
    results_dir: Path = RESULTS_DIR,
) -> dict[str, object]:
    train = load_v3_split("train")
    validation = load_v3_split("validation")
    overlap = check_v3_split_overlap(train, validation)
    if any(overlap.values()):
        raise AssertionError(f"Dataset V3 development overlap: {overlap}")

    required_paths = {
        "model_comparison.csv",
        "validation_predictions.csv",
        "multimodal_validation_predictions.csv",
        "multimodal_per_category.csv",
        "paired_comparisons.csv",
        "run_info.json",
        "training_summary.json",
        "environment_lock.json",
        "environment_lock.txt",
        "torch_text_dataset_v3_architecture.txt",
        "torch_text_dataset_v3_training_history.csv",
        "torch_cnn_image_dataset_v3_architecture.txt",
        "torch_cnn_image_dataset_v3_training_history.csv",
        "torch_multimodal_dataset_v3_architecture.txt",
        "torch_multimodal_dataset_v3_training_history.csv",
        "models/torch_text_dataset_v3_state.pt",
        "models/torch_cnn_image_dataset_v3_state.pt",
        "models/torch_multimodal_dataset_v3_state.pt",
    }
    missing = sorted(
        relative
        for relative in required_paths
        if not (results_dir / relative).is_file()
    )
    if missing:
        raise AssertionError(f"Missing Dataset V3 training artifacts: {missing}")

    table = pd.read_csv(results_dir / "model_comparison.csv")
    predictions = pd.read_csv(results_dir / "validation_predictions.csv")
    main_predictions = pd.read_csv(
        results_dir / "multimodal_validation_predictions.csv"
    )
    per_category = pd.read_csv(results_dir / "multimodal_per_category.csv")
    paired = pd.read_csv(results_dir / "paired_comparisons.csv")
    run_info = _read_json(results_dir / "run_info.json")
    training_summary = _read_json(results_dir / "training_summary.json")
    environment = _read_json(results_dir / "environment_lock.json")

    if len(table) != len(EXPECTED_MODEL_SLUGS):
        raise AssertionError("Unexpected Dataset V3 model count.")
    if set(table["model_slug"]) != EXPECTED_MODEL_SLUGS:
        raise AssertionError("Unexpected Dataset V3 model inventory.")

    expected_sorted = table.sort_values(
        ["validation_macro_f1", "validation_accuracy", "model"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    assert_frame_equal(
        table.reset_index(drop=True),
        expected_sorted,
        check_dtype=False,
    )

    if len(predictions) != len(validation) * len(EXPECTED_MODEL_SLUGS):
        raise AssertionError("Unexpected Dataset V3 prediction row count.")
    if set(predictions["source"]) != {"dataset_v3"}:
        raise AssertionError("Unexpected source in Dataset V3 predictions.")
    if predictions["image_path"].astype(str).str.replace(
        "\\", "/", regex=False
    ).str.startswith("data/locked_test/").any():
        raise AssertionError("Locked-test path found in Dataset V3 predictions.")

    validation_ids = set(validation["sample_id"])
    indexed_table = table.set_index("model_slug")
    for slug, group in predictions.groupby("model_slug"):
        if set(group["sample_id"]) != validation_ids:
            raise AssertionError(f"Incomplete validation coverage for {slug}.")
        row = indexed_table.loc[slug]
        accuracy = float(
            accuracy_score(group["true_label"], group["predicted_label"])
        )
        macro_f1 = float(
            f1_score(
                group["true_label"],
                group["predicted_label"],
                average="macro",
                zero_division=0,
            )
        )
        if not math.isclose(
            accuracy,
            float(row["validation_accuracy"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Accuracy mismatch for {slug}.")
        if not math.isclose(
            macro_f1,
            float(row["validation_macro_f1"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise AssertionError(f"Macro F1 mismatch for {slug}.")
        if int(group["is_correct"].sum()) != int(row["correct_predictions"]):
            raise AssertionError(f"Correct-count mismatch for {slug}.")

    expected_main = predictions[
        predictions["model_slug"].eq(MAIN_MODEL_SLUG)
    ].reset_index(drop=True)
    assert_frame_equal(main_predictions, expected_main, check_dtype=False)

    if set(per_category["part_category"]) != set(CATEGORIES):
        raise AssertionError("Unexpected Dataset V3 per-category inventory.")
    if set(per_category["samples"]) != {60}:
        raise AssertionError("Each Dataset V3 category must have 60 relation rows.")
    if set(per_category["independent_images"]) != {10}:
        raise AssertionError("Each Dataset V3 category must have 10 validation images.")

    expected_pairs = pd.DataFrame(
        [
            grouped_paired_randomization(
                predictions,
                MAIN_MODEL_SLUG,
                "image_text_logistic_regression",
            ),
            grouped_paired_randomization(
                predictions,
                MAIN_MODEL_SLUG,
                "majority",
            ),
        ]
    )
    assert_frame_equal(paired, expected_pairs, check_dtype=False)
    if set(paired["independent_groups"]) != {80}:
        raise AssertionError("Paired comparisons must use 80 image groups.")
    if set(paired["method"]) != {"monte_carlo_image_group_sign_flip"}:
        raise AssertionError("Dataset V3 comparisons must use grouped Monte Carlo.")

    for slug in (
        "torch_text_dataset_v3",
        "torch_cnn_image_dataset_v3",
        MAIN_MODEL_SLUG,
    ):
        history = pd.read_csv(results_dir / f"{slug}_training_history.csv")
        if history.empty:
            raise AssertionError(f"Empty training history for {slug}.")
        required_columns = {"epoch", "loss", "accuracy", "val_loss", "val_accuracy"}
        if not required_columns.issubset(history.columns):
            raise AssertionError(f"Incomplete training history for {slug}.")
        architecture = (results_dir / f"{slug}_architecture.txt").read_text(
            encoding="utf-8"
        )
        if "Total parameters" not in architecture or "\\n" in architecture:
            raise AssertionError(f"Invalid architecture summary for {slug}.")

        checkpoint = torch.load(
            results_dir / "models" / f"{slug}_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["model_slug"] != slug:
            raise AssertionError(f"Checkpoint slug mismatch for {slug}.")
        if checkpoint["dataset_version"] != DATASET_VERSION:
            raise AssertionError(f"Checkpoint dataset mismatch for {slug}.")
        if tuple(checkpoint["categories"]) != tuple(sorted(CATEGORIES)):
            raise AssertionError(f"Checkpoint category mismatch for {slug}.")
        if not checkpoint["state_dict"]:
            raise AssertionError(f"Checkpoint state is empty for {slug}.")

    if run_info["dataset_version"] != DATASET_VERSION:
        raise AssertionError("Unexpected Dataset V3 run-info version.")
    if base_torch_version(str(run_info["torch_version"])) != CANONICAL_TORCH_VERSION:
        raise AssertionError("Training did not use the canonical PyTorch version.")
    if run_info["training_rows"] != 2880 or run_info["training_images"] != 480:
        raise AssertionError("Unexpected Dataset V3 training size.")
    if run_info["validation_rows"] != 480 or run_info["validation_images"] != 80:
        raise AssertionError("Unexpected Dataset V3 validation size.")
    if run_info["saved_model_count"] != 7:
        raise AssertionError("Unexpected Dataset V3 saved-model count.")
    if run_info["main_model_slug"] != MAIN_MODEL_SLUG:
        raise AssertionError("Unexpected Dataset V3 main model.")
    if run_info["test_lock_status"] != (
        "LOCKED_NOT_AUTHORIZED_FOR_TRAINING_OR_SELECTION"
    ):
        raise AssertionError("Unexpected Dataset V3 test-lock status.")
    if run_info["test_lock_sha256"] != EXPECTED_TEST_LOCK_SHA256:
        raise AssertionError("Unexpected Dataset V3 test-lock fingerprint.")
    for flag in (
        "test_manifest_read",
        "test_images_read",
        "test_split_used",
        "test_evaluation_permitted",
        "test_evaluation_executed",
    ):
        if run_info[flag] is not False:
            raise AssertionError(f"Dataset V3 run-info flag must be false: {flag}")

    if environment["python_version"] != run_info["python_version"]:
        raise AssertionError("Environment lock and run-info Python versions differ.")
    if base_torch_version(environment["packages"]["torch"]) != CANONICAL_TORCH_VERSION:
        raise AssertionError("Environment lock has the wrong PyTorch version.")

    main = indexed_table.loc[MAIN_MODEL_SLUG]
    if training_summary["status"] != (
        "PASS_DATASET_V3_DEVELOPMENT_TRAINING_ARTIFACTS_READY"
    ):
        raise AssertionError("Unexpected Dataset V3 training-summary status.")
    if training_summary["main_model_slug"] != MAIN_MODEL_SLUG:
        raise AssertionError("Training summary has the wrong main model.")
    if not math.isclose(
        float(training_summary["main_validation_accuracy"]),
        float(main["validation_accuracy"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Training-summary accuracy mismatch.")
    if not math.isclose(
        float(training_summary["main_validation_macro_f1"]),
        float(main["validation_macro_f1"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Training-summary macro F1 mismatch.")
    if training_summary["test_lock_sha256"] != EXPECTED_TEST_LOCK_SHA256:
        raise AssertionError("Training summary has the wrong test-lock fingerprint.")

    return {
        "status": "PASS_DATASET_V3_DEVELOPMENT_TRAINING_VERIFIED",
        "dataset_version": DATASET_VERSION,
        "models": len(table),
        "prediction_rows": len(predictions),
        "validation_rows": len(validation),
        "validation_images": int(validation["image_id"].nunique()),
        "main_model_slug": MAIN_MODEL_SLUG,
        "main_validation_accuracy": float(main["validation_accuracy"]),
        "main_validation_macro_f1": float(main["validation_macro_f1"]),
        "main_correct_predictions": int(main["correct_predictions"]),
        "main_total_predictions": int(main["total_predictions"]),
        "paired_comparisons": len(paired),
        "paired_independent_groups": 80,
        "identity_overlap": overlap,
        "test_lock_status": run_info["test_lock_status"],
        "test_lock_sha256": EXPECTED_TEST_LOCK_SHA256,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "test_evaluation_executed": False,
        "artifact_consistency": {
            "saved_metrics_match_predictions": True,
            "main_predictions_match": True,
            "per_category_table_valid": True,
            "paired_comparisons_match": True,
            "neural_checkpoints_valid": True,
            "environment_lock_matches": True,
        },
    }


def main() -> None:
    summary = verify_training_artifacts()
    (RESULTS_DIR / "verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
