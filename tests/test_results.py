from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import accuracy_score, f1_score

from src.data import DATA_DIR, RESULTS_DIR
from src.train import MAIN_MODEL_SLUG


def test_result_table_contains_all_eight_models() -> None:
    table = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    assert len(table) == 8
    assert set(table["modality"]) == {"none", "text", "image", "image + text"}
    assert MAIN_MODEL_SLUG in set(table["model_slug"])
    assert "torch_multimodal_real_only" in set(table["model_slug"])


def test_model_comparison_is_sorted_by_reported_scores() -> None:
    table = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    expected = table.sort_values(
        ["validation_macro_f1", "validation_accuracy", "model"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    assert_frame_equal(table.reset_index(drop=True), expected, check_dtype=False)


def test_saved_predictions_are_validation_only() -> None:
    predictions = pd.read_csv(RESULTS_DIR / "validation_predictions.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    assert len(predictions) == 8 * len(validation) == 240
    for _, group in predictions.groupby("model_slug"):
        assert set(group["sample_id"]) == set(validation["sample_id"])
        assert set(group["source"]) == {"wikimedia"}


def test_saved_metrics_match_saved_predictions() -> None:
    table = pd.read_csv(RESULTS_DIR / "model_comparison.csv").set_index("model_slug")
    predictions = pd.read_csv(RESULTS_DIR / "validation_predictions.csv")
    for slug, group in predictions.groupby("model_slug"):
        row = table.loc[slug]
        assert abs(accuracy_score(group["true_label"], group["predicted_label"]) - row["validation_accuracy"]) < 1e-12
        assert abs(
            f1_score(group["true_label"], group["predicted_label"], average="macro", zero_division=0)
            - row["validation_macro_f1"]
        ) < 1e-12
        assert int(group["is_correct"].sum()) == row["correct_predictions"]


def test_main_prediction_file_is_derived_from_all_predictions() -> None:
    all_predictions = pd.read_csv(RESULTS_DIR / "validation_predictions.csv")
    expected = all_predictions[all_predictions["model_slug"].eq(MAIN_MODEL_SLUG)].reset_index(drop=True)
    saved = pd.read_csv(RESULTS_DIR / "multimodal_validation_predictions.csv")
    assert_frame_equal(saved, expected, check_dtype=False)


def test_synthetic_ablation_contains_exactly_two_models() -> None:
    table = pd.read_csv(RESULTS_DIR / "synthetic_ablation.csv")
    assert set(table["model_slug"]) == {
        "torch_multimodal_real_only",
        "torch_multimodal_real_plus_synthetic",
    }


def test_training_histories_and_architectures_are_nonempty() -> None:
    slugs = [
        "torch_text",
        "torch_cnn_image",
        "torch_multimodal_real_only",
        "torch_multimodal_real_plus_synthetic",
    ]
    for slug in slugs:
        history = pd.read_csv(RESULTS_DIR / f"{slug}_training_history.csv")
        assert len(history) >= 1
        assert {"epoch", "loss", "accuracy", "val_loss", "val_accuracy"}.issubset(history.columns)
        architecture = (RESULTS_DIR / f"{slug}_architecture.txt").read_text(encoding="utf-8")
        assert "Total parameters" in architecture
        assert "\\n" not in architecture


def test_run_info_records_real_validation_and_locked_test() -> None:
    info = json.loads((RESULTS_DIR / "run_info.json").read_text(encoding="utf-8"))
    assert info["deep_learning_framework"] == "PyTorch"
    assert info["validation_rows"] == 30
    assert info["validation_images"] == 10
    assert info["test_split_used"] is False
    assert info["test_evaluation_permitted"] is False
