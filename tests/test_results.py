import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import accuracy_score, f1_score

from src.data import PROJECT_ROOT


def test_model_comparison_is_sorted_by_reported_scores() -> None:
    table = pd.read_csv(PROJECT_ROOT / "results/model_comparison.csv")
    expected = table.sort_values(
        ["real_image_macro_f1", "full_validation_macro_f1"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    assert_frame_equal(
        table.reset_index(drop=True),
        expected,
        check_dtype=False,
    )


def test_saved_predictions_are_validation_only() -> None:
    predictions = pd.read_csv(PROJECT_ROOT / "results/validation_predictions.csv")
    validation = pd.read_csv(PROJECT_ROOT / "data/validation.csv")
    assert set(predictions["sample_id"]) == set(validation["sample_id"])
    assert len(predictions) == 7 * len(validation)


def test_result_table_contains_all_models() -> None:
    table = pd.read_csv(PROJECT_ROOT / "results/model_comparison.csv")
    assert len(table) == 7
    assert set(table["modality"]) == {"none", "text", "image", "image + text"}
    assert "Image + text Logistic Regression" in set(table["model"])


def test_saved_metrics_match_saved_predictions() -> None:
    table = pd.read_csv(PROJECT_ROOT / "results/model_comparison.csv").set_index("model")
    predictions = pd.read_csv(PROJECT_ROOT / "results/validation_predictions.csv")

    for model, group in predictions.groupby("model"):
        row = table.loc[model]
        assert abs(accuracy_score(group["true_label"], group["predicted_label"]) - row["full_validation_accuracy"]) < 1e-12
        assert abs(
            f1_score(group["true_label"], group["predicted_label"], average="macro", zero_division=0)
            - row["full_validation_macro_f1"]
        ) < 1e-12

        real = group[group["source"].eq("wikimedia")]
        assert abs(accuracy_score(real["true_label"], real["predicted_label"]) - row["real_image_accuracy"]) < 1e-12
        assert abs(
            f1_score(real["true_label"], real["predicted_label"], average="macro", zero_division=0)
            - row["real_image_macro_f1"]
        ) < 1e-12


def test_multimodal_prediction_file_is_derived_from_all_predictions() -> None:
    all_predictions = pd.read_csv(PROJECT_ROOT / "results/validation_predictions.csv")
    expected = all_predictions[all_predictions["model_slug"].eq("keras_multimodal")].reset_index(drop=True)
    saved = pd.read_csv(PROJECT_ROOT / "results/multimodal_validation_predictions.csv")
    assert_frame_equal(saved, expected, check_dtype=False)


def test_multimodal_category_table_matches_predictions() -> None:
    predictions = pd.read_csv(PROJECT_ROOT / "results/multimodal_validation_predictions.csv")
    saved = pd.read_csv(PROJECT_ROOT / "results/multimodal_per_category.csv").set_index("part_category")

    for category, group in predictions.groupby("part_category"):
        assert saved.loc[category, "samples"] == len(group)
        assert abs(
            saved.loc[category, "accuracy"]
            - accuracy_score(group["true_label"], group["predicted_label"])
        ) < 1e-12
        assert abs(
            saved.loc[category, "macro_f1"]
            - f1_score(group["true_label"], group["predicted_label"], average="macro", zero_division=0)
        ) < 1e-12
