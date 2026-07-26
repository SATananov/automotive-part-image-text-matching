import pandas as pd

from src.data import PROJECT_ROOT


def test_multimodal_model_is_best_on_real_images() -> None:
    table = pd.read_csv(PROJECT_ROOT / "results/model_comparison.csv")
    best = table.sort_values("real_image_macro_f1", ascending=False).iloc[0]
    assert best["model"] == "Keras multimodal model"
    assert round(best["real_image_accuracy"], 4) == 0.4667
    assert round(best["real_image_macro_f1"], 4) == 0.4626


def test_saved_predictions_are_validation_only() -> None:
    predictions = pd.read_csv(PROJECT_ROOT / "results/validation_predictions.csv")
    validation = pd.read_csv(PROJECT_ROOT / "data/validation.csv")
    assert set(predictions["sample_id"]) == set(validation["sample_id"])
    assert len(predictions) == 6 * len(validation)


def test_result_table_contains_all_models() -> None:
    table = pd.read_csv(PROJECT_ROOT / "results/model_comparison.csv")
    assert len(table) == 6
    assert set(table["modality"]) == {"none", "text", "image", "image + text"}
