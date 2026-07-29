from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import exact_grouped_paired_randomization


def _prediction_table() -> pd.DataFrame:
    rows = []
    left = {
        "image_1": [True, True, False],
        "image_2": [True, False, False],
        "image_3": [True, True, True],
    }
    right = {
        "image_1": [True, False, False],
        "image_2": [True, False, False],
        "image_3": [True, False, False],
    }
    for model_slug, values in (("left", left), ("right", right)):
        for image_id, correctness in values.items():
            for row_index, is_correct in enumerate(correctness):
                rows.append(
                    {
                        "sample_id": f"{image_id}_{row_index}",
                        "image_id": image_id,
                        "model_slug": model_slug,
                        "is_correct": is_correct,
                    }
                )
    return pd.DataFrame(rows)


def test_grouped_paired_randomization_uses_images_as_units() -> None:
    result = exact_grouped_paired_randomization(_prediction_table(), "left", "right")
    assert result["method"] == "exact_image_group_sign_flip"
    assert result["independent_groups"] == 3
    assert result["paired_rows"] == 9
    assert result["randomization_assignments"] == 8
    assert result["observed_correct_difference"] == 3
    assert result["left_better_groups"] == 2
    assert result["right_better_groups"] == 0
    assert result["tied_groups"] == 1
    assert result["grouped_exact_two_sided_p_value"] == 0.5


def test_grouped_paired_randomization_is_deterministic() -> None:
    table = _prediction_table()
    first = exact_grouped_paired_randomization(table, "left", "right")
    second = exact_grouped_paired_randomization(table, "left", "right")
    assert first == second


def test_grouped_paired_randomization_rejects_group_mismatch() -> None:
    table = _prediction_table()
    mask = table["model_slug"].eq("right") & table["sample_id"].eq("image_1_0")
    table.loc[mask, "image_id"] = "different_image"
    with pytest.raises(ValueError, match="group identity"):
        exact_grouped_paired_randomization(table, "left", "right")


def test_grouped_paired_randomization_requires_multiple_groups() -> None:
    table = _prediction_table()
    table = table[table["image_id"].eq("image_1")]
    with pytest.raises(ValueError, match="At least two"):
        exact_grouped_paired_randomization(table, "left", "right")
