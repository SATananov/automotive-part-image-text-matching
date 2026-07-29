from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import (
    exact_grouped_paired_randomization,
    grouped_paired_randomization,
    monte_carlo_grouped_paired_randomization,
)


def _prediction_table(groups: int = 3) -> pd.DataFrame:
    rows = []
    for model_slug in ("left", "right"):
        for group_index in range(groups):
            image_id = f"image_{group_index:03d}"
            for row_index in range(3):
                left_correct = row_index <= group_index % 3
                right_correct = row_index == 0
                rows.append({
                    "sample_id": f"{image_id}_{row_index}",
                    "image_id": image_id,
                    "model_slug": model_slug,
                    "is_correct": left_correct if model_slug == "left" else right_correct,
                })
    return pd.DataFrame(rows)


def test_exact_grouped_randomization_uses_images_as_units() -> None:
    result = exact_grouped_paired_randomization(_prediction_table(), "left", "right")
    assert result["method"] == "exact_image_group_sign_flip"
    assert result["independent_groups"] == 3
    assert result["paired_rows"] == 9
    assert result["randomization_assignments"] == 8
    assert 0 <= result["grouped_two_sided_p_value"] <= 1


def test_dispatcher_uses_monte_carlo_for_dataset_v2_scale() -> None:
    table = _prediction_table(groups=60)
    first = grouped_paired_randomization(table, "left", "right", monte_carlo_repeats=5000)
    second = grouped_paired_randomization(table, "left", "right", monte_carlo_repeats=5000)
    assert first == second
    assert first["method"] == "monte_carlo_image_group_sign_flip"
    assert first["independent_groups"] == 60
    assert first["randomization_assignments"] == 5000
    assert first["randomization_seed"] == 42
    assert 0 < first["grouped_two_sided_p_value"] <= 1


def test_monte_carlo_rejects_nonpositive_repeat_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        monte_carlo_grouped_paired_randomization(
            _prediction_table(groups=21), "left", "right", repeats=0
        )


def test_grouped_randomization_rejects_group_mismatch() -> None:
    table = _prediction_table()
    mask = table["model_slug"].eq("right") & table["sample_id"].eq("image_000_0")
    table.loc[mask, "image_id"] = "different_image"
    with pytest.raises(ValueError, match="group identity"):
        exact_grouped_paired_randomization(table, "left", "right")


def test_grouped_randomization_requires_multiple_groups() -> None:
    table = _prediction_table(groups=1)
    with pytest.raises(ValueError, match="At least two"):
        exact_grouped_paired_randomization(table, "left", "right")
