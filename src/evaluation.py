from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

MAX_EXACT_GROUPS = 20


def _coerce_correct(values: pd.Series) -> pd.Series:
    """Return a strict boolean correctness series from in-memory or CSV data."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        examples = sorted(normalized[invalid].unique())[:3]
        raise ValueError(f"Invalid is_correct values: {examples}")
    return normalized.eq("true")


def exact_grouped_paired_randomization(
    predictions: pd.DataFrame,
    left_slug: str,
    right_slug: str,
    *,
    group_column: str = "image_id",
) -> dict[str, object]:
    """Compare two models with an exact sign-flip test over independent groups.

    Each image contributes several dependent image-text rows. Swapping the two
    model labels for a complete image group is the correct paired permutation
    unit; swapping individual rows would overstate the effective sample size.

    The test statistic is the total difference in correct predictions across
    all rows. For ``G`` independent image groups, all ``2**G`` group-level label
    swaps are enumerated. The returned p-value is two-sided.
    """
    required = {"sample_id", "model_slug", "is_correct", group_column}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")

    left = predictions[predictions["model_slug"].eq(left_slug)].set_index("sample_id")
    right = predictions[predictions["model_slug"].eq(right_slug)].set_index("sample_id")
    if left.empty or right.empty:
        raise ValueError("Both compared models must have prediction rows")
    if set(left.index) != set(right.index):
        raise ValueError("Paired models do not cover the same validation rows")

    left = left.sort_index()
    right = right.loc[left.index]
    if not left[group_column].astype(str).equals(right[group_column].astype(str)):
        raise ValueError("Paired prediction rows disagree on group identity")

    left_correct = _coerce_correct(left["is_correct"])
    right_correct = _coerce_correct(right["is_correct"])

    grouped = pd.DataFrame(
        {
            "group": left[group_column].astype(str),
            "left_correct": left_correct.astype(int),
            "right_correct": right_correct.astype(int),
        },
        index=left.index,
    ).groupby("group", sort=True)[["left_correct", "right_correct"]].sum()

    group_count = len(grouped)
    if group_count < 2:
        raise ValueError("At least two independent groups are required")
    if group_count > MAX_EXACT_GROUPS:
        raise ValueError(
            f"Exact enumeration supports at most {MAX_EXACT_GROUPS} groups; got {group_count}"
        )

    group_differences = (
        grouped["left_correct"] - grouped["right_correct"]
    ).to_numpy(dtype=np.int64)
    observed_difference = int(group_differences.sum())
    observed_absolute = abs(observed_difference)

    assignments = 2**group_count
    extreme = 0
    for signs in product((-1, 1), repeat=group_count):
        randomized = int(np.dot(group_differences, np.asarray(signs, dtype=np.int64)))
        if abs(randomized) >= observed_absolute:
            extreme += 1
    p_value = float(extreme / assignments)

    left_row = left_correct.to_numpy(dtype=bool)
    right_row = right_correct.to_numpy(dtype=bool)
    left_only_rows = int(np.sum(left_row & ~right_row))
    right_only_rows = int(np.sum(~left_row & right_row))
    total_rows = len(left)

    return {
        "left_model_slug": left_slug,
        "right_model_slug": right_slug,
        "method": "exact_image_group_sign_flip",
        "group_column": group_column,
        "independent_groups": int(group_count),
        "paired_rows": int(total_rows),
        "left_correct_total": int(left_correct.sum()),
        "right_correct_total": int(right_correct.sum()),
        "observed_correct_difference": observed_difference,
        "observed_accuracy_difference": float(observed_difference / total_rows),
        "left_better_groups": int(np.sum(group_differences > 0)),
        "right_better_groups": int(np.sum(group_differences < 0)),
        "tied_groups": int(np.sum(group_differences == 0)),
        "left_correct_right_wrong_rows": left_only_rows,
        "left_wrong_right_correct_rows": right_only_rows,
        "row_discordant_predictions": left_only_rows + right_only_rows,
        "randomization_assignments": int(assignments),
        "grouped_exact_two_sided_p_value": p_value,
    }
