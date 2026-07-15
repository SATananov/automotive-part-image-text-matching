from __future__ import annotations

import pandas as pd

from src.dataset_config import LABELS
from src.create_grouped_split import (
    load_metadata,
    split_grouped_dataframe,
)


def create_splits() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    original_dataframe = load_metadata()

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = split_grouped_dataframe(original_dataframe)

    return (
        original_dataframe,
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    )


def test_expected_group_counts() -> None:
    (
        _,
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = create_splits()

    assert train_dataframe["part_group_id"].nunique() == 12
    assert validation_dataframe["part_group_id"].nunique() == 4
    assert test_dataframe["part_group_id"].nunique() == 4


def test_part_groups_do_not_overlap() -> None:
    (
        _,
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = create_splits()

    train_groups = set(train_dataframe["part_group_id"])
    validation_groups = set(
        validation_dataframe["part_group_id"]
    )
    test_groups = set(test_dataframe["part_group_id"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_all_samples_are_assigned_once() -> None:
    (
        original_dataframe,
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = create_splits()

    combined_dataframe = pd.concat(
        [
            train_dataframe,
            validation_dataframe,
            test_dataframe,
        ],
        ignore_index=True,
    )

    assert len(combined_dataframe) == len(original_dataframe)

    assert set(combined_dataframe["sample_id"]) == set(
        original_dataframe["sample_id"]
    )

    assert combined_dataframe["sample_id"].is_unique


def test_each_image_keeps_all_labels() -> None:
    (
        _,
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = create_splits()

    for dataframe in (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ):
        labels_by_image = (
            dataframe
            .groupby("image_id")["label"]
            .apply(set)
        )

        for image_labels in labels_by_image:
            assert image_labels == set(LABELS)


def test_each_split_is_label_balanced() -> None:
    (
        _,
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = create_splits()

    for dataframe in (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ):
        label_counts = [
            int(dataframe["label"].eq(label).sum())
            for label in LABELS
        ]

        assert len(set(label_counts)) == 1


def test_grouped_split_is_reproducible() -> None:
    original_dataframe = load_metadata()

    first_split = split_grouped_dataframe(
        original_dataframe
    )

    second_split = split_grouped_dataframe(
        original_dataframe
    )

    for first_dataframe, second_dataframe in zip(
        first_split,
        second_split,
        strict=True,
    ):
        assert list(first_dataframe["sample_id"]) == list(
            second_dataframe["sample_id"]
        )