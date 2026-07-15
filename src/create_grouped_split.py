from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.dataset_config import LABELS, METADATA_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "metadata.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_PATH = OUTPUT_DIRECTORY / "development_train.csv"
VALIDATION_PATH = OUTPUT_DIRECTORY / "development_validation.csv"
TEST_PATH = OUTPUT_DIRECTORY / "development_test.csv"
MANIFEST_PATH = OUTPUT_DIRECTORY / "development_split_manifest.csv"

JSON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "development_grouped_split.json"
)

MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "development_grouped_split.md"
)

RANDOM_STATE = 42

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20


def load_metadata() -> pd.DataFrame:
    dataframe = pd.read_csv(INPUT_METADATA_PATH)

    actual_columns = tuple(dataframe.columns)

    if actual_columns != METADATA_COLUMNS:
        raise ValueError(
            "The development metadata schema is not valid."
        )

    return dataframe


def sort_split(dataframe: pd.DataFrame) -> pd.DataFrame:
    return (
        dataframe
        .sort_values(
            by=[
                "part_group_id",
                "image_id",
                "label",
            ]
        )
        .reset_index(drop=True)
    )


def split_grouped_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
    )

    train_indices, remaining_indices = next(
        first_splitter.split(
            dataframe,
            groups=dataframe["part_group_id"],
        )
    )

    train_dataframe = dataframe.iloc[train_indices].copy()
    remaining_dataframe = dataframe.iloc[remaining_indices].copy()

    relative_test_ratio = (
        TEST_RATIO
        / (VALIDATION_RATIO + TEST_RATIO)
    )

    second_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=relative_test_ratio,
        random_state=RANDOM_STATE,
    )

    validation_indices, test_indices = next(
        second_splitter.split(
            remaining_dataframe,
            groups=remaining_dataframe["part_group_id"],
        )
    )

    validation_dataframe = (
        remaining_dataframe
        .iloc[validation_indices]
        .copy()
    )

    test_dataframe = (
        remaining_dataframe
        .iloc[test_indices]
        .copy()
    )

    return (
        sort_split(train_dataframe),
        sort_split(validation_dataframe),
        sort_split(test_dataframe),
    )


def validate_split_integrity(
    original_dataframe: pd.DataFrame,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    split_dataframes = {
        "train": train_dataframe,
        "validation": validation_dataframe,
        "test": test_dataframe,
    }

    group_sets = {
        split_name: set(dataframe["part_group_id"])
        for split_name, dataframe in split_dataframes.items()
    }

    if group_sets["train"] & group_sets["validation"]:
        raise ValueError(
            "Train and validation contain overlapping part groups."
        )

    if group_sets["train"] & group_sets["test"]:
        raise ValueError(
            "Train and test contain overlapping part groups."
        )

    if group_sets["validation"] & group_sets["test"]:
        raise ValueError(
            "Validation and test contain overlapping part groups."
        )

    assigned_groups = set().union(*group_sets.values())
    original_groups = set(original_dataframe["part_group_id"])

    if assigned_groups != original_groups:
        raise ValueError(
            "Not all part groups were assigned to a split."
        )

    assigned_sample_ids: set[str] = set()

    for split_name, dataframe in split_dataframes.items():
        split_sample_ids = set(dataframe["sample_id"])

        if assigned_sample_ids & split_sample_ids:
            raise ValueError(
                f"Duplicate samples detected in {split_name}."
            )

        assigned_sample_ids.update(split_sample_ids)

        labels_by_image = (
            dataframe
            .groupby("image_id")["label"]
            .apply(set)
        )

        for image_id, image_labels in labels_by_image.items():
            if image_labels != set(LABELS):
                raise ValueError(
                    f"Image {image_id} does not contain "
                    f"all three labels in {split_name}."
                )

    original_sample_ids = set(original_dataframe["sample_id"])

    if assigned_sample_ids != original_sample_ids:
        raise ValueError(
            "Not all samples were assigned exactly once."
        )


def create_manifest(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    manifest_parts: list[pd.DataFrame] = []

    split_dataframes = {
        "train": train_dataframe,
        "validation": validation_dataframe,
        "test": test_dataframe,
    }

    for split_name, dataframe in split_dataframes.items():
        group_summary = (
            dataframe
            .groupby(
                [
                    "part_group_id",
                    "part_family",
                    "part_category",
                ],
                as_index=False,
            )
            .agg(
                image_count=("image_id", "nunique"),
                sample_count=("sample_id", "count"),
            )
        )

        group_summary.insert(
            loc=0,
            column="split",
            value=split_name,
        )

        manifest_parts.append(group_summary)

    return (
        pd.concat(
            manifest_parts,
            ignore_index=True,
        )
        .sort_values(
            by=[
                "split",
                "part_category",
                "part_group_id",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_split(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    return {
        "samples": int(len(dataframe)),
        "images": int(dataframe["image_id"].nunique()),
        "part_groups": int(
            dataframe["part_group_id"].nunique()
        ),
        "label_distribution": {
            label: int(
                dataframe["label"].eq(label).sum()
            )
            for label in LABELS
        },
        "category_distribution": {
            category: int(count)
            for category, count in (
                dataframe["part_category"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
    }


def build_report(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> dict[str, object]:
    return {
        "status": "PASS",
        "random_state": RANDOM_STATE,
        "requested_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "splits": {
            "train": summarize_split(train_dataframe),
            "validation": summarize_split(
                validation_dataframe
            ),
            "test": summarize_split(test_dataframe),
        },
        "group_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
    }


def write_outputs(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_dataframe.to_csv(
        TRAIN_PATH,
        index=False,
    )

    validation_dataframe.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    test_dataframe.to_csv(
        TEST_PATH,
        index=False,
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )


def write_reports(
    report: dict[str, object],
) -> None:
    JSON_REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown_lines = [
        "# Development Grouped Split",
        "",
        f"**Status:** {report['status']}",
        "",
        f"- Random state: {report['random_state']}",
        "- Grouping column: `part_group_id`",
        "",
        "## Split summary",
        "",
        "| Split | Samples | Images | Part groups |",
        "|---|---:|---:|---:|",
    ]

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_summary = report["splits"][split_name]

        markdown_lines.append(
            f"| {split_name.title()} "
            f"| {split_summary['samples']} "
            f"| {split_summary['images']} "
            f"| {split_summary['part_groups']} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Label distribution",
            "",
            "| Split | MATCH | PARTIAL_MATCH | MISMATCH |",
            "|---|---:|---:|---:|",
        ]
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        label_distribution = (
            report["splits"][split_name][
                "label_distribution"
            ]
        )

        markdown_lines.append(
            f"| {split_name.title()} "
            f"| {label_distribution['MATCH']} "
            f"| {label_distribution['PARTIAL_MATCH']} "
            f"| {label_distribution['MISMATCH']} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Leakage check",
            "",
            "- Train and validation group overlap: 0",
            "- Train and test group overlap: 0",
            "- Validation and test group overlap: 0",
            "",
            "All rows belonging to the same physical part "
            "remain in one subset.",
            "",
            "This split is intended for development and "
            "pipeline testing.",
        ]
    )

    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    original_dataframe = load_metadata()

    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = split_grouped_dataframe(original_dataframe)

    validate_split_integrity(
        original_dataframe=original_dataframe,
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
    )

    manifest = create_manifest(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
    )

    report = build_report(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
    )

    write_outputs(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        test_dataframe=test_dataframe,
        manifest=manifest,
    )

    write_reports(report)

    print("Grouped split created successfully.")

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_summary = report["splits"][split_name]

        print(
            f"{split_name.title()}: "
            f"{split_summary['samples']} samples, "
            f"{split_summary['part_groups']} groups"
        )

    print("Part group overlap: 0")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()