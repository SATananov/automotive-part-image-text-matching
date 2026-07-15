from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.dataset_config import (
    LABELS,
    METADATA_COLUMNS,
    PART_CATEGORIES,
    PART_FAMILIES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "development" / "metadata.csv"

JSON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "development_dataset_validation.json"
)

MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "development_dataset_validation.md"
)


def validate_metadata_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []

    actual_columns = tuple(dataframe.columns)

    if actual_columns != METADATA_COLUMNS:
        errors.append(
            "Metadata columns do not match the expected schema."
        )

    return errors


def validate_missing_values(
    dataframe: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []

    missing_counts = dataframe.isna().sum()

    for column, count in missing_counts.items():
        if count > 0:
            errors.append(
                f"Column '{column}' contains {count} missing values."
            )

    empty_string_counts = (
        dataframe
        .astype(str)
        .apply(lambda column: column.str.strip().eq("").sum())
    )

    for column, count in empty_string_counts.items():
        if count > 0:
            errors.append(
                f"Column '{column}' contains {count} empty values."
            )

    return errors


def validate_identifiers(
    dataframe: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []

    duplicate_sample_ids = dataframe["sample_id"].duplicated().sum()

    if duplicate_sample_ids:
        errors.append(
            f"Found {duplicate_sample_ids} duplicate sample IDs."
        )

    image_label_counts = (
        dataframe
        .groupby("image_id")["label"]
        .nunique()
    )

    invalid_image_count = int(
        image_label_counts.ne(len(LABELS)).sum()
    )

    if invalid_image_count:
        errors.append(
            f"{invalid_image_count} images do not have all three labels."
        )

    return errors


def validate_labels_and_categories(
    dataframe: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []

    invalid_labels = (
        set(dataframe["label"])
        - set(LABELS)
    )

    if invalid_labels:
        errors.append(
            f"Invalid labels: {sorted(invalid_labels)}"
        )

    invalid_categories = (
        set(dataframe["part_category"])
        - set(PART_CATEGORIES)
    )

    if invalid_categories:
        errors.append(
            f"Invalid categories: {sorted(invalid_categories)}"
        )

    for row in dataframe.itertuples(index=False):
        if row.part_family not in PART_FAMILIES:
            errors.append(
                f"Invalid part family in sample {row.sample_id}: "
                f"{row.part_family}"
            )
            continue

        valid_family_categories = PART_FAMILIES[row.part_family]

        if row.part_category not in valid_family_categories:
            errors.append(
                f"Category-family mismatch in sample "
                f"{row.sample_id}."
            )

    return errors


def validate_images(
    dataframe: pd.DataFrame,
) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    image_information: dict[str, dict[str, object]] = {}

    unique_images = (
        dataframe[["image_id", "image_path"]]
        .drop_duplicates()
    )

    for row in unique_images.itertuples(index=False):
        image_path = PROJECT_ROOT / row.image_path

        if not image_path.is_file():
            errors.append(
                f"Missing image file: {row.image_path}"
            )
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                width, height = image.size

                image_information[row.image_id] = {
                    "path": row.image_path,
                    "format": image.format,
                    "mode": image.mode,
                    "width": width,
                    "height": height,
                }

                if image.format != "PNG":
                    errors.append(
                        f"Unexpected format for {row.image_id}: "
                        f"{image.format}"
                    )

                if image.mode != "RGB":
                    errors.append(
                        f"Unexpected mode for {row.image_id}: "
                        f"{image.mode}"
                    )

                if image.size != (224, 224):
                    errors.append(
                        f"Unexpected size for {row.image_id}: "
                        f"{image.size}"
                    )

        except (UnidentifiedImageError, OSError) as error:
            errors.append(
                f"Unreadable image {row.image_path}: {error}"
            )

    summary = {
        "checked_images": len(image_information),
        "formats": dict(
            Counter(
                information["format"]
                for information in image_information.values()
            )
        ),
        "modes": dict(
            Counter(
                information["mode"]
                for information in image_information.values()
            )
        ),
        "dimensions": dict(
            Counter(
                f"{information['width']}x{information['height']}"
                for information in image_information.values()
            )
        ),
    }

    return errors, summary


def build_report(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    validation_errors: list[str] = []

    validation_errors.extend(
        validate_metadata_columns(dataframe)
    )
    validation_errors.extend(
        validate_missing_values(dataframe)
    )
    validation_errors.extend(
        validate_identifiers(dataframe)
    )
    validation_errors.extend(
        validate_labels_and_categories(dataframe)
    )

    image_errors, image_summary = validate_images(dataframe)
    validation_errors.extend(image_errors)

    labels_by_image: dict[str, list[str]] = defaultdict(list)

    for row in dataframe.itertuples(index=False):
        labels_by_image[row.image_id].append(row.label)

    return {
        "status": "PASS" if not validation_errors else "FAIL",
        "metadata_path": str(METADATA_PATH),
        "sample_count": len(dataframe),
        "image_count": dataframe["image_id"].nunique(),
        "part_group_count": dataframe["part_group_id"].nunique(),
        "label_distribution": (
            dataframe["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "category_distribution": (
            dataframe["part_category"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
        "image_summary": image_summary,
        "errors": validation_errors,
    }


def write_reports(report: dict[str, object]) -> None:
    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    markdown_lines = [
        "# Development Dataset Validation",
        "",
        f"**Status:** {report['status']}",
        "",
        "## Dataset summary",
        "",
        f"- Samples: {report['sample_count']}",
        f"- Images: {report['image_count']}",
        f"- Physical part groups: {report['part_group_count']}",
        "",
        "## Label distribution",
        "",
    ]

    for label, count in report["label_distribution"].items():
        markdown_lines.append(f"- {label}: {count}")

    markdown_lines.extend(
        [
            "",
            "## Image validation",
            "",
            (
                "- Checked images: "
                f"{report['image_summary']['checked_images']}"
            ),
            (
                "- Formats: "
                f"{report['image_summary']['formats']}"
            ),
            (
                "- Modes: "
                f"{report['image_summary']['modes']}"
            ),
            (
                "- Dimensions: "
                f"{report['image_summary']['dimensions']}"
            ),
            "",
            "## Validation errors",
            "",
        ]
    )

    errors = report["errors"]

    if errors:
        for error in errors:
            markdown_lines.append(f"- {error}")
    else:
        markdown_lines.append("- No validation errors found.")

    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    dataframe = pd.read_csv(METADATA_PATH)

    report = build_report(dataframe)
    write_reports(report)

    print(f"Validation status: {report['status']}")
    print(f"Samples checked: {report['sample_count']}")
    print(f"Images checked: {report['image_count']}")
    print(f"Errors found: {len(report['errors'])}")
    print(f"Report: {MARKDOWN_REPORT_PATH}")

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()