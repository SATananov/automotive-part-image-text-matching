from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from src.dataset_config import (
    LABELS,
    METADATA_COLUMNS,
    PART_CATEGORIES,
)
from src.external_dataset_integration_config import (
    APPROVED_EXTERNAL_CATALOG_COLUMNS,
    APPROVED_EXTERNAL_CATALOG_PATH,
    CATEGORY_DESCRIPTIONS,
    CATEGORY_TO_FAMILY,
    DEVELOPMENT_TEST_PATH,
    DEVELOPMENT_TRAIN_PATH,
    DEVELOPMENT_VALIDATION_PATH,
    EXTERNAL_APPROVED_PER_CATEGORY,
    EXTERNAL_INTEGRATION_JSON_PATH,
    EXTERNAL_INTEGRATION_MARKDOWN_PATH,
    EXTERNAL_METADATA_PATH,
    EXTERNAL_SOURCE_NAME,
    EXTERNAL_SPLIT_MANIFEST_PATH,
    EXTERNAL_TEST_GROUPS_PER_CATEGORY,
    EXTERNAL_TEST_PATH,
    EXTERNAL_TRAIN_GROUPS_PER_CATEGORY,
    EXTERNAL_TRAIN_PATH,
    EXTERNAL_VALIDATION_GROUPS_PER_CATEGORY,
    EXTERNAL_VALIDATION_PATH,
    INTEGRATED_SPLIT_MANIFEST_PATH,
    INTEGRATED_TEST_LOCK_PATH,
    INTEGRATED_TEST_PATH,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
    MISMATCH_CATEGORY,
    OPEN_LICENSE_MANIFEST_PATH,
    OPEN_LICENSE_REVIEW_PATH,
    PARTIAL_CATEGORY,
    PROJECT_ROOT,
    SPLIT_MANIFEST_COLUMNS,
)
from src.open_license_dataset_config import (
    OPEN_LICENSE_MANIFEST_COLUMNS,
    OPEN_LICENSE_REVIEW_COLUMNS,
)
from src.validate_open_license_dataset import (
    validate_open_license_dataset,
)


class ExternalDatasetIntegrationError(RuntimeError):
    pass


def project_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(
            PROJECT_ROOT.resolve()
        ).as_posix()
    except ValueError as error:
        raise ExternalDatasetIntegrationError(
            f"Path is outside the project root: {path}."
        ) from error


def read_csv_rows(
    path: Path,
    expected_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    if not path.is_file():
        raise ExternalDatasetIntegrationError(
            f"Required CSV file is missing: {path}."
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != expected_columns:
            raise ExternalDatasetIntegrationError(
                f"Unexpected CSV schema in {path}."
            )
        return [
            {
                column: str(row.get(column, "")).strip()
                for column in expected_columns
            }
            for row in reader
        ]


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ExternalDatasetIntegrationError(
            f"Required metadata file is missing: {path}."
        )

    dataframe = pd.read_csv(path, dtype=str).fillna("")
    if tuple(dataframe.columns) != METADATA_COLUMNS:
        raise ExternalDatasetIntegrationError(
            f"Unexpected metadata schema in {path}."
        )
    return dataframe


def safe_identifier(value: str) -> str:
    normalized = "".join(
        character.lower()
        if character.isalnum()
        else "_"
        for character in value.strip()
    )
    normalized = "_".join(
        part
        for part in normalized.split("_")
        if part
    )
    if not normalized:
        raise ExternalDatasetIntegrationError(
            "Cannot build an identifier from a blank value."
        )
    return normalized


def build_approved_catalog(
    manifest_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
) -> pd.DataFrame:
    manifest_by_id = {
        row["asset_id"]: row
        for row in manifest_rows
        if row["asset_id"]
    }
    review_by_id = {
        row["asset_id"]: row
        for row in review_rows
        if row["asset_id"]
    }

    if len(manifest_by_id) != len(manifest_rows):
        raise ExternalDatasetIntegrationError(
            "The open-license manifest contains duplicate or blank asset IDs."
        )
    if len(review_by_id) != len(review_rows):
        raise ExternalDatasetIntegrationError(
            "The review workbook contains duplicate or blank asset IDs."
        )
    if set(manifest_by_id) != set(review_by_id):
        raise ExternalDatasetIntegrationError(
            "Manifest and review workbook asset IDs differ."
        )

    approved_records: list[dict[str, str]] = []
    category_counts = {
        category: 0
        for category in PART_CATEGORIES
    }

    for asset_id in sorted(manifest_by_id):
        review = review_by_id[asset_id]
        decision = review["operator_decision"].lower()
        if decision != "approved":
            continue

        manifest = manifest_by_id[asset_id]
        category = manifest["part_category"]
        if category not in category_counts:
            raise ExternalDatasetIntegrationError(
                f"Unsupported approved category: {category}."
            )

        page_id = safe_identifier(manifest["commons_page_id"])
        category_id = safe_identifier(category)
        image_id = f"external_image_{category_id}_{page_id}"
        group_id = f"external_group_{category_id}_{page_id}"

        approved_records.append(
            {
                "asset_id": asset_id,
                "image_id": image_id,
                "part_group_id": group_id,
                "part_family": manifest["part_family"],
                "part_category": category,
                "image_path": manifest["local_path"],
                "commons_page_id": manifest["commons_page_id"],
                "commons_title": manifest["commons_title"],
                "description_url": manifest["description_url"],
                "author": manifest["author"],
                "credit": manifest["credit"],
                "license_short_name": (
                    manifest["license_short_name"]
                ),
                "license_url": manifest["license_url"],
                "sha256": manifest["sha256"],
                "width": manifest["width"],
                "height": manifest["height"],
                "format": manifest["format"],
                "source": EXTERNAL_SOURCE_NAME,
            }
        )
        category_counts[category] += 1

    expected_total = (
        len(PART_CATEGORIES)
        * EXTERNAL_APPROVED_PER_CATEGORY
    )
    if len(approved_records) != expected_total:
        raise ExternalDatasetIntegrationError(
            f"Expected {expected_total} approved external images, "
            f"found {len(approved_records)}."
        )

    for category, count in category_counts.items():
        if count != EXTERNAL_APPROVED_PER_CATEGORY:
            raise ExternalDatasetIntegrationError(
                f"Category '{category}' contains {count} approved images; "
                f"expected {EXTERNAL_APPROVED_PER_CATEGORY}."
            )

    catalog = pd.DataFrame(
        approved_records,
        columns=APPROVED_EXTERNAL_CATALOG_COLUMNS,
    )

    if not catalog["image_id"].is_unique:
        raise ExternalDatasetIntegrationError(
            "Approved external image IDs are not unique."
        )
    if not catalog["part_group_id"].is_unique:
        raise ExternalDatasetIntegrationError(
            "Approved external part-group IDs are not unique."
        )
    if not catalog["sha256"].is_unique:
        raise ExternalDatasetIntegrationError(
            "Approved external image hashes are not unique."
        )

    return (
        catalog
        .sort_values(
            by=[
                "part_category",
                "part_group_id",
            ]
        )
        .reset_index(drop=True)
    )


def description_category_for_label(
    part_category: str,
    label: str,
) -> str:
    if label == "MATCH":
        return part_category
    if label == "PARTIAL_MATCH":
        return PARTIAL_CATEGORY[part_category]
    if label == "MISMATCH":
        return MISMATCH_CATEGORY[part_category]
    raise ExternalDatasetIntegrationError(
        f"Unsupported label: {label}."
    )


def build_external_metadata(
    approved_catalog: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, str]] = []

    for row in approved_catalog.to_dict(orient="records"):
        category = str(row["part_category"])
        page_id = safe_identifier(str(row["commons_page_id"]))

        for label in LABELS:
            description_category = description_category_for_label(
                category,
                label,
            )
            records.append(
                {
                    "sample_id": (
                        f"external_sample_{category}_{page_id}_"
                        f"{label.lower()}"
                    ),
                    "image_id": str(row["image_id"]),
                    "part_group_id": str(row["part_group_id"]),
                    "image_path": str(row["image_path"]),
                    "part_family": str(row["part_family"]),
                    "part_category": category,
                    "description": CATEGORY_DESCRIPTIONS[
                        description_category
                    ],
                    "label": label,
                    "source": EXTERNAL_SOURCE_NAME,
                }
            )

    dataframe = pd.DataFrame(
        records,
        columns=METADATA_COLUMNS,
    )

    if len(dataframe) != len(approved_catalog) * len(LABELS):
        raise ExternalDatasetIntegrationError(
            "External sample generation produced an unexpected row count."
        )
    if not dataframe["sample_id"].is_unique:
        raise ExternalDatasetIntegrationError(
            "External sample IDs are not unique."
        )

    labels_by_image = (
        dataframe
        .groupby("image_id")["label"]
        .apply(set)
    )
    for image_id, image_labels in labels_by_image.items():
        if image_labels != set(LABELS):
            raise ExternalDatasetIntegrationError(
                f"Image {image_id} does not contain all three labels."
            )

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


def split_external_metadata(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    split_groups: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    expected_groups = (
        EXTERNAL_TRAIN_GROUPS_PER_CATEGORY
        + EXTERNAL_VALIDATION_GROUPS_PER_CATEGORY
        + EXTERNAL_TEST_GROUPS_PER_CATEGORY
    )

    for category in PART_CATEGORIES:
        groups = sorted(
            dataframe.loc[
                dataframe["part_category"].eq(category),
                "part_group_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        if len(groups) != expected_groups:
            raise ExternalDatasetIntegrationError(
                f"Category '{category}' contains {len(groups)} groups; "
                f"expected {expected_groups}."
            )

        train_end = EXTERNAL_TRAIN_GROUPS_PER_CATEGORY
        validation_end = (
            train_end
            + EXTERNAL_VALIDATION_GROUPS_PER_CATEGORY
        )

        split_groups["train"].extend(
            groups[:train_end]
        )
        split_groups["validation"].extend(
            groups[train_end:validation_end]
        )
        split_groups["test"].extend(
            groups[validation_end:]
        )

    splits: dict[str, pd.DataFrame] = {}
    for split_name, groups in split_groups.items():
        splits[split_name] = (
            dataframe[
                dataframe["part_group_id"].isin(groups)
            ]
            .copy()
            .sort_values(
                by=[
                    "part_group_id",
                    "image_id",
                    "label",
                ]
            )
            .reset_index(drop=True)
        )

    validate_split_integrity(
        original_dataframe=dataframe,
        splits=splits,
    )
    return splits


def validate_split_integrity(
    original_dataframe: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
) -> None:
    required_names = {
        "train",
        "validation",
        "test",
    }
    if set(splits) != required_names:
        raise ExternalDatasetIntegrationError(
            "External split names are not complete."
        )

    group_sets = {
        name: set(dataframe["part_group_id"])
        for name, dataframe in splits.items()
    }

    if group_sets["train"] & group_sets["validation"]:
        raise ExternalDatasetIntegrationError(
            "Train and validation part groups overlap."
        )
    if group_sets["train"] & group_sets["test"]:
        raise ExternalDatasetIntegrationError(
            "Train and test part groups overlap."
        )
    if group_sets["validation"] & group_sets["test"]:
        raise ExternalDatasetIntegrationError(
            "Validation and test part groups overlap."
        )

    assigned_groups = set().union(*group_sets.values())
    original_groups = set(original_dataframe["part_group_id"])
    if assigned_groups != original_groups:
        raise ExternalDatasetIntegrationError(
            "Not all external part groups were assigned."
        )

    assigned_samples: set[str] = set()
    for split_name, dataframe in splits.items():
        split_samples = set(dataframe["sample_id"])
        if assigned_samples & split_samples:
            raise ExternalDatasetIntegrationError(
                f"Duplicate sample assignment in {split_name}."
            )
        assigned_samples.update(split_samples)

        if set(dataframe["part_category"]) != set(PART_CATEGORIES):
            raise ExternalDatasetIntegrationError(
                f"{split_name} does not contain all categories."
            )

        label_counts = [
            int(dataframe["label"].eq(label).sum())
            for label in LABELS
        ]
        if len(set(label_counts)) != 1:
            raise ExternalDatasetIntegrationError(
                f"{split_name} is not label-balanced."
            )

        labels_by_image = (
            dataframe
            .groupby("image_id")["label"]
            .apply(set)
        )
        for image_id, image_labels in labels_by_image.items():
            if image_labels != set(LABELS):
                raise ExternalDatasetIntegrationError(
                    f"Image {image_id} lost a label in {split_name}."
                )

    if assigned_samples != set(original_dataframe["sample_id"]):
        raise ExternalDatasetIntegrationError(
            "Not all external samples were assigned exactly once."
        )


def combine_split(
    development_dataframe: pd.DataFrame,
    external_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if set(development_dataframe["sample_id"]) & set(
        external_dataframe["sample_id"]
    ):
        raise ExternalDatasetIntegrationError(
            "Development and external sample IDs overlap."
        )
    if set(development_dataframe["image_id"]) & set(
        external_dataframe["image_id"]
    ):
        raise ExternalDatasetIntegrationError(
            "Development and external image IDs overlap."
        )
    if set(development_dataframe["part_group_id"]) & set(
        external_dataframe["part_group_id"]
    ):
        raise ExternalDatasetIntegrationError(
            "Development and external group IDs overlap."
        )

    combined = pd.concat(
        [
            development_dataframe,
            external_dataframe,
        ],
        ignore_index=True,
    )

    if not combined["sample_id"].is_unique:
        raise ExternalDatasetIntegrationError(
            "Combined sample IDs are not unique."
        )

    return (
        combined
        .sort_values(
            by=[
                "part_group_id",
                "image_id",
                "label",
            ]
        )
        .reset_index(drop=True)
    )


def create_split_manifest(
    splits: dict[str, pd.DataFrame],
    *,
    dataset_origin: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        dataframe = splits[split_name]
        summary = (
            dataframe
            .groupby(
                [
                    "part_group_id",
                    "image_id",
                    "part_family",
                    "part_category",
                    "source",
                ],
                as_index=False,
            )
            .agg(
                image_count=("image_id", "nunique"),
                sample_count=("sample_id", "count"),
            )
        )
        summary.insert(
            0,
            "split",
            split_name,
        )
        summary.insert(
            0,
            "dataset_origin",
            dataset_origin,
        )
        parts.append(summary)

    manifest = pd.concat(
        parts,
        ignore_index=True,
    )
    manifest = manifest.loc[
        :,
        SPLIT_MANIFEST_COLUMNS,
    ]

    return (
        manifest
        .sort_values(
            by=[
                "split",
                "part_category",
                "part_group_id",
            ]
        )
        .reset_index(drop=True)
    )


def create_integrated_manifest(
    integrated_splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        dataframe = integrated_splits[split_name].copy()
        dataframe["dataset_origin"] = dataframe["source"].map(
            lambda source: (
                "open_license_external"
                if source == EXTERNAL_SOURCE_NAME
                else "development"
            )
        )

        summary = (
            dataframe
            .groupby(
                [
                    "dataset_origin",
                    "part_group_id",
                    "image_id",
                    "part_family",
                    "part_category",
                    "source",
                ],
                as_index=False,
            )
            .agg(
                image_count=("image_id", "nunique"),
                sample_count=("sample_id", "count"),
            )
        )
        summary.insert(
            1,
            "split",
            split_name,
        )
        parts.append(summary)

    manifest = pd.concat(
        parts,
        ignore_index=True,
    )
    manifest = manifest.loc[
        :,
        SPLIT_MANIFEST_COLUMNS,
    ]

    return (
        manifest
        .sort_values(
            by=[
                "split",
                "dataset_origin",
                "part_category",
                "part_group_id",
            ]
        )
        .reset_index(drop=True)
    )


def dataframe_csv_bytes(
    dataframe: pd.DataFrame,
) -> bytes:
    return dataframe.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def summarize_dataframe(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "samples": int(len(dataframe)),
        "images": int(dataframe["image_id"].nunique()),
        "part_groups": int(
            dataframe["part_group_id"].nunique()
        ),
        "categories": int(
            dataframe["part_category"].nunique()
        ),
        "label_distribution": {
            label: int(dataframe["label"].eq(label).sum())
            for label in LABELS
        },
        "source_distribution": {
            str(source): int(count)
            for source, count in (
                dataframe["source"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
    }


def build_integration_report(
    approved_catalog: pd.DataFrame,
    external_metadata: pd.DataFrame,
    external_splits: dict[str, pd.DataFrame],
    integrated_splits: dict[str, pd.DataFrame],
    *,
    external_test_sha256: str,
    integrated_test_sha256: str,
) -> dict[str, Any]:
    return {
        "status": "PASS",
        "readiness": "READY_FOR_TRAINING_VALIDATION",
        "approved_external_images": int(
            len(approved_catalog)
        ),
        "external_samples": int(
            len(external_metadata)
        ),
        "approved_per_category": {
            category: int(
                approved_catalog[
                    "part_category"
                ].eq(category).sum()
            )
            for category in PART_CATEGORIES
        },
        "external_splits": {
            name: summarize_dataframe(dataframe)
            for name, dataframe in external_splits.items()
        },
        "integrated_splits": {
            name: summarize_dataframe(dataframe)
            for name, dataframe in integrated_splits.items()
        },
        "group_overlap": {
            "external_train_validation": 0,
            "external_train_test": 0,
            "external_validation_test": 0,
            "integrated_train_validation": 0,
            "integrated_train_test": 0,
            "integrated_validation_test": 0,
        },
        "test_lock": {
            "test_locked": True,
            "test_evaluation_permitted": False,
            "hash_normalization": "utf-8-lf",
            "external_test_sha256": external_test_sha256,
            "integrated_test_sha256": integrated_test_sha256,
            "training_inputs": [
                project_relative_path(INTEGRATED_TRAIN_PATH),
                project_relative_path(INTEGRATED_VALIDATION_PATH),
            ],
            "locked_test_paths": [
                project_relative_path(EXTERNAL_TEST_PATH),
                project_relative_path(INTEGRATED_TEST_PATH),
            ],
        },
    }


def render_integration_markdown(
    report: dict[str, Any],
) -> str:
    lines = [
        "# Step 010.2 — External Dataset Integration",
        "",
        f"- Status: **{report['status']}**",
        f"- Readiness: **{report['readiness']}**",
        (
            "- Approved external images: "
            f"**{report['approved_external_images']}**"
        ),
        f"- External samples: **{report['external_samples']}**",
        "- Grouping column: `part_group_id`",
        "- Test evaluation permitted: **no**",
        "",
        "## External grouped split",
        "",
        "| Split | Samples | Images | Part groups |",
        "|---|---:|---:|---:|",
    ]

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        summary = report["external_splits"][split_name]
        lines.append(
            f"| {split_name.title()} "
            f"| {summary['samples']} "
            f"| {summary['images']} "
            f"| {summary['part_groups']} |"
        )

    lines.extend(
        [
            "",
            "## Integrated development + external split",
            "",
            "| Split | Samples | Images | Part groups |",
            "|---|---:|---:|---:|",
        ]
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        summary = report["integrated_splits"][split_name]
        lines.append(
            f"| {split_name.title()} "
            f"| {summary['samples']} "
            f"| {summary['images']} "
            f"| {summary['part_groups']} |"
        )

    lines.extend(
        [
            "",
            "## Leakage and test-lock policy",
            "",
            "- Train and validation group overlap: 0",
            "- Train and test group overlap: 0",
            "- Validation and test group overlap: 0",
            "- External groups use a dedicated `external_group_` namespace.",
            "- Training-ready inputs include only integrated train and "
            "validation CSV files.",
            "- External and integrated test CSV files are fingerprinted and "
            "locked.",
            "- No model is trained or evaluated by Step 010.2.",
            "",
            "The 29 rejected open-license candidates remain in the Step 010.1 "
            "audit workbook but are excluded from every integrated dataset.",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def snapshot_files(
    paths: tuple[Path, ...],
) -> dict[Path, bytes | None]:
    return {
        path: path.read_bytes() if path.is_file() else None
        for path in paths
    }


def restore_files(
    snapshots: dict[Path, bytes | None],
) -> None:
    for path, content in snapshots.items():
        if content is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def atomic_write_bytes(
    path: Path,
    content: bytes,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def integrate_external_dataset() -> dict[str, Any]:
    open_license_report = validate_open_license_dataset()
    if (
        open_license_report.get("status") != "PASS"
        or open_license_report.get("readiness")
        != "READY_FOR_EXTERNAL_DATASET"
    ):
        raise ExternalDatasetIntegrationError(
            "Step 010.1 must be READY_FOR_EXTERNAL_DATASET "
            "before Step 010.2 integration."
        )

    manifest_rows = read_csv_rows(
        OPEN_LICENSE_MANIFEST_PATH,
        OPEN_LICENSE_MANIFEST_COLUMNS,
    )
    review_rows = read_csv_rows(
        OPEN_LICENSE_REVIEW_PATH,
        OPEN_LICENSE_REVIEW_COLUMNS,
    )

    approved_catalog = build_approved_catalog(
        manifest_rows,
        review_rows,
    )
    external_metadata = build_external_metadata(
        approved_catalog
    )
    external_splits = split_external_metadata(
        external_metadata
    )

    development_splits = {
        "train": load_metadata(DEVELOPMENT_TRAIN_PATH),
        "validation": load_metadata(
            DEVELOPMENT_VALIDATION_PATH
        ),
        "test": load_metadata(DEVELOPMENT_TEST_PATH),
    }

    integrated_splits = {
        split_name: combine_split(
            development_splits[split_name],
            external_splits[split_name],
        )
        for split_name in (
            "train",
            "validation",
            "test",
        )
    }

    validate_split_integrity(
        original_dataframe=pd.concat(
            list(integrated_splits.values()),
            ignore_index=True,
        ),
        splits=integrated_splits,
    )

    external_manifest = create_split_manifest(
        external_splits,
        dataset_origin="open_license_external",
    )
    integrated_manifest = create_integrated_manifest(
        integrated_splits
    )

    output_bytes: dict[Path, bytes] = {
        APPROVED_EXTERNAL_CATALOG_PATH: dataframe_csv_bytes(
            approved_catalog
        ),
        EXTERNAL_METADATA_PATH: dataframe_csv_bytes(
            external_metadata
        ),
        EXTERNAL_TRAIN_PATH: dataframe_csv_bytes(
            external_splits["train"]
        ),
        EXTERNAL_VALIDATION_PATH: dataframe_csv_bytes(
            external_splits["validation"]
        ),
        EXTERNAL_TEST_PATH: dataframe_csv_bytes(
            external_splits["test"]
        ),
        EXTERNAL_SPLIT_MANIFEST_PATH: dataframe_csv_bytes(
            external_manifest
        ),
        INTEGRATED_TRAIN_PATH: dataframe_csv_bytes(
            integrated_splits["train"]
        ),
        INTEGRATED_VALIDATION_PATH: dataframe_csv_bytes(
            integrated_splits["validation"]
        ),
        INTEGRATED_TEST_PATH: dataframe_csv_bytes(
            integrated_splits["test"]
        ),
        INTEGRATED_SPLIT_MANIFEST_PATH: dataframe_csv_bytes(
            integrated_manifest
        ),
    }

    external_test_sha256 = sha256_bytes(
        output_bytes[EXTERNAL_TEST_PATH]
    )
    integrated_test_sha256 = sha256_bytes(
        output_bytes[INTEGRATED_TEST_PATH]
    )

    report = build_integration_report(
        approved_catalog=approved_catalog,
        external_metadata=external_metadata,
        external_splits=external_splits,
        integrated_splits=integrated_splits,
        external_test_sha256=external_test_sha256,
        integrated_test_sha256=integrated_test_sha256,
    )

    test_lock = {
        **report["test_lock"],
        "external_test_rows": int(
            len(external_splits["test"])
        ),
        "external_test_groups": int(
            external_splits["test"][
                "part_group_id"
            ].nunique()
        ),
        "integrated_test_rows": int(
            len(integrated_splits["test"])
        ),
        "integrated_test_groups": int(
            integrated_splits["test"][
                "part_group_id"
            ].nunique()
        ),
        "policy": (
            "The test split remains locked until model selection "
            "and the final evaluation procedure are fixed."
        ),
    }

    output_bytes[INTEGRATED_TEST_LOCK_PATH] = (
        json.dumps(
            test_lock,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    output_bytes[EXTERNAL_INTEGRATION_JSON_PATH] = (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    output_bytes[EXTERNAL_INTEGRATION_MARKDOWN_PATH] = (
        render_integration_markdown(report).encode("utf-8")
    )

    output_paths = tuple(output_bytes)
    snapshots = snapshot_files(output_paths)

    try:
        for path, content in output_bytes.items():
            atomic_write_bytes(path, content)
    except Exception:
        restore_files(snapshots)
        raise

    return report


def main() -> None:
    try:
        report = integrate_external_dataset()
    except Exception as error:
        print("Step 010.2 external dataset integration")
        print("- Status: FAIL")
        print(f"- Error: {error}")
        raise SystemExit(1) from error

    print("Step 010.2 external dataset integration")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(
        "- Approved external images: "
        f"{report['approved_external_images']}"
    )
    print(
        "- External samples: "
        f"{report['external_samples']}"
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        external = report["external_splits"][split_name]
        integrated = report["integrated_splits"][split_name]
        print(
            f"- {split_name}: "
            f"external={external['samples']} samples/"
            f"{external['part_groups']} groups, "
            f"integrated={integrated['samples']} samples/"
            f"{integrated['part_groups']} groups"
        )

    print("- Test evaluation permitted: no")
    print(
        "- Integrated train: "
        f"{project_relative_path(INTEGRATED_TRAIN_PATH)}"
    )
    print(
        "- Integrated validation: "
        f"{project_relative_path(INTEGRATED_VALIDATION_PATH)}"
    )
    print(
        "- Locked integrated test: "
        f"{project_relative_path(INTEGRATED_TEST_PATH)}"
    )


if __name__ == "__main__":
    main()
