from __future__ import annotations

import hashlib
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
    DEVELOPMENT_TEST_PATH,
    DEVELOPMENT_TRAIN_PATH,
    DEVELOPMENT_VALIDATION_PATH,
    EXTERNAL_APPROVED_PER_CATEGORY,
    EXTERNAL_METADATA_PATH,
    EXTERNAL_SOURCE_NAME,
    EXTERNAL_SPLIT_MANIFEST_PATH,
    EXTERNAL_TEST_PATH,
    EXTERNAL_TRAIN_PATH,
    EXTERNAL_TRAINING_READINESS_JSON_PATH,
    EXTERNAL_TRAINING_READINESS_MARKDOWN_PATH,
    EXTERNAL_VALIDATION_PATH,
    INTEGRATED_SPLIT_MANIFEST_PATH,
    INTEGRATED_TEST_LOCK_PATH,
    INTEGRATED_TEST_PATH,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
    SPLIT_MANIFEST_COLUMNS,
)
from src.integrate_external_dataset import (
    ExternalDatasetIntegrationError,
    load_metadata,
    project_relative_path,
    validate_split_integrity,
)
from src.open_license_dataset_config import (
    OPEN_LICENSE_REVIEW_COLUMNS,
    OPEN_LICENSE_REVIEW_PATH,
)
from src.validate_open_license_dataset import (
    read_csv_exact,
    validate_open_license_dataset,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_canonical_csv(path: Path) -> str:
    content = path.read_bytes()
    canonical = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def atomic_write_text(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            content,
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_dataframe_exact(
    path: Path,
    columns: tuple[str, ...],
    errors: list[str],
) -> pd.DataFrame:
    if not path.is_file():
        errors.append(f"Missing file: {path}.")
        return pd.DataFrame(columns=columns)

    try:
        dataframe = pd.read_csv(
            path,
            dtype=str,
        ).fillna("")
    except Exception as error:
        errors.append(f"Cannot read {path}: {error}.")
        return pd.DataFrame(columns=columns)

    if tuple(dataframe.columns) != columns:
        errors.append(f"Invalid schema: {path}.")
        return pd.DataFrame(columns=columns)

    return dataframe


def validate_label_triplets(
    dataframe: pd.DataFrame,
    *,
    name: str,
    errors: list[str],
) -> None:
    if dataframe.empty:
        errors.append(f"{name} is empty.")
        return

    if not dataframe["sample_id"].is_unique:
        errors.append(f"{name} contains duplicate sample IDs.")

    labels_by_image = (
        dataframe
        .groupby("image_id")["label"]
        .apply(set)
    )
    for image_id, image_labels in labels_by_image.items():
        if image_labels != set(LABELS):
            errors.append(
                f"{name}: image {image_id} does not contain "
                "all three labels."
            )

    sample_counts = (
        dataframe
        .groupby("image_id")["sample_id"]
        .count()
    )
    for image_id, count in sample_counts.items():
        if int(count) != len(LABELS):
            errors.append(
                f"{name}: image {image_id} has {count} samples; "
                f"expected {len(LABELS)}."
            )


def validate_grouped_splits(
    *,
    original_dataframe: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    name: str,
    errors: list[str],
) -> None:
    try:
        validate_split_integrity(
            original_dataframe=original_dataframe,
            splits=splits,
        )
    except ExternalDatasetIntegrationError as error:
        errors.append(f"{name}: {error}")


def read_test_lock(
    errors: list[str],
) -> dict[str, Any]:
    if not INTEGRATED_TEST_LOCK_PATH.is_file():
        errors.append(
            f"Missing test lock: {INTEGRATED_TEST_LOCK_PATH}."
        )
        return {}

    try:
        payload = json.loads(
            INTEGRATED_TEST_LOCK_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        errors.append(
            f"Cannot read test lock: {error}."
        )
        return {}

    if not isinstance(payload, dict):
        errors.append("The integrated test lock is not a JSON object.")
        return {}

    return payload


def validate_external_training_readiness() -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    open_license_report = validate_open_license_dataset()
    if (
        open_license_report.get("status") != "PASS"
        or open_license_report.get("readiness")
        != "READY_FOR_EXTERNAL_DATASET"
    ):
        errors.append(
            "Step 010.1 is not READY_FOR_EXTERNAL_DATASET."
        )

    approved_catalog = read_dataframe_exact(
        APPROVED_EXTERNAL_CATALOG_PATH,
        APPROVED_EXTERNAL_CATALOG_COLUMNS,
        errors,
    )
    external_metadata = read_dataframe_exact(
        EXTERNAL_METADATA_PATH,
        METADATA_COLUMNS,
        errors,
    )

    external_splits = {
        "train": read_dataframe_exact(
            EXTERNAL_TRAIN_PATH,
            METADATA_COLUMNS,
            errors,
        ),
        "validation": read_dataframe_exact(
            EXTERNAL_VALIDATION_PATH,
            METADATA_COLUMNS,
            errors,
        ),
        "test": read_dataframe_exact(
            EXTERNAL_TEST_PATH,
            METADATA_COLUMNS,
            errors,
        ),
    }

    development_splits = {
        "train": read_dataframe_exact(
            DEVELOPMENT_TRAIN_PATH,
            METADATA_COLUMNS,
            errors,
        ),
        "validation": read_dataframe_exact(
            DEVELOPMENT_VALIDATION_PATH,
            METADATA_COLUMNS,
            errors,
        ),
        "test": read_dataframe_exact(
            DEVELOPMENT_TEST_PATH,
            METADATA_COLUMNS,
            errors,
        ),
    }

    integrated_splits = {
        "train": read_dataframe_exact(
            INTEGRATED_TRAIN_PATH,
            METADATA_COLUMNS,
            errors,
        ),
        "validation": read_dataframe_exact(
            INTEGRATED_VALIDATION_PATH,
            METADATA_COLUMNS,
            errors,
        ),
        "test": read_dataframe_exact(
            INTEGRATED_TEST_PATH,
            METADATA_COLUMNS,
            errors,
        ),
    }

    external_manifest = read_dataframe_exact(
        EXTERNAL_SPLIT_MANIFEST_PATH,
        SPLIT_MANIFEST_COLUMNS,
        errors,
    )
    integrated_manifest = read_dataframe_exact(
        INTEGRATED_SPLIT_MANIFEST_PATH,
        SPLIT_MANIFEST_COLUMNS,
        errors,
    )

    expected_approved = (
        len(PART_CATEGORIES)
        * EXTERNAL_APPROVED_PER_CATEGORY
    )
    if len(approved_catalog) != expected_approved:
        errors.append(
            f"Approved external catalog has {len(approved_catalog)} rows; "
            f"expected {expected_approved}."
        )

    if not approved_catalog.empty:
        for column in (
            "asset_id",
            "image_id",
            "part_group_id",
            "sha256",
        ):
            if not approved_catalog[column].is_unique:
                errors.append(
                    f"Approved external catalog column "
                    f"'{column}' is not unique."
                )

        for category in PART_CATEGORIES:
            count = int(
                approved_catalog[
                    "part_category"
                ].eq(category).sum()
            )
            if count != EXTERNAL_APPROVED_PER_CATEGORY:
                errors.append(
                    f"Approved category '{category}' has {count} images; "
                    f"expected {EXTERNAL_APPROVED_PER_CATEGORY}."
                )

        if set(approved_catalog["source"]) != {
            EXTERNAL_SOURCE_NAME
        }:
            errors.append(
                "Approved external catalog contains an unexpected source."
            )

        project_root = APPROVED_EXTERNAL_CATALOG_PATH.parents[3]
        for row in approved_catalog.to_dict(orient="records"):
            image_path = project_root / str(row["image_path"])
            if not image_path.is_file():
                errors.append(
                    f"Approved image is missing: {row['image_path']}."
                )
                continue
            if sha256_file(image_path) != str(row["sha256"]):
                errors.append(
                    f"Approved image hash differs: {row['asset_id']}."
                )

    review_rows, review_errors = read_csv_exact(
        OPEN_LICENSE_REVIEW_PATH,
        OPEN_LICENSE_REVIEW_COLUMNS,
    )
    errors.extend(review_errors)

    approved_review_ids = {
        row["asset_id"]
        for row in review_rows
        if row["operator_decision"].lower() == "approved"
    }
    rejected_review_ids = {
        row["asset_id"]
        for row in review_rows
        if row["operator_decision"].lower() == "rejected"
    }
    catalog_ids = set(approved_catalog["asset_id"])

    if catalog_ids != approved_review_ids:
        errors.append(
            "Approved external catalog does not match approved review rows."
        )

    if catalog_ids & rejected_review_ids:
        errors.append(
            "Rejected audit candidates leaked into the approved catalog."
        )

    expected_external_samples = expected_approved * len(LABELS)
    if len(external_metadata) != expected_external_samples:
        errors.append(
            f"External metadata has {len(external_metadata)} rows; "
            f"expected {expected_external_samples}."
        )

    if not external_metadata.empty:
        validate_label_triplets(
            external_metadata,
            name="external metadata",
            errors=errors,
        )
        if set(external_metadata["source"]) != {
            EXTERNAL_SOURCE_NAME
        }:
            errors.append(
                "External metadata contains an unexpected source."
            )
        if (
            external_metadata["part_group_id"].nunique()
            != expected_approved
        ):
            errors.append(
                "External metadata does not contain exactly one group "
                "per approved image."
            )

    validate_grouped_splits(
        original_dataframe=external_metadata,
        splits=external_splits,
        name="external grouped split",
        errors=errors,
    )

    expected_external_counts = {
        "train": (90, 30),
        "validation": (30, 10),
        "test": (30, 10),
    }
    for split_name, (samples, groups) in (
        expected_external_counts.items()
    ):
        dataframe = external_splits[split_name]
        if len(dataframe) != samples:
            errors.append(
                f"External {split_name} has {len(dataframe)} samples; "
                f"expected {samples}."
            )
        if dataframe["part_group_id"].nunique() != groups:
            errors.append(
                f"External {split_name} has "
                f"{dataframe['part_group_id'].nunique()} groups; "
                f"expected {groups}."
            )
        validate_label_triplets(
            dataframe,
            name=f"external {split_name}",
            errors=errors,
        )

    integrated_original = pd.concat(
        list(integrated_splits.values()),
        ignore_index=True,
    )
    validate_grouped_splits(
        original_dataframe=integrated_original,
        splits=integrated_splits,
        name="integrated grouped split",
        errors=errors,
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        development = development_splits[split_name]
        external = external_splits[split_name]
        integrated = integrated_splits[split_name]

        expected_rows = len(development) + len(external)
        expected_groups = (
            development["part_group_id"].nunique()
            + external["part_group_id"].nunique()
        )
        if len(integrated) != expected_rows:
            errors.append(
                f"Integrated {split_name} has {len(integrated)} rows; "
                f"expected {expected_rows}."
            )
        if integrated["part_group_id"].nunique() != expected_groups:
            errors.append(
                f"Integrated {split_name} has "
                f"{integrated['part_group_id'].nunique()} groups; "
                f"expected {expected_groups}."
            )

        external_rows = int(
            integrated["source"].eq(
                EXTERNAL_SOURCE_NAME
            ).sum()
        )
        if external_rows != len(external):
            errors.append(
                f"Integrated {split_name} external-row count differs "
                "from the external split."
            )

        if set(development["sample_id"]) & set(
            external["sample_id"]
        ):
            errors.append(
                f"Development and external {split_name} sample IDs overlap."
            )
        if set(development["part_group_id"]) & set(
            external["part_group_id"]
        ):
            errors.append(
                f"Development and external {split_name} groups overlap."
            )

        validate_label_triplets(
            integrated,
            name=f"integrated {split_name}",
            errors=errors,
        )

    if len(external_manifest) != expected_approved:
        errors.append(
            "External split manifest does not contain one row "
            "per approved image."
        )
    if len(integrated_manifest) != (
        sum(
            dataframe["part_group_id"].nunique()
            for dataframe in development_splits.values()
        )
        + expected_approved
    ):
        errors.append(
            "Integrated split manifest has an unexpected row count."
        )

    test_lock = read_test_lock(errors)
    if test_lock:
        if test_lock.get("test_locked") is not True:
            errors.append("The integrated test split is not locked.")
        if (
            test_lock.get("test_evaluation_permitted")
            is not False
        ):
            errors.append(
                "The test lock permits test evaluation."
            )

        training_inputs = [
            str(value)
            for value in test_lock.get(
                "training_inputs",
                [],
            )
        ]
        required_training_inputs = {
            project_relative_path(INTEGRATED_TRAIN_PATH),
            project_relative_path(INTEGRATED_VALIDATION_PATH),
        }
        if set(training_inputs) != required_training_inputs:
            errors.append(
                "Training inputs do not contain exactly integrated "
                "train and validation."
            )

        forbidden_test_paths = {
            project_relative_path(EXTERNAL_TEST_PATH),
            project_relative_path(INTEGRATED_TEST_PATH),
        }
        if set(training_inputs) & forbidden_test_paths:
            errors.append(
                "A locked test path appears in training inputs."
            )

        if test_lock.get("hash_normalization") != "utf-8-lf":
            errors.append(
                "The test lock does not declare canonical UTF-8/LF hashing."
            )

        if EXTERNAL_TEST_PATH.is_file():
            actual_hash = sha256_canonical_csv(EXTERNAL_TEST_PATH)
            if (
                test_lock.get("external_test_sha256")
                != actual_hash
            ):
                errors.append(
                    "External test fingerprint differs from the lock."
                )

        if INTEGRATED_TEST_PATH.is_file():
            actual_hash = sha256_canonical_csv(INTEGRATED_TEST_PATH)
            if (
                test_lock.get("integrated_test_sha256")
                != actual_hash
            ):
                errors.append(
                    "Integrated test fingerprint differs from the lock."
                )

    if errors:
        status = "FAIL"
        readiness = "TRAINING_READINESS_BLOCKED"
    else:
        status = "PASS"
        readiness = "READY_FOR_TRAINING"

    return {
        "status": status,
        "readiness": readiness,
        "test_locked": bool(
            test_lock.get("test_locked")
        ),
        "test_evaluation_permitted": bool(
            test_lock.get(
                "test_evaluation_permitted",
                True,
            )
        ),
        "approved_external_images": int(
            len(approved_catalog)
        ),
        "external_samples": int(
            len(external_metadata)
        ),
        "external_split_rows": {
            name: int(len(dataframe))
            for name, dataframe in external_splits.items()
        },
        "external_split_groups": {
            name: int(
                dataframe["part_group_id"].nunique()
            )
            for name, dataframe in external_splits.items()
        },
        "integrated_split_rows": {
            name: int(len(dataframe))
            for name, dataframe in integrated_splits.items()
        },
        "integrated_split_groups": {
            name: int(
                dataframe["part_group_id"].nunique()
            )
            for name, dataframe in integrated_splits.items()
        },
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def render_readiness_markdown(
    report: dict[str, Any],
) -> str:
    lines = [
        "# Step 010.2 — External Training Readiness",
        "",
        f"- Status: **{report['status']}**",
        f"- Readiness: **{report['readiness']}**",
        (
            "- Approved external images: "
            f"**{report['approved_external_images']}**"
        ),
        (
            "- Generated external samples: "
            f"**{report['external_samples']}**"
        ),
        f"- Test locked: **{str(report['test_locked']).lower()}**",
        (
            "- Test evaluation permitted: "
            f"**{str(report['test_evaluation_permitted']).lower()}**"
        ),
        "",
        "## External split",
        "",
        "| Split | Samples | Groups |",
        "|---|---:|---:|",
    ]

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        lines.append(
            f"| {split_name.title()} "
            f"| {report['external_split_rows'][split_name]} "
            f"| {report['external_split_groups'][split_name]} |"
        )

    lines.extend(
        [
            "",
            "## Integrated split",
            "",
            "| Split | Samples | Groups |",
            "|---|---:|---:|",
        ]
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        lines.append(
            f"| {split_name.title()} "
            f"| {report['integrated_split_rows'][split_name]} "
            f"| {report['integrated_split_groups'][split_name]} |"
        )

    lines.extend(
        [
            "",
            "## Training policy",
            "",
            "- Model training may use only `integrated_train.csv`.",
            "- Model selection may use only `integrated_validation.csv`.",
            "- `integrated_test.csv` remains fingerprinted and locked.",
            "- Step 010.2 performs structural validation only; it does not "
            "train or evaluate a model.",
        ]
    )

    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(
            f"- {error}"
            for error in report["errors"]
        )

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(
            f"- {warning}"
            for warning in report["warnings"]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_readiness_outputs(
    report: dict[str, Any],
) -> None:
    atomic_write_text(
        EXTERNAL_TRAINING_READINESS_JSON_PATH,
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    atomic_write_text(
        EXTERNAL_TRAINING_READINESS_MARKDOWN_PATH,
        render_readiness_markdown(report),
    )


def main() -> None:
    report = validate_external_training_readiness()
    write_readiness_outputs(report)

    print("Step 010.2 external training readiness")
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
    print(
        "- External split: "
        f"{report['external_split_rows']}"
    )
    print(
        "- Integrated split: "
        f"{report['integrated_split_rows']}"
    )
    print(
        "- Test evaluation permitted: "
        f"{str(report['test_evaluation_permitted']).lower()}"
    )

    for error in report["errors"]:
        print(f"ERROR: {error}")

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
