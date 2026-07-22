from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import src.integrate_external_dataset as integration
import src.validate_external_training_readiness as readiness
from src.dataset_config import (
    LABELS,
    METADATA_COLUMNS,
    PART_CATEGORIES,
)
from src.external_dataset_integration_config import (
    APPROVED_EXTERNAL_CATALOG_COLUMNS,
    EXTERNAL_SOURCE_NAME,
    SPLIT_MANIFEST_COLUMNS,
)
from src.open_license_dataset_config import (
    OPEN_LICENSE_MANIFEST_COLUMNS,
    OPEN_LICENSE_REVIEW_COLUMNS,
)


def write_csv_rows(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def synthetic_manifest_and_review(
    *,
    rejected_asset: str | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    manifest_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []

    page_id = 1000
    for category_index, category in enumerate(PART_CATEGORIES):
        family = integration.CATEGORY_TO_FAMILY[category]
        for item_index in range(5):
            page_id += 1
            asset_id = f"commons_{category}_{page_id}"
            local_path = (
                f"data/external/open_license/images/"
                f"{category}/{asset_id}.jpg"
            )
            manifest = {
                column: ""
                for column in OPEN_LICENSE_MANIFEST_COLUMNS
            }
            manifest.update(
                {
                    "asset_id": asset_id,
                    "part_family": family,
                    "part_category": category,
                    "search_query": category,
                    "commons_page_id": str(page_id),
                    "commons_title": f"File:{asset_id}.jpg",
                    "description_url": (
                        "https://commons.wikimedia.org/wiki/"
                        f"File:{asset_id}.jpg"
                    ),
                    "original_url": (
                        "https://upload.wikimedia.org/"
                        f"{asset_id}.jpg"
                    ),
                    "download_url": (
                        "https://upload.wikimedia.org/thumb/"
                        f"{asset_id}.jpg"
                    ),
                    "author": "Example Author",
                    "credit": "",
                    "license_short_name": "CC BY-SA 4.0",
                    "license_url": (
                        "https://creativecommons.org/licenses/"
                        "by-sa/4.0/"
                    ),
                    "attribution_required": "yes",
                    "usage_terms": "CC BY-SA 4.0",
                    "local_path": local_path,
                    "sha256": (
                        f"{category_index:02d}"
                        f"{item_index:02d}"
                        + "a" * 60
                    ),
                    "file_size_bytes": "100",
                    "width": "640",
                    "height": "480",
                    "format": "JPEG",
                    "downloaded_at_utc": (
                        "2026-07-19T00:00:00+00:00"
                    ),
                    "modifications": "Thumbnail only.",
                }
            )
            manifest_rows.append(manifest)

            review = {
                column: ""
                for column in OPEN_LICENSE_REVIEW_COLUMNS
            }
            review.update(
                {
                    "asset_id": asset_id,
                    "part_family": family,
                    "part_category": category,
                    "local_path": local_path,
                    "commons_title": f"File:{asset_id}.jpg",
                    "author": "Example Author",
                    "license_short_name": "CC BY-SA 4.0",
                    "license_url": (
                        "https://creativecommons.org/licenses/"
                        "by-sa/4.0/"
                    ),
                    "description_url": (
                        "https://commons.wikimedia.org/wiki/"
                        f"File:{asset_id}.jpg"
                    ),
                    "operator_decision": (
                        "rejected"
                        if asset_id == rejected_asset
                        else "approved"
                    ),
                    "rejection_reason": (
                        "Wrong category."
                        if asset_id == rejected_asset
                        else ""
                    ),
                    "operator_notes": "Reviewed.",
                }
            )
            review_rows.append(review)

    return manifest_rows, review_rows


def development_split(
    split_name: str,
) -> pd.DataFrame:
    counts = {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    offsets = {
        "train": 0,
        "validation": 3,
        "test": 4,
    }

    records: list[dict[str, str]] = []
    for category in PART_CATEGORIES:
        family = integration.CATEGORY_TO_FAMILY[category]
        for index in range(
            offsets[split_name],
            offsets[split_name] + counts[split_name],
        ):
            group_id = f"dev_group_{category}_{index + 1:02d}"
            image_id = f"dev_image_{category}_{index + 1:02d}"

            for label in LABELS:
                records.append(
                    {
                        "sample_id": (
                            f"dev_sample_{category}_{index + 1:02d}_"
                            f"{label.lower()}"
                        ),
                        "image_id": image_id,
                        "part_group_id": group_id,
                        "image_path": (
                            f"data/development/images/{image_id}.png"
                        ),
                        "part_family": family,
                        "part_category": category,
                        "description": (
                            integration.CATEGORY_DESCRIPTIONS[category]
                        ),
                        "label": label,
                        "source": "generated_development",
                    }
                )

    return pd.DataFrame(
        records,
        columns=METADATA_COLUMNS,
    )


def test_approved_catalog_contains_exact_balanced_50():
    manifest_rows, review_rows = synthetic_manifest_and_review()

    catalog = integration.build_approved_catalog(
        manifest_rows,
        review_rows,
    )

    assert tuple(catalog.columns) == (
        APPROVED_EXTERNAL_CATALOG_COLUMNS
    )
    assert len(catalog) == 50
    assert catalog["image_id"].is_unique
    assert catalog["part_group_id"].is_unique
    assert set(catalog["source"]) == {
        EXTERNAL_SOURCE_NAME
    }

    counts = catalog["part_category"].value_counts()
    assert set(counts.index) == set(PART_CATEGORIES)
    assert set(counts.tolist()) == {5}


def test_rejected_candidate_is_excluded_and_blocks_balance():
    manifest_rows, review_rows = synthetic_manifest_and_review()
    rejected_asset = review_rows[0]["asset_id"]
    review_rows[0]["operator_decision"] = "rejected"
    review_rows[0]["rejection_reason"] = "Wrong category."

    with pytest.raises(
        integration.ExternalDatasetIntegrationError,
        match="Expected 50 approved",
    ):
        integration.build_approved_catalog(
            manifest_rows,
            review_rows,
        )

    assert rejected_asset not in {
        row["asset_id"]
        for row in review_rows
        if row["operator_decision"] == "approved"
    }


def test_external_metadata_builds_three_semantic_labels_per_image():
    manifest_rows, review_rows = synthetic_manifest_and_review()
    catalog = integration.build_approved_catalog(
        manifest_rows,
        review_rows,
    )

    metadata = integration.build_external_metadata(catalog)

    assert tuple(metadata.columns) == METADATA_COLUMNS
    assert len(metadata) == 150
    assert metadata["sample_id"].is_unique
    assert metadata["image_id"].nunique() == 50
    assert metadata["part_group_id"].nunique() == 50

    for image_id, rows in metadata.groupby("image_id"):
        assert set(rows["label"]) == set(LABELS)
        category = rows["part_category"].iloc[0]
        descriptions = {
            row.label: row.description
            for row in rows.itertuples()
        }
        assert descriptions["MATCH"] == (
            integration.CATEGORY_DESCRIPTIONS[category]
        )
        assert descriptions["PARTIAL_MATCH"] == (
            integration.CATEGORY_DESCRIPTIONS[
                integration.PARTIAL_CATEGORY[category]
            ]
        )
        assert descriptions["MISMATCH"] == (
            integration.CATEGORY_DESCRIPTIONS[
                integration.MISMATCH_CATEGORY[category]
            ]
        )


def test_external_grouped_split_is_balanced_and_disjoint():
    manifest_rows, review_rows = synthetic_manifest_and_review()
    catalog = integration.build_approved_catalog(
        manifest_rows,
        review_rows,
    )
    metadata = integration.build_external_metadata(catalog)

    splits = integration.split_external_metadata(metadata)

    assert len(splits["train"]) == 90
    assert len(splits["validation"]) == 30
    assert len(splits["test"]) == 30

    assert splits["train"]["part_group_id"].nunique() == 30
    assert splits["validation"]["part_group_id"].nunique() == 10
    assert splits["test"]["part_group_id"].nunique() == 10

    train_groups = set(splits["train"]["part_group_id"])
    validation_groups = set(
        splits["validation"]["part_group_id"]
    )
    test_groups = set(splits["test"]["part_group_id"])

    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)

    for dataframe in splits.values():
        assert set(dataframe["part_category"]) == set(PART_CATEGORIES)
        assert {
            int(dataframe["label"].eq(label).sum())
            for label in LABELS
        } == {len(dataframe) // len(LABELS)}


def test_combined_split_preserves_namespaces():
    manifest_rows, review_rows = synthetic_manifest_and_review()
    catalog = integration.build_approved_catalog(
        manifest_rows,
        review_rows,
    )
    metadata = integration.build_external_metadata(catalog)
    external_splits = integration.split_external_metadata(
        metadata
    )

    for split_name in ("train", "validation", "test"):
        development = development_split(split_name)
        combined = integration.combine_split(
            development,
            external_splits[split_name],
        )

        assert len(combined) == (
            len(development)
            + len(external_splits[split_name])
        )
        assert combined["sample_id"].is_unique
        assert not (
            set(development["part_group_id"])
            & set(
                external_splits[split_name]["part_group_id"]
            )
        )


def patch_integration_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    paths = {
        "OPEN_LICENSE_MANIFEST_PATH": (
            tmp_path
            / "data"
            / "external"
            / "open_license"
            / "open_license_manifest.csv"
        ),
        "OPEN_LICENSE_REVIEW_PATH": (
            tmp_path
            / "data"
            / "external"
            / "open_license"
            / "open_license_review.csv"
        ),
        "APPROVED_EXTERNAL_CATALOG_PATH": (
            tmp_path
            / "data"
            / "external"
            / "integrated"
            / "approved_external_images.csv"
        ),
        "EXTERNAL_METADATA_PATH": (
            tmp_path
            / "data"
            / "external"
            / "integrated"
            / "external_matching_metadata.csv"
        ),
        "EXTERNAL_TRAIN_PATH": (
            tmp_path
            / "data"
            / "external"
            / "integrated"
            / "external_train.csv"
        ),
        "EXTERNAL_VALIDATION_PATH": (
            tmp_path
            / "data"
            / "external"
            / "integrated"
            / "external_validation.csv"
        ),
        "EXTERNAL_TEST_PATH": (
            tmp_path
            / "data"
            / "external"
            / "integrated"
            / "external_test.csv"
        ),
        "EXTERNAL_SPLIT_MANIFEST_PATH": (
            tmp_path
            / "data"
            / "external"
            / "integrated"
            / "external_split_manifest.csv"
        ),
        "DEVELOPMENT_TRAIN_PATH": (
            tmp_path / "data" / "processed" / "development_train.csv"
        ),
        "DEVELOPMENT_VALIDATION_PATH": (
            tmp_path
            / "data"
            / "processed"
            / "development_validation.csv"
        ),
        "DEVELOPMENT_TEST_PATH": (
            tmp_path / "data" / "processed" / "development_test.csv"
        ),
        "INTEGRATED_TRAIN_PATH": (
            tmp_path / "data" / "processed" / "integrated_train.csv"
        ),
        "INTEGRATED_VALIDATION_PATH": (
            tmp_path
            / "data"
            / "processed"
            / "integrated_validation.csv"
        ),
        "INTEGRATED_TEST_PATH": (
            tmp_path / "data" / "processed" / "integrated_test.csv"
        ),
        "INTEGRATED_SPLIT_MANIFEST_PATH": (
            tmp_path
            / "data"
            / "processed"
            / "integrated_split_manifest.csv"
        ),
        "INTEGRATED_TEST_LOCK_PATH": (
            tmp_path
            / "data"
            / "processed"
            / "integrated_test_lock.json"
        ),
        "EXTERNAL_INTEGRATION_JSON_PATH": (
            tmp_path
            / "reports"
            / "external_dataset"
            / "external_integration_summary.json"
        ),
        "EXTERNAL_INTEGRATION_MARKDOWN_PATH": (
            tmp_path
            / "reports"
            / "external_dataset"
            / "external_integration_summary.md"
        ),
    }

    monkeypatch.setattr(
        integration,
        "PROJECT_ROOT",
        tmp_path,
    )

    for name, value in paths.items():
        monkeypatch.setattr(integration, name, value)

    return paths


def test_end_to_end_integration_writes_locked_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = patch_integration_paths(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(
        integration,
        "validate_open_license_dataset",
        lambda: {
            "status": "PASS",
            "readiness": "READY_FOR_EXTERNAL_DATASET",
        },
    )

    manifest_rows, review_rows = synthetic_manifest_and_review()
    write_csv_rows(
        paths["OPEN_LICENSE_MANIFEST_PATH"],
        OPEN_LICENSE_MANIFEST_COLUMNS,
        manifest_rows,
    )
    write_csv_rows(
        paths["OPEN_LICENSE_REVIEW_PATH"],
        OPEN_LICENSE_REVIEW_COLUMNS,
        review_rows,
    )

    for split_name, path_name in (
        ("train", "DEVELOPMENT_TRAIN_PATH"),
        ("validation", "DEVELOPMENT_VALIDATION_PATH"),
        ("test", "DEVELOPMENT_TEST_PATH"),
    ):
        dataframe = development_split(split_name)
        path = paths[path_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(
            path,
            index=False,
            lineterminator="\n",
        )

    report = integration.integrate_external_dataset()

    assert report["status"] == "PASS"
    assert report["approved_external_images"] == 50
    assert report["external_samples"] == 150

    integrated_train = pd.read_csv(
        paths["INTEGRATED_TRAIN_PATH"]
    )
    integrated_validation = pd.read_csv(
        paths["INTEGRATED_VALIDATION_PATH"]
    )
    integrated_test = pd.read_csv(
        paths["INTEGRATED_TEST_PATH"]
    )

    assert len(integrated_train) == 180
    assert len(integrated_validation) == 60
    assert len(integrated_test) == 60

    lock = json.loads(
        paths["INTEGRATED_TEST_LOCK_PATH"].read_text(
            encoding="utf-8"
        )
    )
    assert lock["test_locked"] is True
    assert lock["test_evaluation_permitted"] is False
    assert set(lock["training_inputs"]) == {
        "data/processed/integrated_train.csv",
        "data/processed/integrated_validation.csv",
    }
    assert "data/processed/integrated_test.csv" not in (
        lock["training_inputs"]
    )

    assert lock["hash_normalization"] == "utf-8-lf"
    assert lock["integrated_test_sha256"] == (
        readiness.sha256_canonical_csv(
            paths["INTEGRATED_TEST_PATH"]
        )
    )


def test_project_relative_path_uses_portable_posix_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        integration,
        "PROJECT_ROOT",
        tmp_path,
    )

    nested_path = (
        tmp_path
        / "data"
        / "processed"
        / "integrated_train.csv"
    )

    assert integration.project_relative_path(nested_path) == (
        "data/processed/integrated_train.csv"
    )


def test_split_manifest_schema_and_group_rows():
    manifest_rows, review_rows = synthetic_manifest_and_review()
    catalog = integration.build_approved_catalog(
        manifest_rows,
        review_rows,
    )
    metadata = integration.build_external_metadata(catalog)
    splits = integration.split_external_metadata(metadata)

    manifest = integration.create_split_manifest(
        splits,
        dataset_origin="open_license_external",
    )

    assert tuple(manifest.columns) == SPLIT_MANIFEST_COLUMNS
    assert len(manifest) == 50
    assert set(manifest["split"]) == {
        "train",
        "validation",
        "test",
    }
    assert set(manifest["sample_count"].astype(int)) == {3}
    assert set(manifest["image_count"].astype(int)) == {1}


def test_canonical_csv_hash_is_newline_independent(
    tmp_path: Path,
) -> None:
    lf_path = tmp_path / "lf.csv"
    crlf_path = tmp_path / "crlf.csv"
    lf_path.write_bytes(b"a,b\n1,2\n")
    crlf_path.write_bytes(b"a,b\r\n1,2\r\n")

    assert readiness.sha256_canonical_csv(lf_path) == (
        readiness.sha256_canonical_csv(crlf_path)
    )
