from __future__ import annotations

from pathlib import Path

from src.dataset_config import (
    LABELS,
    METADATA_COLUMNS,
    PART_CATEGORIES,
    PART_FAMILIES,
)
from src.open_license_dataset_config import (
    OPEN_LICENSE_MANIFEST_PATH,
    OPEN_LICENSE_REVIEW_PATH,
)
from src.real_dataset_config import PROJECT_ROOT

EXTERNAL_INTEGRATION_ROOT = (
    PROJECT_ROOT / "data" / "external" / "integrated"
)

APPROVED_EXTERNAL_CATALOG_PATH = (
    EXTERNAL_INTEGRATION_ROOT / "approved_external_images.csv"
)
EXTERNAL_METADATA_PATH = (
    EXTERNAL_INTEGRATION_ROOT / "external_matching_metadata.csv"
)
EXTERNAL_TRAIN_PATH = (
    EXTERNAL_INTEGRATION_ROOT / "external_train.csv"
)
EXTERNAL_VALIDATION_PATH = (
    EXTERNAL_INTEGRATION_ROOT / "external_validation.csv"
)
EXTERNAL_TEST_PATH = (
    EXTERNAL_INTEGRATION_ROOT / "external_test.csv"
)
EXTERNAL_SPLIT_MANIFEST_PATH = (
    EXTERNAL_INTEGRATION_ROOT / "external_split_manifest.csv"
)

DEVELOPMENT_PROCESSED_ROOT = (
    PROJECT_ROOT / "data" / "processed"
)
DEVELOPMENT_TRAIN_PATH = (
    DEVELOPMENT_PROCESSED_ROOT / "development_train.csv"
)
DEVELOPMENT_VALIDATION_PATH = (
    DEVELOPMENT_PROCESSED_ROOT / "development_validation.csv"
)
DEVELOPMENT_TEST_PATH = (
    DEVELOPMENT_PROCESSED_ROOT / "development_test.csv"
)

INTEGRATED_TRAIN_PATH = (
    DEVELOPMENT_PROCESSED_ROOT / "integrated_train.csv"
)
INTEGRATED_VALIDATION_PATH = (
    DEVELOPMENT_PROCESSED_ROOT / "integrated_validation.csv"
)
INTEGRATED_TEST_PATH = (
    DEVELOPMENT_PROCESSED_ROOT / "integrated_test.csv"
)
INTEGRATED_SPLIT_MANIFEST_PATH = (
    DEVELOPMENT_PROCESSED_ROOT / "integrated_split_manifest.csv"
)
INTEGRATED_TEST_LOCK_PATH = (
    DEVELOPMENT_PROCESSED_ROOT / "integrated_test_lock.json"
)

EXTERNAL_REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "external_dataset"
)
EXTERNAL_INTEGRATION_JSON_PATH = (
    EXTERNAL_REPORT_ROOT / "external_integration_summary.json"
)
EXTERNAL_INTEGRATION_MARKDOWN_PATH = (
    EXTERNAL_REPORT_ROOT / "external_integration_summary.md"
)
EXTERNAL_TRAINING_READINESS_JSON_PATH = (
    EXTERNAL_REPORT_ROOT / "external_training_readiness.json"
)
EXTERNAL_TRAINING_READINESS_MARKDOWN_PATH = (
    EXTERNAL_REPORT_ROOT / "external_training_readiness.md"
)

EXTERNAL_SOURCE_NAME = "wikimedia_commons_open_license"
DEVELOPMENT_SOURCE_NAME = "generated_development"

EXTERNAL_APPROVED_PER_CATEGORY = 5
EXTERNAL_TRAIN_GROUPS_PER_CATEGORY = 3
EXTERNAL_VALIDATION_GROUPS_PER_CATEGORY = 1
EXTERNAL_TEST_GROUPS_PER_CATEGORY = 1

APPROVED_EXTERNAL_CATALOG_COLUMNS = (
    "asset_id",
    "image_id",
    "part_group_id",
    "part_family",
    "part_category",
    "image_path",
    "commons_page_id",
    "commons_title",
    "description_url",
    "author",
    "credit",
    "license_short_name",
    "license_url",
    "sha256",
    "width",
    "height",
    "format",
    "source",
)

SPLIT_MANIFEST_COLUMNS = (
    "dataset_origin",
    "split",
    "part_group_id",
    "image_id",
    "part_family",
    "part_category",
    "source",
    "image_count",
    "sample_count",
)

CATEGORY_DESCRIPTIONS = {
    "starter": "Automotive starter motor.",
    "alternator": "Automotive alternator.",
    "brake_disc": "Automotive brake disc.",
    "brake_pad": "Automotive brake pad.",
    "shock_absorber": "Automotive shock absorber.",
    "coil_spring": "Automotive suspension coil spring.",
    "headlight": "Automotive headlight assembly.",
    "taillight": "Automotive taillight assembly.",
    "oil_filter": "Automotive engine oil filter.",
    "air_filter": "Automotive engine air filter.",
}

PARTIAL_CATEGORY = {
    "starter": "alternator",
    "alternator": "starter",
    "brake_disc": "brake_pad",
    "brake_pad": "brake_disc",
    "shock_absorber": "coil_spring",
    "coil_spring": "shock_absorber",
    "headlight": "taillight",
    "taillight": "headlight",
    "oil_filter": "air_filter",
    "air_filter": "oil_filter",
}

MISMATCH_CATEGORY = {
    "starter": "brake_disc",
    "alternator": "brake_pad",
    "brake_disc": "shock_absorber",
    "brake_pad": "coil_spring",
    "shock_absorber": "headlight",
    "coil_spring": "taillight",
    "headlight": "oil_filter",
    "taillight": "air_filter",
    "oil_filter": "starter",
    "air_filter": "alternator",
}

CATEGORY_TO_FAMILY = {
    category: family
    for family, categories in PART_FAMILIES.items()
    for category in categories
}

if tuple(CATEGORY_DESCRIPTIONS) != PART_CATEGORIES:
    raise RuntimeError(
        "External integration descriptions differ from project categories."
    )

if tuple(PARTIAL_CATEGORY) != PART_CATEGORIES:
    raise RuntimeError(
        "Partial-match mapping differs from project categories."
    )

if tuple(MISMATCH_CATEGORY) != PART_CATEGORIES:
    raise RuntimeError(
        "Mismatch mapping differs from project categories."
    )

for category in PART_CATEGORIES:
    partial = PARTIAL_CATEGORY[category]
    mismatch = MISMATCH_CATEGORY[category]
    if CATEGORY_TO_FAMILY[partial] != CATEGORY_TO_FAMILY[category]:
        raise RuntimeError(
            f"Partial mapping for {category} crosses part families."
        )
    if CATEGORY_TO_FAMILY[mismatch] == CATEGORY_TO_FAMILY[category]:
        raise RuntimeError(
            f"Mismatch mapping for {category} stays in the same family."
        )

if LABELS != ("MATCH", "PARTIAL_MATCH", "MISMATCH"):
    raise RuntimeError("Unexpected project label order.")

if METADATA_COLUMNS != (
    "sample_id",
    "image_id",
    "part_group_id",
    "image_path",
    "part_family",
    "part_category",
    "description",
    "label",
    "source",
):
    raise RuntimeError("Unexpected project metadata schema.")
