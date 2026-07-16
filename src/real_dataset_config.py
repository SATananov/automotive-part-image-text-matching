from __future__ import annotations

from pathlib import Path

from src.dataset_config import PART_CATEGORIES, PART_FAMILIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET_ROOT = PROJECT_ROOT / "data" / "real"
REAL_ANNOTATIONS_DIRECTORY = REAL_DATASET_ROOT / "annotations"
REAL_PART_GROUPS_PATH = REAL_ANNOTATIONS_DIRECTORY / "part_groups.csv"
REAL_IMAGES_PATH = REAL_ANNOTATIONS_DIRECTORY / "images.csv"
REAL_PROCESSED_DIRECTORY = REAL_DATASET_ROOT / "processed"
REAL_PROCESSED_IMAGES_DIRECTORY = REAL_PROCESSED_DIRECTORY / "images"
REAL_IMAGE_MANIFEST_PATH = (
    REAL_PROCESSED_DIRECTORY / "real_image_manifest.csv"
)
DEVELOPMENT_IMAGES_DIRECTORY = (
    PROJECT_ROOT / "data" / "development" / "images"
)

PART_GROUP_COLUMNS = (
    "part_group_id",
    "part_family",
    "part_category",
    "match_description",
    "partial_description",
    "mismatch_description",
    "source",
    "approved",
    "notes",
)

IMAGE_MANIFEST_COLUMNS = (
    "image_id",
    "part_group_id",
    "image_path",
    "view",
    "approved",
)

REAL_IMAGE_INTAKE_MANIFEST_COLUMNS = (
    "image_id",
    "part_group_id",
    "image_path",
    "part_family",
    "part_category",
    "view",
    "source",
    "approved",
    "sha256",
    "file_size_bytes",
    "width",
    "height",
    "mode",
    "format",
)

ALLOWED_IMAGE_VIEWS = (
    "front",
    "rear",
    "left",
    "right",
    "top",
    "detail",
    "other",
)

ALLOWED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
)

APPROVAL_VALUES = (
    "yes",
    "no",
)

CATEGORY_TO_FAMILY = {
    category: family
    for family, categories in PART_FAMILIES.items()
    for category in categories
}

REAL_DATASET_CATEGORIES = PART_CATEGORIES
REAL_ID_PREFIX = "real_"

TARGET_GROUPS_PER_CATEGORY = 10
TARGET_IMAGES_PER_GROUP = 2

TARGET_PART_GROUPS = (
    len(REAL_DATASET_CATEGORIES)
    * TARGET_GROUPS_PER_CATEGORY
)

TARGET_IMAGES = (
    TARGET_PART_GROUPS
    * TARGET_IMAGES_PER_GROUP
)

TARGET_IMAGE_TEXT_SAMPLES = TARGET_IMAGES * 3
