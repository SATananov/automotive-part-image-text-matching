from __future__ import annotations

from src.dataset_config import PART_CATEGORIES, PART_FAMILIES


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

ALLOWED_IMAGE_VIEWS = (
    "front",
    "rear",
    "left",
    "right",
    "top",
    "detail",
    "other",
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