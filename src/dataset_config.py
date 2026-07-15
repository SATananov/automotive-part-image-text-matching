from __future__ import annotations


LABELS = (
    "MATCH",
    "PARTIAL_MATCH",
    "MISMATCH",
)

PART_FAMILIES = {
    "electrical": (
        "starter",
        "alternator",
    ),
    "braking": (
        "brake_disc",
        "brake_pad",
    ),
    "suspension": (
        "shock_absorber",
        "coil_spring",
    ),
    "lighting": (
        "headlight",
        "taillight",
    ),
    "filtration": (
        "oil_filter",
        "air_filter",
    ),
}

PART_CATEGORIES = tuple(
    category
    for categories in PART_FAMILIES.values()
    for category in categories
)

METADATA_COLUMNS = (
    "sample_id",
    "image_id",
    "part_group_id",
    "image_path",
    "part_family",
    "part_category",
    "description",
    "label",
    "source",
)
