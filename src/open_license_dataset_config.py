from __future__ import annotations

from pathlib import Path

from src.dataset_config import PART_CATEGORIES, PART_FAMILIES
from src.real_dataset_config import PROJECT_ROOT

OPEN_LICENSE_ROOT = (
    PROJECT_ROOT / "data" / "external" / "open_license"
)
OPEN_LICENSE_IMAGES_DIRECTORY = OPEN_LICENSE_ROOT / "images"
OPEN_LICENSE_RUNTIME_DIRECTORY = OPEN_LICENSE_ROOT / "runtime"
OPEN_LICENSE_MANIFEST_PATH = (
    OPEN_LICENSE_ROOT / "open_license_manifest.csv"
)
OPEN_LICENSE_REVIEW_PATH = (
    OPEN_LICENSE_ROOT / "open_license_review.csv"
)
OPEN_LICENSE_ATTRIBUTION_PATH = (
    OPEN_LICENSE_ROOT / "ATTRIBUTION.md"
)

OPEN_LICENSE_REPORT_DIRECTORY = (
    PROJECT_ROOT / "reports" / "external_dataset"
)
OPEN_LICENSE_COLLECTION_REPORT_PATH = (
    OPEN_LICENSE_REPORT_DIRECTORY
    / "open_license_collection_summary.md"
)
OPEN_LICENSE_VALIDATION_REPORT_PATH = (
    OPEN_LICENSE_REPORT_DIRECTORY
    / "open_license_validation_summary.md"
)
OPEN_LICENSE_REVIEW_GALLERY_PATH = (
    OPEN_LICENSE_REPORT_DIRECTORY
    / "open_license_review_gallery.html"
)

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
COMMONS_USER_AGENT = (
    "automotive-part-image-text-matching/0.1 "
    "(educational open-license dataset; "
    "https://github.com/SATananov/"
    "automotive-part-image-text-matching)"
)

OPEN_LICENSE_TARGET_PER_CATEGORY = 5
OPEN_LICENSE_THUMBNAIL_WIDTH = 1024
OPEN_LICENSE_SEARCH_LIMIT = 30
OPEN_LICENSE_MIN_WIDTH = 256
OPEN_LICENSE_MIN_HEIGHT = 256
OPEN_LICENSE_MAX_ASPECT_RATIO = 4.0

OPEN_LICENSE_SEARCH_QUERIES = {
    "starter": (
        "automotive starter motor",
        "car starter motor",
    ),
    "alternator": (
        "automotive alternator",
        "car alternator",
    ),
    "brake_disc": (
        "automobile brake disc",
        "car brake rotor",
    ),
    "brake_pad": (
        "automobile brake pad",
        "car brake pads",
    ),
    "shock_absorber": (
        "automobile shock absorber",
        "car shock absorber",
    ),
    "coil_spring": (
        "automobile suspension coil spring",
        "car coil spring",
    ),
    "headlight": (
        "automobile headlight assembly",
        "car headlamp",
    ),
    "taillight": (
        "automobile tail light assembly",
        "car taillight",
    ),
    "oil_filter": (
        "automotive oil filter",
        "car oil filter",
    ),
    "air_filter": (
        "automotive air filter",
        "car engine air filter",
    ),
}

OPEN_LICENSE_MANIFEST_COLUMNS = (
    "asset_id",
    "part_family",
    "part_category",
    "search_query",
    "commons_page_id",
    "commons_title",
    "description_url",
    "original_url",
    "download_url",
    "author",
    "credit",
    "license_short_name",
    "license_url",
    "attribution_required",
    "usage_terms",
    "local_path",
    "sha256",
    "file_size_bytes",
    "width",
    "height",
    "format",
    "downloaded_at_utc",
    "modifications",
)

OPEN_LICENSE_REVIEW_COLUMNS = (
    "asset_id",
    "part_family",
    "part_category",
    "local_path",
    "commons_title",
    "author",
    "license_short_name",
    "license_url",
    "description_url",
    "operator_decision",
    "rejection_reason",
    "operator_notes",
)

OPEN_LICENSE_REVIEW_DECISIONS = (
    "pending",
    "approved",
    "rejected",
)

CATEGORY_TO_FAMILY = {
    category: family
    for family, categories in PART_FAMILIES.items()
    for category in categories
}

if tuple(OPEN_LICENSE_SEARCH_QUERIES) != PART_CATEGORIES:
    raise RuntimeError(
        "Open-license search categories differ from the project "
        "part categories."
    )
