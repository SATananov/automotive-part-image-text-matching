from __future__ import annotations

from pathlib import Path

from src.dataset_config import PART_CATEGORIES, PART_FAMILIES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET_ROOT = PROJECT_ROOT / "data" / "real"
REAL_ORIGINALS_DIRECTORY = REAL_DATASET_ROOT / "originals"
REAL_CAPTURE_INBOX_DIRECTORY = REAL_DATASET_ROOT / "capture_inbox"
REAL_STAGING_DIRECTORY = REAL_DATASET_ROOT / "staging"
REAL_ANNOTATIONS_DIRECTORY = REAL_DATASET_ROOT / "annotations"
REAL_PART_GROUPS_PATH = REAL_ANNOTATIONS_DIRECTORY / "part_groups.csv"
REAL_IMAGES_PATH = REAL_ANNOTATIONS_DIRECTORY / "images.csv"
REAL_SAMPLE_INTAKE_PATH = (
    REAL_ANNOTATIONS_DIRECTORY / "sample_intake.csv"
)
REAL_PROCESSED_DIRECTORY = REAL_DATASET_ROOT / "processed"
REAL_PROCESSED_IMAGES_DIRECTORY = REAL_PROCESSED_DIRECTORY / "images"
REAL_IMAGE_MANIFEST_PATH = (
    REAL_PROCESSED_DIRECTORY / "real_image_manifest.csv"
)
REAL_APPROVAL_LOG_PATH = (
    REAL_PROCESSED_DIRECTORY / "approval_log.csv"
)

FIRST_BATCH_PLAN_PATH = (
    REAL_ANNOTATIONS_DIRECTORY / "first_batch_plan.csv"
)
FIRST_BATCH_CAPTURE_FILE_MAP_PATH = (
    REAL_ANNOTATIONS_DIRECTORY / "first_batch_capture_file_map.csv"
)
FIRST_BATCH_PREVIEW_PATH = (
    REAL_PROCESSED_DIRECTORY / "first_batch_queue_preview.csv"
)
FIRST_BATCH_CAPTURE_INBOX_DIRECTORY = (
    REAL_CAPTURE_INBOX_DIRECTORY / "batch_001"
)
FIRST_BATCH_ORIGINALS_DIRECTORY = (
    REAL_ORIGINALS_DIRECTORY / "batch_001"
)
FIRST_BATCH_CAPTURE_INVENTORY_PATH = (
    REAL_PROCESSED_DIRECTORY / "first_batch_capture_inventory.csv"
)
FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH = (
    REAL_PROCESSED_DIRECTORY / "first_batch_review_queue_draft.csv"
)
FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH = (
    REAL_PROCESSED_DIRECTORY / "first_batch_local_import_inventory.csv"
)
FIRST_BATCH_CAPTURE_SESSION_PATH = (
    REAL_PROCESSED_DIRECTORY / "first_batch_capture_session.csv"
)
FIRST_BATCH_CAPTURE_PROGRESS_PATH = (
    REAL_PROCESSED_DIRECTORY / "first_batch_capture_progress.csv"
)

FIRST_BATCH_PLAN_COLUMNS = (
    "batch_id",
    "batch_item_id",
    "intake_id",
    "staging_path",
    "part_group_id",
    "part_family",
    "part_category",
    "view",
    "source",
    "match_description",
    "partial_description",
    "mismatch_description",
    "notes",
)

FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS = (
    "batch_id",
    "batch_item_id",
    "intake_id",
    "capture_filename",
    "part_group_id",
    "part_category",
    "view",
    "staging_path",
)

FIRST_BATCH_LOCAL_IMPORT_INVENTORY_COLUMNS = (
    *FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    "inbox_source_path",
    "inbox_source_status",
    "original_destination_path",
    "import_status",
    "sha256",
    "file_size_bytes",
    "width",
    "height",
    "mode",
    "format",
)

FIRST_BATCH_CAPTURE_SESSION_COLUMNS = (
    "batch_id",
    "sequence",
    "part_group_id",
    "part_category",
    "front_filename",
    "detail_filename",
    "front_inbox_status",
    "detail_inbox_status",
    "front_original_status",
    "detail_original_status",
    "pair_status",
    "next_action",
)

FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS = (
    "batch_id",
    "batch_item_id",
    "intake_id",
    "part_group_id",
    "part_category",
    "view",
    "capture_filename",
    "capture_status",
    "import_status",
    "staging_status",
    "review_status",
    "queue_status",
    "decision_status",
    "approval_status",
    "pipeline_stage",
    "stage_index",
    "progress_percent",
    "next_action",
)

FIRST_BATCH_PREVIEW_COLUMNS = (
    *FIRST_BATCH_PLAN_COLUMNS,
    "file_present",
    "queue_status",
    "review_status",
    "review_errors",
    "review_warnings",
)

FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS = (
    *FIRST_BATCH_PLAN_COLUMNS,
    "capture_source_path",
    "capture_source_status",
    "staging_status",
    "staged_sha256",
    "width",
    "height",
    "mode",
    "format",
    "review_status",
    "review_errors",
    "review_warnings",
    "ready_for_queue",
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

SAMPLE_INTAKE_COLUMNS = (
    "intake_id",
    "staging_path",
    "part_group_id",
    "part_family",
    "part_category",
    "view",
    "source",
    "match_description",
    "partial_description",
    "mismatch_description",
    "decision",
    "rejection_reason",
    "notes",
)

APPROVAL_LOG_COLUMNS = (
    "intake_id",
    "decision",
    "part_group_id",
    "image_id",
    "processed_image_path",
    "sha256",
    "width",
    "height",
    "mode",
    "format",
    "quality_status",
    "processed_at_utc",
    "rejection_reason",
    "notes",
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

INTAKE_DECISION_VALUES = (
    "pending",
    "approved",
    "rejected",
)

CATEGORY_TO_FAMILY = {
    category: family
    for family, categories in PART_FAMILIES.items()
    for category in categories
}

REAL_DATASET_CATEGORIES = PART_CATEGORIES
REAL_ID_PREFIX = "real_"
INTAKE_ID_PREFIX = "intake_"

FIRST_BATCH_ID = "batch_001"
FIRST_BATCH_GROUP_NUMBER = "001"
FIRST_BATCH_VIEWS = ("front", "detail")
FIRST_BATCH_EXPECTED_GROUPS = len(REAL_DATASET_CATEGORIES)
FIRST_BATCH_EXPECTED_IMAGES = (
    FIRST_BATCH_EXPECTED_GROUPS * len(FIRST_BATCH_VIEWS)
)

MIN_IMAGE_WIDTH = 128
MIN_IMAGE_HEIGHT = 128
RECOMMENDED_IMAGE_WIDTH = 512
RECOMMENDED_IMAGE_HEIGHT = 512
MAX_IMAGE_ASPECT_RATIO = 4.0
LOW_LUMINANCE_WARNING = 20.0
HIGH_LUMINANCE_WARNING = 235.0
LOW_CONTRAST_WARNING = 10.0

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
