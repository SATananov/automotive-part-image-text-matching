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
OPEN_LICENSE_SEARCH_LIMIT = 80
OPEN_LICENSE_MIN_WIDTH = 256
OPEN_LICENSE_MIN_HEIGHT = 256
OPEN_LICENSE_MAX_ASPECT_RATIO = 4.0

OPEN_LICENSE_SEARCH_QUERIES = {
    "starter": (
        'intitle:"Automobile starter 2.JPG"',
        'intitle:"Automobile starter.JPG"',
        'intitle:"Starter motor.JPG"',
        '"MOTOR STARTER.jpg"',
        '"Starter motor.JPG"',
        '"Motor starter.jpg"',
        'incategory:"Electric starter motors"',
        'incategory:"Engine starters" "starter motor"',
        'incategory:"Automobile electrics" "starter motor"',
        '"Automobile starter 2.JPG"',
        '"automobile starter motor"',
        '"automotive starter motor"',
        '"car starter motor"',
        '"electric starter motor" automobile',
        '"motor de arranque" automobile',
        'incategory:"Automobile parts" "starter motor"',
    ),
    "alternator": (
        'incategory:"Automobile alternators"',
        'incategory:"Automobile engine parts" alternator',
        '"automobile alternator"',
        '"automotive alternator"',
        '"vehicle alternator"',
        '"car charging alternator"',
    ),
    "brake_disc": (
        'intitle:"Brake Discs.jpg"',
        'intitle:"Disk brake dsc03682.jpg"',
        'intitle:"Disc brake car.jpg"',
        '"Disc brakes.jpg"',
        '"Disc brake car.jpg"',
        '"Scheibenbremse(Kfz).JPG"',
        '"Hamulec tarczowy.jpg"',
        '"Detail of - AMC Pacer - right front disc brake and suspension system.jpg"',
        'incategory:"Automobile disk brakes" "brake disc"',
        'incategory:"Automobile disk brakes" rotor',
        'incategory:"Automobile disk brakes"',
        '"automobile brake disc"',
        '"car brake rotor"',
        '"passenger car brake disc"',
        'incategory:"Brake disks" automobile',
        'incategory:"Disk brakes" automobile rotor',
        'incategory:"Brake disks"',
    ),
    "brake_pad": (
        '"Brake pads.JPG"',
        '"Performance Disk Brake Pads.jpg"',
        '"Bremsbeläge-abgefahren.JPG"',
        '"Replacing brake pads 140205-A-GJ352-044.jpg"',
        '"Automobile brake pad.jpg"',
        '"Brake pads.JPG"',
        '"Brakepad.jpg"',
        '"Performance Disk Brake Pads.jpg"',
        '"Brake pads fitted with spring anchor.jpg"',
        'incategory:"Automobile disk brakes" "brake pad"',
        'incategory:"Disk brakes" "automobile brake pad"',
        'incategory:"Brake blocks" automobile',
        '"automobile brake pads"',
        '"car brake pads"',
        '"passenger car brake pad"',
        '"disc brake pad" automobile',
        'incategory:"Brake pads"',
    ),
    "shock_absorber": (
        'incategory:"Automobile shock absorbers"',
        'incategory:"Shock absorbers" automobile',
        '"automobile shock absorber"',
        '"vehicle shock absorber"',
        '"car suspension damper"',
        '"automotive shock absorber"',
    ),
    "coil_spring": (
        'incategory:"Coil spring automobile suspension"',
        'incategory:"Automobile springs" "coil spring"',
        'incategory:"Automobile suspension" "coil spring"',
        'incategory:"Coil spring vehicle suspension"',
        'incategory:"Coil springs" automobile',
        '"automobile suspension coil spring"',
        '"automobile coil spring"',
        '"vehicle suspension coil spring"',
        '"car suspension spring"',
        '"automotive coil spring"',
        '"coilover spring" automobile',
        '"MacPherson strut spring"',
    ),
    "headlight": (
        'incategory:"Automobile headlamps"',
        'incategory:"Automobile lights" headlamp',
        '"automobile headlamp"',
        '"automobile headlight"',
        '"vehicle headlamp assembly"',
        '"car headlight assembly"',
    ),
    "taillight": (
        'incategory:"Automobile rear lights"',
        'incategory:"Automobile lights" "rear light"',
        '"automobile rear light"',
        '"automobile tail lamp"',
        '"vehicle taillight assembly"',
        '"car rear lamp assembly"',
    ),
    "oil_filter": (
        'incategory:"Automobile oil filters"',
        'incategory:"Oil filters" automobile',
        'incategory:"Paper filters" "oil filter"',
        '"automotive oil filter"',
        '"automobile oil filter"',
        '"engine oil filter" car',
    ),
    "air_filter": (
        'incategory:"Automobile engine air filters"',
        'incategory:"Automobile air filters"',
        'incategory:"Paper filters" "air filter"',
        '"automotive engine air filter"',
        '"automobile air filter"',
        '"car engine air filter"',
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
