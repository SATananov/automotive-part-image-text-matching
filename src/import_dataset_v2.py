from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.data import DATA_DIR, PROJECT_ROOT

KAGGLE_HANDLE = "gpiosenka/car-parts-40-classes"
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes"
KAGGLE_LICENSE = "Apache-2.0"
KAGGLE_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_DATASET_URL = "https://commons.wikimedia.org/"
USER_AGENT = "AutomotivePartImageTextMatching/2.0 (educational exam project)"
TARGET_DIR = DATA_DIR / "images" / "dataset_v2"
MANIFEST_PATH = DATA_DIR / "dataset_v2_manifest.csv"
SUMMARY_PATH = DATA_DIR / "dataset_v2_import_summary.json"
CACHE_DIR = PROJECT_ROOT / ".cache" / "dataset_v2_source"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_SIZE = (224, 224)
# Dataset V2 uses the largest balanced quota that every category can support
# after exact and near-duplicate filtering. The canonical upper bound remains
# 20 train + 5 validation images per category, while sparse public-source
# categories may use a smaller, explicitly recorded balanced quota.
TRAIN_PER_CATEGORY = 20  # maximum imported training images per category
VALIDATION_PER_CATEGORY = 5  # maximum imported validation images per category
TOTAL_PER_CATEGORY = TRAIN_PER_CATEGORY + VALIDATION_PER_CATEGORY
MIN_TRAIN_PER_CATEGORY = 3
MIN_VALIDATION_PER_CATEGORY = 1
MIN_TOTAL_PER_CATEGORY = MIN_TRAIN_PER_CATEGORY + MIN_VALIDATION_PER_CATEGORY
DUPLICATE_DHASH_DISTANCE = 2
DUPLICATE_COSINE_LIMIT = 0.995
MIN_STANDARDIZED_PIXEL_STD = 8.0
MIN_STANDARDIZED_LUMA_ENTROPY = 2.0
MAX_STANDARDIZED_SINGLE_LUMA_FRACTION = 0.985

KAGGLE_CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "air_filter": ("AIR FILTER", "AIR FILTERS", "ENGINE AIR FILTER", "AIR CLEANER"),
    "alternator": ("ALTERNATOR",),
    "brake_disc": ("BRAKE ROTOR", "BRAKE DISC", "BRAKE DISK"),
    "brake_pad": ("BRAKE PAD", "BRAKE PADS"),
    "coil_spring": ("COIL SPRING",),
    "headlight": ("HEADLIGHTS", "HEADLIGHT", "HEAD LAMP"),
    "oil_filter": ("OIL FILTER",),
    "shock_absorber": (
        "SHOCK ABSORBER",
        "SHOCK ABSORBERS",
        "SUSPENSION STRUT",
        "STRUT",
        "DAMPER",
    ),
    "starter": ("STARTER", "STARTER MOTOR"),
    "taillight": ("TAILLIGHTS", "TAIL LIGHT", "TAILLIGHT"),
}
COMMONS_SEARCH_TERMS: dict[str, tuple[str, ...]] = {
    "air_filter": (
        'incategory:"Automobile engine air filters"',
        'incategory:"Automobile air filters"',
        '"automobile air filter"',
        '"car air filter"',
        '"engine air filter"',
    ),
    "shock_absorber": (
        'incategory:"Automobile shock absorbers"',
        'incategory:"Vehicle shock absorbers"',
        '"automobile shock absorber"',
        '"car shock absorber"',
        '"telescopic shock absorber"',
    ),
}
COMMONS_FALLBACK_CATEGORIES = frozenset(COMMONS_SEARCH_TERMS)
REQUIRED_KAGGLE_CATEGORIES = frozenset(KAGGLE_CATEGORY_ALIASES) - COMMONS_FALLBACK_CATEGORIES
ALL_CATEGORIES = tuple(sorted(KAGGLE_CATEGORY_ALIASES))
MANIFEST_COLUMNS = (
    "asset_id",
    "part_category",
    "assigned_split",
    "object_group_id",
    "local_path",
    "provider",
    "source_dataset",
    "dataset_url",
    "source_relative_path",
    "source_original_split",
    "source_title",
    "description_url",
    "author",
    "credit",
    "license_short_name",
    "license_url",
    "source_sha256",
    "sha256",
    "dhash",
    "modifications",
)


def normalize_name(value: str) -> str:
    return " ".join(
        "".join(character if character.isalnum() else " " for character in value.upper()).split()
    )


ALIAS_LOOKUP = {
    normalize_name(alias): category
    for category, aliases in KAGGLE_CATEGORY_ALIASES.items()
    for alias in aliases
}


def plain_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    return " ".join(html.unescape(value).split())


@dataclass(frozen=True)
class Candidate:
    category: str
    path: Path
    provider: str
    original_split: str
    original_sha256: str
    dhash: str
    gray_vector: np.ndarray
    source_dataset: str
    dataset_url: str
    source_relative_path: str
    source_title: str
    description_url: str
    author: str
    credit: str
    license_short_name: str
    license_url: str


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def difference_hash(path: Path) -> str:
    with Image.open(path) as image:
        pixels = np.asarray(
            ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    bits = (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def normalized_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        vector = np.asarray(
            ImageOps.exif_transpose(image).convert("L").resize((48, 48), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).reshape(-1)
    vector -= vector.mean()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def hamming_hex(left: str, right: str) -> int:
    return int((int(left, 16) ^ int(right, 16)).bit_count())


def infer_original_split(path: Path, source_root: Path) -> str:
    relative_parts = [normalize_name(part) for part in path.relative_to(source_root).parts[:-1]]
    for part in reversed(relative_parts):
        if part in {"TRAIN", "TRAINING"}:
            return "train"
        if part in {"VALID", "VALIDATION", "VAL"}:
            return "validation"
        if part in {"TEST", "TESTING"}:
            return "test"
    return "unspecified"


def safe_extract_zip(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        handle.extractall(destination)
    return destination


def discover_kaggle_class_directories(source_root: Path) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {category: [] for category in KAGGLE_CATEGORY_ALIASES}
    for directory in sorted(path for path in source_root.rglob("*") if path.is_dir()):
        category = ALIAS_LOOKUP.get(normalize_name(directory.name))
        if category is None:
            continue
        images = [
            path
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        discovered[category].extend(images)
    return discovered


def resolve_kaggle_root(downloaded: Path) -> Path:
    if downloaded.is_file() and zipfile.is_zipfile(downloaded):
        extracted = CACHE_DIR / "kaggle_extracted"
        if extracted.exists():
            shutil.rmtree(extracted)
        return safe_extract_zip(downloaded, extracted)
    if not downloaded.is_dir():
        raise RuntimeError(f"Unexpected Kaggle download path: {downloaded}")
    if any(discover_kaggle_class_directories(downloaded).values()):
        return downloaded
    archives = sorted(downloaded.rglob("*.zip"), key=lambda path: path.stat().st_size, reverse=True)
    if archives:
        extracted = CACHE_DIR / "kaggle_extracted"
        if extracted.exists():
            shutil.rmtree(extracted)
        return safe_extract_zip(archives[0], extracted)
    return downloaded


def download_kaggle_source() -> Path:
    # Use KaggleHub's managed cache rather than output_dir. A direct output_dir
    # is required to be empty and turns an interrupted first attempt into a
    # FileExistsError on every retry. The managed cache is retry-safe and can
    # be reused by a later invocation of the pipeline.
    managed_cache = CACHE_DIR / "kagglehub"
    managed_cache.mkdir(parents=True, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = str(managed_cache.resolve())

    try:
        import kagglehub  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "kagglehub is required for automatic acquisition. Install requirements-acquisition.txt."
        ) from error

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            downloaded = Path(kagglehub.dataset_download(KAGGLE_HANDLE))
            return resolve_kaggle_root(downloaded)
        except Exception as error:  # pragma: no cover - external network defense
            last_error = error
            if attempt < 3:
                delay = 2**attempt
                print(
                    f"Kaggle download attempt {attempt} failed: {error}. "
                    f"Retrying in {delay} seconds...",
                    file=sys.stderr,
                )
                time.sleep(delay)
    raise RuntimeError(
        "Kaggle dataset download failed after three attempts. "
        "Re-run the resume script to reuse the managed cache, or pass --source-root "
        "with a manually downloaded Kaggle archive/directory."
    ) from last_error


def fetch_json(params: dict[str, str]) -> dict[str, Any]:
    url = f"{COMMONS_API}?{urllib.parse.urlencode(params)}"
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # pragma: no cover - external network defense
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Wikimedia Commons API request failed: {url}") from last_error


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            if temporary.exists():
                temporary.unlink()
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            temporary.replace(destination)
            return
        except Exception as error:  # pragma: no cover - external network defense
            last_error = error
            if temporary.exists():
                temporary.unlink()
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Image download failed after three attempts: {url}") from last_error


def allowed_commons_license(short_name: str) -> bool:
    normalized = short_name.upper().replace(" ", "")
    return any(token in normalized for token in ("CCBY", "CC0", "PUBLICDOMAIN", "PDM"))


def commons_pages(category: str) -> list[dict[str, Any]]:
    pages: dict[int, dict[str, Any]] = {}
    for term in COMMONS_SEARCH_TERMS[category]:
        continuation: dict[str, str] = {}
        while len(pages) < 180:
            params = {
                "action": "query",
                "generator": "search",
                "gsrnamespace": "6",
                "gsrsearch": f"{term} filetype:bitmap",
                "gsrlimit": "50",
                "prop": "imageinfo|categories",
                "iiprop": "url|mime|size|extmetadata",
                "cllimit": "max",
                "iiurlwidth": "1200",
                "format": "json",
                "formatversion": "2",
                **continuation,
            }
            payload = fetch_json(params)
            for page in payload.get("query", {}).get("pages", []):
                pages[int(page["pageid"])] = page
            if "continue" not in payload:
                break
            continuation = {key: str(value) for key, value in payload["continue"].items()}
        if len(pages) >= 100:
            break
    return [pages[key] for key in sorted(pages)]


def commons_candidate_relevant(
    category: str,
    title: str,
    description: str,
    categories: tuple[str, ...] | list[str] = (),
) -> bool:
    haystack = " ".join((title, description, *categories)).lower()
    excluded = (
        "advert",
        "advertisement",
        "logo",
        "icon",
        "washing machine",
        "taipei",
        "skyscraper",
        "tuned mass damper",
        "fall arrest",
        "safety lanyard",
        "steelprotection",
        "aircraft",
        "landing gear",
        "industrial shock absorber",
        "railway",
        "bicycle",
        "motorcycle",
    )
    if any(term in haystack for term in excluded):
        return False
    if category == "air_filter":
        part_match = "air filter" in haystack or "luftfilter" in haystack
        vehicle_context = any(
            term in haystack
            for term in ("automobile", "automotive", "motor vehicle", "car ", "engine air filter")
        )
        return part_match and vehicle_context
    automotive_context = (
        "automobile shock absorber",
        "automobile shock absorbers",
        "automobile suspension",
        "automotive suspension",
        "vehicle shock absorber",
        "vehicle shock absorbers",
        "vehicle suspension",
        "car shock absorber",
        "macpherson strut",
        "lever arm shock absorber",
        "lever arm shock absorbers",
        "friction disk shock absorber",
        "friction disc shock absorber",
        "coilover",
    )
    return any(term in haystack for term in automotive_context)


def build_commons_candidates_from_api(category: str) -> list[Candidate]:
    cache = CACHE_DIR / "commons" / category
    candidates: list[Candidate] = []
    for page in commons_pages(category):
        imageinfo = (page.get("imageinfo") or [None])[0]
        if not imageinfo:
            continue
        mime = str(imageinfo.get("mime", ""))
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        if int(imageinfo.get("width", 0)) < 400 or int(imageinfo.get("height", 0)) < 400:
            continue
        metadata = imageinfo.get("extmetadata", {})
        value = lambda key: str(metadata.get(key, {}).get("value", ""))
        license_name = plain_text(value("LicenseShortName") or value("UsageTerms"))
        license_url = plain_text(value("LicenseUrl"))
        if not allowed_commons_license(license_name) or not license_url:
            continue
        title = str(page.get("title", ""))
        description = plain_text(value("ImageDescription"))
        categories = tuple(
            str(item.get("title", "")) for item in page.get("categories", [])
        )
        if not commons_candidate_relevant(category, title, description, categories):
            continue
        source_urls = [
            str(value)
            for value in (imageinfo.get("thumburl"), imageinfo.get("url"))
            if value
        ]
        source_urls = list(dict.fromkeys(source_urls))
        if not source_urls:
            continue
        suffix = ".png" if mime == "image/png" else ".webp" if mime == "image/webp" else ".jpg"
        path = cache / f"commons_{int(page['pageid'])}{suffix}"
        try:
            if not path.is_file():
                download_errors: list[str] = []
                for source_url in source_urls:
                    try:
                        download_file(source_url, path)
                        break
                    except Exception as error:
                        download_errors.append(str(error))
                else:
                    raise RuntimeError("; ".join(download_errors))
            with Image.open(path) as image:
                image.verify()
            digest = sha256(path)
            candidates.append(
                Candidate(
                    category=category,
                    path=path,
                    provider="Wikimedia Commons",
                    original_split="unspecified",
                    original_sha256=digest,
                    dhash=difference_hash(path),
                    gray_vector=normalized_gray(path),
                    source_dataset=f"Wikimedia Commons search: {category}",
                    dataset_url=COMMONS_DATASET_URL,
                    source_relative_path=title,
                    source_title=title,
                    description_url=(str(imageinfo.get("descriptionurl", "")) or f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"),
                    author=plain_text(value("Artist")) or "See Wikimedia Commons source page",
                    credit=plain_text(value("Credit")) or "See Wikimedia Commons source page",
                    license_short_name=license_name,
                    license_url=license_url,
                )
            )
        except Exception as error:  # pragma: no cover - network and third-party data defense
            print(f"Skipping Commons image {title}: {error}", file=sys.stderr)
    return candidates


def build_commons_candidates_from_root(category: str, root: Path) -> list[Candidate]:
    directory = root / category
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing offline Commons category directory: {directory}")
    rows: list[Candidate] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        with Image.open(path) as image:
            image.verify()
        digest = sha256(path)
        rows.append(
            Candidate(
                category=category,
                path=path,
                provider="Offline Wikimedia-compatible source",
                original_split="unspecified",
                original_sha256=digest,
                dhash=difference_hash(path),
                gray_vector=normalized_gray(path),
                source_dataset="Offline source supplied to Dataset V2 importer",
                dataset_url="local://commons-source-root",
                source_relative_path=path.relative_to(root).as_posix(),
                source_title=path.name,
                description_url="local://commons-source-root",
                author="Offline test/source metadata",
                credit="Offline test/source metadata",
                license_short_name="CC0-1.0",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            )
        )
    return rows


def build_kaggle_candidates(source_root: Path) -> dict[str, list[Candidate]]:
    discovered = discover_kaggle_class_directories(source_root)
    missing = [
        category
        for category in sorted(REQUIRED_KAGGLE_CATEGORIES)
        if not discovered.get(category)
    ]
    if missing:
        available = sorted({normalize_name(path.name) for path in source_root.rglob("*") if path.is_dir()})
        raise ValueError(f"Missing required Kaggle classes: {missing}. Available: {available[:100]}")
    seen_exact: set[str] = set()
    candidates: dict[str, list[Candidate]] = {category: [] for category in discovered}
    for category, paths in discovered.items():
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                digest = sha256(path)
                if digest in seen_exact:
                    continue
                seen_exact.add(digest)
                candidates[category].append(
                    Candidate(
                        category=category,
                        path=path,
                        provider="Kaggle",
                        original_split=infer_original_split(path, source_root),
                        original_sha256=digest,
                        dhash=difference_hash(path),
                        gray_vector=normalized_gray(path),
                        source_dataset=KAGGLE_HANDLE,
                        dataset_url=KAGGLE_DATASET_URL,
                        source_relative_path=path.relative_to(source_root).as_posix(),
                        source_title=path.name,
                        description_url=KAGGLE_DATASET_URL,
                        author="Gpiosenka dataset contributor",
                        credit="50 Types of Car Parts dataset on Kaggle",
                        license_short_name=KAGGLE_LICENSE,
                        license_url=KAGGLE_LICENSE_URL,
                    )
                )
            except Exception as error:  # pragma: no cover
                print(f"Skipping unreadable Kaggle image {path}: {error}", file=sys.stderr)
    return candidates


def current_reference_features() -> dict[str, list[tuple[str, np.ndarray]]]:
    references: dict[str, list[tuple[str, np.ndarray]]] = {category: [] for category in ALL_CATEGORIES}
    manifest_path = DATA_DIR / "image_manifest.csv"
    if not manifest_path.exists():
        return references
    manifest = pd.read_csv(manifest_path)
    for row in manifest.itertuples(index=False):
        if getattr(row, "source", None) == "dataset_v2" or row.part_category not in references:
            continue
        path = PROJECT_ROOT / row.image_path
        if path.is_file():
            references[row.part_category].append((difference_hash(path), normalized_gray(path)))
    return references


def is_near_duplicate(candidate: Candidate, accepted: list[tuple[str, np.ndarray]]) -> bool:
    for existing_hash, existing_vector in accepted:
        if hamming_hex(candidate.dhash, existing_hash) <= DUPLICATE_DHASH_DISTANCE:
            return True
        if float(np.dot(candidate.gray_vector, existing_vector)) >= DUPLICATE_COSINE_LIMIT:
            return True
    return False


def stable_order(candidates: list[Candidate], preferred_splits: set[str]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda item: (
            0 if item.original_split in preferred_splits else 1,
            item.original_sha256,
            item.path.as_posix().lower(),
        ),
    )


def balanced_selection_quota(unique_counts: dict[str, int]) -> tuple[int, int]:
    """Return the largest deterministic quota supported by every category.

    The importer keeps category counts exactly balanced so relation-pair
    construction cannot learn a category-frequency shortcut. Validation uses
    roughly one fifth to one quarter of the imported images, capped at five.
    """
    missing = [category for category in ALL_CATEGORIES if category not in unique_counts]
    if missing:
        raise ValueError(f"Missing deduplicated category counts: {missing}")
    bottleneck = min(unique_counts.values())
    if bottleneck < MIN_TOTAL_PER_CATEGORY:
        raise ValueError(
            "Dataset V2 cannot form the minimum balanced quota: "
            f"smallest category has {bottleneck} non-duplicate images, "
            f"but at least {MIN_TOTAL_PER_CATEGORY} are required "
            f"({MIN_TRAIN_PER_CATEGORY} train + {MIN_VALIDATION_PER_CATEGORY} validation)."
        )

    validation_quota = min(
        VALIDATION_PER_CATEGORY,
        max(MIN_VALIDATION_PER_CATEGORY, bottleneck // 4),
    )
    train_quota = min(TRAIN_PER_CATEGORY, bottleneck - validation_quota)
    if train_quota < MIN_TRAIN_PER_CATEGORY:
        validation_quota = bottleneck - MIN_TRAIN_PER_CATEGORY
        train_quota = MIN_TRAIN_PER_CATEGORY
    return train_quota, validation_quota


def deduplicate_candidates(
    candidates: dict[str, list[Candidate]],
) -> dict[str, list[Candidate]]:
    references = current_reference_features()
    unique_by_category: dict[str, list[Candidate]] = {}
    globally_seen_source_hashes: set[str] = set()
    for category in ALL_CATEGORIES:
        rows = candidates.get(category, [])
        accepted_features = list(references[category])
        unique_rows: list[Candidate] = []
        seen_source_hashes: set[str] = set()
        for candidate in stable_order(rows, {"train", "validation", "test"}):
            if (
                candidate.original_sha256 in seen_source_hashes
                or candidate.original_sha256 in globally_seen_source_hashes
            ):
                continue
            seen_source_hashes.add(candidate.original_sha256)
            if is_near_duplicate(candidate, accepted_features):
                continue
            unique_rows.append(candidate)
            globally_seen_source_hashes.add(candidate.original_sha256)
            accepted_features.append((candidate.dhash, candidate.gray_vector))
        unique_by_category[category] = unique_rows
    return unique_by_category


def select_candidates(candidates: dict[str, list[Candidate]]) -> dict[str, dict[str, list[Candidate]]]:
    unique_by_category = deduplicate_candidates(candidates)
    unique_counts = {category: len(rows) for category, rows in unique_by_category.items()}
    train_quota, validation_quota = balanced_selection_quota(unique_counts)
    print(
        "Balanced Dataset V2 quota after duplicate filtering: "
        f"train={train_quota}, validation={validation_quota} per category; "
        f"bottleneck={min(unique_counts.values())} non-duplicate images."
    )

    selected: dict[str, dict[str, list[Candidate]]] = {}
    for category in ALL_CATEGORIES:
        unique_rows = unique_by_category[category]
        validation_pool = stable_order(unique_rows, {"validation", "test"})
        validation = validation_pool[:validation_quota]
        validation_hashes = {item.original_sha256 for item in validation}
        remaining = [item for item in unique_rows if item.original_sha256 not in validation_hashes]
        train = stable_order(remaining, {"train"})[:train_quota]
        if len(train) != train_quota or len(validation) != validation_quota:
            raise ValueError(
                f"Balanced selection failed for {category}: "
                f"train={len(train)}/{train_quota}, validation={len(validation)}/{validation_quota}, "
                f"discovered={len(candidates.get(category, []))}, after_dedup={len(unique_rows)}"
            )
        selected[category] = {"train": train, "validation": validation}
    return selected


def prepare_rgb_image(image: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return oriented.convert("RGB")


def image_content_metrics(image: Image.Image) -> tuple[float, float, float]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    counts = np.bincount(gray.reshape(-1), minlength=256)
    probabilities = counts[counts > 0].astype(np.float64) / gray.size
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return float(gray.std()), entropy, float(counts.max() / gray.size)


def validate_standardized_image(image_or_path: Image.Image | Path) -> tuple[float, float, float]:
    if isinstance(image_or_path, Path):
        with Image.open(image_or_path) as image:
            if image.size != TARGET_SIZE or image.mode != "RGB":
                raise ValueError(
                    f"Standardized image must be RGB {TARGET_SIZE[0]}x{TARGET_SIZE[1]}: {image_or_path}"
                )
            metrics = image_content_metrics(image)
    else:
        if image_or_path.size != TARGET_SIZE or image_or_path.mode != "RGB":
            raise ValueError(
                f"Standardized image must be RGB {TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
            )
        metrics = image_content_metrics(image_or_path)
    pixel_std, entropy, dominant_fraction = metrics
    if pixel_std < MIN_STANDARDIZED_PIXEL_STD:
        raise ValueError(f"Image pixel standard deviation is too low: {pixel_std:.4f}")
    if entropy < MIN_STANDARDIZED_LUMA_ENTROPY:
        raise ValueError(f"Image luminance entropy is too low: {entropy:.4f}")
    if dominant_fraction > MAX_STANDARDIZED_SINGLE_LUMA_FRACTION:
        raise ValueError(
            f"One luminance value occupies too much of the image: {dominant_fraction:.4f}"
        )
    return metrics


def standardize_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        transformed = ImageOps.fit(
            prepare_rgb_image(image),
            TARGET_SIZE,
            method=Image.Resampling.LANCZOS,
        )
        validate_standardized_image(transformed)
        transformed.save(destination, format="JPEG", quality=92, optimize=True)


def write_selected(selected: dict[str, dict[str, list[Candidate]]], *, force: bool) -> pd.DataFrame:
    if TARGET_DIR.exists():
        if not force:
            raise FileExistsError(f"{TARGET_DIR} exists. Re-run with --force.")
        shutil.rmtree(TARGET_DIR)
    rows: list[dict[str, object]] = []
    for category in sorted(selected):
        counter = 0
        for assigned_split in ("train", "validation"):
            for candidate in selected[category][assigned_split]:
                counter += 1
                asset_id = f"dataset_v2_{category}_{counter:03d}"
                destination = TARGET_DIR / category / f"{asset_id}.jpg"
                standardize_image(candidate.path, destination)
                rows.append(
                    {
                        "asset_id": asset_id,
                        "part_category": category,
                        "assigned_split": assigned_split,
                        "object_group_id": f"dataset_v2_object_{category}_{counter:03d}",
                        "local_path": destination.relative_to(PROJECT_ROOT).as_posix(),
                        "provider": candidate.provider,
                        "source_dataset": candidate.source_dataset,
                        "dataset_url": candidate.dataset_url,
                        "source_relative_path": candidate.source_relative_path,
                        "source_original_split": candidate.original_split,
                        "source_title": candidate.source_title,
                        "description_url": candidate.description_url,
                        "author": candidate.author,
                        "credit": candidate.credit,
                        "license_short_name": candidate.license_short_name,
                        "license_url": candidate.license_url,
                        "source_sha256": candidate.original_sha256,
                        "sha256": sha256(destination),
                        "dhash": difference_hash(destination),
                        "modifications": "EXIF orientation applied; transparency composited on white; RGB conversion; center crop and resize to 224x224; JPEG quality 92; low-information content validation passed.",
                    }
                )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(
        ["assigned_split", "part_category", "asset_id"], kind="stable"
    )
    manifest.to_csv(MANIFEST_PATH, index=False, lineterminator="\n")
    return manifest


def validate_import(manifest: pd.DataFrame) -> dict[str, object]:
    if manifest.empty:
        raise AssertionError("Dataset V2 manifest is empty")
    if set(manifest["part_category"]) != set(ALL_CATEGORIES):
        raise AssertionError("Dataset V2 manifest does not cover all project categories")

    counts = manifest.groupby(["assigned_split", "part_category"]).size().unstack(fill_value=0)
    if set(counts.index) != {"train", "validation"}:
        raise AssertionError(f"Unexpected Dataset V2 splits: {list(counts.index)}")
    if counts.loc["train"].nunique() != 1 or counts.loc["validation"].nunique() != 1:
        raise AssertionError("Dataset V2 category counts are not balanced")

    train_per_category = int(counts.loc["train"].iloc[0])
    validation_per_category = int(counts.loc["validation"].iloc[0])
    if not MIN_TRAIN_PER_CATEGORY <= train_per_category <= TRAIN_PER_CATEGORY:
        raise AssertionError(f"Unexpected imported training quota: {train_per_category}")
    if not MIN_VALIDATION_PER_CATEGORY <= validation_per_category <= VALIDATION_PER_CATEGORY:
        raise AssertionError(f"Unexpected imported validation quota: {validation_per_category}")

    expected_total = len(ALL_CATEGORIES) * (train_per_category + validation_per_category)
    if len(manifest) != expected_total:
        raise AssertionError(f"Expected {expected_total} imported images, got {len(manifest)}")
    if manifest["sha256"].duplicated().any():
        raise AssertionError("Duplicate standardized image hash in Dataset V2")
    for row in manifest.itertuples(index=False):
        path = PROJECT_ROOT / row.local_path
        if not path.is_file() or sha256(path) != row.sha256:
            raise AssertionError(f"Imported image hash mismatch: {row.asset_id}")
        try:
            validate_standardized_image(path)
        except ValueError as error:
            raise AssertionError(f"Imported image content check failed: {row.asset_id}: {error}") from error
    summary = {
        "status": "PASS",
        "selection_policy": "largest_balanced_quota_after_exact_and_near_duplicate_filtering",
        "providers": manifest["provider"].value_counts().to_dict(),
        "source_datasets": sorted(manifest["source_dataset"].unique()),
        "recorded_licenses": sorted(manifest["license_short_name"].unique()),
        "imported_images": int(len(manifest)),
        "training_images": int(manifest["assigned_split"].eq("train").sum()),
        "validation_images": int(manifest["assigned_split"].eq("validation").sum()),
        "categories": list(ALL_CATEGORIES),
        "per_category_train": train_per_category,
        "per_category_validation": validation_per_category,
        "maximum_per_category_train": TRAIN_PER_CATEGORY,
        "maximum_per_category_validation": VALIDATION_PER_CATEGORY,
        "minimum_per_category_train": MIN_TRAIN_PER_CATEGORY,
        "minimum_per_category_validation": MIN_VALIDATION_PER_CATEGORY,
        "duplicate_dhash_threshold": DUPLICATE_DHASH_DISTANCE,
        "duplicate_cosine_threshold": DUPLICATE_COSINE_LIMIT,
        "minimum_pixel_standard_deviation": MIN_STANDARDIZED_PIXEL_STD,
        "minimum_luminance_entropy": MIN_STANDARDIZED_LUMA_ENTROPY,
        "maximum_single_luminance_fraction": MAX_STANDARDIZED_SINGLE_LUMA_FRACTION,
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def import_dataset(
    kaggle_root: Path,
    *,
    commons_root: Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    kaggle_root = resolve_kaggle_root(kaggle_root.resolve())
    if not kaggle_root.is_dir():
        raise FileNotFoundError(f"Kaggle source directory does not exist: {kaggle_root}")
    candidates = build_kaggle_candidates(kaggle_root)
    for category in sorted(COMMONS_FALLBACK_CATEGORIES):
        kaggle_count = len(candidates.get(category, []))
        if kaggle_count >= TOTAL_PER_CATEGORY:
            print(
                f"Using Kaggle for {category}: discovered {kaggle_count} candidate images; "
                "Wikimedia fallback is not required."
            )
            continue

        print(
            f"Kaggle provided {kaggle_count} candidate images for {category}; "
            "supplementing with the Wikimedia-compatible fallback."
        )
        fallback_candidates = (
            build_commons_candidates_from_root(category, commons_root.resolve())
            if commons_root is not None
            else build_commons_candidates_from_api(category)
        )
        candidates.setdefault(category, []).extend(fallback_candidates)
    selected = select_candidates(candidates)
    return validate_import(write_selected(selected, force=force))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire, deduplicate, standardize, and select the 10-category Dataset V2 subset."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Use an already downloaded 50 Types of Car Parts directory instead of KaggleHub.",
    )
    parser.add_argument(
        "--commons-source-root",
        type=Path,
        default=None,
        help="Offline testing/fallback root containing air_filter and shock_absorber directories.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kaggle_root = args.source_root if args.source_root is not None else download_kaggle_source()
    summary = import_dataset(
        kaggle_root,
        commons_root=args.commons_source_root,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
