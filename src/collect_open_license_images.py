from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

from src.open_license_dataset_config import (
    CATEGORY_TO_FAMILY,
    COMMONS_API_URL,
    COMMONS_USER_AGENT,
    OPEN_LICENSE_ATTRIBUTION_PATH,
    OPEN_LICENSE_COLLECTION_REPORT_PATH,
    OPEN_LICENSE_IMAGES_DIRECTORY,
    OPEN_LICENSE_MANIFEST_COLUMNS,
    OPEN_LICENSE_MANIFEST_PATH,
    OPEN_LICENSE_MAX_ASPECT_RATIO,
    OPEN_LICENSE_MIN_HEIGHT,
    OPEN_LICENSE_MIN_WIDTH,
    OPEN_LICENSE_REVIEW_COLUMNS,
    OPEN_LICENSE_REVIEW_PATH,
    OPEN_LICENSE_SEARCH_LIMIT,
    OPEN_LICENSE_SEARCH_QUERIES,
    OPEN_LICENSE_TARGET_PER_CATEGORY,
    OPEN_LICENSE_THUMBNAIL_WIDTH,
)


class OpenLicenseCollectionError(RuntimeError):
    pass


def clean_metadata_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def metadata_value(
    extmetadata: dict[str, Any],
    key: str,
) -> str:
    item = extmetadata.get(key, {})
    if isinstance(item, dict):
        return clean_metadata_text(item.get("value", ""))
    return clean_metadata_text(item)


def license_is_allowed(license_short_name: str) -> bool:
    normalized = " ".join(license_short_name.upper().split())
    return (
        normalized == "PUBLIC DOMAIN"
        or normalized.startswith("CC0")
        or normalized.startswith("CC BY ")
        or normalized.startswith("CC BY-SA ")
    )


def derived_license_url(license_short_name: str) -> str:
    normalized = " ".join(license_short_name.split())
    upper = normalized.upper()

    if upper == "PUBLIC DOMAIN":
        return (
            "https://commons.wikimedia.org/wiki/"
            "Commons:Public_domain"
        )
    if upper.startswith("CC0"):
        return (
            "https://creativecommons.org/publicdomain/zero/1.0/"
        )

    match = re.fullmatch(
        r"CC (BY|BY-SA) ([0-9.]+)",
        upper,
    )
    if not match:
        return ""

    license_code = match.group(1).lower()
    version = match.group(2)
    return (
        f"https://creativecommons.org/licenses/"
        f"{license_code}/{version}/"
    )


def parse_boolean(value: object) -> bool:
    normalized = clean_metadata_text(value).strip().lower()
    return normalized in {"1", "true", "yes", "required"}


def api_request_json(
    params: dict[str, object],
    *,
    timeout: float = 30.0,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{COMMONS_API_URL}?{query}",
        headers={
            "User-Agent": COMMONS_USER_AGENT,
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        payload = response.read()
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise OpenLicenseCollectionError(
            "Wikimedia Commons returned an invalid JSON payload."
        )
    return decoded


def download_url_bytes(
    url: str,
    *,
    timeout: float = 45.0,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": COMMONS_USER_AGENT},
    )
    with urllib.request.urlopen(
        request,
        timeout=timeout,
    ) as response:
        return response.read()


def search_commons(
    search_query: str,
    *,
    request_json: Callable[
        [dict[str, object]],
        dict[str, Any],
    ] = api_request_json,
) -> list[dict[str, Any]]:
    params: dict[str, object] = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "search",
        "gsrsearch": search_query,
        "gsrnamespace": "6",
        "gsrlimit": str(OPEN_LICENSE_SEARCH_LIMIT),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata",
        "iiurlwidth": str(OPEN_LICENSE_THUMBNAIL_WIDTH),
    }
    payload = request_json(params)
    pages = payload.get("query", {}).get("pages", [])
    if not isinstance(pages, list):
        return []
    return [
        page
        for page in pages
        if isinstance(page, dict)
    ]


def candidate_from_page(
    page: dict[str, Any],
    *,
    part_category: str,
    search_query: str,
) -> dict[str, str] | None:
    imageinfo_items = page.get("imageinfo", [])
    if not isinstance(imageinfo_items, list) or not imageinfo_items:
        return None

    imageinfo = imageinfo_items[0]
    if not isinstance(imageinfo, dict):
        return None

    mime_type = clean_metadata_text(imageinfo.get("mime", ""))
    if mime_type not in {"image/jpeg", "image/png"}:
        return None

    extmetadata = imageinfo.get("extmetadata", {})
    if not isinstance(extmetadata, dict):
        extmetadata = {}

    license_short_name = metadata_value(
        extmetadata,
        "LicenseShortName",
    )
    if not license_is_allowed(license_short_name):
        return None

    author = metadata_value(extmetadata, "Artist")
    credit = metadata_value(extmetadata, "Credit")
    attribution_required = parse_boolean(
        metadata_value(extmetadata, "AttributionRequired")
    )
    if attribution_required and not (author or credit):
        return None

    license_url = metadata_value(extmetadata, "LicenseUrl")
    if not license_url:
        license_url = derived_license_url(license_short_name)
    if not license_url:
        return None

    download_url = clean_metadata_text(
        imageinfo.get("thumburl")
        or imageinfo.get("url")
        or ""
    )
    original_url = clean_metadata_text(imageinfo.get("url", ""))
    description_url = clean_metadata_text(
        imageinfo.get("descriptionurl", "")
    )
    title = clean_metadata_text(page.get("title", ""))
    page_id = clean_metadata_text(page.get("pageid", ""))

    if not (
        download_url
        and original_url
        and description_url
        and title
        and page_id
    ):
        return None

    return {
        "part_family": CATEGORY_TO_FAMILY[part_category],
        "part_category": part_category,
        "search_query": search_query,
        "commons_page_id": page_id,
        "commons_title": title,
        "description_url": description_url,
        "original_url": original_url,
        "download_url": download_url,
        "author": author,
        "credit": credit,
        "license_short_name": license_short_name,
        "license_url": license_url,
        "attribution_required": (
            "yes" if attribution_required else "no"
        ),
        "usage_terms": metadata_value(
            extmetadata,
            "UsageTerms",
        ),
    }


def inspect_image_bytes(
    image_bytes: bytes,
) -> tuple[int, int, str, str]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source_image:
            image_format = str(
                source_image.format or ""
            ).upper()
            image = ImageOps.exif_transpose(source_image)
            image.load()
            width, height = image.size
    except Exception as error:
        raise OpenLicenseCollectionError(
            f"Downloaded file is not a readable image: {error}"
        ) from error

    if image_format not in {"JPEG", "PNG"}:
        raise OpenLicenseCollectionError(
            f"Unsupported downloaded image format: {image_format}."
        )
    if (
        width < OPEN_LICENSE_MIN_WIDTH
        or height < OPEN_LICENSE_MIN_HEIGHT
    ):
        raise OpenLicenseCollectionError(
            f"Image is too small: {width}x{height}."
        )

    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > OPEN_LICENSE_MAX_ASPECT_RATIO:
        raise OpenLicenseCollectionError(
            f"Image aspect ratio is too extreme: {aspect_ratio:.2f}."
        )

    extension = ".jpg" if image_format == "JPEG" else ".png"
    return width, height, image_format, extension


def read_csv_rows(
    path: Path,
    expected_columns: tuple[str, ...],
) -> list[dict[str, str]]:
    if not path.is_file():
        return []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_columns:
            raise OpenLicenseCollectionError(
                f"Unexpected CSV schema in {path}."
            )
        return [
            {
                column: str(row.get(column, "")).strip()
                for column in expected_columns
            }
            for row in reader
        ]


def atomic_write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open(
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
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            text,
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_review_rows(
    manifest_rows: list[dict[str, str]],
    existing_review_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    existing_by_id = {
        row["asset_id"]: row
        for row in existing_review_rows
    }
    review_rows: list[dict[str, str]] = []

    for manifest_row in manifest_rows:
        asset_id = manifest_row["asset_id"]
        existing = existing_by_id.get(asset_id, {})
        review_rows.append(
            {
                "asset_id": asset_id,
                "part_family": manifest_row["part_family"],
                "part_category": manifest_row["part_category"],
                "local_path": manifest_row["local_path"],
                "commons_title": manifest_row["commons_title"],
                "author": manifest_row["author"],
                "license_short_name": (
                    manifest_row["license_short_name"]
                ),
                "license_url": manifest_row["license_url"],
                "description_url": (
                    manifest_row["description_url"]
                ),
                "operator_decision": existing.get(
                    "operator_decision",
                    "pending",
                ),
                "rejection_reason": existing.get(
                    "rejection_reason",
                    "",
                ),
                "operator_notes": existing.get(
                    "operator_notes",
                    "",
                ),
            }
        )

    return review_rows


def render_attribution(
    manifest_rows: list[dict[str, str]],
) -> str:
    lines = [
        "# Open-License Image Attribution",
        "",
        (
            "Each image below keeps the license stated on its "
            "Wikimedia Commons file page."
        ),
        (
            "The local copy is a Wikimedia thumbnail resized to a "
            "maximum width of 1024 pixels; otherwise it is unmodified."
        ),
        "",
    ]

    current_category = ""
    for row in sorted(
        manifest_rows,
        key=lambda item: (
            item["part_category"],
            item["asset_id"],
        ),
    ):
        category = row["part_category"]
        if category != current_category:
            current_category = category
            lines.extend(
                [
                    f"## {category}",
                    "",
                ]
            )

        creator = (
            row["author"]
            or row["credit"]
            or "Creator listed on the Commons file page"
        )
        lines.extend(
            [
                f"### {row['asset_id']}",
                "",
                f"- Title: {row['commons_title']}",
                f"- Creator/credit: {creator}",
                f"- License: {row['license_short_name']}",
                f"- License URL: {row['license_url']}",
                f"- Source page: {row['description_url']}",
                f"- Local file: `{row['local_path']}`",
                f"- Modifications: {row['modifications']}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def render_collection_summary(
    *,
    manifest_rows: list[dict[str, str]],
    added_count: int,
    warnings: list[str],
) -> str:
    category_counts = {
        category: 0
        for category in OPEN_LICENSE_SEARCH_QUERIES
    }
    for row in manifest_rows:
        category_counts[row["part_category"]] += 1

    complete_categories = sum(
        count >= OPEN_LICENSE_TARGET_PER_CATEGORY
        for count in category_counts.values()
    )
    readiness = (
        "READY_FOR_MANUAL_REVIEW"
        if complete_categories == len(category_counts)
        else "COLLECTION_INCOMPLETE"
    )

    lines = [
        "# Step 010.1 — Open-License Image Collection",
        "",
        "- Status: **PASS**",
        f"- Readiness: **{readiness}**",
        f"- Total images: **{len(manifest_rows)}**",
        f"- Added in this run: **{added_count}**",
        (
            "- Complete categories: "
            f"**{complete_categories} / {len(category_counts)}**"
        ),
        "",
        "## Category counts",
        "",
    ]
    lines.extend(
        f"- {category}: {count}"
        for category, count in category_counts.items()
    )

    lines.extend(
        [
            "",
            "All downloaded rows remain `pending` until manual review.",
            (
                "The collection is separate from the warehouse-photo "
                "real-data workflow."
            ),
        ]
    )

    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines).rstrip() + "\n"


def snapshot_files(
    paths: tuple[Path, ...],
) -> dict[Path, bytes | None]:
    return {
        path: path.read_bytes() if path.is_file() else None
        for path in paths
    }


def restore_files(
    snapshots: dict[Path, bytes | None],
) -> None:
    for path, content in snapshots.items():
        if content is None:
            if path.exists():
                path.unlink()
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def collect_open_license_images(
    *,
    request_json: Callable[
        [dict[str, object]],
        dict[str, Any],
    ] = api_request_json,
    download_bytes: Callable[[str], bytes] = download_url_bytes,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    existing_manifest = read_csv_rows(
        OPEN_LICENSE_MANIFEST_PATH,
        OPEN_LICENSE_MANIFEST_COLUMNS,
    )
    existing_review = read_csv_rows(
        OPEN_LICENSE_REVIEW_PATH,
        OPEN_LICENSE_REVIEW_COLUMNS,
    )

    existing_by_page_category = {
        (
            row["part_category"],
            row["commons_page_id"],
        )
        for row in existing_manifest
    }
    existing_hashes = {
        row["sha256"]
        for row in existing_manifest
        if row["sha256"]
    }
    review_decision_by_asset_id = {
        row["asset_id"]: row["operator_decision"].strip().lower()
        for row in existing_review
        if row["asset_id"]
    }
    counts = {
        category: 0
        for category in OPEN_LICENSE_SEARCH_QUERIES
    }
    for row in existing_manifest:
        decision = review_decision_by_asset_id.get(
            row["asset_id"],
            "pending",
        )
        if decision != "rejected":
            counts[row["part_category"]] += 1

    metadata_paths = (
        OPEN_LICENSE_MANIFEST_PATH,
        OPEN_LICENSE_REVIEW_PATH,
        OPEN_LICENSE_ATTRIBUTION_PATH,
        OPEN_LICENSE_COLLECTION_REPORT_PATH,
    )
    metadata_snapshots = snapshot_files(metadata_paths)

    temporary_root = (
        OPEN_LICENSE_IMAGES_DIRECTORY.parent
        / f".collect_tmp_{uuid.uuid4().hex}"
    )
    temporary_root.mkdir(parents=True, exist_ok=False)

    new_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    downloaded_paths: list[tuple[Path, Path]] = []

    try:
        for category, queries in OPEN_LICENSE_SEARCH_QUERIES.items():
            if counts[category] >= OPEN_LICENSE_TARGET_PER_CATEGORY:
                continue

            for search_query in queries:
                if counts[category] >= OPEN_LICENSE_TARGET_PER_CATEGORY:
                    break

                try:
                    pages = search_commons(
                        search_query,
                        request_json=request_json,
                    )
                except Exception as error:
                    warnings.append(
                        f"{category}: search failed for "
                        f"'{search_query}': {error}"
                    )
                    continue

                for page in pages:
                    if counts[category] >= OPEN_LICENSE_TARGET_PER_CATEGORY:
                        break

                    candidate = candidate_from_page(
                        page,
                        part_category=category,
                        search_query=search_query,
                    )
                    if candidate is None:
                        continue

                    page_key = (
                        category,
                        candidate["commons_page_id"],
                    )
                    if page_key in existing_by_page_category:
                        continue

                    try:
                        image_bytes = download_bytes(
                            candidate["download_url"]
                        )
                        (
                            width,
                            height,
                            image_format,
                            extension,
                        ) = inspect_image_bytes(image_bytes)
                    except Exception as error:
                        warnings.append(
                            f"{candidate['commons_title']}: {error}"
                        )
                        continue

                    digest = hashlib.sha256(
                        image_bytes
                    ).hexdigest()
                    if digest in existing_hashes:
                        continue

                    asset_id = (
                        f"commons_{category}_"
                        f"{candidate['commons_page_id']}"
                    )
                    relative_path = (
                        Path("data")
                        / "external"
                        / "open_license"
                        / "images"
                        / category
                        / f"{asset_id}{extension}"
                    )
                    final_path = (
                        OPEN_LICENSE_IMAGES_DIRECTORY
                        / category
                        / f"{asset_id}{extension}"
                    )
                    temporary_path = (
                        temporary_root
                        / category
                        / f"{asset_id}{extension}"
                    )
                    temporary_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    temporary_path.write_bytes(image_bytes)

                    row = {
                        "asset_id": asset_id,
                        **candidate,
                        "local_path": relative_path.as_posix(),
                        "sha256": digest,
                        "file_size_bytes": str(len(image_bytes)),
                        "width": str(width),
                        "height": str(height),
                        "format": image_format,
                        "downloaded_at_utc": (
                            datetime.now(timezone.utc).isoformat()
                        ),
                        "modifications": (
                            "Wikimedia thumbnail resized to maximum "
                            "1024 px width; otherwise unmodified."
                        ),
                    }
                    new_rows.append(row)
                    downloaded_paths.append(
                        (temporary_path, final_path)
                    )
                    existing_by_page_category.add(page_key)
                    existing_hashes.add(digest)
                    counts[category] += 1

                sleep(0.2)

        merged_manifest = sorted(
            [*existing_manifest, *new_rows],
            key=lambda row: (
                row["part_category"],
                row["asset_id"],
            ),
        )
        merged_review = build_review_rows(
            merged_manifest,
            existing_review,
        )

        for temporary_path, final_path in downloaded_paths:
            final_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            if final_path.exists():
                raise OpenLicenseCollectionError(
                    f"Refusing to overwrite {final_path}."
                )
            shutil.move(
                str(temporary_path),
                str(final_path),
            )

        atomic_write_csv(
            OPEN_LICENSE_MANIFEST_PATH,
            OPEN_LICENSE_MANIFEST_COLUMNS,
            merged_manifest,
        )
        atomic_write_csv(
            OPEN_LICENSE_REVIEW_PATH,
            OPEN_LICENSE_REVIEW_COLUMNS,
            merged_review,
        )
        atomic_write_text(
            OPEN_LICENSE_ATTRIBUTION_PATH,
            render_attribution(merged_manifest),
        )
        atomic_write_text(
            OPEN_LICENSE_COLLECTION_REPORT_PATH,
            render_collection_summary(
                manifest_rows=merged_manifest,
                added_count=len(new_rows),
                warnings=warnings,
            ),
        )
    except Exception:
        for _, final_path in downloaded_paths:
            if final_path.exists():
                final_path.unlink()
        restore_files(metadata_snapshots)
        raise
    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    return {
        "status": "PASS",
        "added": len(new_rows),
        "total": len(merged_manifest),
        "category_counts": counts,
        "warnings": warnings,
    }


def main() -> None:
    try:
        report = collect_open_license_images()
    except Exception as error:
        print("Step 010.1 open-license collection")
        print("- Status: FAIL")
        print(f"- Error: {error}")
        raise SystemExit(1) from error

    print("Step 010.1 open-license collection")
    print(f"- Status: {report['status']}")
    print(f"- Added: {report['added']}")
    print(f"- Total: {report['total']}")
    for category, count in report["category_counts"].items():
        print(f"- {category}: {count}")
    print(
        "- Manifest: "
        f"{OPEN_LICENSE_MANIFEST_PATH.relative_to(Path.cwd())}"
    )
    print(
        "- Review workbook: "
        f"{OPEN_LICENSE_REVIEW_PATH.relative_to(Path.cwd())}"
    )


if __name__ == "__main__":
    main()
