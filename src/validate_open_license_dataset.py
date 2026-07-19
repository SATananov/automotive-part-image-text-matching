from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image

from src.open_license_dataset_config import (
    OPEN_LICENSE_ATTRIBUTION_PATH,
    OPEN_LICENSE_MANIFEST_COLUMNS,
    OPEN_LICENSE_MANIFEST_PATH,
    OPEN_LICENSE_REVIEW_COLUMNS,
    OPEN_LICENSE_REVIEW_DECISIONS,
    OPEN_LICENSE_REVIEW_PATH,
    OPEN_LICENSE_SEARCH_QUERIES,
    OPEN_LICENSE_TARGET_PER_CATEGORY,
    OPEN_LICENSE_VALIDATION_REPORT_PATH,
)
from src.collect_open_license_images import (
    atomic_write_text,
    license_is_allowed,
)


class OpenLicenseValidationError(RuntimeError):
    pass


def read_csv_exact(
    path: Path,
    columns: tuple[str, ...],
) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        return [], [f"Missing file: {path}."]

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != columns:
                return [], [f"Invalid schema: {path}."]
            rows = [
                {
                    column: str(row.get(column, "")).strip()
                    for column in columns
                }
                for row in reader
            ]
    except Exception as error:
        return [], [f"Cannot read {path}: {error}."]

    return rows, []


def validate_open_license_dataset() -> dict[str, Any]:
    if (
        not OPEN_LICENSE_MANIFEST_PATH.exists()
        and not OPEN_LICENSE_REVIEW_PATH.exists()
    ):
        return {
            "status": "PASS",
            "readiness": "AWAITING_COLLECTION",
            "total": 0,
            "pending": 0,
            "rejected": 0,
            "category_counts": {
                category: {
                    "collected": 0,
                    "approved": 0,
                    "pending": 0,
                    "rejected": 0,
                }
                for category in OPEN_LICENSE_SEARCH_QUERIES
            },
            "errors": [],
            "warnings": [],
        }

    manifest_rows, manifest_errors = read_csv_exact(
        OPEN_LICENSE_MANIFEST_PATH,
        OPEN_LICENSE_MANIFEST_COLUMNS,
    )
    review_rows, review_errors = read_csv_exact(
        OPEN_LICENSE_REVIEW_PATH,
        OPEN_LICENSE_REVIEW_COLUMNS,
    )
    errors = [*manifest_errors, *review_errors]
    warnings: list[str] = []

    manifest_by_id = {
        row["asset_id"]: row
        for row in manifest_rows
        if row["asset_id"]
    }
    review_by_id = {
        row["asset_id"]: row
        for row in review_rows
        if row["asset_id"]
    }

    if len(manifest_by_id) != len(manifest_rows):
        errors.append(
            "The manifest contains blank or duplicate asset IDs."
        )
    if len(review_by_id) != len(review_rows):
        errors.append(
            "The review workbook contains blank or duplicate asset IDs."
        )

    if set(manifest_by_id) != set(review_by_id):
        errors.append(
            "Manifest and review workbook asset IDs differ."
        )

    hashes: dict[str, str] = {}
    category_counts = {
        category: {
            "collected": 0,
            "approved": 0,
            "pending": 0,
            "rejected": 0,
        }
        for category in OPEN_LICENSE_SEARCH_QUERIES
    }

    for asset_id, row in manifest_by_id.items():
        category = row["part_category"]
        if category not in category_counts:
            errors.append(
                f"{asset_id}: unsupported category '{category}'."
            )
            continue

        category_counts[category]["collected"] += 1

        required_fields = (
            "description_url",
            "original_url",
            "download_url",
            "license_short_name",
            "license_url",
            "local_path",
            "sha256",
            "width",
            "height",
            "format",
        )
        for field in required_fields:
            if not row[field]:
                errors.append(
                    f"{asset_id}: required field '{field}' is blank."
                )

        if not license_is_allowed(row["license_short_name"]):
            errors.append(
                f"{asset_id}: license is not in the allowlist: "
                f"{row['license_short_name']}."
            )

        if (
            row["attribution_required"] == "yes"
            and not (row["author"] or row["credit"])
        ):
            errors.append(
                f"{asset_id}: attribution is required but author "
                "and credit are blank."
            )

        local_path = Path(row["local_path"])
        if local_path.is_absolute() or ".." in local_path.parts:
            errors.append(
                f"{asset_id}: unsafe local path."
            )
            continue

        project_root = OPEN_LICENSE_MANIFEST_PATH.parents[3]
        image_path = project_root / local_path
        if not image_path.is_file():
            errors.append(
                f"{asset_id}: image file is missing: "
                f"{row['local_path']}."
            )
            continue

        digest = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()
        if digest != row["sha256"]:
            errors.append(
                f"{asset_id}: SHA-256 differs from the manifest."
            )

        other_asset = hashes.get(digest)
        if other_asset and other_asset != asset_id:
            errors.append(
                f"{asset_id}: duplicate image content also used by "
                f"{other_asset}."
            )
        hashes[digest] = asset_id

        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                image_format = str(image.format or "").upper()
        except Exception as error:
            errors.append(
                f"{asset_id}: unreadable image: {error}."
            )
            continue

        if str(width) != row["width"]:
            errors.append(
                f"{asset_id}: width differs from the manifest."
            )
        if str(height) != row["height"]:
            errors.append(
                f"{asset_id}: height differs from the manifest."
            )
        if image_format != row["format"]:
            errors.append(
                f"{asset_id}: format differs from the manifest."
            )

        review = review_by_id.get(asset_id)
        if review is None:
            continue

        decision = review["operator_decision"].lower()
        if decision not in OPEN_LICENSE_REVIEW_DECISIONS:
            errors.append(
                f"{asset_id}: unsupported review decision "
                f"'{decision}'."
            )
            continue

        category_counts[category][decision] += 1
        if decision == "rejected" and not review["rejection_reason"]:
            errors.append(
                f"{asset_id}: rejected row requires a reason."
            )
        if decision == "approved" and review["rejection_reason"]:
            errors.append(
                f"{asset_id}: approved row must not have a "
                "rejection reason."
            )

    if manifest_rows and not OPEN_LICENSE_ATTRIBUTION_PATH.is_file():
        errors.append("ATTRIBUTION.md is missing.")

    collected_complete = all(
        counts["collected"] >= OPEN_LICENSE_TARGET_PER_CATEGORY
        for counts in category_counts.values()
    )
    approved_complete = all(
        counts["approved"] >= OPEN_LICENSE_TARGET_PER_CATEGORY
        for counts in category_counts.values()
    )
    pending = sum(
        counts["pending"]
        for counts in category_counts.values()
    )
    rejected = sum(
        counts["rejected"]
        for counts in category_counts.values()
    )

    if errors:
        status = "FAIL"
        readiness = "VALIDATION_BLOCKED"
    elif not manifest_rows:
        status = "PASS"
        readiness = "AWAITING_COLLECTION"
    elif not collected_complete:
        status = "PASS"
        readiness = "COLLECTION_INCOMPLETE"
    elif pending:
        status = "PASS"
        readiness = "MANUAL_REVIEW_REQUIRED"
    elif approved_complete:
        status = "PASS"
        readiness = "READY_FOR_EXTERNAL_DATASET"
    elif rejected:
        status = "PASS"
        readiness = "REPLACEMENT_IMAGES_REQUIRED"
    else:
        status = "PASS"
        readiness = "MANUAL_REVIEW_REQUIRED"

    return {
        "status": status,
        "readiness": readiness,
        "total": len(manifest_rows),
        "pending": pending,
        "rejected": rejected,
        "category_counts": category_counts,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def render_validation_summary(
    report: dict[str, Any],
) -> str:
    lines = [
        "# Step 010.1 — Open-License Dataset Validation",
        "",
        f"- Status: **{report['status']}**",
        f"- Readiness: **{report['readiness']}**",
        f"- Total images: **{report['total']}**",
        f"- Pending review: **{report['pending']}**",
        f"- Rejected: **{report['rejected']}**",
        "",
        "## Category counts",
        "",
    ]

    for category, counts in report["category_counts"].items():
        lines.append(
            f"- {category}: collected={counts['collected']}, "
            f"approved={counts['approved']}, "
            f"pending={counts['pending']}, "
            f"rejected={counts['rejected']}"
        )

    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(
            f"- {error}"
            for error in report["errors"]
        )

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(
            f"- {warning}"
            for warning in report["warnings"]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    report = validate_open_license_dataset()
    atomic_write_text(
        OPEN_LICENSE_VALIDATION_REPORT_PATH,
        render_validation_summary(report),
    )

    print("Step 010.1 open-license validation")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Total: {report['total']}")
    print(f"- Pending: {report['pending']}")
    print(f"- Rejected: {report['rejected']}")

    for error in report["errors"]:
        print(f"ERROR: {error}")

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
