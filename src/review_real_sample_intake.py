from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

from src.real_dataset_config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_IMAGE_VIEWS,
    APPROVAL_LOG_COLUMNS,
    CATEGORY_TO_FAMILY,
    DEVELOPMENT_IMAGES_DIRECTORY,
    HIGH_LUMINANCE_WARNING,
    IMAGE_MANIFEST_COLUMNS,
    INTAKE_DECISION_VALUES,
    LOW_CONTRAST_WARNING,
    LOW_LUMINANCE_WARNING,
    MAX_IMAGE_ASPECT_RATIO,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    PART_GROUP_COLUMNS,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_DATASET_CATEGORIES,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_PROCESSED_IMAGES_DIRECTORY,
    REAL_SAMPLE_INTAKE_PATH,
    REAL_STAGING_DIRECTORY,
    RECOMMENDED_IMAGE_HEIGHT,
    RECOMMENDED_IMAGE_WIDTH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.validate_real_dataset import (
    normalized_text,
    sha256_file,
    validate_part_groups,
)


JSON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_review.json"
)

MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_review.md"
)

INTAKE_ID_PATTERN = re.compile(r"^intake_[0-9]{6,}$")


def read_csv_exact(
    path: Path,
    expected_columns: tuple[str, ...],
    label: str,
) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        return pd.DataFrame(columns=expected_columns), [
            f"Missing {label}: {path.relative_to(PROJECT_ROOT)}."
        ]

    try:
        dataframe = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        return pd.DataFrame(columns=expected_columns), [
            f"Cannot read {label}: {error}."
        ]

    errors: list[str] = []
    actual_columns = tuple(dataframe.columns)

    if actual_columns != expected_columns:
        errors.append(
            f"{label} columns are {actual_columns}; expected "
            f"{expected_columns}."
        )

    return dataframe, errors


def derive_image_id(part_group_id: str, view: str) -> str:
    return f"{part_group_id}_{view}"


def row_value(row: pd.Series, column: str) -> str:
    return normalized_text(row.get(column, ""))


def make_group_row(row: pd.Series) -> dict[str, str]:
    return {
        "part_group_id": row_value(row, "part_group_id"),
        "part_family": row_value(row, "part_family"),
        "part_category": row_value(row, "part_category"),
        "match_description": row_value(row, "match_description"),
        "partial_description": row_value(row, "partial_description"),
        "mismatch_description": row_value(row, "mismatch_description"),
        "source": row_value(row, "source"),
        "approved": "yes",
        "notes": row_value(row, "notes"),
    }


def quality_status(errors: list[str], warnings: list[str]) -> str:
    if errors:
        return "FAIL"
    if warnings:
        return "WARN"
    return "PASS"


def validate_staging_path(
    intake_id: str,
    staging_path_text: str,
    require_file: bool,
) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    relative_path = Path(staging_path_text)

    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None, [
            f"Unsafe staging path '{staging_path_text}'."
        ]

    expected_prefix = Path("data/real/staging")

    try:
        path_inside_staging = relative_path.relative_to(expected_prefix)
    except ValueError:
        return None, [
            f"Staging path must be under data/real/staging/: "
            f"'{staging_path_text}'."
        ]

    if len(path_inside_staging.parts) != 1:
        errors.append(
            "Staging files must be placed directly under "
            "data/real/staging/."
        )

    if relative_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        errors.append(
            f"Unsupported staged image extension "
            f"'{relative_path.suffix}'."
        )

    if relative_path.stem != intake_id:
        errors.append(
            f"Staged filename stem must equal intake_id "
            f"'{intake_id}'."
        )

    absolute_path = PROJECT_ROOT / relative_path

    if require_file and not absolute_path.is_file():
        errors.append(
            f"Staged image file is missing: {staging_path_text}."
        )
        return absolute_path, errors

    if absolute_path.exists():
        try:
            resolved_path = absolute_path.resolve(strict=True)
            staging_root = REAL_STAGING_DIRECTORY.resolve(strict=True)

            if not resolved_path.is_relative_to(staging_root):
                errors.append(
                    f"Staged image escapes the staging directory: "
                    f"{staging_path_text}."
                )
        except OSError as error:
            errors.append(
                f"Cannot resolve staged image '{staging_path_text}': "
                f"{error}."
            )

    return absolute_path, errors


def inspect_staged_image(
    path: Path,
) -> tuple[dict[str, object], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, object] = {}

    try:
        with Image.open(path) as image:
            image_format = image.format or ""
            normalized_image = ImageOps.exif_transpose(image)
            width, height = normalized_image.size
            mode = normalized_image.mode or ""
            grayscale = normalized_image.convert("L")
            statistics = ImageStat.Stat(grayscale)
            mean_luminance = float(statistics.mean[0])
            contrast_stddev = float(statistics.stddev[0])
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return metrics, [
            f"Unreadable staged image '{path.name}': {error}."
        ], warnings

    aspect_ratio = max(width / height, height / width)
    file_size_bytes = path.stat().st_size

    metrics = {
        "sha256": sha256_file(path),
        "file_size_bytes": file_size_bytes,
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "mean_luminance": round(mean_luminance, 3),
        "contrast_stddev": round(contrast_stddev, 3),
        "aspect_ratio": round(aspect_ratio, 3),
    }

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        errors.append(
            f"Image dimensions {width}x{height} are below the minimum "
            f"{MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}."
        )
    elif (
        width < RECOMMENDED_IMAGE_WIDTH
        or height < RECOMMENDED_IMAGE_HEIGHT
    ):
        warnings.append(
            f"Image dimensions {width}x{height} are below the "
            f"recommended {RECOMMENDED_IMAGE_WIDTH}x"
            f"{RECOMMENDED_IMAGE_HEIGHT}."
        )

    if aspect_ratio > MAX_IMAGE_ASPECT_RATIO:
        warnings.append(
            f"Extreme image aspect ratio {aspect_ratio:.2f}."
        )

    if mode not in {"RGB", "RGBA"}:
        warnings.append(
            f"Image mode '{mode}' will be converted to RGB on approval."
        )

    if mean_luminance < LOW_LUMINANCE_WARNING:
        warnings.append("Image may be too dark.")
    elif mean_luminance > HIGH_LUMINANCE_WARNING:
        warnings.append("Image may be overexposed.")

    if contrast_stddev < LOW_CONTRAST_WARNING:
        warnings.append("Image has very low luminance contrast.")

    if file_size_bytes < 1024:
        warnings.append("Image file is unusually small.")

    return metrics, errors, warnings


def existing_hashes_from_manifest() -> tuple[dict[str, list[str]], list[str]]:
    hashes: dict[str, list[str]] = defaultdict(list)
    errors: list[str] = []

    if not REAL_IMAGE_MANIFEST_PATH.is_file():
        return hashes, [
            "Real image manifest is missing; existing real hashes "
            "cannot be checked."
        ]

    try:
        manifest = pd.read_csv(
            REAL_IMAGE_MANIFEST_PATH,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        return hashes, [
            f"Cannot read real image manifest: {error}."
        ]

    if not {"image_id", "sha256"}.issubset(manifest.columns):
        return hashes, [
            "Real image manifest is missing image_id or sha256."
        ]

    for row in manifest.itertuples(index=False):
        file_hash = normalized_text(row.sha256)
        image_id = normalized_text(row.image_id)

        if file_hash:
            hashes[file_hash].append(image_id)

    return hashes, errors


def development_hashes() -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = defaultdict(list)

    if not DEVELOPMENT_IMAGES_DIRECTORY.is_dir():
        return hashes

    for path in sorted(DEVELOPMENT_IMAGES_DIRECTORY.iterdir()):
        if path.is_file():
            hashes[sha256_file(path)].append(path.name)

    return hashes


def validate_existing_group(
    group_row: dict[str, str],
    existing_groups: pd.DataFrame,
) -> list[str]:
    group_id = group_row["part_group_id"]

    if existing_groups.empty or "part_group_id" not in existing_groups:
        return []

    matches = existing_groups.loc[
        existing_groups["part_group_id"] == group_id
    ]

    if matches.empty:
        return []

    existing = matches.iloc[0]
    fields = (
        "part_family",
        "part_category",
        "match_description",
        "partial_description",
        "mismatch_description",
        "source",
    )
    mismatches = [
        field
        for field in fields
        if normalized_text(existing.get(field, ""))
        != group_row[field]
    ]

    if mismatches:
        return [
            f"Existing group '{group_id}' conflicts on fields: "
            f"{mismatches}."
        ]

    return []


def build_review_report(
    intake: pd.DataFrame,
    part_groups: pd.DataFrame,
    images: pd.DataFrame,
    approval_log: pd.DataFrame,
    initial_errors: Iterable[str] = (),
) -> dict[str, object]:
    errors = list(initial_errors)
    warnings: list[str] = []
    items: list[dict[str, object]] = []

    missing_columns = sorted(
        set(SAMPLE_INTAKE_COLUMNS) - set(intake.columns)
    )

    if missing_columns:
        errors.append(
            f"sample_intake.csv is missing columns: {missing_columns}."
        )

    if tuple(intake.columns) != SAMPLE_INTAKE_COLUMNS:
        errors.append(
            "sample_intake.csv must use the exact configured column "
            "order."
        )

    if missing_columns:
        return {
            "status": "FAIL",
            "readiness": "REVIEW_BLOCKED",
            "counts": {
                "rows": int(len(intake)),
                "pending": 0,
                "approved": 0,
                "rejected": 0,
            },
            "items": items,
            "errors": errors,
            "warnings": warnings,
        }

    for column, label in (
        ("intake_id", "intake_id"),
        ("staging_path", "staging_path"),
    ):
        duplicate_values = sorted(
            value
            for value, count in Counter(
                normalized_text(value) for value in intake[column]
            ).items()
            if value and count > 1
        )

        if duplicate_values:
            errors.append(
                f"Duplicate {label} values in sample_intake.csv: "
                f"{duplicate_values}."
            )

    processed_intake_ids = (
        set(approval_log.get("intake_id", []))
        if "intake_id" in approval_log.columns
        else set()
    )
    existing_image_ids = set(images.get("image_id", []))
    existing_image_paths = set(images.get("image_path", []))
    real_hashes, hash_errors = existing_hashes_from_manifest()
    errors.extend(hash_errors)
    dev_hashes = development_hashes()
    queue_hashes: dict[str, list[str]] = defaultdict(list)
    queue_group_signatures: dict[str, tuple[str, ...]] = {}
    queue_image_ids: dict[str, list[str]] = defaultdict(list)

    for row_index, row in intake.iterrows():
        row_number = int(row_index) + 2
        intake_id = row_value(row, "intake_id")
        staging_path = row_value(row, "staging_path")
        decision = row_value(row, "decision")
        group_id = row_value(row, "part_group_id")
        family = row_value(row, "part_family")
        category = row_value(row, "part_category")
        view = row_value(row, "view")
        rejection_reason = row_value(row, "rejection_reason")
        item_errors: list[str] = []
        item_warnings: list[str] = []
        metrics: dict[str, object] = {}
        image_id = derive_image_id(group_id, view) if group_id and view else ""

        if not intake_id:
            item_errors.append("intake_id is required.")
        elif INTAKE_ID_PATTERN.fullmatch(intake_id) is None:
            item_errors.append(
                f"Invalid intake_id '{intake_id}'; use intake_<number> "
                "with at least six digits."
            )

        if intake_id in processed_intake_ids:
            item_errors.append(
                f"intake_id '{intake_id}' already exists in the "
                "approval log."
            )

        if not staging_path:
            item_errors.append("staging_path is required.")

        if decision not in INTAKE_DECISION_VALUES:
            item_errors.append(
                f"decision must be one of {INTAKE_DECISION_VALUES}."
            )

        if decision == "rejected" and not rejection_reason:
            item_errors.append(
                "rejection_reason is required for rejected rows."
            )

        requires_candidate = decision in {"pending", "approved"}

        if requires_candidate:
            required_fields = (
                "part_group_id",
                "part_family",
                "part_category",
                "view",
                "source",
                "match_description",
                "partial_description",
                "mismatch_description",
            )
            missing_values = [
                field for field in required_fields if not row_value(row, field)
            ]

            if missing_values:
                item_errors.append(
                    f"Missing required candidate fields: {missing_values}."
                )
            else:
                group_row = make_group_row(row)
                group_signature = tuple(
                    group_row[column]
                    for column in (
                        "part_family",
                        "part_category",
                        "match_description",
                        "partial_description",
                        "mismatch_description",
                        "source",
                    )
                )
                previous_signature = queue_group_signatures.get(group_id)

                if (
                    previous_signature is not None
                    and previous_signature != group_signature
                ):
                    item_errors.append(
                        f"Queue rows for group '{group_id}' contain "
                        "conflicting metadata."
                    )
                else:
                    queue_group_signatures[group_id] = group_signature

                queue_image_ids[image_id].append(intake_id)
                group_frame = pd.DataFrame(
                    [group_row], columns=PART_GROUP_COLUMNS
                )
                item_errors.extend(validate_part_groups(group_frame))
                item_errors.extend(
                    validate_existing_group(group_row, part_groups)
                )

                if category not in REAL_DATASET_CATEGORIES:
                    item_errors.append(
                        f"Unknown part category '{category}'."
                    )
                elif family != CATEGORY_TO_FAMILY[category]:
                    item_errors.append(
                        f"Category '{category}' must use family "
                        f"'{CATEGORY_TO_FAMILY[category]}'."
                    )

                if view not in ALLOWED_IMAGE_VIEWS:
                    item_errors.append(
                        f"Unknown image view '{view}'."
                    )

                processed_path = (
                    "data/real/processed/images/"
                    f"{image_id}.png"
                )

                if image_id in existing_image_ids:
                    item_errors.append(
                        f"Derived image_id '{image_id}' already exists."
                    )

                if processed_path in existing_image_paths:
                    item_errors.append(
                        f"Processed image path '{processed_path}' "
                        "already exists in images.csv."
                    )

                if (PROJECT_ROOT / processed_path).exists():
                    item_errors.append(
                        f"Processed destination already exists: "
                        f"{processed_path}."
                    )

        staged_file: Path | None = None
        path_errors: list[str] = []

        if staging_path:
            staged_file, path_errors = validate_staging_path(
                intake_id=intake_id,
                staging_path_text=staging_path,
                require_file=requires_candidate,
            )
            item_errors.extend(path_errors)

        if (
            requires_candidate
            and staged_file is not None
            and staged_file.is_file()
            and not path_errors
        ):
            metrics, image_errors, image_warnings = inspect_staged_image(
                staged_file
            )
            item_errors.extend(image_errors)
            item_warnings.extend(image_warnings)

            file_hash = normalized_text(metrics.get("sha256", ""))

            if file_hash:
                queue_hashes[file_hash].append(intake_id)

                if file_hash in dev_hashes:
                    item_errors.append(
                        f"Staged image duplicates development content: "
                        f"{dev_hashes[file_hash]}."
                    )

                if file_hash in real_hashes:
                    item_errors.append(
                        f"Staged image duplicates approved real content: "
                        f"{real_hashes[file_hash]}."
                    )

        item_status = quality_status(item_errors, item_warnings)
        items.append(
            {
                "row_number": row_number,
                "intake_id": intake_id,
                "decision": decision,
                "part_group_id": group_id,
                "image_id": image_id,
                "staging_path": staging_path,
                "status": item_status,
                "metrics": metrics,
                "errors": item_errors,
                "warnings": item_warnings,
            }
        )
        errors.extend(
            f"Row {row_number} ({intake_id or 'missing intake_id'}): "
            f"{error}"
            for error in item_errors
        )
        warnings.extend(
            f"Row {row_number} ({intake_id or 'missing intake_id'}): "
            f"{warning}"
            for warning in item_warnings
        )

    for file_hash, intake_ids in sorted(queue_hashes.items()):
        if len(intake_ids) > 1:
            errors.append(
                f"Duplicate staged image content {file_hash}: "
                f"{intake_ids}."
            )

    for image_id, intake_ids in sorted(queue_image_ids.items()):
        if image_id and len(intake_ids) > 1:
            errors.append(
                f"Duplicate derived image_id '{image_id}' in queue: "
                f"{intake_ids}."
            )

    counts = {
        "rows": int(len(intake)),
        "pending": int((intake["decision"] == "pending").sum()),
        "approved": int((intake["decision"] == "approved").sum()),
        "rejected": int((intake["decision"] == "rejected").sum()),
    }

    if intake.empty:
        readiness = "EMPTY_QUEUE"
    elif errors:
        readiness = "REVIEW_BLOCKED"
    else:
        readiness = "REVIEW_READY"

    return {
        "status": "PASS" if not errors else "FAIL",
        "readiness": readiness,
        "counts": counts,
        "items": items,
        "errors": errors,
        "warnings": warnings,
    }


def load_review_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    intake, intake_errors = read_csv_exact(
        REAL_SAMPLE_INTAKE_PATH,
        SAMPLE_INTAKE_COLUMNS,
        "sample_intake.csv",
    )
    part_groups, group_errors = read_csv_exact(
        REAL_PART_GROUPS_PATH,
        PART_GROUP_COLUMNS,
        "part_groups.csv",
    )
    images, image_errors = read_csv_exact(
        REAL_IMAGES_PATH,
        IMAGE_MANIFEST_COLUMNS,
        "images.csv",
    )
    approval_log, approval_errors = read_csv_exact(
        REAL_APPROVAL_LOG_PATH,
        APPROVAL_LOG_COLUMNS,
        "approval_log.csv",
    )

    return (
        intake,
        part_groups,
        images,
        approval_log,
        [
            *intake_errors,
            *group_errors,
            *image_errors,
            *approval_errors,
        ],
    )


def write_review_outputs(report: dict[str, object]) -> None:
    JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    counts = report["counts"]
    lines = [
        "# Real Dataset Sample Intake Review",
        "",
        f"**Status:** {report['status']}",
        f"**Readiness:** {report['readiness']}",
        "",
        "## Queue counts",
        "",
        f"- Rows: {counts['rows']}",
        f"- Pending: {counts['pending']}",
        f"- Approved decisions: {counts['approved']}",
        f"- Rejected decisions: {counts['rejected']}",
        "",
        "## Items",
        "",
    ]

    if report["items"]:
        for item in report["items"]:
            lines.extend(
                [
                    f"### {item['intake_id'] or 'Missing intake ID'}",
                    "",
                    f"- Decision: {item['decision']}",
                    f"- Status: {item['status']}",
                    f"- Part group: {item['part_group_id']}",
                    f"- Image ID: {item['image_id']}",
                    f"- Staging path: {item['staging_path']}",
                    f"- Metrics: `{json.dumps(item['metrics'], sort_keys=True)}`",
                    "",
                ]
            )
    else:
        lines.append("- Intake queue is empty.")

    lines.extend(["", "## Errors", ""])
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- No review errors found.")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No review warnings found.")

    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    intake, part_groups, images, approval_log, errors = (
        load_review_inputs()
    )
    report = build_review_report(
        intake,
        part_groups,
        images,
        approval_log,
        initial_errors=errors,
    )
    write_review_outputs(report)

    print("Real dataset sample intake review")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Queue rows: {report['counts']['rows']}")
    print(f"- Pending: {report['counts']['pending']}")
    print(f"- Approved decisions: {report['counts']['approved']}")
    print(f"- Rejected decisions: {report['counts']['rejected']}")
    print(f"- Errors: {len(report['errors'])}")
    print(f"- Warnings: {len(report['warnings'])}")
    print(
        "- Report: "
        f"{MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
