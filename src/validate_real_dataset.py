from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.dataset_config import PART_FAMILIES
from src.real_dataset_config import (
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_IMAGE_VIEWS,
    APPROVAL_VALUES,
    CATEGORY_TO_FAMILY,
    DEVELOPMENT_IMAGES_DIRECTORY,
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    PROJECT_ROOT,
    REAL_DATASET_CATEGORIES,
    REAL_ID_PREFIX,
    REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_PROCESSED_IMAGES_DIRECTORY,
)


JSON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "intake_validation.json"
)

MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "intake_validation.md"
)

GROUP_ID_PATTERN = re.compile(
    r"^real_(?P<category>[a-z0-9]+(?:_[a-z0-9]+)*)_(?P<number>[0-9]{3,})$"
)

IMAGE_ID_PATTERN = re.compile(
    r"^(?P<group_id>real_[a-z0-9]+(?:_[a-z0-9]+)*_[0-9]{3,})_"
    r"(?P<view>[a-z0-9]+(?:_[a-z0-9]+)*)$"
)


def read_annotation_csv(
    path: Path,
    expected_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []

    if not path.is_file():
        return pd.DataFrame(columns=expected_columns), [
            f"Missing annotation file: {path.relative_to(PROJECT_ROOT)}"
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
            f"Cannot read {path.relative_to(PROJECT_ROOT)}: {error}"
        ]

    actual_columns = tuple(dataframe.columns)

    if actual_columns != expected_columns:
        errors.append(
            f"{path.relative_to(PROJECT_ROOT)} columns are "
            f"{actual_columns}; expected {expected_columns}."
        )

    return dataframe, errors


def normalized_text(value: object) -> str:
    return str(value).strip()


def validate_required_values(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    table_name: str,
) -> list[str]:
    errors: list[str] = []

    for column in required_columns:
        if column not in dataframe.columns:
            continue

        empty_rows = [
            str(index + 2)
            for index, value in dataframe[column].items()
            if not normalized_text(value)
        ]

        if empty_rows:
            errors.append(
                f"{table_name} column '{column}' contains empty values "
                f"on CSV rows: {', '.join(empty_rows)}."
            )

    return errors


def validate_approval_values(
    dataframe: pd.DataFrame,
    table_name: str,
) -> list[str]:
    errors: list[str] = []

    if "approved" not in dataframe.columns:
        return errors

    invalid_values = sorted(
        {
            normalized_text(value)
            for value in dataframe["approved"]
            if normalized_text(value) not in APPROVAL_VALUES
        }
    )

    if invalid_values:
        errors.append(
            f"{table_name} contains invalid approval values: "
            f"{invalid_values}."
        )

    return errors


def mentioned_categories(description: str) -> set[str]:
    normalized_description = " ".join(
        description.lower().replace("_", " ").split()
    )

    return {
        category
        for category in REAL_DATASET_CATEGORIES
        if category.replace("_", " ") in normalized_description
    }


def validate_part_groups(
    part_groups: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []

    errors.extend(
        validate_required_values(
            part_groups,
            required_columns=(
                "part_group_id",
                "part_family",
                "part_category",
                "match_description",
                "partial_description",
                "mismatch_description",
                "source",
                "approved",
            ),
            table_name="part_groups.csv",
        )
    )
    errors.extend(
        validate_approval_values(part_groups, "part_groups.csv")
    )

    missing_columns = sorted(
        set(PART_GROUP_COLUMNS) - set(part_groups.columns)
    )

    if missing_columns:
        errors.append(
            f"part_groups.csv is missing columns: {missing_columns}."
        )
        return errors

    if "part_group_id" in part_groups.columns:
        duplicates = sorted(
            part_groups.loc[
                part_groups["part_group_id"].duplicated(keep=False),
                "part_group_id",
            ].unique()
        )

        if duplicates:
            errors.append(
                f"Duplicate part_group_id values: {duplicates}."
            )

    for row_number, row in enumerate(
        part_groups.itertuples(index=False),
        start=2,
    ):
        group_id = normalized_text(row.part_group_id)
        family = normalized_text(row.part_family)
        category = normalized_text(row.part_category)
        approval = normalized_text(row.approved)

        group_match = GROUP_ID_PATTERN.fullmatch(group_id)

        if group_match is None:
            errors.append(
                f"part_groups.csv row {row_number} has invalid "
                f"part_group_id '{group_id}'. Use "
                f"real_<category>_<number>."
            )
        elif group_match.group("category") != category:
            errors.append(
                f"part_groups.csv row {row_number} group ID category "
                f"does not match '{category}'."
            )

        if not group_id.startswith(REAL_ID_PREFIX):
            errors.append(
                f"part_groups.csv row {row_number} is missing the "
                f"'{REAL_ID_PREFIX}' real-data prefix."
            )

        if category not in REAL_DATASET_CATEGORIES:
            errors.append(
                f"part_groups.csv row {row_number} has invalid "
                f"category '{category}'."
            )
            continue

        expected_family = CATEGORY_TO_FAMILY[category]

        if family != expected_family:
            errors.append(
                f"part_groups.csv row {row_number} maps category "
                f"'{category}' to '{family}', expected "
                f"'{expected_family}'."
            )

        if approval != "yes":
            continue

        descriptions = {
            "match": normalized_text(row.match_description),
            "partial": normalized_text(row.partial_description),
            "mismatch": normalized_text(row.mismatch_description),
        }

        if len({value.casefold() for value in descriptions.values()}) != 3:
            errors.append(
                f"Approved group '{group_id}' must contain three "
                f"distinct descriptions."
            )

        match_categories = mentioned_categories(descriptions["match"])
        partial_categories = mentioned_categories(
            descriptions["partial"]
        )
        mismatch_categories = mentioned_categories(
            descriptions["mismatch"]
        )

        if category not in match_categories:
            errors.append(
                f"Approved group '{group_id}' match_description must "
                f"name category '{category}'."
            )

        same_family_categories = set(PART_FAMILIES[expected_family])
        expected_partial_categories = same_family_categories - {category}

        if not (partial_categories & expected_partial_categories):
            errors.append(
                f"Approved group '{group_id}' partial_description must "
                f"name another category from family '{expected_family}'."
            )

        different_family_categories = (
            set(REAL_DATASET_CATEGORIES) - same_family_categories
        )

        if not (mismatch_categories & different_family_categories):
            errors.append(
                f"Approved group '{group_id}' mismatch_description must "
                "name a category from a different family."
            )

    return errors


def validate_image_annotations(
    images: pd.DataFrame,
    part_groups: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []

    errors.extend(
        validate_required_values(
            images,
            required_columns=IMAGE_MANIFEST_COLUMNS,
            table_name="images.csv",
        )
    )
    errors.extend(validate_approval_values(images, "images.csv"))

    missing_columns = sorted(
        set(IMAGE_MANIFEST_COLUMNS) - set(images.columns)
    )

    if missing_columns:
        errors.append(
            f"images.csv is missing columns: {missing_columns}."
        )
        return errors

    if "image_id" in images.columns:
        duplicate_ids = sorted(
            images.loc[
                images["image_id"].duplicated(keep=False),
                "image_id",
            ].unique()
        )

        if duplicate_ids:
            errors.append(f"Duplicate image_id values: {duplicate_ids}.")

    if "image_path" in images.columns:
        duplicate_paths = sorted(
            images.loc[
                images["image_path"].duplicated(keep=False),
                "image_path",
            ].unique()
        )

        if duplicate_paths:
            errors.append(
                f"Duplicate image_path values: {duplicate_paths}."
            )

    known_groups = (
        set(part_groups["part_group_id"])
        if "part_group_id" in part_groups.columns
        else set()
    )

    group_approval = (
        part_groups.set_index("part_group_id")["approved"].to_dict()
        if {"part_group_id", "approved"}.issubset(part_groups.columns)
        and not part_groups["part_group_id"].duplicated().any()
        else {}
    )

    for row_number, row in enumerate(
        images.itertuples(index=False),
        start=2,
    ):
        image_id = normalized_text(row.image_id)
        group_id = normalized_text(row.part_group_id)
        image_path_text = normalized_text(row.image_path)
        view = normalized_text(row.view)
        approval = normalized_text(row.approved)

        image_match = IMAGE_ID_PATTERN.fullmatch(image_id)

        if image_match is None:
            errors.append(
                f"images.csv row {row_number} has invalid image_id "
                f"'{image_id}'."
            )
        else:
            if image_match.group("group_id") != group_id:
                errors.append(
                    f"images.csv row {row_number} image_id does not "
                    f"start with part_group_id '{group_id}'."
                )

            if image_match.group("view") != view:
                errors.append(
                    f"images.csv row {row_number} image_id view does "
                    f"not match '{view}'."
                )

        if group_id not in known_groups:
            errors.append(
                f"images.csv row {row_number} references unknown group "
                f"'{group_id}'."
            )

        if view not in ALLOWED_IMAGE_VIEWS:
            errors.append(
                f"images.csv row {row_number} has invalid view "
                f"'{view}'."
            )

        path = Path(image_path_text)

        if path.is_absolute() or ".." in path.parts:
            errors.append(
                f"images.csv row {row_number} uses unsafe image path "
                f"'{image_path_text}'."
            )
            continue

        expected_prefix = Path("data/real/processed/images")

        try:
            relative_to_images = path.relative_to(expected_prefix)
        except ValueError:
            errors.append(
                f"images.csv row {row_number} path must be under "
                "data/real/processed/images/."
            )
            continue

        if len(relative_to_images.parts) != 1:
            errors.append(
                f"images.csv row {row_number} path must point directly "
                "to a file under processed/images/."
            )

        if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            errors.append(
                f"images.csv row {row_number} uses unsupported extension "
                f"'{path.suffix}'."
            )

        if path.stem != image_id:
            errors.append(
                f"images.csv row {row_number} filename stem must equal "
                f"image_id '{image_id}'."
            )

        if approval == "yes" and group_approval.get(group_id) != "yes":
            errors.append(
                f"Approved image '{image_id}' belongs to a group that "
                "is not approved."
            )

    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def collect_development_hashes() -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = defaultdict(list)

    if not DEVELOPMENT_IMAGES_DIRECTORY.is_dir():
        return hashes

    for path in sorted(DEVELOPMENT_IMAGES_DIRECTORY.iterdir()):
        if not path.is_file():
            continue

        hashes[sha256_file(path)].append(
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        )

    return hashes


def inspect_approved_images(
    images: pd.DataFrame,
    part_groups: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_rows: list[dict[str, object]] = []

    if not set(IMAGE_MANIFEST_COLUMNS).issubset(images.columns):
        return (
            pd.DataFrame(columns=REAL_IMAGE_INTAKE_MANIFEST_COLUMNS),
            errors,
            warnings,
        )

    if not set(PART_GROUP_COLUMNS).issubset(part_groups.columns):
        return (
            pd.DataFrame(columns=REAL_IMAGE_INTAKE_MANIFEST_COLUMNS),
            errors,
            warnings,
        )

    if images.empty:
        return (
            pd.DataFrame(columns=REAL_IMAGE_INTAKE_MANIFEST_COLUMNS),
            errors,
            warnings,
        )

    group_lookup = (
        part_groups.set_index("part_group_id").to_dict("index")
        if not part_groups.empty
        and "part_group_id" in part_groups.columns
        and not part_groups["part_group_id"].duplicated().any()
        else {}
    )

    development_hashes = collect_development_hashes()
    real_hash_to_images: dict[str, list[tuple[str, str]]] = defaultdict(list)
    views_by_group: dict[str, list[str]] = defaultdict(list)

    for row in images.itertuples(index=False):
        if normalized_text(row.approved) != "yes":
            continue

        image_id = normalized_text(row.image_id)
        group_id = normalized_text(row.part_group_id)
        image_path_text = normalized_text(row.image_path)
        view = normalized_text(row.view)
        image_path = PROJECT_ROOT / image_path_text

        if not image_path.is_file():
            errors.append(
                f"Approved image file is missing: {image_path_text}."
            )
            continue

        try:
            resolved_path = image_path.resolve(strict=True)
            processed_root = REAL_PROCESSED_IMAGES_DIRECTORY.resolve(
                strict=True
            )

            if not resolved_path.is_relative_to(processed_root):
                errors.append(
                    f"Approved image escapes the real processed directory: "
                    f"{image_path_text}."
                )
                continue
        except OSError as error:
            errors.append(
                f"Cannot resolve approved image '{image_path_text}': "
                f"{error}."
            )
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode or ""
                image_format = image.format or ""
        except (UnidentifiedImageError, OSError) as error:
            errors.append(
                f"Unreadable approved image '{image_path_text}': {error}."
            )
            continue

        file_hash = sha256_file(image_path)
        real_hash_to_images[file_hash].append((image_id, group_id))
        views_by_group[group_id].append(view)

        if file_hash in development_hashes:
            errors.append(
                f"Approved real image '{image_id}' duplicates development "
                f"content: {development_hashes[file_hash]}."
            )

        group_information = group_lookup.get(group_id)

        if group_information is None:
            continue

        manifest_rows.append(
            {
                "image_id": image_id,
                "part_group_id": group_id,
                "image_path": image_path_text.replace("\\", "/"),
                "part_family": normalized_text(
                    group_information["part_family"]
                ),
                "part_category": normalized_text(
                    group_information["part_category"]
                ),
                "view": view,
                "source": normalized_text(group_information["source"]),
                "approved": "yes",
                "sha256": file_hash,
                "file_size_bytes": image_path.stat().st_size,
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
            }
        )

    for file_hash, image_references in sorted(real_hash_to_images.items()):
        if len(image_references) < 2:
            continue

        image_ids = [image_id for image_id, _ in image_references]
        group_ids = {group_id for _, group_id in image_references}

        if len(group_ids) > 1:
            errors.append(
                f"Cross-group duplicate image hash {file_hash}: "
                f"{image_ids}."
            )
        else:
            errors.append(
                f"Duplicate approved image content within group "
                f"'{next(iter(group_ids))}': {image_ids}."
            )

    for group_id, views in sorted(views_by_group.items()):
        duplicate_views = sorted(
            view
            for view, count in Counter(views).items()
            if count > 1
        )

        if duplicate_views:
            errors.append(
                f"Approved group '{group_id}' repeats views: "
                f"{duplicate_views}."
            )

        if len(views) < 2:
            warnings.append(
                f"Approved group '{group_id}' has {len(views)} approved "
                "image; the collection target is at least 2."
            )

    manifest = pd.DataFrame(
        manifest_rows,
        columns=REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    )

    if not manifest.empty:
        manifest = manifest.sort_values(
            by=["part_category", "part_group_id", "image_id"]
        ).reset_index(drop=True)

    return manifest, errors, warnings


def validate_development_real_separation(
    part_groups: pd.DataFrame,
    images: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    development_metadata_path = (
        PROJECT_ROOT / "data" / "development" / "metadata.csv"
    )

    if not development_metadata_path.is_file():
        return [
            "Development metadata is missing; real/development ID "
            "separation cannot be verified."
        ]

    development = pd.read_csv(
        development_metadata_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    real_group_ids = set(part_groups.get("part_group_id", []))
    real_image_ids = set(images.get("image_id", []))
    development_group_ids = set(development.get("part_group_id", []))
    development_image_ids = set(development.get("image_id", []))

    group_overlap = sorted(real_group_ids & development_group_ids)
    image_overlap = sorted(real_image_ids & development_image_ids)

    if group_overlap:
        errors.append(
            f"Real/development part_group_id overlap: {group_overlap}."
        )

    if image_overlap:
        errors.append(
            f"Real/development image_id overlap: {image_overlap}."
        )

    non_prefixed_groups = sorted(
        group_id
        for group_id in real_group_ids
        if group_id and not group_id.startswith(REAL_ID_PREFIX)
    )
    non_prefixed_images = sorted(
        image_id
        for image_id in real_image_ids
        if image_id and not image_id.startswith(REAL_ID_PREFIX)
    )

    if non_prefixed_groups:
        errors.append(
            f"Real groups without '{REAL_ID_PREFIX}' prefix: "
            f"{non_prefixed_groups}."
        )

    if non_prefixed_images:
        errors.append(
            f"Real images without '{REAL_ID_PREFIX}' prefix: "
            f"{non_prefixed_images}."
        )

    return errors


def build_report(
    part_groups: pd.DataFrame,
    images: pd.DataFrame,
    initial_errors: Iterable[str] = (),
) -> tuple[dict[str, object], pd.DataFrame]:
    errors = list(initial_errors)
    warnings: list[str] = []

    errors.extend(validate_part_groups(part_groups))
    errors.extend(validate_image_annotations(images, part_groups))
    errors.extend(
        validate_development_real_separation(part_groups, images)
    )

    manifest, image_errors, image_warnings = inspect_approved_images(
        images,
        part_groups,
    )
    errors.extend(image_errors)
    warnings.extend(image_warnings)

    approved_groups = (
        int(part_groups["approved"].eq("yes").sum())
        if "approved" in part_groups.columns
        else 0
    )
    approved_images = (
        int(images["approved"].eq("yes").sum())
        if "approved" in images.columns
        else 0
    )

    readiness = (
        "EMPTY_FOUNDATION"
        if len(part_groups) == 0 and len(images) == 0
        else "INTAKE_IN_PROGRESS"
    )

    if approved_groups and approved_images:
        readiness = "APPROVED_DATA_AVAILABLE"

    report = {
        "status": "PASS" if not errors else "FAIL",
        "readiness": readiness,
        "annotation_counts": {
            "part_groups": int(len(part_groups)),
            "images": int(len(images)),
            "approved_part_groups": approved_groups,
            "approved_images": approved_images,
        },
        "manifest_rows": int(len(manifest)),
        "category_distribution": (
            part_groups["part_category"]
            .value_counts()
            .sort_index()
            .to_dict()
            if "part_category" in part_groups.columns
            else {}
        ),
        "errors": errors,
        "warnings": warnings,
    }

    return report, manifest


def write_outputs(
    report: dict[str, object],
    manifest: pd.DataFrame,
) -> None:
    REAL_IMAGE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(
        REAL_IMAGE_MANIFEST_PATH,
        index=False,
        encoding="utf-8",
    )

    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Real Dataset Intake Validation",
        "",
        f"**Status:** {report['status']}",
        f"**Readiness:** {report['readiness']}",
        "",
        "## Counts",
        "",
        f"- Annotated part groups: {report['annotation_counts']['part_groups']}",
        f"- Annotated images: {report['annotation_counts']['images']}",
        f"- Approved part groups: {report['annotation_counts']['approved_part_groups']}",
        f"- Approved images: {report['annotation_counts']['approved_images']}",
        f"- Intake manifest rows: {report['manifest_rows']}",
        "",
        "## Errors",
        "",
    ]

    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- No validation errors found.")

    lines.extend(["", "## Warnings", ""])

    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No validation warnings found.")

    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    part_groups, part_group_read_errors = read_annotation_csv(
        REAL_PART_GROUPS_PATH,
        PART_GROUP_COLUMNS,
    )
    images, image_read_errors = read_annotation_csv(
        REAL_IMAGES_PATH,
        IMAGE_MANIFEST_COLUMNS,
    )

    report, manifest = build_report(
        part_groups,
        images,
        initial_errors=(
            *part_group_read_errors,
            *image_read_errors,
        ),
    )
    write_outputs(report, manifest)

    print("Real dataset intake validation")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(
        "- Part groups: "
        f"{report['annotation_counts']['part_groups']}"
    )
    print(f"- Images: {report['annotation_counts']['images']}")
    print(f"- Manifest rows: {report['manifest_rows']}")
    print(f"- Errors: {len(report['errors'])}")
    print(f"- Warnings: {len(report['warnings'])}")
    print(
        "- Manifest: "
        f"{REAL_IMAGE_MANIFEST_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "- Report: "
        f"{MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
