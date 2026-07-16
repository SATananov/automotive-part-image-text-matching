from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image, ImageOps

from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_PROCESSED_IMAGES_DIRECTORY,
    REAL_SAMPLE_INTAKE_PATH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.review_real_sample_intake import (
    build_review_report,
    derive_image_id,
    development_hashes,
    existing_hashes_from_manifest,
    load_review_inputs,
    make_group_row,
    normalized_text,
    quality_status,
    row_value,
    write_review_outputs,
)
from src.validate_real_dataset import (
    build_report as build_intake_validation_report,
    sha256_file,
    validate_development_real_separation,
    validate_image_annotations,
    validate_part_groups,
    write_outputs as write_intake_validation_outputs,
)


JSON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_apply.json"
)

MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "sample_intake_apply.md"
)


class IntakeApplyError(RuntimeError):
    pass


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def atomic_write_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    dataframe.to_csv(temporary_path, index=False, encoding="utf-8")
    os.replace(temporary_path, path)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, path)


def normalize_to_png(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        width, height = normalized.size
        normalized.save(destination, format="PNG", optimize=False)

    return {
        "sha256": sha256_file(destination),
        "width": width,
        "height": height,
        "mode": "RGB",
        "format": "PNG",
    }


def dataframe_with_row(
    dataframe: pd.DataFrame,
    row: dict[str, str],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return pd.concat(
        [dataframe, pd.DataFrame([row], columns=columns)],
        ignore_index=True,
    )


def snapshot_files(paths: list[Path]) -> dict[Path, bytes | None]:
    return {
        path: path.read_bytes() if path.is_file() else None
        for path in paths
    }


def restore_files(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def validate_prospective_annotations(
    part_groups: pd.DataFrame,
    images: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    errors.extend(validate_part_groups(part_groups))
    errors.extend(validate_image_annotations(images, part_groups))
    errors.extend(
        validate_development_real_separation(part_groups, images)
    )
    return errors


def build_apply_plan(
    intake: pd.DataFrame,
    part_groups: pd.DataFrame,
    images: pd.DataFrame,
    approval_log: pd.DataFrame,
    review_report: dict[str, object],
    temporary_directory: Path,
    timestamp_factory: Callable[[], str] = utc_timestamp,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[tuple[Path, Path]],
    list[dict[str, object]],
    list[str],
]:
    prospective_groups = part_groups.copy()
    prospective_images = images.copy()
    prospective_log = approval_log.copy()
    handled_intake_ids: set[str] = set()
    staged_moves: list[tuple[Path, Path]] = []
    applied_items: list[dict[str, object]] = []
    errors: list[str] = []

    review_items = {
        item["intake_id"]: item for item in review_report["items"]
    }
    existing_real_hashes, hash_errors = existing_hashes_from_manifest()
    errors.extend(hash_errors)
    existing_development_hashes = development_hashes()
    normalized_hashes: dict[str, list[str]] = {}

    for _, row in intake.iterrows():
        intake_id = row_value(row, "intake_id")
        decision = row_value(row, "decision")

        if decision == "pending":
            continue

        handled_intake_ids.add(intake_id)
        item = review_items.get(intake_id, {})
        timestamp = timestamp_factory()

        if decision == "rejected":
            log_row = {
                "intake_id": intake_id,
                "decision": "rejected",
                "part_group_id": row_value(row, "part_group_id"),
                "image_id": "",
                "processed_image_path": "",
                "sha256": "",
                "width": "",
                "height": "",
                "mode": "",
                "format": "",
                "quality_status": "REJECTED",
                "processed_at_utc": timestamp,
                "rejection_reason": row_value(row, "rejection_reason"),
                "notes": row_value(row, "notes"),
            }
            prospective_log = dataframe_with_row(
                prospective_log,
                log_row,
                APPROVAL_LOG_COLUMNS,
            )
            applied_items.append(log_row)
            continue

        group_row = make_group_row(row)
        group_id = group_row["part_group_id"]
        view = row_value(row, "view")
        image_id = derive_image_id(group_id, view)
        processed_relative = (
            f"data/real/processed/images/{image_id}.png"
        )
        destination = PROJECT_ROOT / processed_relative
        temporary_image = temporary_directory / f"{image_id}.png"
        source = PROJECT_ROOT / row_value(row, "staging_path")

        if destination.exists():
            errors.append(
                f"Processed destination already exists: "
                f"{processed_relative}."
            )
            continue

        image_metadata = normalize_to_png(source, temporary_image)
        normalized_hash = str(image_metadata["sha256"])
        normalized_hashes.setdefault(normalized_hash, []).append(intake_id)

        if normalized_hash in existing_real_hashes:
            errors.append(
                f"Normalized image '{intake_id}' duplicates approved "
                f"real content: {existing_real_hashes[normalized_hash]}."
            )

        if normalized_hash in existing_development_hashes:
            errors.append(
                f"Normalized image '{intake_id}' duplicates "
                f"development content: "
                f"{existing_development_hashes[normalized_hash]}."
            )

        existing_group_mask = (
            prospective_groups["part_group_id"] == group_id
            if "part_group_id" in prospective_groups.columns
            else pd.Series(dtype=bool)
        )

        if not existing_group_mask.any():
            prospective_groups = dataframe_with_row(
                prospective_groups,
                group_row,
                PART_GROUP_COLUMNS,
            )
        else:
            prospective_groups.loc[
                existing_group_mask, "approved"
            ] = "yes"

        image_row = {
            "image_id": image_id,
            "part_group_id": group_id,
            "image_path": processed_relative,
            "view": view,
            "approved": "yes",
        }
        prospective_images = dataframe_with_row(
            prospective_images,
            image_row,
            IMAGE_MANIFEST_COLUMNS,
        )

        item_errors = list(item.get("errors", []))
        item_warnings = list(item.get("warnings", []))
        log_row = {
            "intake_id": intake_id,
            "decision": "approved",
            "part_group_id": group_id,
            "image_id": image_id,
            "processed_image_path": processed_relative,
            "sha256": normalized_hash,
            "width": str(image_metadata["width"]),
            "height": str(image_metadata["height"]),
            "mode": str(image_metadata["mode"]),
            "format": str(image_metadata["format"]),
            "quality_status": quality_status(
                item_errors,
                item_warnings,
            ),
            "processed_at_utc": timestamp,
            "rejection_reason": "",
            "notes": row_value(row, "notes"),
        }
        prospective_log = dataframe_with_row(
            prospective_log,
            log_row,
            APPROVAL_LOG_COLUMNS,
        )
        staged_moves.append((temporary_image, destination))
        applied_items.append(log_row)

    for file_hash, intake_ids in normalized_hashes.items():
        if len(intake_ids) > 1:
            errors.append(
                f"Approved rows normalize to duplicate image content "
                f"{file_hash}: {intake_ids}."
            )

    remaining_intake = intake.loc[
        ~intake["intake_id"].isin(handled_intake_ids)
    ].copy()
    remaining_intake = remaining_intake.reset_index(drop=True)

    errors.extend(
        validate_prospective_annotations(
            prospective_groups,
            prospective_images,
        )
    )

    return (
        prospective_groups,
        prospective_images,
        remaining_intake,
        prospective_log,
        staged_moves,
        applied_items,
        errors,
    )


def write_apply_outputs(report: dict[str, object]) -> None:
    atomic_write_text(
        JSON_REPORT_PATH,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )

    counts = report["counts"]
    lines = [
        "# Real Dataset Sample Intake Apply",
        "",
        f"**Status:** {report['status']}",
        f"**Result:** {report['result']}",
        "",
        "## Counts",
        "",
        f"- Approved and applied: {counts['approved']}",
        f"- Rejected and logged: {counts['rejected']}",
        f"- Remaining pending: {counts['remaining_pending']}",
        "",
        "## Applied records",
        "",
    ]

    if report["items"]:
        for item in report["items"]:
            lines.append(
                f"- `{item['intake_id']}` — {item['decision']} — "
                f"{item['image_id'] or 'no processed image'}"
            )
    else:
        lines.append("- No decisions were applied.")

    lines.extend(["", "## Errors", ""])
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- No apply errors found.")

    atomic_write_text(
        MARKDOWN_REPORT_PATH,
        "\n".join(lines) + "\n",
    )


def apply_intake(
    timestamp_factory: Callable[[], str] = utc_timestamp,
) -> dict[str, object]:
    intake, part_groups, images, approval_log, read_errors = (
        load_review_inputs()
    )
    review_report = build_review_report(
        intake,
        part_groups,
        images,
        approval_log,
        initial_errors=read_errors,
    )
    write_review_outputs(review_report)

    if review_report["status"] != "PASS":
        raise IntakeApplyError(
            "Sample intake review failed. Resolve the review report "
            "before applying decisions."
        )

    decided = intake.loc[
        intake["decision"].isin({"approved", "rejected"})
    ]

    if decided.empty:
        report = {
            "status": "PASS",
            "result": "NO_DECISIONS",
            "counts": {
                "approved": 0,
                "rejected": 0,
                "remaining_pending": int(
                    (intake["decision"] == "pending").sum()
                ),
            },
            "items": [],
            "errors": [],
        }
        write_apply_outputs(report)
        return report

    REAL_PROCESSED_IMAGES_DIRECTORY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory(
        prefix=".automotive_step009_1_",
        dir=REAL_PROCESSED_IMAGES_DIRECTORY.parent,
    ) as temporary_name:
        temporary_directory = Path(temporary_name)
        (
            prospective_groups,
            prospective_images,
            remaining_intake,
            prospective_log,
            staged_moves,
            applied_items,
            plan_errors,
        ) = build_apply_plan(
            intake,
            part_groups,
            images,
            approval_log,
            review_report,
            temporary_directory,
            timestamp_factory=timestamp_factory,
        )

        if plan_errors:
            raise IntakeApplyError(
                "Apply plan failed: " + " | ".join(plan_errors)
            )

        files_to_snapshot = [
            REAL_PART_GROUPS_PATH,
            REAL_IMAGES_PATH,
            REAL_SAMPLE_INTAKE_PATH,
            REAL_APPROVAL_LOG_PATH,
            PROJECT_ROOT
            / "data"
            / "real"
            / "processed"
            / "real_image_manifest.csv",
            PROJECT_ROOT
            / "reports"
            / "real_dataset"
            / "intake_validation.json",
            PROJECT_ROOT
            / "reports"
            / "real_dataset"
            / "intake_validation.md",
            JSON_REPORT_PATH,
            MARKDOWN_REPORT_PATH,
        ]
        snapshot = snapshot_files(files_to_snapshot)
        created_images: list[Path] = []

        try:
            for temporary_image, destination in staged_moves:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary_image, destination)
                created_images.append(destination)

            atomic_write_dataframe(
                prospective_groups,
                REAL_PART_GROUPS_PATH,
            )
            atomic_write_dataframe(
                prospective_images,
                REAL_IMAGES_PATH,
            )
            atomic_write_dataframe(
                remaining_intake,
                REAL_SAMPLE_INTAKE_PATH,
            )
            atomic_write_dataframe(
                prospective_log,
                REAL_APPROVAL_LOG_PATH,
            )

            validation_report, manifest = (
                build_intake_validation_report(
                    prospective_groups,
                    prospective_images,
                )
            )
            write_intake_validation_outputs(
                validation_report,
                manifest,
            )

            if validation_report["status"] != "PASS":
                raise IntakeApplyError(
                    "Final real-dataset validation failed: "
                    + " | ".join(validation_report["errors"])
                )

            approved_count = sum(
                item["decision"] == "approved"
                for item in applied_items
            )
            rejected_count = sum(
                item["decision"] == "rejected"
                for item in applied_items
            )
            report = {
                "status": "PASS",
                "result": "APPLIED",
                "counts": {
                    "approved": approved_count,
                    "rejected": rejected_count,
                    "remaining_pending": int(
                        (remaining_intake["decision"] == "pending").sum()
                    ),
                },
                "items": applied_items,
                "errors": [],
            }
            write_apply_outputs(report)
        except Exception:
            for path in created_images:
                path.unlink(missing_ok=True)
            restore_files(snapshot)
            raise

    return report


def main() -> None:
    try:
        report = apply_intake()
    except IntakeApplyError as error:
        print("Real dataset sample intake apply")
        print("- Status: FAIL")
        print(f"- Error: {error}")
        raise SystemExit(1) from error

    print("Real dataset sample intake apply")
    print(f"- Status: {report['status']}")
    print(f"- Result: {report['result']}")
    print(f"- Approved: {report['counts']['approved']}")
    print(f"- Rejected: {report['counts']['rejected']}")
    print(
        "- Remaining pending: "
        f"{report['counts']['remaining_pending']}"
    )
    print(
        "- Report: "
        f"{MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
