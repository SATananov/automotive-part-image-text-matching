from __future__ import annotations

import hashlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps

import src.prepare_first_real_batch as batch_prepare
from src.prepare_first_real_batch import plan_row_to_intake
from src.real_dataset_config import (
    ALLOWED_IMAGE_EXTENSIONS,
    FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS,
    FIRST_BATCH_CAPTURE_INVENTORY_PATH,
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_ID,
    FIRST_BATCH_ORIGINALS_DIRECTORY,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PLAN_PATH,
    FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_SAMPLE_INTAKE_PATH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.review_real_sample_intake import (
    development_hashes,
    existing_hashes_from_manifest,
    inspect_staged_image,
    read_csv_exact,
    row_value,
)


JSON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_capture_readiness.json"
)
MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_capture_readiness.md"
)

JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_jpeg_bytes(source: Path) -> tuple[bytes, dict[str, Any]]:
    with Image.open(source) as image:
        image.load()
        oriented = ImageOps.exif_transpose(image)
        rgb = oriented.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(
            buffer,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
            optimize=False,
            progressive=False,
        )
        content = buffer.getvalue()
        metrics = {
            "width": int(rgb.width),
            "height": int(rgb.height),
            "mode": "RGB",
            "format": "JPEG",
            "sha256": sha256_bytes(content),
            "file_size_bytes": len(content),
        }
    return content, metrics


def read_plan() -> tuple[pd.DataFrame, list[str]]:
    return read_csv_exact(
        FIRST_BATCH_PLAN_PATH,
        FIRST_BATCH_PLAN_COLUMNS,
        "first batch plan",
    )


def find_capture_candidates(intake_id: str) -> list[Path]:
    if not FIRST_BATCH_ORIGINALS_DIRECTORY.exists():
        return []
    return sorted(
        path
        for path in FIRST_BATCH_ORIGINALS_DIRECTORY.iterdir()
        if path.is_file()
        and path.stem == intake_id
        and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def capture_relative_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def unexpected_capture_files(expected_ids: set[str]) -> list[str]:
    if not FIRST_BATCH_ORIGINALS_DIRECTORY.exists():
        return []

    unexpected = []
    for path in sorted(FIRST_BATCH_ORIGINALS_DIRECTORY.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        if path.stem not in expected_ids:
            unexpected.append(capture_relative_path(path))
    return unexpected


def current_live_state() -> dict[str, str]:
    paths = (
        REAL_SAMPLE_INTAKE_PATH,
        REAL_APPROVAL_LOG_PATH,
        REAL_IMAGE_MANIFEST_PATH,
        REAL_PART_GROUPS_PATH,
        REAL_IMAGES_PATH,
    )
    state: dict[str, str] = {}
    for path in paths:
        key = capture_relative_path(path)
        state[key] = sha256_file(path) if path.is_file() else "MISSING"
    return state


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.step0093.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def restore_staging(snapshot: dict[Path, bytes | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            atomic_write_bytes(path, content)


def source_duplicate_errors(
    source_hashes: dict[str, list[str]],
    development: dict[str, list[str]],
    approved_real: dict[str, list[str]],
) -> list[str]:
    errors: list[str] = []
    for file_hash, intake_ids in sorted(source_hashes.items()):
        if len(intake_ids) > 1:
            errors.append(
                "Duplicate original capture content "
                f"{file_hash}: {intake_ids}."
            )
        if file_hash in development:
            errors.append(
                "Original capture duplicates development content "
                f"{development[file_hash]}: {intake_ids}."
            )
        if file_hash in approved_real:
            errors.append(
                "Original capture duplicates approved real content "
                f"{approved_real[file_hash]}: {intake_ids}."
            )
    return errors


def build_queue_draft(
    plan: pd.DataFrame,
    preview: pd.DataFrame,
) -> pd.DataFrame:
    preview_by_id = {
        row_value(row, "intake_id"): row
        for _, row in preview.iterrows()
    }
    rows: list[dict[str, str]] = []
    for _, plan_row in plan.iterrows():
        intake_id = row_value(plan_row, "intake_id")
        preview_row = preview_by_id.get(intake_id)
        if preview_row is None:
            continue
        review_status = row_value(preview_row, "review_status")
        queue_status = row_value(preview_row, "queue_status")
        if (
            row_value(preview_row, "file_present") == "yes"
            and review_status in {"PASS", "WARN"}
            and queue_status == "not_queued"
        ):
            rows.append(plan_row_to_intake(plan_row))
    return pd.DataFrame(rows, columns=SAMPLE_INTAKE_COLUMNS)


def build_inventory(
    plan: pd.DataFrame,
    preview: pd.DataFrame,
    source_paths: dict[str, Path | None],
    source_statuses: dict[str, str],
    staging_statuses: dict[str, str],
) -> pd.DataFrame:
    preview_by_id = {
        row_value(row, "intake_id"): row
        for _, row in preview.iterrows()
    }
    rows: list[dict[str, str]] = []

    for _, plan_row in plan.iterrows():
        intake_id = row_value(plan_row, "intake_id")
        staged_path = PROJECT_ROOT / row_value(plan_row, "staging_path")
        preview_row = preview_by_id.get(intake_id)
        metrics: dict[str, Any] = {}
        if staged_path.is_file():
            metrics, _, _ = inspect_staged_image(staged_path)

        review_status = (
            row_value(preview_row, "review_status")
            if preview_row is not None
            else "NOT_REVIEWED"
        )
        review_errors = (
            row_value(preview_row, "review_errors")
            if preview_row is not None
            else ""
        )
        review_warnings = (
            row_value(preview_row, "review_warnings")
            if preview_row is not None
            else ""
        )
        queue_status = (
            row_value(preview_row, "queue_status")
            if preview_row is not None
            else "not_queued"
        )
        ready_for_queue = (
            "yes"
            if staged_path.is_file()
            and review_status in {"PASS", "WARN"}
            and queue_status == "not_queued"
            else "no"
        )
        row = {
            column: row_value(plan_row, column)
            for column in FIRST_BATCH_PLAN_COLUMNS
        }
        row.update(
            {
                "capture_source_path": capture_relative_path(
                    source_paths.get(intake_id)
                ),
                "capture_source_status": source_statuses.get(
                    intake_id, "missing"
                ),
                "staging_status": staging_statuses.get(
                    intake_id,
                    "present" if staged_path.is_file() else "missing",
                ),
                "staged_sha256": str(metrics.get("sha256", "")),
                "width": str(metrics.get("width", "")),
                "height": str(metrics.get("height", "")),
                "mode": str(metrics.get("mode", "")),
                "format": str(metrics.get("format", "")),
                "review_status": review_status,
                "review_errors": review_errors,
                "review_warnings": review_warnings,
                "ready_for_queue": ready_for_queue,
            }
        )
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS,
    )


def determine_readiness(
    errors: list[str],
    staged_count: int,
    draft_count: int,
    queued_count: int,
    processed_count: int,
) -> str:
    if errors:
        return "CAPTURE_BLOCKED"
    if processed_count == FIRST_BATCH_EXPECTED_IMAGES:
        return "BATCH_COMPLETE"
    if staged_count == 0:
        return "AWAITING_CAPTURE"
    if staged_count < FIRST_BATCH_EXPECTED_IMAGES:
        return "CAPTURE_IN_PROGRESS"
    if draft_count + queued_count + processed_count < FIRST_BATCH_EXPECTED_IMAGES:
        return "REVIEW_BLOCKED"
    if draft_count:
        return "READY_FOR_MANUAL_QUEUE_IMPORT"
    return "READY_FOR_CONTROLLED_INTAKE"


def write_outputs(
    report: dict[str, Any],
    inventory: pd.DataFrame,
    queue_draft: pd.DataFrame,
) -> None:
    JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIRST_BATCH_CAPTURE_INVENTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    inventory.to_csv(
        FIRST_BATCH_CAPTURE_INVENTORY_PATH,
        index=False,
        encoding="utf-8",
    )
    queue_draft.to_csv(
        FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
        index=False,
        encoding="utf-8",
    )
    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    counts = report["counts"]
    lines = [
        "# First Real Batch Capture, Staging and Review Readiness",
        "",
        f"**Status:** {report['status']}",
        f"**Readiness:** {report['readiness']}",
        f"**Batch:** {report['batch_id']}",
        f"**Live queue unchanged:** {report['live_queue_unchanged']}",
        "",
        "## Counts",
        "",
        f"- Planned images: {counts['planned_images']}",
        f"- Original captures found: {counts['originals_found']}",
        f"- Newly staged images: {counts['newly_staged']}",
        f"- Total staged images: {counts['staged_files']}",
        f"- Review-ready queue draft rows: {counts['queue_draft_rows']}",
        f"- Rows already queued: {counts['queued_rows']}",
        f"- Rows already processed: {counts['processed_rows']}",
        "",
        "## Safety",
        "",
        "- Original photographs remain under the ignored originals tree.",
        "- Staging writes are transactional and never overwrite conflicts.",
        "- The generated queue draft contains only pending decisions.",
        "- The live sample intake queue is not modified.",
        "- No image is approved or copied to processed images.",
        "",
        "## Errors",
        "",
    ]
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- No capture or staging errors found.")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No capture or staging warnings found.")

    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def stage_first_batch_capture() -> dict[str, Any]:
    FIRST_BATCH_ORIGINALS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    plan, read_errors = read_plan()
    errors = list(read_errors)
    warnings: list[str] = []
    errors.extend(batch_prepare.validate_plan(plan))

    if tuple(plan.columns) != FIRST_BATCH_PLAN_COLUMNS:
        inventory = pd.DataFrame(
            columns=FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS
        )
        queue_draft = pd.DataFrame(columns=SAMPLE_INTAKE_COLUMNS)
        report = {
            "status": "FAIL",
            "readiness": "CAPTURE_BLOCKED",
            "batch_id": FIRST_BATCH_ID,
            "live_queue_unchanged": "PASS",
            "counts": {
                "planned_images": int(len(plan)),
                "originals_found": 0,
                "newly_staged": 0,
                "staged_files": 0,
                "queue_draft_rows": 0,
                "queued_rows": 0,
                "processed_rows": 0,
            },
            "errors": errors,
            "warnings": warnings,
        }
        write_outputs(report, inventory, queue_draft)
        return report

    expected_ids = set(plan["intake_id"])
    unexpected = unexpected_capture_files(expected_ids)
    if unexpected:
        errors.append(
            "Unexpected image files in the first-batch originals directory: "
            f"{unexpected}."
        )

    source_paths: dict[str, Path | None] = {}
    source_statuses: dict[str, str] = {}
    staging_statuses: dict[str, str] = {}
    normalized_by_destination: dict[Path, bytes] = {}
    source_hashes: dict[str, list[str]] = defaultdict(list)
    normalized_hashes: dict[str, list[str]] = defaultdict(list)

    for _, row in plan.iterrows():
        intake_id = row_value(row, "intake_id")
        staging_path = PROJECT_ROOT / row_value(row, "staging_path")
        candidates = find_capture_candidates(intake_id)
        source_paths[intake_id] = candidates[0] if len(candidates) == 1 else None

        if len(candidates) > 1:
            source_statuses[intake_id] = "multiple_candidates"
            errors.append(
                f"Multiple original capture files found for {intake_id}: "
                f"{[capture_relative_path(path) for path in candidates]}."
            )
            staging_statuses[intake_id] = (
                "present" if staging_path.is_file() else "missing"
            )
            continue

        if not candidates:
            source_statuses[intake_id] = "missing"
            staging_statuses[intake_id] = (
                "existing" if staging_path.is_file() else "missing"
            )
            if staging_path.is_file():
                normalized_hashes[sha256_file(staging_path)].append(intake_id)
            continue

        source = candidates[0]
        source_statuses[intake_id] = "captured"
        source_hashes[sha256_file(source)].append(intake_id)
        try:
            normalized, _ = normalized_jpeg_bytes(source)
        except (OSError, ValueError) as error:
            errors.append(
                f"Cannot normalize original capture {capture_relative_path(source)}: "
                f"{error}."
            )
            staging_statuses[intake_id] = "normalization_failed"
            continue

        normalized_hash = sha256_bytes(normalized)
        normalized_hashes[normalized_hash].append(intake_id)
        if staging_path.is_file():
            if staging_path.read_bytes() == normalized:
                staging_statuses[intake_id] = "already_staged"
            else:
                staging_statuses[intake_id] = "conflict"
                errors.append(
                    f"Staging destination already exists with different "
                    f"content: {capture_relative_path(staging_path)}."
                )
        else:
            normalized_by_destination[staging_path] = normalized
            staging_statuses[intake_id] = "ready_to_stage"

    approved_hashes, approved_hash_errors = existing_hashes_from_manifest()
    errors.extend(approved_hash_errors)
    errors.extend(
        source_duplicate_errors(
            source_hashes,
            development_hashes(),
            approved_hashes,
        )
    )
    for file_hash, intake_ids in sorted(normalized_hashes.items()):
        if len(intake_ids) > 1:
            errors.append(
                "Duplicate normalized staging content "
                f"{file_hash}: {intake_ids}."
            )

    live_before = current_live_state()
    snapshot = {
        path: path.read_bytes() if path.is_file() else None
        for path in normalized_by_destination
    }
    newly_staged = 0
    preparation_report: dict[str, Any]
    preview = pd.DataFrame(columns=batch_prepare.FIRST_BATCH_PREVIEW_COLUMNS)

    if not errors:
        try:
            for destination, content in normalized_by_destination.items():
                atomic_write_bytes(destination, content)
                staging_statuses[destination.stem] = "staged"
                newly_staged += 1
            preparation_report = batch_prepare.prepare_first_batch()
            if preparation_report["status"] != "PASS":
                errors.extend(preparation_report["errors"])
                raise RuntimeError("First-batch preparation failed after staging.")
            preview = pd.read_csv(
                batch_prepare.FIRST_BATCH_PREVIEW_PATH,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        except Exception as error:
            restore_staging(snapshot)
            if not errors:
                errors.append(str(error))
            newly_staged = 0
            preparation_report = batch_prepare.prepare_first_batch()
            if preparation_report.get("status") != "PASS":
                errors.extend(preparation_report.get("errors", []))
            if batch_prepare.FIRST_BATCH_PREVIEW_PATH.is_file():
                preview = pd.read_csv(
                    batch_prepare.FIRST_BATCH_PREVIEW_PATH,
                    dtype=str,
                    keep_default_na=False,
                    encoding="utf-8-sig",
                )
    else:
        preparation_report = batch_prepare.prepare_first_batch()
        if preparation_report.get("status") != "PASS":
            errors.extend(preparation_report.get("errors", []))
        if batch_prepare.FIRST_BATCH_PREVIEW_PATH.is_file():
            preview = pd.read_csv(
                batch_prepare.FIRST_BATCH_PREVIEW_PATH,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )

    warnings.extend(preparation_report.get("warnings", []))
    queue_draft = build_queue_draft(plan, preview)
    inventory = build_inventory(
        plan,
        preview,
        source_paths,
        source_statuses,
        staging_statuses,
    )

    staged_count = int(
        sum(
            (PROJECT_ROOT / row_value(row, "staging_path")).is_file()
            for _, row in plan.iterrows()
        )
    )
    queued_count = int(
        preview.get("queue_status", pd.Series(dtype=str))
        .isin({"pending", "approved", "rejected"})
        .sum()
    )
    processed_count = int(
        (preview.get("queue_status", pd.Series(dtype=str)) == "processed").sum()
    )
    live_after = current_live_state()
    live_unchanged = "PASS" if live_before == live_after else "FAIL"
    if live_unchanged != "PASS":
        errors.append(
            "Live annotations, queue, approval log, or manifest changed "
            "unexpectedly."
        )
        restore_staging(snapshot)
        newly_staged = 0

    readiness = determine_readiness(
        errors=errors,
        staged_count=staged_count,
        draft_count=int(len(queue_draft)),
        queued_count=queued_count,
        processed_count=processed_count,
    )
    report = {
        "status": "PASS" if not errors else "FAIL",
        "readiness": readiness,
        "batch_id": FIRST_BATCH_ID,
        "live_queue_unchanged": live_unchanged,
        "counts": {
            "planned_images": int(len(plan)),
            "originals_found": int(
                sum(status == "captured" for status in source_statuses.values())
            ),
            "newly_staged": newly_staged,
            "staged_files": staged_count,
            "queue_draft_rows": int(len(queue_draft)),
            "queued_rows": queued_count,
            "processed_rows": processed_count,
        },
        "preparation": {
            "status": preparation_report.get("status", "UNKNOWN"),
            "readiness": preparation_report.get("readiness", "UNKNOWN"),
        },
        "errors": errors,
        "warnings": sorted(set(warnings)),
    }
    write_outputs(report, inventory, queue_draft)
    return report


def main() -> None:
    report = stage_first_batch_capture()
    counts = report["counts"]
    print("First real batch capture, staging and review readiness")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Original captures: {counts['originals_found']}")
    print(f"- Newly staged: {counts['newly_staged']}")
    print(f"- Total staged: {counts['staged_files']}")
    print(f"- Queue draft rows: {counts['queue_draft_rows']}")
    print(f"- Live queue unchanged: {report['live_queue_unchanged']}")
    print(
        "- Inventory: "
        f"{FIRST_BATCH_CAPTURE_INVENTORY_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "- Queue draft: "
        f"{FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(f"- Report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
