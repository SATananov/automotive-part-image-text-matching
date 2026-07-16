from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from src.real_dataset_config import (
    ALLOWED_IMAGE_EXTENSIONS,
    FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    FIRST_BATCH_CAPTURE_FILE_MAP_PATH,
    FIRST_BATCH_CAPTURE_INBOX_DIRECTORY,
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_LOCAL_IMPORT_INVENTORY_COLUMNS,
    FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH,
    FIRST_BATCH_ORIGINALS_DIRECTORY,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PLAN_PATH,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_SAMPLE_INTAKE_PATH,
    REAL_STAGING_DIRECTORY,
)
from src.review_real_sample_intake import read_csv_exact, row_value


JSON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_local_import_readiness.json"
)
MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_local_import_readiness.md"
)

CAPTURE_FILENAME_PATTERN = re.compile(
    r"^real_[a-z0-9_]+_(front|detail)\.jpg$"
)
TEMP_SUFFIX = ".local_import.tmp"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_capture_file_map() -> tuple[pd.DataFrame, list[str]]:
    return read_csv_exact(
        FIRST_BATCH_CAPTURE_FILE_MAP_PATH,
        FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
        "first batch capture file map",
    )


def read_plan() -> tuple[pd.DataFrame, list[str]]:
    return read_csv_exact(
        FIRST_BATCH_PLAN_PATH,
        FIRST_BATCH_PLAN_COLUMNS,
        "first batch plan",
    )


def validate_capture_file_map(
    file_map: pd.DataFrame,
    plan: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if tuple(file_map.columns) != FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS:
        errors.append(
            "first_batch_capture_file_map.csv must use the exact "
            "configured column order."
        )
        return errors

    if len(file_map) != FIRST_BATCH_EXPECTED_IMAGES:
        errors.append(
            "First batch capture file map must contain "
            f"{FIRST_BATCH_EXPECTED_IMAGES} rows; found {len(file_map)}."
        )

    for column in FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS:
        missing = [
            str(index + 2)
            for index, value in file_map[column].items()
            if not str(value).strip()
        ]
        if missing:
            errors.append(
                f"Capture file map column '{column}' has empty values "
                f"on rows: {', '.join(missing)}."
            )

    for column in ("intake_id", "capture_filename"):
        duplicated = sorted(
            value
            for value, count in file_map[column].value_counts().items()
            if value and count > 1
        )
        if duplicated:
            errors.append(
                f"Duplicate {column} values in capture file map: "
                f"{duplicated}."
            )

    for row_number, row in enumerate(file_map.itertuples(index=False), start=2):
        capture_filename = str(row.capture_filename).strip()
        expected_filename = f"{row.part_group_id}_{row.view}.jpg"
        if capture_filename != expected_filename:
            errors.append(
                f"Capture file map row {row_number} must use filename "
                f"'{expected_filename}'."
            )
        if CAPTURE_FILENAME_PATTERN.fullmatch(capture_filename) is None:
            errors.append(
                f"Capture file map row {row_number} has an unsafe or "
                f"unclear filename '{capture_filename}'."
            )

    if tuple(plan.columns) == FIRST_BATCH_PLAN_COLUMNS:
        plan_by_id = {
            row_value(row, "intake_id"): row
            for _, row in plan.iterrows()
        }
        for row_number, row in enumerate(
            file_map.itertuples(index=False),
            start=2,
        ):
            plan_row = plan_by_id.get(str(row.intake_id))
            if plan_row is None:
                errors.append(
                    f"Capture file map row {row_number} references unknown "
                    f"intake ID '{row.intake_id}'."
                )
                continue
            comparisons = (
                ("batch_id", row.batch_id),
                ("batch_item_id", row.batch_item_id),
                ("part_group_id", row.part_group_id),
                ("part_category", row.part_category),
                ("view", row.view),
                ("staging_path", row.staging_path),
            )
            for field, actual in comparisons:
                expected = row_value(plan_row, field)
                if str(actual) != expected:
                    errors.append(
                        f"Capture file map row {row_number} field '{field}' "
                        f"does not match first_batch_plan.csv."
                    )
    return errors


def matching_paths(directory: Path, capture_filename: str) -> list[Path]:
    if not directory.exists():
        return []
    stem = Path(capture_filename).stem
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.stem == stem
        and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def candidate_paths(capture_filename: str) -> list[Path]:
    return matching_paths(
        FIRST_BATCH_CAPTURE_INBOX_DIRECTORY,
        capture_filename,
    )


def original_paths(capture_filename: str) -> list[Path]:
    return matching_paths(
        FIRST_BATCH_ORIGINALS_DIRECTORY,
        capture_filename,
    )


def unexpected_inbox_files(expected_stems: set[str]) -> list[str]:
    if not FIRST_BATCH_CAPTURE_INBOX_DIRECTORY.exists():
        return []
    unexpected: list[str] = []
    for path in sorted(FIRST_BATCH_CAPTURE_INBOX_DIRECTORY.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        if path.stem not in expected_stems:
            unexpected.append(relative_path(path))
    return unexpected


def inspect_source(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        with Image.open(path) as image:
            image.load()
            metrics = {
                "width": int(image.width),
                "height": int(image.height),
                "mode": str(image.mode),
                "format": str(image.format or "UNKNOWN"),
            }
    except (OSError, ValueError) as error:
        return {}, [f"Cannot read local capture '{relative_path(path)}': {error}."]
    return metrics, []


def live_state() -> dict[str, str]:
    paths = (
        REAL_STAGING_DIRECTORY,
        REAL_SAMPLE_INTAKE_PATH,
        REAL_APPROVAL_LOG_PATH,
        REAL_IMAGE_MANIFEST_PATH,
        REAL_PART_GROUPS_PATH,
        REAL_IMAGES_PATH,
    )
    state: dict[str, str] = {}
    for path in paths:
        key = relative_path(path)
        if path.is_file():
            state[key] = sha256_file(path)
        elif path.is_dir():
            entries = []
            for item in sorted(path.rglob("*")):
                if item.is_file():
                    entries.append(
                        f"{item.relative_to(path).as_posix()}:{sha256_file(item)}"
                    )
            state[key] = sha256_bytes("\n".join(entries).encode("utf-8"))
        else:
            state[key] = "MISSING"
    return state


def atomic_copy_bytes(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}{TEMP_SUFFIX}")
    temporary.write_bytes(content)
    os.replace(temporary, destination)


def restore_destinations(snapshot: dict[Path, bytes | None]) -> None:
    for destination, previous in snapshot.items():
        if previous is None:
            destination.unlink(missing_ok=True)
        else:
            atomic_copy_bytes(destination, previous)


def determine_readiness(imported: int, errors: list[str]) -> str:
    if errors:
        return "LOCAL_IMPORT_BLOCKED"
    if imported == 0:
        return "AWAITING_LOCAL_FILES"
    if imported < FIRST_BATCH_EXPECTED_IMAGES:
        return "LOCAL_IMPORT_IN_PROGRESS"
    return "READY_FOR_STAGING"


def build_inventory(
    file_map: pd.DataFrame,
    sources: dict[str, Path | None],
    source_statuses: dict[str, str],
    destinations: dict[str, Path],
    import_statuses: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in file_map.iterrows():
        intake_id = row_value(row, "intake_id")
        source = sources.get(intake_id)
        destination = destinations.get(intake_id)
        metrics: dict[str, Any] = {}
        inspected_path = source if source and source.is_file() else destination
        if inspected_path and inspected_path.is_file():
            metrics, _ = inspect_source(inspected_path)
        content_hash = (
            sha256_file(inspected_path)
            if inspected_path and inspected_path.is_file()
            else ""
        )
        size = (
            str(inspected_path.stat().st_size)
            if inspected_path and inspected_path.is_file()
            else ""
        )
        output = {
            column: row_value(row, column)
            for column in FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS
        }
        output.update(
            {
                "inbox_source_path": relative_path(source),
                "inbox_source_status": source_statuses.get(
                    intake_id,
                    "missing",
                ),
                "original_destination_path": relative_path(destination),
                "import_status": import_statuses.get(intake_id, "missing"),
                "sha256": content_hash,
                "file_size_bytes": size,
                "width": str(metrics.get("width", "")),
                "height": str(metrics.get("height", "")),
                "mode": str(metrics.get("mode", "")),
                "format": str(metrics.get("format", "")),
            }
        )
        rows.append(output)
    return pd.DataFrame(
        rows,
        columns=FIRST_BATCH_LOCAL_IMPORT_INVENTORY_COLUMNS,
    )


def write_outputs(report: dict[str, Any], inventory: pd.DataFrame) -> None:
    FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    inventory.to_csv(
        FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH,
        index=False,
        encoding="utf-8",
    )
    JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    counts = report["counts"]
    lines = [
        "# First Batch Local Import Readiness",
        "",
        f"- Status: **{report['status']}**",
        f"- Readiness: **{report['readiness']}**",
        f"- Planned files: **{counts['planned_files']}**",
        f"- Inbox files found: **{counts['inbox_files_found']}**",
        f"- Newly imported: **{counts['newly_imported']}**",
        f"- Originals available: **{counts['originals_available']}**",
        f"- Live state unchanged: **{report['live_state_unchanged']}**",
        "",
        "## Errors",
        "",
    ]
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- No local import errors found.")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No local import warnings found.")
    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def import_first_real_batch() -> dict[str, Any]:
    FIRST_BATCH_CAPTURE_INBOX_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIRST_BATCH_ORIGINALS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    file_map, map_errors = read_capture_file_map()
    plan, plan_errors = read_plan()
    errors = [*map_errors, *plan_errors]
    warnings: list[str] = []
    errors.extend(validate_capture_file_map(file_map, plan))

    expected_stems = {
        Path(value).stem for value in file_map.get("capture_filename", [])
    }
    unexpected = unexpected_inbox_files(expected_stems)
    if unexpected:
        errors.append(
            "Unexpected image files in the first-batch capture inbox: "
            f"{unexpected}."
        )

    sources: dict[str, Path | None] = {}
    source_statuses: dict[str, str] = {}
    destinations: dict[str, Path] = {}
    import_statuses: dict[str, str] = {}
    content_to_ids: dict[str, list[str]] = defaultdict(list)
    pending_writes: dict[Path, bytes] = {}

    for _, row in file_map.iterrows():
        intake_id = row_value(row, "intake_id")
        capture_filename = row_value(row, "capture_filename")
        candidates = candidate_paths(capture_filename)
        existing_originals = original_paths(capture_filename)
        source = candidates[0] if len(candidates) == 1 else None
        sources[intake_id] = source

        if len(candidates) > 1:
            source_statuses[intake_id] = "multiple_candidates"
            import_statuses[intake_id] = "blocked"
            errors.append(
                f"Multiple local capture files found for {capture_filename}: "
                f"{[relative_path(path) for path in candidates]}."
            )
            destinations[intake_id] = (
                existing_originals[0]
                if len(existing_originals) == 1
                else FIRST_BATCH_ORIGINALS_DIRECTORY / capture_filename
            )
            continue

        if len(existing_originals) > 1:
            source_statuses[intake_id] = (
                "found" if source is not None else "missing"
            )
            import_statuses[intake_id] = "blocked"
            destinations[intake_id] = existing_originals[0]
            errors.append(
                f"Multiple original files already exist for {capture_filename}: "
                f"{[relative_path(path) for path in existing_originals]}."
            )
            continue

        existing = existing_originals[0] if existing_originals else None
        if source is None:
            source_statuses[intake_id] = "missing"
            destination = (
                existing
                if existing is not None
                else FIRST_BATCH_ORIGINALS_DIRECTORY / capture_filename
            )
            destinations[intake_id] = destination
            import_statuses[intake_id] = (
                "already_imported" if existing is not None else "missing"
            )
            continue

        source_statuses[intake_id] = "found"
        destination = (
            existing
            if existing is not None
            else FIRST_BATCH_ORIGINALS_DIRECTORY / source.name
        )
        destinations[intake_id] = destination
        metrics, inspect_errors = inspect_source(source)
        del metrics
        if inspect_errors:
            errors.extend(inspect_errors)
            import_statuses[intake_id] = "invalid_image"
            continue

        content = source.read_bytes()
        content_hash = sha256_bytes(content)
        content_to_ids[content_hash].append(intake_id)
        if destination.is_file():
            if destination.read_bytes() == content:
                import_statuses[intake_id] = "already_imported"
            else:
                import_statuses[intake_id] = "conflict"
                errors.append(
                    "Original destination already exists with different "
                    f"content: {relative_path(destination)}."
                )
        else:
            pending_writes[destination] = content
            import_statuses[intake_id] = "ready_to_import"

    for content_hash, intake_ids in sorted(content_to_ids.items()):
        if len(intake_ids) > 1:
            errors.append(
                "Duplicate local capture content "
                f"{content_hash}: {intake_ids}."
            )

    live_before = live_state()
    snapshot = {
        destination: (
            destination.read_bytes() if destination.is_file() else None
        )
        for destination in pending_writes
    }
    newly_imported = 0
    if not errors:
        try:
            for destination, content in pending_writes.items():
                atomic_copy_bytes(destination, content)
                newly_imported += 1
                for intake_id, mapped_destination in destinations.items():
                    if mapped_destination == destination:
                        import_statuses[intake_id] = "imported"
                        break
        except Exception as error:
            restore_destinations(snapshot)
            newly_imported = 0
            errors.append(f"Local import transaction failed: {error}.")

    live_after = live_state()
    live_unchanged = "PASS" if live_before == live_after else "FAIL"
    if live_unchanged != "PASS":
        restore_destinations(snapshot)
        newly_imported = 0
        errors.append(
            "Staging, annotations, queue, approval log, or manifest changed "
            "during local import."
        )

    originals_available = int(
        sum(
            destination.is_file()
            for destination in destinations.values()
        )
    )
    inventory = build_inventory(
        file_map,
        sources,
        source_statuses,
        destinations,
        import_statuses,
    )
    report = {
        "status": "PASS" if not errors else "FAIL",
        "readiness": determine_readiness(originals_available, errors),
        "live_state_unchanged": live_unchanged,
        "counts": {
            "planned_files": int(len(file_map)),
            "inbox_files_found": int(
                sum(status == "found" for status in source_statuses.values())
            ),
            "newly_imported": newly_imported,
            "originals_available": originals_available,
        },
        "errors": errors,
        "warnings": warnings,
    }
    write_outputs(report, inventory)
    return report


def main() -> None:
    report = import_first_real_batch()
    counts = report["counts"]
    print("First real batch local import")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Inbox files found: {counts['inbox_files_found']}")
    print(f"- Newly imported: {counts['newly_imported']}")
    print(f"- Originals available: {counts['originals_available']}")
    print(f"- Live state unchanged: {report['live_state_unchanged']}")
    print(
        "- Inventory: "
        f"{FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(f"- Report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
