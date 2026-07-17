from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from src.import_first_real_batch import (
    matching_paths,
    validate_capture_file_map,
)
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    FIRST_BATCH_CAPTURE_FILE_MAP_PATH,
    FIRST_BATCH_CAPTURE_INBOX_DIRECTORY,
    FIRST_BATCH_CAPTURE_SESSION_COLUMNS,
    FIRST_BATCH_CAPTURE_SESSION_PATH,
    FIRST_BATCH_EXPECTED_GROUPS,
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_ORIGINALS_DIRECTORY,
    FIRST_BATCH_PLAN_COLUMNS,
    FIRST_BATCH_PLAN_PATH,
    FIRST_BATCH_VIEWS,
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
    / "first_batch_capture_session_readiness.json"
)
MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_capture_session_readiness.md"
)


def relative_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_fingerprint(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.exists():
        return "MISSING"
    entries = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            entries.append(
                f"{item.relative_to(path).as_posix()}:{sha256_file(item)}"
            )
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def protected_state() -> dict[str, str]:
    paths = (
        FIRST_BATCH_CAPTURE_INBOX_DIRECTORY,
        FIRST_BATCH_ORIGINALS_DIRECTORY,
        REAL_STAGING_DIRECTORY,
        REAL_SAMPLE_INTAKE_PATH,
        REAL_APPROVAL_LOG_PATH,
        REAL_IMAGE_MANIFEST_PATH,
        REAL_PART_GROUPS_PATH,
        REAL_IMAGES_PATH,
    )
    return {
        relative_path(path): path_fingerprint(path)
        for path in paths
    }


def inspect_image(path: Path) -> list[str]:
    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, ValueError) as error:
        return [f"Cannot read capture file '{relative_path(path)}': {error}."]
    return []


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


def validate_session_map(file_map: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    if tuple(file_map.columns) != FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS:
        return ["First-batch capture file map has an unexpected schema."]
    if len(file_map) != FIRST_BATCH_EXPECTED_IMAGES:
        errors.append(
            "First-batch capture session requires "
            f"{FIRST_BATCH_EXPECTED_IMAGES} mapped images; found "
            f"{len(file_map)}."
        )

    group_count = int(file_map["part_group_id"].nunique())
    if group_count != FIRST_BATCH_EXPECTED_GROUPS:
        errors.append(
            "First-batch capture session requires "
            f"{FIRST_BATCH_EXPECTED_GROUPS} physical groups; found "
            f"{group_count}."
        )

    for group_id, group in file_map.groupby("part_group_id", sort=False):
        views = tuple(sorted(group["view"].astype(str)))
        if views != tuple(sorted(FIRST_BATCH_VIEWS)):
            errors.append(
                f"Physical group '{group_id}' must contain exactly "
                f"{list(FIRST_BATCH_VIEWS)}; found {list(views)}."
            )
        if group["part_category"].nunique() != 1:
            errors.append(
                f"Physical group '{group_id}' uses multiple categories."
            )
    return errors


def resolve_slot(
    capture_filename: str,
    errors: list[str],
) -> dict[str, Any]:
    inbox_paths = matching_paths(
        FIRST_BATCH_CAPTURE_INBOX_DIRECTORY,
        capture_filename,
    )
    original_paths = matching_paths(
        FIRST_BATCH_ORIGINALS_DIRECTORY,
        capture_filename,
    )

    if len(inbox_paths) > 1:
        errors.append(
            f"Multiple inbox files found for '{capture_filename}': "
            f"{[relative_path(path) for path in inbox_paths]}."
        )
    if len(original_paths) > 1:
        errors.append(
            f"Multiple original files found for '{capture_filename}': "
            f"{[relative_path(path) for path in original_paths]}."
        )

    inbox_path = inbox_paths[0] if len(inbox_paths) == 1 else None
    original_path = (
        original_paths[0] if len(original_paths) == 1 else None
    )
    for path in (inbox_path, original_path):
        if path is not None:
            errors.extend(inspect_image(path))

    selected_path = original_path or inbox_path
    return {
        "inbox_status": "AVAILABLE" if inbox_path else "MISSING",
        "original_status": "IMPORTED" if original_path else "MISSING",
        "available": selected_path is not None,
        "selected_path": selected_path,
    }


def pair_status(front: dict[str, Any], detail: dict[str, Any]) -> str:
    available = int(front["available"]) + int(detail["available"])
    imported = int(front["original_status"] == "IMPORTED") + int(
        detail["original_status"] == "IMPORTED"
    )
    if imported == 2:
        return "READY_FOR_STAGING"
    if available == 2:
        return "READY_FOR_LOCAL_IMPORT"
    if available == 1:
        return "CAPTURE_IN_PROGRESS"
    return "AWAITING_CAPTURE"


def next_action_for_pair(
    status: str,
    front: dict[str, Any],
    detail: dict[str, Any],
) -> str:
    if status == "READY_FOR_STAGING":
        return "Run stage-first-real-batch-capture"
    if status == "READY_FOR_LOCAL_IMPORT":
        return "Run import-first-real-batch"
    missing = []
    if not front["available"]:
        missing.append("front")
    if not detail["available"]:
        missing.append("detail")
    if missing:
        return f"Capture missing view(s): {', '.join(missing)}"
    return "Review capture files"


def determine_readiness(
    captured_slots: int,
    originals_available: int,
    errors: list[str],
) -> str:
    if errors:
        return "CAPTURE_SESSION_BLOCKED"
    if captured_slots == 0:
        return "AWAITING_CAPTURE"
    if captured_slots < FIRST_BATCH_EXPECTED_IMAGES:
        return "CAPTURE_SESSION_IN_PROGRESS"
    if originals_available < FIRST_BATCH_EXPECTED_IMAGES:
        return "READY_FOR_LOCAL_IMPORT"
    return "READY_FOR_STAGING"


def build_session() -> tuple[pd.DataFrame, dict[str, Any]]:
    FIRST_BATCH_CAPTURE_INBOX_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIRST_BATCH_ORIGINALS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    before = protected_state()
    file_map, map_errors = read_capture_file_map()
    plan, plan_errors = read_plan()
    errors = [*map_errors, *plan_errors]
    errors.extend(validate_capture_file_map(file_map, plan))
    errors.extend(validate_session_map(file_map))

    rows: list[dict[str, str]] = []
    content_to_slots: dict[str, list[str]] = defaultdict(list)
    captured_slots = 0
    originals_available = 0
    complete_pairs = 0

    for sequence, (group_id, group) in enumerate(
        file_map.groupby("part_group_id", sort=False),
        start=1,
    ):
        by_view = {
            row_value(row, "view"): row
            for _, row in group.iterrows()
        }
        front_row = by_view.get("front")
        detail_row = by_view.get("detail")
        if front_row is None or detail_row is None:
            continue

        front_filename = row_value(front_row, "capture_filename")
        detail_filename = row_value(detail_row, "capture_filename")
        front = resolve_slot(front_filename, errors)
        detail = resolve_slot(detail_filename, errors)

        for filename, slot in (
            (front_filename, front),
            (detail_filename, detail),
        ):
            if slot["available"]:
                captured_slots += 1
            if slot["original_status"] == "IMPORTED":
                originals_available += 1
            selected_path = slot["selected_path"]
            if selected_path is not None:
                content_to_slots[sha256_file(selected_path)].append(filename)

        status = pair_status(front, detail)
        if front["available"] and detail["available"]:
            complete_pairs += 1
        rows.append(
            {
                "batch_id": row_value(front_row, "batch_id"),
                "sequence": str(sequence),
                "part_group_id": group_id,
                "part_category": row_value(front_row, "part_category"),
                "front_filename": front_filename,
                "detail_filename": detail_filename,
                "front_inbox_status": front["inbox_status"],
                "detail_inbox_status": detail["inbox_status"],
                "front_original_status": front["original_status"],
                "detail_original_status": detail["original_status"],
                "pair_status": status,
                "next_action": next_action_for_pair(
                    status,
                    front,
                    detail,
                ),
            }
        )

    for content_hash, filenames in sorted(content_to_slots.items()):
        if len(filenames) > 1:
            errors.append(
                "Duplicate capture content "
                f"{content_hash}: {sorted(filenames)}."
            )

    session = pd.DataFrame(
        rows,
        columns=FIRST_BATCH_CAPTURE_SESSION_COLUMNS,
    )
    after = protected_state()
    live_state_unchanged = "PASS" if before == after else "FAIL"
    if live_state_unchanged != "PASS":
        errors.append(
            "Capture-session preparation changed inbox, originals, staging, "
            "annotations, queue, approval log, or manifest."
        )

    missing_files = max(FIRST_BATCH_EXPECTED_IMAGES - captured_slots, 0)
    next_capture = []
    if not session.empty:
        incomplete = session[
            session["pair_status"].isin(
                ["AWAITING_CAPTURE", "CAPTURE_IN_PROGRESS"]
            )
        ]
        if not incomplete.empty:
            first = incomplete.iloc[0]
            next_capture = [
                value
                for value, status in (
                    (
                        first["front_filename"],
                        first["front_inbox_status"] == "AVAILABLE"
                        or first["front_original_status"] == "IMPORTED",
                    ),
                    (
                        first["detail_filename"],
                        first["detail_inbox_status"] == "AVAILABLE"
                        or first["detail_original_status"] == "IMPORTED",
                    ),
                )
                if not status
            ]

    report = {
        "status": "PASS" if not errors else "FAIL",
        "readiness": determine_readiness(
            captured_slots,
            originals_available,
            errors,
        ),
        "live_state_unchanged": live_state_unchanged,
        "counts": {
            "planned_groups": int(len(session)),
            "planned_files": FIRST_BATCH_EXPECTED_IMAGES,
            "captured_slots": captured_slots,
            "missing_files": missing_files,
            "complete_pairs": complete_pairs,
            "originals_available": originals_available,
        },
        "next_capture": next_capture,
        "errors": errors,
        "warnings": [],
    }
    return session, report


def write_outputs(
    session: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    FIRST_BATCH_CAPTURE_SESSION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    session.to_csv(
        FIRST_BATCH_CAPTURE_SESSION_PATH,
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
        "# First Batch Capture Session Readiness",
        "",
        f"- Status: **{report['status']}**",
        f"- Readiness: **{report['readiness']}**",
        f"- Planned physical parts: **{counts['planned_groups']}**",
        f"- Planned photographs: **{counts['planned_files']}**",
        f"- Captured slots: **{counts['captured_slots']}**",
        f"- Missing photographs: **{counts['missing_files']}**",
        f"- Complete front/detail pairs: **{counts['complete_pairs']}**",
        f"- Originals available: **{counts['originals_available']}**",
        f"- Protected state unchanged: **{report['live_state_unchanged']}**",
        "",
        "## Next capture",
        "",
    ]
    if report["next_capture"]:
        lines.extend(f"- `{name}`" for name in report["next_capture"])
    else:
        lines.append("- No missing capture is currently selected.")
    lines.extend(["", "## Errors", ""])
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- No capture-session errors found.")
    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def prepare_first_batch_capture_session() -> dict[str, Any]:
    session, report = build_session()
    write_outputs(session, report)
    return report


def main() -> None:
    report = prepare_first_batch_capture_session()
    counts = report["counts"]
    print("First real batch capture session preparation")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Physical parts: {counts['planned_groups']}")
    print(f"- Captured slots: {counts['captured_slots']}")
    print(f"- Missing photographs: {counts['missing_files']}")
    if report["next_capture"]:
        print(f"- Next capture: {', '.join(report['next_capture'])}")
    print(
        "- Session worksheet: "
        f"{FIRST_BATCH_CAPTURE_SESSION_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(f"- Report: {MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
