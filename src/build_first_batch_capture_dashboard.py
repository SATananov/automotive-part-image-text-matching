from __future__ import annotations

import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.import_first_real_batch import matching_paths
from src.prepare_first_batch_capture_session import path_fingerprint
from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
    FIRST_BATCH_CAPTURE_FILE_MAP_PATH,
    FIRST_BATCH_CAPTURE_INBOX_DIRECTORY,
    FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS,
    FIRST_BATCH_CAPTURE_INVENTORY_PATH,
    FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS,
    FIRST_BATCH_CAPTURE_PROGRESS_PATH,
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_ORIGINALS_DIRECTORY,
    FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
    IMAGE_MANIFEST_COLUMNS,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_SAMPLE_INTAKE_PATH,
    REAL_STAGING_DIRECTORY,
    REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
    SAMPLE_INTAKE_COLUMNS,
    PROJECT_ROOT,
)
from src.review_real_sample_intake import read_csv_exact, row_value


REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "real_dataset"
DASHBOARD_HTML_PATH = REPORT_DIRECTORY / "first_batch_capture_dashboard.html"
DASHBOARD_JSON_PATH = REPORT_DIRECTORY / "first_batch_capture_dashboard.json"
DASHBOARD_MARKDOWN_PATH = (
    REPORT_DIRECTORY / "first_batch_capture_progress_summary.md"
)
DASHBOARD_GUIDE_PATH = (
    REPORT_DIRECTORY
    / "first_batch_capture_dashboard_and_progress_tracking.md"
)

PIPELINE_STAGES = (
    "AWAITING_CAPTURE",
    "CAPTURED",
    "IMPORTED",
    "STAGED",
    "REVIEW_READY",
    "QUEUED_FOR_DECISION",
    "DECISION_RECORDED",
    "APPROVED_DATASET",
)


def relative_path(path: Path) -> str:
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


def protected_state() -> dict[str, str]:
    paths = (
        FIRST_BATCH_CAPTURE_FILE_MAP_PATH,
        FIRST_BATCH_CAPTURE_INBOX_DIRECTORY,
        FIRST_BATCH_ORIGINALS_DIRECTORY,
        REAL_STAGING_DIRECTORY,
        FIRST_BATCH_CAPTURE_INVENTORY_PATH,
        FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
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


def read_required_csv(
    path: Path,
    columns: tuple[str, ...],
    label: str,
) -> tuple[pd.DataFrame, list[str]]:
    return read_csv_exact(path, columns, label)


def lookup_by_intake(frame: pd.DataFrame) -> dict[str, pd.Series]:
    if frame.empty or "intake_id" not in frame.columns:
        return {}
    return {
        row_value(row, "intake_id"): row
        for _, row in frame.iterrows()
        if row_value(row, "intake_id")
    }


def latest_approval_rows(frame: pd.DataFrame) -> dict[str, pd.Series]:
    rows: dict[str, pd.Series] = {}
    if frame.empty:
        return rows
    for _, row in frame.iterrows():
        intake_id = row_value(row, "intake_id")
        if intake_id:
            rows[intake_id] = row
    return rows


def stage_for_row(
    *,
    captured: bool,
    imported: bool,
    staged: bool,
    review_ready: bool,
    queued: bool,
    decision_recorded: bool,
    approved: bool,
) -> tuple[str, int]:
    flags = (
        (approved, "APPROVED_DATASET", 7),
        (decision_recorded, "DECISION_RECORDED", 6),
        (queued, "QUEUED_FOR_DECISION", 5),
        (review_ready, "REVIEW_READY", 4),
        (staged, "STAGED", 3),
        (imported, "IMPORTED", 2),
        (captured, "CAPTURED", 1),
    )
    for condition, label, index in flags:
        if condition:
            return label, index
    return "AWAITING_CAPTURE", 0


def next_action_for_stage(
    stage: str,
    capture_filename: str,
    decision_status: str,
) -> str:
    actions = {
        "AWAITING_CAPTURE": f"Capture {capture_filename}",
        "CAPTURED": "Run import-first-real-batch",
        "IMPORTED": "Run stage-first-real-batch-capture",
        "STAGED": "Review capture inventory and queue draft",
        "REVIEW_READY": "Add reviewed row to sample_intake.csv",
        "QUEUED_FOR_DECISION": "Set explicit approved or rejected decision",
        "APPROVED_DATASET": "Complete",
    }
    if stage == "DECISION_RECORDED":
        if decision_status == "REJECTED":
            return "Capture a replacement only if the sample is needed"
        return "Verify approval transaction and manifest consistency"
    return actions[stage]


def determine_readiness(
    counts: dict[str, int],
    errors: list[str],
) -> str:
    if errors:
        return "CAPTURE_DASHBOARD_BLOCKED"
    if counts["approved"] == FIRST_BATCH_EXPECTED_IMAGES:
        return "BATCH_APPROVED"
    if counts["queued"] or counts["decision_recorded"]:
        return "REVIEW_IN_PROGRESS"
    if counts["review_ready"]:
        return "READY_FOR_MANUAL_REVIEW"
    if counts["staged"]:
        return "STAGING_IN_PROGRESS"
    if counts["imported"] == FIRST_BATCH_EXPECTED_IMAGES:
        return "READY_FOR_STAGING"
    if counts["captured"] == FIRST_BATCH_EXPECTED_IMAGES:
        return "READY_FOR_LOCAL_IMPORT"
    if counts["captured"]:
        return "CAPTURE_SESSION_IN_PROGRESS"
    return "AWAITING_CAPTURE"


def build_progress() -> tuple[pd.DataFrame, dict[str, Any]]:
    before = protected_state()
    errors: list[str] = []
    warnings: list[str] = []

    file_map, file_map_errors = read_required_csv(
        FIRST_BATCH_CAPTURE_FILE_MAP_PATH,
        FIRST_BATCH_CAPTURE_FILE_MAP_COLUMNS,
        "first batch capture file map",
    )
    capture_inventory, capture_errors = read_required_csv(
        FIRST_BATCH_CAPTURE_INVENTORY_PATH,
        FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS,
        "first batch capture inventory",
    )
    queue_draft, draft_errors = read_required_csv(
        FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
        SAMPLE_INTAKE_COLUMNS,
        "first batch review queue draft",
    )
    live_queue, queue_errors = read_required_csv(
        REAL_SAMPLE_INTAKE_PATH,
        SAMPLE_INTAKE_COLUMNS,
        "real sample intake queue",
    )
    approval_log, approval_errors = read_required_csv(
        REAL_APPROVAL_LOG_PATH,
        APPROVAL_LOG_COLUMNS,
        "real approval log",
    )
    manifest, manifest_errors = read_required_csv(
        REAL_IMAGE_MANIFEST_PATH,
        REAL_IMAGE_INTAKE_MANIFEST_COLUMNS,
        "real image manifest",
    )
    errors.extend(
        file_map_errors
        + capture_errors
        + draft_errors
        + queue_errors
        + approval_errors
        + manifest_errors
    )

    if len(file_map) != FIRST_BATCH_EXPECTED_IMAGES:
        errors.append(
            "Capture dashboard requires "
            f"{FIRST_BATCH_EXPECTED_IMAGES} mapped files; found "
            f"{len(file_map)}."
        )
    if not file_map.empty and file_map["intake_id"].duplicated().any():
        errors.append("Capture dashboard file map contains duplicate intake IDs.")

    inventory_by_intake = lookup_by_intake(capture_inventory)
    draft_by_intake = lookup_by_intake(queue_draft)
    queue_by_intake = lookup_by_intake(live_queue)
    approval_by_intake = latest_approval_rows(approval_log)
    approved_image_ids = set(
        manifest.loc[
            manifest["approved"].str.lower() == "yes",
            "image_id",
        ].astype(str)
    )

    rows: list[dict[str, str]] = []
    for _, source_row in file_map.iterrows():
        intake_id = row_value(source_row, "intake_id")
        capture_filename = row_value(source_row, "capture_filename")
        staging_value = row_value(source_row, "staging_path")
        staging_path = PROJECT_ROOT / Path(staging_value)

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
                f"Multiple inbox files found for '{capture_filename}'."
            )
        if len(original_paths) > 1:
            errors.append(
                f"Multiple original files found for '{capture_filename}'."
            )

        captured = bool(inbox_paths or original_paths)
        imported = bool(original_paths)
        staged = staging_path.is_file()

        inventory_row = inventory_by_intake.get(intake_id)
        review_status = (
            row_value(inventory_row, "review_status")
            if inventory_row is not None
            else "NOT_REVIEWED"
        )
        review_ready = (
            inventory_row is not None
            and row_value(inventory_row, "ready_for_queue").lower() == "yes"
        )
        draft_ready = intake_id in draft_by_intake
        queue_row = queue_by_intake.get(intake_id)
        queue_decision = (
            row_value(queue_row, "decision").upper()
            if queue_row is not None
            else ""
        )
        queued = queue_row is not None

        approval_row = approval_by_intake.get(intake_id)
        approval_decision = (
            row_value(approval_row, "decision").upper()
            if approval_row is not None
            else ""
        )
        decision_status = approval_decision or queue_decision or "NOT_DECIDED"
        decision_recorded = approval_row is not None or queue_decision in {
            "APPROVED",
            "REJECTED",
        }
        image_id = (
            row_value(approval_row, "image_id")
            if approval_row is not None
            else ""
        )
        approved = bool(image_id and image_id in approved_image_ids)

        queue_status = "NOT_QUEUED"
        if queued:
            queue_status = f"QUEUED_{queue_decision or 'PENDING'}"
        elif draft_ready:
            queue_status = "DRAFT_READY"

        stage, stage_index = stage_for_row(
            captured=captured,
            imported=imported,
            staged=staged,
            review_ready=review_ready or draft_ready,
            queued=queued,
            decision_recorded=decision_recorded,
            approved=approved,
        )
        rows.append(
            {
                "batch_id": row_value(source_row, "batch_id"),
                "batch_item_id": row_value(source_row, "batch_item_id"),
                "intake_id": intake_id,
                "part_group_id": row_value(source_row, "part_group_id"),
                "part_category": row_value(source_row, "part_category"),
                "view": row_value(source_row, "view"),
                "capture_filename": capture_filename,
                "capture_status": "CAPTURED" if captured else "MISSING",
                "import_status": "IMPORTED" if imported else "NOT_IMPORTED",
                "staging_status": "STAGED" if staged else "NOT_STAGED",
                "review_status": review_status,
                "queue_status": queue_status,
                "decision_status": decision_status,
                "approval_status": (
                    "APPROVED_DATASET" if approved else "NOT_APPROVED"
                ),
                "pipeline_stage": stage,
                "stage_index": str(stage_index),
                "progress_percent": str(round(stage_index / 7 * 100)),
                "next_action": next_action_for_stage(
                    stage,
                    capture_filename,
                    decision_status,
                ),
            }
        )

    progress = pd.DataFrame(
        rows,
        columns=FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS,
    )
    stage_counts = Counter(progress.get("pipeline_stage", pd.Series(dtype=str)))
    counts = {
        "planned": int(len(progress)),
        "captured": int((progress["capture_status"] == "CAPTURED").sum()),
        "imported": int((progress["import_status"] == "IMPORTED").sum()),
        "staged": int((progress["staging_status"] == "STAGED").sum()),
        "review_ready": int(
            progress["pipeline_stage"].isin(
                [
                    "REVIEW_READY",
                    "QUEUED_FOR_DECISION",
                    "DECISION_RECORDED",
                    "APPROVED_DATASET",
                ]
            ).sum()
        ),
        "queued": int(
            progress["pipeline_stage"].isin(
                [
                    "QUEUED_FOR_DECISION",
                    "DECISION_RECORDED",
                    "APPROVED_DATASET",
                ]
            ).sum()
        ),
        "decision_recorded": int(
            progress["pipeline_stage"].isin(
                ["DECISION_RECORDED", "APPROVED_DATASET"]
            ).sum()
        ),
        "approved": int(
            (progress["pipeline_stage"] == "APPROVED_DATASET").sum()
        ),
    }
    stage_index_total = sum(int(value) for value in progress["stage_index"])
    denominator = max(len(progress) * 7, 1)
    overall_progress = round(stage_index_total / denominator * 100, 1)

    category_progress = []
    for category, group in progress.groupby("part_category", sort=False):
        average = round(group["progress_percent"].astype(int).mean(), 1)
        category_progress.append(
            {
                "part_category": category,
                "completed_slots": int(
                    (group["pipeline_stage"] == "APPROVED_DATASET").sum()
                ),
                "planned_slots": int(len(group)),
                "progress_percent": average,
            }
        )

    next_actions = (
        progress.loc[
            progress["pipeline_stage"] != "APPROVED_DATASET",
            ["capture_filename", "next_action"],
        ]
        .head(5)
        .to_dict(orient="records")
    )
    after = protected_state()
    live_state_unchanged = "PASS" if before == after else "FAIL"
    if live_state_unchanged != "PASS":
        errors.append(
            "Dashboard scan changed capture, staging, queue, approval, "
            "annotation, or manifest inputs."
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "readiness": determine_readiness(counts, errors),
        "live_state_unchanged": live_state_unchanged,
        "overall_progress_percent": overall_progress,
        "counts": counts,
        "stage_counts": {
            stage: int(stage_counts.get(stage, 0))
            for stage in PIPELINE_STAGES
        },
        "category_progress": category_progress,
        "next_actions": next_actions,
        "errors": errors,
        "warnings": warnings,
    }
    return progress, report


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8", lineterminator="\n")
    temporary.replace(path)


def render_markdown(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# First Batch Capture Progress Summary",
        "",
        f"- Status: **{report['status']}**",
        f"- Readiness: **{report['readiness']}**",
        f"- Overall progress: **{report['overall_progress_percent']}%**",
        f"- Captured: **{counts['captured']} / {counts['planned']}**",
        f"- Imported: **{counts['imported']} / {counts['planned']}**",
        f"- Staged: **{counts['staged']} / {counts['planned']}**",
        f"- Review-ready: **{counts['review_ready']} / {counts['planned']}**",
        f"- Queued: **{counts['queued']} / {counts['planned']}**",
        f"- Approved: **{counts['approved']} / {counts['planned']}**",
        f"- Live state unchanged: **{report['live_state_unchanged']}**",
        "",
        "## Next actions",
        "",
    ]
    if report["next_actions"]:
        lines.extend(
            f"- `{item['capture_filename']}`: {item['next_action']}"
            for item in report["next_actions"]
        )
    else:
        lines.append("- No remaining action. The batch is complete.")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def status_class(stage: str) -> str:
    if stage == "APPROVED_DATASET":
        return "done"
    if stage in {"DECISION_RECORDED", "QUEUED_FOR_DECISION"}:
        return "review"
    if stage in {"STAGED", "REVIEW_READY"}:
        return "active"
    return "waiting"


def render_html(progress: pd.DataFrame, report: dict[str, Any]) -> str:
    counts = report["counts"]
    cards = [
        ("Overall", f"{report['overall_progress_percent']}%"),
        ("Captured", f"{counts['captured']} / {counts['planned']}"),
        ("Imported", f"{counts['imported']} / {counts['planned']}"),
        ("Staged", f"{counts['staged']} / {counts['planned']}"),
        ("Review-ready", f"{counts['review_ready']} / {counts['planned']}"),
        ("Approved", f"{counts['approved']} / {counts['planned']}"),
    ]
    card_html = "".join(
        "<article class='card'><span>"
        + html.escape(label)
        + "</span><strong>"
        + html.escape(value)
        + "</strong></article>"
        for label, value in cards
    )
    category_html = "".join(
        "<div class='category'><div><strong>"
        + html.escape(item["part_category"].replace("_", " ").title())
        + "</strong><span>"
        + f"{item['progress_percent']}%"
        + "</span></div><progress max='100' value='"
        + str(item["progress_percent"])
        + "'></progress></div>"
        for item in report["category_progress"]
    )
    rows = []
    for _, row in progress.iterrows():
        stage = row_value(row, "pipeline_stage")
        rows.append(
            "<tr>"
            f"<td>{html.escape(row_value(row, 'part_category'))}</td>"
            f"<td>{html.escape(row_value(row, 'view'))}</td>"
            f"<td><code>{html.escape(row_value(row, 'capture_filename'))}</code></td>"
            f"<td><span class='badge {status_class(stage)}'>"
            f"{html.escape(stage)}</span></td>"
            f"<td>{html.escape(row_value(row, 'progress_percent'))}%</td>"
            f"<td>{html.escape(row_value(row, 'next_action'))}</td>"
            "</tr>"
        )
    errors = ""
    if report["errors"]:
        errors = "<section class='errors'><h2>Blocking errors</h2><ul>" + "".join(
            f"<li>{html.escape(error)}</li>" for error in report["errors"]
        ) + "</ul></section>"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>First Batch Capture Dashboard</title>
<style>
:root {{ color-scheme: light; --bg:#f4f6f8; --panel:#fff; --ink:#17212b;
--muted:#647383; --line:#dce3e8; --accent:#176b87; --ok:#237a57;
--warn:#9a6500; --wait:#6b7280; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:15px/1.45 system-ui, sans-serif; background:var(--bg);
color:var(--ink); }}
main {{ max-width:1400px; margin:auto; padding:28px; }}
header {{ display:flex; justify-content:space-between; gap:20px; align-items:end;
margin-bottom:22px; }}
h1 {{ margin:0; font-size:30px; }}
.subtitle {{ color:var(--muted); }}
.readiness {{ padding:9px 13px; border-radius:999px; background:#e7f3f7;
color:var(--accent); font-weight:700; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:14px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
padding:16px; box-shadow:0 4px 14px rgba(23,33,43,.05); }}
.card span {{ display:block; color:var(--muted); margin-bottom:8px; }}
.card strong {{ font-size:25px; }}
.panel {{ margin-top:18px; background:var(--panel); border:1px solid var(--line);
border-radius:14px; padding:18px; overflow:auto; }}
.category {{ margin:12px 0; }}
.category div {{ display:flex; justify-content:space-between; gap:16px; }}
progress {{ width:100%; height:12px; accent-color:var(--accent); }}
table {{ width:100%; border-collapse:collapse; min-width:980px; }}
th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:11px 9px;
vertical-align:top; }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase;
letter-spacing:.04em; }}
code {{ font-size:12px; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:999px;
font-size:11px; font-weight:800; }}
.badge.waiting {{ background:#eef0f2; color:var(--wait); }}
.badge.active {{ background:#e7f3f7; color:var(--accent); }}
.badge.review {{ background:#fff1cf; color:var(--warn); }}
.badge.done {{ background:#e4f4ec; color:var(--ok); }}
.errors {{ margin-top:18px; padding:16px; border-radius:14px; background:#fff0f0;
border:1px solid #efb4b4; }}
@media (max-width:700px) {{ main {{ padding:16px; }} header {{ align-items:start;
flex-direction:column; }} }}
</style>
</head>
<body>
<main>
<header><div><h1>First Batch Capture Dashboard</h1>
<div class="subtitle">20 photographs across 10 physical automotive parts</div></div>
<div class="readiness">{html.escape(report['readiness'])}</div></header>
<section class="cards">{card_html}</section>
<section class="panel"><h2>Category progress</h2>{category_html}</section>
<section class="panel"><h2>Photograph pipeline</h2>
<table><thead><tr><th>Category</th><th>View</th><th>Filename</th>
<th>Stage</th><th>Progress</th><th>Next action</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></section>
{errors}
</main>
</body>
</html>
"""


def write_outputs(
    progress: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    atomic_write_csv(FIRST_BATCH_CAPTURE_PROGRESS_PATH, progress)
    atomic_write_text(
        DASHBOARD_JSON_PATH,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(DASHBOARD_MARKDOWN_PATH, render_markdown(report))
    atomic_write_text(DASHBOARD_HTML_PATH, render_html(progress, report))


def build_first_batch_capture_dashboard() -> dict[str, Any]:
    progress, report = build_progress()
    write_outputs(progress, report)
    return report


def main() -> None:
    report = build_first_batch_capture_dashboard()
    counts = report["counts"]
    print("First real batch capture dashboard")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Overall progress: {report['overall_progress_percent']}%")
    print(f"- Captured: {counts['captured']} / {counts['planned']}")
    print(f"- Imported: {counts['imported']} / {counts['planned']}")
    print(f"- Staged: {counts['staged']} / {counts['planned']}")
    print(f"- Review-ready: {counts['review_ready']} / {counts['planned']}")
    print(f"- Approved: {counts['approved']} / {counts['planned']}")
    print(f"- Live state unchanged: {report['live_state_unchanged']}")
    print(f"- Dashboard: {relative_path(DASHBOARD_HTML_PATH)}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
