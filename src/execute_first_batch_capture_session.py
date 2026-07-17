from __future__ import annotations

import csv
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import src.build_first_batch_capture_dashboard as dashboard
import src.import_first_real_batch as local_import
import src.prepare_first_batch_capture_session as session
import src.prepare_first_real_batch as batch_prepare
import src.stage_first_real_batch_capture as staging
from src.prepare_first_batch_capture_session import path_fingerprint
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_INBOX_DIRECTORY,
    FIRST_BATCH_EXECUTION_JOURNAL_COLUMNS,
    FIRST_BATCH_EXECUTION_JOURNAL_PATH,
    FIRST_BATCH_LIVE_DASHBOARD_JSON_PATH,
    FIRST_BATCH_LIVE_DASHBOARD_PATH,
    FIRST_BATCH_LIVE_PROGRESS_PATH,
    FIRST_BATCH_LIVE_PROGRESS_SUMMARY_PATH,
    FIRST_BATCH_LIVE_STATUS_PATH,
    FIRST_BATCH_LIVE_SUMMARY_PATH,
    FIRST_BATCH_ORIGINALS_DIRECTORY,
    FIRST_BATCH_RUNTIME_DIRECTORY,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_SAMPLE_INTAKE_PATH,
    REAL_STAGING_DIRECTORY,
)


RUNTIME_IMPORT_INVENTORY_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "local_import_inventory.csv"
)
RUNTIME_IMPORT_JSON_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "local_import_status.json"
)
RUNTIME_IMPORT_MARKDOWN_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "local_import_status.md"
)
RUNTIME_PREPARATION_PREVIEW_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "queue_preview.csv"
)
RUNTIME_PREPARATION_JSON_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "preparation_status.json"
)
RUNTIME_PREPARATION_MARKDOWN_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "preparation_status.md"
)
RUNTIME_CAPTURE_INVENTORY_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "capture_inventory.csv"
)
RUNTIME_QUEUE_DRAFT_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "review_queue_draft.csv"
)
RUNTIME_STAGING_JSON_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "staging_status.json"
)
RUNTIME_STAGING_MARKDOWN_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "staging_status.md"
)
RUNTIME_SESSION_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "capture_session.csv"
)
RUNTIME_SESSION_JSON_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "capture_session_status.json"
)
RUNTIME_SESSION_MARKDOWN_PATH = (
    FIRST_BATCH_RUNTIME_DIRECTORY / "capture_session_status.md"
)


RUNTIME_PATH_BINDINGS = (
    (
        local_import,
        "FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH",
        RUNTIME_IMPORT_INVENTORY_PATH,
    ),
    (local_import, "JSON_REPORT_PATH", RUNTIME_IMPORT_JSON_PATH),
    (local_import, "MARKDOWN_REPORT_PATH", RUNTIME_IMPORT_MARKDOWN_PATH),
    (batch_prepare, "FIRST_BATCH_PREVIEW_PATH", RUNTIME_PREPARATION_PREVIEW_PATH),
    (batch_prepare, "JSON_REPORT_PATH", RUNTIME_PREPARATION_JSON_PATH),
    (batch_prepare, "MARKDOWN_REPORT_PATH", RUNTIME_PREPARATION_MARKDOWN_PATH),
    (staging, "FIRST_BATCH_CAPTURE_INVENTORY_PATH", RUNTIME_CAPTURE_INVENTORY_PATH),
    (staging, "FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH", RUNTIME_QUEUE_DRAFT_PATH),
    (staging, "JSON_REPORT_PATH", RUNTIME_STAGING_JSON_PATH),
    (staging, "MARKDOWN_REPORT_PATH", RUNTIME_STAGING_MARKDOWN_PATH),
    (session, "FIRST_BATCH_CAPTURE_SESSION_PATH", RUNTIME_SESSION_PATH),
    (session, "JSON_REPORT_PATH", RUNTIME_SESSION_JSON_PATH),
    (session, "MARKDOWN_REPORT_PATH", RUNTIME_SESSION_MARKDOWN_PATH),
    (dashboard, "FIRST_BATCH_CAPTURE_INVENTORY_PATH", RUNTIME_CAPTURE_INVENTORY_PATH),
    (dashboard, "FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH", RUNTIME_QUEUE_DRAFT_PATH),
    (dashboard, "FIRST_BATCH_CAPTURE_PROGRESS_PATH", FIRST_BATCH_LIVE_PROGRESS_PATH),
    (dashboard, "DASHBOARD_HTML_PATH", FIRST_BATCH_LIVE_DASHBOARD_PATH),
    (dashboard, "DASHBOARD_JSON_PATH", FIRST_BATCH_LIVE_DASHBOARD_JSON_PATH),
    (
        dashboard,
        "DASHBOARD_MARKDOWN_PATH",
        FIRST_BATCH_LIVE_PROGRESS_SUMMARY_PATH,
    ),
)


TRACKED_OPERATIONAL_OUTPUTS = (
    local_import.FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH,
    local_import.JSON_REPORT_PATH,
    local_import.MARKDOWN_REPORT_PATH,
    batch_prepare.FIRST_BATCH_PREVIEW_PATH,
    batch_prepare.JSON_REPORT_PATH,
    batch_prepare.MARKDOWN_REPORT_PATH,
    staging.FIRST_BATCH_CAPTURE_INVENTORY_PATH,
    staging.FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
    staging.JSON_REPORT_PATH,
    staging.MARKDOWN_REPORT_PATH,
    session.FIRST_BATCH_CAPTURE_SESSION_PATH,
    session.JSON_REPORT_PATH,
    session.MARKDOWN_REPORT_PATH,
    dashboard.FIRST_BATCH_CAPTURE_PROGRESS_PATH,
    dashboard.DASHBOARD_HTML_PATH,
    dashboard.DASHBOARD_JSON_PATH,
    dashboard.DASHBOARD_MARKDOWN_PATH,
)


LIVE_DATASET_PATHS = (
    REAL_SAMPLE_INTAKE_PATH,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_IMAGES_PATH,
)


@contextmanager
def runtime_output_paths() -> Iterator[None]:
    previous: list[tuple[object, str, object]] = []
    FIRST_BATCH_RUNTIME_DIRECTORY.mkdir(parents=True, exist_ok=True)
    try:
        for module, attribute, runtime_path in RUNTIME_PATH_BINDINGS:
            previous.append((module, attribute, getattr(module, attribute)))
            setattr(module, attribute, runtime_path)
        yield
    finally:
        for module, attribute, original_value in reversed(previous):
            setattr(module, attribute, original_value)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def files_fingerprint(paths: tuple[Path, ...]) -> dict[str, str]:
    return {
        relative_path(path): path_fingerprint(path)
        for path in paths
    }


def file_snapshot(paths: tuple[Path, ...]) -> dict[Path, bytes | None]:
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


def directory_snapshot(path: Path) -> dict[str, bytes]:
    if not path.exists():
        return {}
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def restore_directory(path: Path, snapshot: dict[str, bytes]) -> None:
    if path.exists():
        for item in sorted(path.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                try:
                    item.rmdir()
                except OSError:
                    pass
    path.mkdir(parents=True, exist_ok=True)
    for relative_name, content in snapshot.items():
        destination = path / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def planned_staging_snapshot() -> dict[str, bytes]:
    if not REAL_STAGING_DIRECTORY.exists():
        return {}
    return {
        item.relative_to(REAL_STAGING_DIRECTORY).as_posix(): item.read_bytes()
        for item in sorted(REAL_STAGING_DIRECTORY.rglob("*"))
        if item.is_file()
    }


def seed_runtime_operational_inputs() -> None:
    FIRST_BATCH_RUNTIME_DIRECTORY.mkdir(parents=True, exist_ok=True)
    seeds = (
        (
            staging.FIRST_BATCH_CAPTURE_INVENTORY_PATH,
            RUNTIME_CAPTURE_INVENTORY_PATH,
        ),
        (
            staging.FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH,
            RUNTIME_QUEUE_DRAFT_PATH,
        ),
    )
    for source, destination in seeds:
        if destination.is_file() or not source.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def clear_runtime_outputs() -> None:
    FIRST_BATCH_RUNTIME_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path in (
        RUNTIME_IMPORT_INVENTORY_PATH,
        RUNTIME_IMPORT_JSON_PATH,
        RUNTIME_IMPORT_MARKDOWN_PATH,
        RUNTIME_PREPARATION_PREVIEW_PATH,
        RUNTIME_PREPARATION_JSON_PATH,
        RUNTIME_PREPARATION_MARKDOWN_PATH,
        RUNTIME_CAPTURE_INVENTORY_PATH,
        RUNTIME_QUEUE_DRAFT_PATH,
        RUNTIME_STAGING_JSON_PATH,
        RUNTIME_STAGING_MARKDOWN_PATH,
        RUNTIME_SESSION_PATH,
        RUNTIME_SESSION_JSON_PATH,
        RUNTIME_SESSION_MARKDOWN_PATH,
        FIRST_BATCH_LIVE_PROGRESS_PATH,
        FIRST_BATCH_LIVE_DASHBOARD_PATH,
        FIRST_BATCH_LIVE_DASHBOARD_JSON_PATH,
        FIRST_BATCH_LIVE_PROGRESS_SUMMARY_PATH,
        FIRST_BATCH_LIVE_STATUS_PATH,
        FIRST_BATCH_LIVE_SUMMARY_PATH,
    ):
        path.unlink(missing_ok=True)


def runtime_progress_report() -> dict[str, Any]:
    if not FIRST_BATCH_LIVE_STATUS_PATH.is_file():
        return {}
    try:
        return json.loads(
            FIRST_BATCH_LIVE_STATUS_PATH.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}


def determine_result(
    import_report: dict[str, Any],
    staging_report: dict[str, Any],
    dashboard_report: dict[str, Any],
) -> str:
    readiness = str(dashboard_report.get("readiness", ""))
    if readiness == "BATCH_APPROVED":
        return "BATCH_APPROVED"
    if readiness in {"READY_FOR_MANUAL_REVIEW", "REVIEW_IN_PROGRESS"}:
        return "READY_FOR_MANUAL_REVIEW"
    imported = int(import_report.get("counts", {}).get("newly_imported", 0))
    staged = int(staging_report.get("counts", {}).get("newly_staged", 0))
    if imported or staged:
        return "PROGRESS_UPDATED"
    captured = int(dashboard_report.get("counts", {}).get("captured", 0))
    if captured == 0:
        return "NO_CAPTURE_FILES"
    return "NO_NEW_CHANGES"


def render_execution_summary(report: dict[str, Any]) -> str:
    counts = report["counts"]
    lines = [
        "# First Batch Capture Live Execution",
        "",
        f"- Status: **{report['status']}**",
        f"- Result: **{report['result']}**",
        f"- Readiness: **{report['readiness']}**",
        f"- Captured: **{counts['captured']} / {counts['planned']}**",
        f"- Newly imported: **{counts['newly_imported']}**",
        f"- Newly staged: **{counts['newly_staged']}**",
        f"- Review-ready: **{counts['review_ready']}**",
        f"- Overall progress: **{report['overall_progress_percent']}%**",
        f"- Live dataset unchanged: **{report['live_dataset_unchanged']}**",
        f"- Tracked outputs unchanged: **{report['tracked_outputs_unchanged']}**",
        "",
        "## Runtime outputs",
        "",
        f"- `{relative_path(FIRST_BATCH_LIVE_DASHBOARD_PATH)}`",
        f"- `{relative_path(FIRST_BATCH_LIVE_PROGRESS_PATH)}`",
        f"- `{relative_path(FIRST_BATCH_EXECUTION_JOURNAL_PATH)}`",
    ]
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def append_execution_journal(report: dict[str, Any]) -> None:
    FIRST_BATCH_EXECUTION_JOURNAL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    counts = report["counts"]
    row = {
        "cycle_id": report["cycle_id"],
        "executed_at_utc": report["executed_at_utc"],
        "status": report["status"],
        "result": report["result"],
        "readiness": report["readiness"],
        "captured": str(counts["captured"]),
        "newly_imported": str(counts["newly_imported"]),
        "newly_staged": str(counts["newly_staged"]),
        "review_ready": str(counts["review_ready"]),
        "overall_progress_percent": str(report["overall_progress_percent"]),
        "live_dataset_unchanged": report["live_dataset_unchanged"],
        "tracked_outputs_unchanged": report["tracked_outputs_unchanged"],
    }
    exists = FIRST_BATCH_EXECUTION_JOURNAL_PATH.is_file()
    with FIRST_BATCH_EXECUTION_JOURNAL_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIRST_BATCH_EXECUTION_JOURNAL_COLUMNS,
            lineterminator="\n",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def refresh_live_progress() -> dict[str, Any]:
    seed_runtime_operational_inputs()
    tracked_before = files_fingerprint(TRACKED_OPERATIONAL_OUTPUTS)
    live_before = files_fingerprint(LIVE_DATASET_PATHS)
    errors: list[str] = []
    with runtime_output_paths():
        session_report = session.prepare_first_batch_capture_session()
        dashboard_report = dashboard.build_first_batch_capture_dashboard()
    if session_report.get("status") != "PASS":
        errors.extend(session_report.get("errors", []))
    if dashboard_report.get("status") != "PASS":
        errors.extend(dashboard_report.get("errors", []))
    tracked_unchanged = (
        "PASS"
        if tracked_before == files_fingerprint(TRACKED_OPERATIONAL_OUTPUTS)
        else "FAIL"
    )
    live_unchanged = (
        "PASS"
        if live_before == files_fingerprint(LIVE_DATASET_PATHS)
        else "FAIL"
    )
    if tracked_unchanged != "PASS":
        errors.append("Live refresh changed tracked operational outputs.")
    if live_unchanged != "PASS":
        errors.append("Live refresh changed queue, approval, or annotations.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "readiness": dashboard_report.get("readiness", "UNKNOWN"),
        "overall_progress_percent": dashboard_report.get(
            "overall_progress_percent",
            0.0,
        ),
        "counts": dashboard_report.get("counts", {}),
        "live_dataset_unchanged": live_unchanged,
        "tracked_outputs_unchanged": tracked_unchanged,
        "errors": errors,
        "warnings": dashboard_report.get("warnings", []),
    }


def execute_capture_cycle() -> dict[str, Any]:
    FIRST_BATCH_RUNTIME_DIRECTORY.mkdir(parents=True, exist_ok=True)
    clear_runtime_outputs()
    original_snapshot = directory_snapshot(FIRST_BATCH_ORIGINALS_DIRECTORY)
    staging_snapshot = planned_staging_snapshot()
    tracked_snapshot = file_snapshot(TRACKED_OPERATIONAL_OUTPUTS)
    live_snapshot = file_snapshot(LIVE_DATASET_PATHS)
    tracked_before = files_fingerprint(TRACKED_OPERATIONAL_OUTPUTS)
    live_before = files_fingerprint(LIVE_DATASET_PATHS)
    errors: list[str] = []
    import_report: dict[str, Any] = {}
    staging_report: dict[str, Any] = {}
    session_report: dict[str, Any] = {}
    dashboard_report: dict[str, Any] = {}

    try:
        with runtime_output_paths():
            import_report = local_import.import_first_real_batch()
            if import_report.get("status") != "PASS":
                errors.extend(import_report.get("errors", []))
                raise RuntimeError("Local capture import failed.")

            staging_report = staging.stage_first_batch_capture()
            if staging_report.get("status") != "PASS":
                errors.extend(staging_report.get("errors", []))
                raise RuntimeError("Capture staging failed.")

            session_report = session.prepare_first_batch_capture_session()
            if session_report.get("status") != "PASS":
                errors.extend(session_report.get("errors", []))
                raise RuntimeError("Capture-session refresh failed.")

            dashboard_report = dashboard.build_first_batch_capture_dashboard()
            if dashboard_report.get("status") != "PASS":
                errors.extend(dashboard_report.get("errors", []))
                raise RuntimeError("Live dashboard refresh failed.")
    except Exception as error:
        restore_directory(FIRST_BATCH_ORIGINALS_DIRECTORY, original_snapshot)
        restore_directory(REAL_STAGING_DIRECTORY, staging_snapshot)
        if not errors:
            errors.append(str(error))

    live_after = files_fingerprint(LIVE_DATASET_PATHS)
    live_unchanged = "PASS" if live_before == live_after else "FAIL"
    tracked_after = files_fingerprint(TRACKED_OPERATIONAL_OUTPUTS)
    tracked_unchanged = "PASS" if tracked_before == tracked_after else "FAIL"
    if live_unchanged != "PASS":
        restore_directory(FIRST_BATCH_ORIGINALS_DIRECTORY, original_snapshot)
        restore_directory(REAL_STAGING_DIRECTORY, staging_snapshot)
        restore_files(live_snapshot)
        errors.append("Execution changed live queue, approval, or annotations.")
    if tracked_unchanged != "PASS":
        restore_files(tracked_snapshot)
        errors.append("Execution changed tracked operational snapshots.")

    if not dashboard_report:
        try:
            seed_runtime_operational_inputs()
            with runtime_output_paths():
                session_report = session.prepare_first_batch_capture_session()
                dashboard_report = (
                    dashboard.build_first_batch_capture_dashboard()
                )
        except Exception as refresh_error:
            errors.append(
                "Failure-state progress refresh failed: "
                f"{refresh_error}."
            )
            dashboard_report = {
                "status": "FAIL",
                "readiness": "CAPTURE_EXECUTION_BLOCKED",
                "overall_progress_percent": 0.0,
                "counts": {
                    "planned": 0,
                    "captured": 0,
                    "staged": 0,
                    "review_ready": 0,
                    "approved": 0,
                },
                "warnings": [],
            }

    counts = dashboard_report.get("counts", {})
    now = utc_now()
    cycle_seed = (
        f"{now.isoformat()}:{counts.get('captured', 0)}:"
        f"{counts.get('staged', 0)}"
    )
    report = {
        "status": "PASS" if not errors else "FAIL",
        "result": (
            determine_result(import_report, staging_report, dashboard_report)
            if not errors
            else "ROLLED_BACK"
        ),
        "readiness": dashboard_report.get("readiness", "UNKNOWN"),
        "cycle_id": hashlib.sha256(cycle_seed.encode("utf-8")).hexdigest()[:16],
        "executed_at_utc": now.isoformat(),
        "overall_progress_percent": dashboard_report.get(
            "overall_progress_percent",
            0.0,
        ),
        "counts": {
            "planned": int(counts.get("planned", 0)),
            "captured": int(counts.get("captured", 0)),
            "newly_imported": int(
                import_report.get("counts", {}).get("newly_imported", 0)
            ),
            "newly_staged": int(
                staging_report.get("counts", {}).get("newly_staged", 0)
            ),
            "staged": int(counts.get("staged", 0)),
            "review_ready": int(counts.get("review_ready", 0)),
            "approved": int(counts.get("approved", 0)),
        },
        "live_dataset_unchanged": live_unchanged,
        "tracked_outputs_unchanged": tracked_unchanged,
        "errors": errors,
        "warnings": sorted(
            set(
                import_report.get("warnings", [])
                + staging_report.get("warnings", [])
                + dashboard_report.get("warnings", [])
            )
        ),
    }
    atomic_write_text(
        FIRST_BATCH_LIVE_STATUS_PATH,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(FIRST_BATCH_LIVE_SUMMARY_PATH, render_execution_summary(report))
    append_execution_journal(report)
    return report


def main() -> None:
    report = execute_capture_cycle()
    counts = report["counts"]
    print("First real batch capture session execution")
    print(f"- Status: {report['status']}")
    print(f"- Result: {report['result']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Captured: {counts['captured']} / {counts['planned']}")
    print(f"- Newly imported: {counts['newly_imported']}")
    print(f"- Newly staged: {counts['newly_staged']}")
    print(f"- Review-ready: {counts['review_ready']}")
    print(f"- Overall progress: {report['overall_progress_percent']}%")
    print(f"- Live dataset unchanged: {report['live_dataset_unchanged']}")
    print(f"- Tracked outputs unchanged: {report['tracked_outputs_unchanged']}")
    print(f"- Live dashboard: {relative_path(FIRST_BATCH_LIVE_DASHBOARD_PATH)}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
