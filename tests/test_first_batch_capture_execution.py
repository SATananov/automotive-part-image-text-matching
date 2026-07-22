from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import src.execute_first_batch_capture_session as execution
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS,
    FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS,
    FIRST_BATCH_EXECUTION_JOURNAL_COLUMNS,
    SAMPLE_INTAKE_COLUMNS,
)
from src.verification.capture_execution import (
    EXECUTION_GUIDE_PATH,
    build_verification_report,
)


def configure_execution_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    runtime = tmp_path / "data" / "real" / "runtime" / "first_batch_capture"
    inbox = tmp_path / "data" / "real" / "capture_inbox" / "batch_001"
    originals = tmp_path / "data" / "real" / "originals" / "batch_001"
    staging = tmp_path / "data" / "real" / "staging"
    tracked = tmp_path / "tracked"
    protected = tmp_path / "protected"
    for path in (runtime, inbox, originals, staging, tracked, protected):
        path.mkdir(parents=True, exist_ok=True)

    tracked_paths = tuple(tracked / f"snapshot_{index}.txt" for index in range(4))
    for index, path in enumerate(tracked_paths):
        path.write_text(f"tracked-{index}\n", encoding="utf-8")
    canonical_inventory = tracked / "first_batch_capture_inventory.csv"
    canonical_draft = tracked / "first_batch_review_queue_draft.csv"
    pd.DataFrame(columns=FIRST_BATCH_CAPTURE_INVENTORY_COLUMNS).to_csv(
        canonical_inventory,
        index=False,
    )
    pd.DataFrame(columns=SAMPLE_INTAKE_COLUMNS).to_csv(
        canonical_draft,
        index=False,
    )

    live_paths = tuple(protected / f"live_{index}.csv" for index in range(5))
    for index, path in enumerate(live_paths):
        path.write_text(f"live-{index}\n", encoding="utf-8")

    paths = {
        "runtime": runtime,
        "inbox": inbox,
        "originals": originals,
        "staging": staging,
        "live_progress": runtime / "live_progress.csv",
        "live_dashboard": runtime / "live_dashboard.html",
        "live_dashboard_json": runtime / "live_dashboard.json",
        "live_progress_summary": runtime / "live_progress_summary.md",
        "live_status": runtime / "execution_status.json",
        "live_summary": runtime / "execution_summary.md",
        "journal": runtime / "execution_journal.csv",
        "import_inventory": runtime / "local_import_inventory.csv",
        "import_json": runtime / "local_import_status.json",
        "import_md": runtime / "local_import_status.md",
        "preview": runtime / "queue_preview.csv",
        "prepare_json": runtime / "preparation_status.json",
        "prepare_md": runtime / "preparation_status.md",
        "capture_inventory": runtime / "capture_inventory.csv",
        "draft": runtime / "review_queue_draft.csv",
        "stage_json": runtime / "staging_status.json",
        "stage_md": runtime / "staging_status.md",
        "session": runtime / "capture_session.csv",
        "session_json": runtime / "capture_session_status.json",
        "session_md": runtime / "capture_session_status.md",
        "tracked_mutation": tracked_paths[0],
        "live_mutation": live_paths[0],
        "canonical_inventory": canonical_inventory,
        "canonical_draft": canonical_draft,
    }

    monkeypatch.setattr(
        execution.staging,
        "FIRST_BATCH_CAPTURE_INVENTORY_PATH",
        canonical_inventory,
    )
    monkeypatch.setattr(
        execution.staging,
        "FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH",
        canonical_draft,
    )

    direct_mapping = {
        "PROJECT_ROOT": tmp_path,
        "FIRST_BATCH_RUNTIME_DIRECTORY": runtime,
        "FIRST_BATCH_CAPTURE_INBOX_DIRECTORY": inbox,
        "FIRST_BATCH_ORIGINALS_DIRECTORY": originals,
        "REAL_STAGING_DIRECTORY": staging,
        "FIRST_BATCH_LIVE_PROGRESS_PATH": paths["live_progress"],
        "FIRST_BATCH_LIVE_DASHBOARD_PATH": paths["live_dashboard"],
        "FIRST_BATCH_LIVE_DASHBOARD_JSON_PATH": paths["live_dashboard_json"],
        "FIRST_BATCH_LIVE_PROGRESS_SUMMARY_PATH": paths["live_progress_summary"],
        "FIRST_BATCH_LIVE_STATUS_PATH": paths["live_status"],
        "FIRST_BATCH_LIVE_SUMMARY_PATH": paths["live_summary"],
        "FIRST_BATCH_EXECUTION_JOURNAL_PATH": paths["journal"],
        "RUNTIME_IMPORT_INVENTORY_PATH": paths["import_inventory"],
        "RUNTIME_IMPORT_JSON_PATH": paths["import_json"],
        "RUNTIME_IMPORT_MARKDOWN_PATH": paths["import_md"],
        "RUNTIME_PREPARATION_PREVIEW_PATH": paths["preview"],
        "RUNTIME_PREPARATION_JSON_PATH": paths["prepare_json"],
        "RUNTIME_PREPARATION_MARKDOWN_PATH": paths["prepare_md"],
        "RUNTIME_CAPTURE_INVENTORY_PATH": paths["capture_inventory"],
        "RUNTIME_QUEUE_DRAFT_PATH": paths["draft"],
        "RUNTIME_STAGING_JSON_PATH": paths["stage_json"],
        "RUNTIME_STAGING_MARKDOWN_PATH": paths["stage_md"],
        "RUNTIME_SESSION_PATH": paths["session"],
        "RUNTIME_SESSION_JSON_PATH": paths["session_json"],
        "RUNTIME_SESSION_MARKDOWN_PATH": paths["session_md"],
        "TRACKED_OPERATIONAL_OUTPUTS": tracked_paths,
        "LIVE_DATASET_PATHS": live_paths,
    }
    for name, value in direct_mapping.items():
        monkeypatch.setattr(execution, name, value)

    bindings = (
        (
            execution.local_import,
            "FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH",
            paths["import_inventory"],
        ),
        (execution.local_import, "JSON_REPORT_PATH", paths["import_json"]),
        (execution.local_import, "MARKDOWN_REPORT_PATH", paths["import_md"]),
        (execution.batch_prepare, "FIRST_BATCH_PREVIEW_PATH", paths["preview"]),
        (execution.batch_prepare, "JSON_REPORT_PATH", paths["prepare_json"]),
        (execution.batch_prepare, "MARKDOWN_REPORT_PATH", paths["prepare_md"]),
        (
            execution.staging,
            "FIRST_BATCH_CAPTURE_INVENTORY_PATH",
            paths["capture_inventory"],
        ),
        (
            execution.staging,
            "FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH",
            paths["draft"],
        ),
        (execution.staging, "JSON_REPORT_PATH", paths["stage_json"]),
        (execution.staging, "MARKDOWN_REPORT_PATH", paths["stage_md"]),
        (execution.session, "FIRST_BATCH_CAPTURE_SESSION_PATH", paths["session"]),
        (execution.session, "JSON_REPORT_PATH", paths["session_json"]),
        (execution.session, "MARKDOWN_REPORT_PATH", paths["session_md"]),
        (
            execution.dashboard,
            "FIRST_BATCH_CAPTURE_INVENTORY_PATH",
            paths["capture_inventory"],
        ),
        (
            execution.dashboard,
            "FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH",
            paths["draft"],
        ),
        (
            execution.dashboard,
            "FIRST_BATCH_CAPTURE_PROGRESS_PATH",
            paths["live_progress"],
        ),
        (
            execution.dashboard,
            "DASHBOARD_HTML_PATH",
            paths["live_dashboard"],
        ),
        (
            execution.dashboard,
            "DASHBOARD_JSON_PATH",
            paths["live_dashboard_json"],
        ),
        (
            execution.dashboard,
            "DASHBOARD_MARKDOWN_PATH",
            paths["live_progress_summary"],
        ),
    )
    monkeypatch.setattr(execution, "RUNTIME_PATH_BINDINGS", bindings)
    monkeypatch.setattr(
        execution,
        "utc_now",
        lambda: datetime(2026, 7, 17, 9, 30, tzinfo=timezone.utc),
    )
    return paths


def install_success_phases(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
    *,
    import_capture: bool = False,
    readiness: str = "AWAITING_CAPTURE",
    captured: int = 0,
    review_ready: int = 0,
    approved: int = 0,
) -> None:
    def fake_import() -> dict[str, object]:
        newly_imported = 0
        source = paths["inbox"] / "real_starter_001_front.jpg"
        destination = paths["originals"] / source.name
        if import_capture and source.is_file() and not destination.is_file():
            destination.write_bytes(source.read_bytes())
            newly_imported = 1
        execution.local_import.FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH.write_text(
            "intake_id,import_status\n",
            encoding="utf-8",
        )
        return {
            "status": "PASS",
            "readiness": (
                "READY_FOR_STAGING"
                if newly_imported
                else "AWAITING_LOCAL_FILES"
            ),
            "counts": {
                "newly_imported": newly_imported,
                "originals_available": int(destination.is_file()),
            },
            "errors": [],
            "warnings": [],
        }

    def fake_stage() -> dict[str, object]:
        newly_staged = 0
        source = paths["originals"] / "real_starter_001_front.jpg"
        destination = paths["staging"] / "intake_000001.jpg"
        if source.is_file() and not destination.is_file():
            destination.write_bytes(source.read_bytes())
            newly_staged = 1
        execution.staging.FIRST_BATCH_CAPTURE_INVENTORY_PATH.write_text(
            "intake_id,ready_for_queue\n",
            encoding="utf-8",
        )
        execution.staging.FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH.write_text(
            "intake_id,decision\n",
            encoding="utf-8",
        )
        return {
            "status": "PASS",
            "readiness": readiness,
            "counts": {
                "newly_staged": newly_staged,
                "staged_files": int(destination.is_file()),
            },
            "errors": [],
            "warnings": [],
        }

    def fake_session() -> dict[str, object]:
        execution.session.FIRST_BATCH_CAPTURE_SESSION_PATH.write_text(
            "part_group_id,pair_status\n",
            encoding="utf-8",
        )
        return {
            "status": "PASS",
            "readiness": readiness,
            "counts": {"planned_files": 20},
            "errors": [],
            "warnings": [],
        }

    def fake_dashboard() -> dict[str, object]:
        progress = pd.DataFrame(
            [
                {
                    column: ""
                    for column in FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS
                }
            ],
            columns=FIRST_BATCH_CAPTURE_PROGRESS_COLUMNS,
        )
        progress.to_csv(
            execution.dashboard.FIRST_BATCH_CAPTURE_PROGRESS_PATH,
            index=False,
        )
        execution.dashboard.DASHBOARD_HTML_PATH.write_text(
            "<!doctype html><title>Live dashboard</title>\n",
            encoding="utf-8",
        )
        execution.dashboard.DASHBOARD_JSON_PATH.write_text(
            json.dumps({"status": "PASS"}) + "\n",
            encoding="utf-8",
        )
        execution.dashboard.DASHBOARD_MARKDOWN_PATH.write_text(
            "# Live progress\n",
            encoding="utf-8",
        )
        return {
            "status": "PASS",
            "readiness": readiness,
            "overall_progress_percent": 14.3 if captured else 0.0,
            "counts": {
                "planned": 20,
                "captured": captured,
                "imported": captured,
                "staged": captured,
                "review_ready": review_ready,
                "queued": 0,
                "decision_recorded": 0,
                "approved": approved,
            },
            "errors": [],
            "warnings": [],
        }

    monkeypatch.setattr(
        execution.local_import,
        "import_first_real_batch",
        fake_import,
    )
    monkeypatch.setattr(
        execution.staging,
        "stage_first_batch_capture",
        fake_stage,
    )
    monkeypatch.setattr(
        execution.session,
        "prepare_first_batch_capture_session",
        fake_session,
    )
    monkeypatch.setattr(
        execution.dashboard,
        "build_first_batch_capture_dashboard",
        fake_dashboard,
    )


def test_runtime_output_context_restores_module_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    original = execution.local_import.FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH

    with execution.runtime_output_paths():
        assert execution.local_import.FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH == (
            paths["import_inventory"]
        )

    assert execution.local_import.FIRST_BATCH_LOCAL_IMPORT_INVENTORY_PATH == original


def test_empty_execution_is_safe_no_capture_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    install_success_phases(monkeypatch, paths)

    report = execution.execute_capture_cycle()

    assert report["status"] == "PASS"
    assert report["result"] == "NO_CAPTURE_FILES"
    assert report["live_dataset_unchanged"] == "PASS"
    assert report["tracked_outputs_unchanged"] == "PASS"
    assert paths["live_dashboard"].is_file()
    assert paths["journal"].is_file()


def test_execution_imports_and_stages_available_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    paths["inbox"].joinpath("real_starter_001_front.jpg").write_bytes(b"image")
    install_success_phases(
        monkeypatch,
        paths,
        import_capture=True,
        readiness="CAPTURE_SESSION_IN_PROGRESS",
        captured=1,
    )

    report = execution.execute_capture_cycle()

    assert report["status"] == "PASS"
    assert report["result"] == "PROGRESS_UPDATED"
    assert report["counts"]["newly_imported"] == 1
    assert report["counts"]["newly_staged"] == 1
    assert paths["originals"].joinpath("real_starter_001_front.jpg").is_file()
    assert paths["staging"].joinpath("intake_000001.jpg").is_file()


def test_execution_reports_ready_for_manual_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    install_success_phases(
        monkeypatch,
        paths,
        readiness="READY_FOR_MANUAL_REVIEW",
        captured=20,
        review_ready=20,
    )

    report = execution.execute_capture_cycle()

    assert report["result"] == "READY_FOR_MANUAL_REVIEW"
    assert report["counts"]["review_ready"] == 20


def test_execution_reports_approved_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    install_success_phases(
        monkeypatch,
        paths,
        readiness="BATCH_APPROVED",
        captured=20,
        review_ready=20,
        approved=20,
    )

    report = execution.execute_capture_cycle()

    assert report["result"] == "BATCH_APPROVED"
    assert report["counts"]["approved"] == 20


def test_import_failure_rolls_back_originals_and_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    original = paths["originals"] / "existing.jpg"
    staged = paths["staging"] / "existing.jpg"
    original.write_bytes(b"original")
    staged.write_bytes(b"staged")

    def failing_import() -> dict[str, object]:
        original.write_bytes(b"changed")
        paths["originals"].joinpath("new.jpg").write_bytes(b"new")
        return {
            "status": "FAIL",
            "counts": {"newly_imported": 0},
            "errors": ["invalid capture"],
            "warnings": [],
        }

    monkeypatch.setattr(
        execution.local_import,
        "import_first_real_batch",
        failing_import,
    )
    report = execution.execute_capture_cycle()

    assert report["status"] == "FAIL"
    assert report["result"] == "ROLLED_BACK"
    assert original.read_bytes() == b"original"
    assert not paths["originals"].joinpath("new.jpg").exists()
    assert staged.read_bytes() == b"staged"


def test_staging_failure_rolls_back_imported_original(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)

    def successful_import() -> dict[str, object]:
        paths["originals"].joinpath("new.jpg").write_bytes(b"new")
        return {
            "status": "PASS",
            "counts": {"newly_imported": 1},
            "errors": [],
            "warnings": [],
        }

    def failing_stage() -> dict[str, object]:
        paths["staging"].joinpath("new.jpg").write_bytes(b"staged")
        return {
            "status": "FAIL",
            "counts": {"newly_staged": 0},
            "errors": ["staging conflict"],
            "warnings": [],
        }

    monkeypatch.setattr(
        execution.local_import,
        "import_first_real_batch",
        successful_import,
    )
    monkeypatch.setattr(
        execution.staging,
        "stage_first_batch_capture",
        failing_stage,
    )
    report = execution.execute_capture_cycle()

    assert report["status"] == "FAIL"
    assert not paths["originals"].joinpath("new.jpg").exists()
    assert not paths["staging"].joinpath("new.jpg").exists()


def test_unexpected_exception_rolls_back_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)

    def exploding_import() -> dict[str, object]:
        paths["originals"].joinpath("new.jpg").write_bytes(b"new")
        raise RuntimeError("boom")

    monkeypatch.setattr(
        execution.local_import,
        "import_first_real_batch",
        exploding_import,
    )
    report = execution.execute_capture_cycle()

    assert report["status"] == "FAIL"
    assert "boom" in report["errors"]
    assert not paths["originals"].joinpath("new.jpg").exists()


def test_live_dataset_mutation_is_restored_and_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    before = paths["live_mutation"].read_bytes()
    install_success_phases(monkeypatch, paths)
    original_dashboard = execution.dashboard.build_first_batch_capture_dashboard

    def mutating_dashboard() -> dict[str, object]:
        report = original_dashboard()
        paths["live_mutation"].write_text("mutated\n", encoding="utf-8")
        return report

    monkeypatch.setattr(
        execution.dashboard,
        "build_first_batch_capture_dashboard",
        mutating_dashboard,
    )
    report = execution.execute_capture_cycle()

    assert report["status"] == "FAIL"
    assert report["live_dataset_unchanged"] == "FAIL"
    assert paths["live_mutation"].read_bytes() == before


def test_tracked_output_mutation_is_restored_and_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    before = paths["tracked_mutation"].read_bytes()
    install_success_phases(monkeypatch, paths)
    original_dashboard = execution.dashboard.build_first_batch_capture_dashboard

    def mutating_dashboard() -> dict[str, object]:
        report = original_dashboard()
        paths["tracked_mutation"].write_text("mutated\n", encoding="utf-8")
        return report

    monkeypatch.setattr(
        execution.dashboard,
        "build_first_batch_capture_dashboard",
        mutating_dashboard,
    )
    report = execution.execute_capture_cycle()

    assert report["status"] == "FAIL"
    assert report["tracked_outputs_unchanged"] == "FAIL"
    assert paths["tracked_mutation"].read_bytes() == before


def test_execution_journal_appends_cycles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    install_success_phases(monkeypatch, paths)

    execution.execute_capture_cycle()
    execution.execute_capture_cycle()

    with paths["journal"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert tuple(rows[0]) == FIRST_BATCH_EXECUTION_JOURNAL_COLUMNS


def test_refresh_seeds_missing_runtime_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    source_inventory = paths["canonical_inventory"]
    source_draft = paths["canonical_draft"]
    source_inventory.write_text("inventory\n", encoding="utf-8")
    source_draft.write_text("draft\n", encoding="utf-8")

    execution.seed_runtime_operational_inputs()

    assert paths["capture_inventory"].read_text(encoding="utf-8") == (
        "inventory\n"
    )
    assert paths["draft"].read_text(encoding="utf-8") == "draft\n"


def test_refresh_does_not_call_import_or_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    install_success_phases(monkeypatch, paths)
    monkeypatch.setattr(
        execution.local_import,
        "import_first_real_batch",
        lambda: pytest.fail("refresh called import"),
    )
    monkeypatch.setattr(
        execution.staging,
        "stage_first_batch_capture",
        lambda: pytest.fail("refresh called staging"),
    )

    report = execution.refresh_live_progress()

    assert report["status"] == "PASS"
    assert report["tracked_outputs_unchanged"] == "PASS"


def test_refresh_keeps_live_dataset_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    before = paths["live_mutation"].read_bytes()
    install_success_phases(monkeypatch, paths)

    report = execution.refresh_live_progress()

    assert report["live_dataset_unchanged"] == "PASS"
    assert paths["live_mutation"].read_bytes() == before


def test_runtime_outputs_are_written_below_runtime_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    install_success_phases(monkeypatch, paths)

    execution.execute_capture_cycle()

    for key in ("live_progress", "live_dashboard", "live_status", "journal"):
        assert paths[key].is_relative_to(paths["runtime"])
        assert paths[key].is_file()


def test_execution_status_has_required_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_execution_project(monkeypatch, tmp_path)
    install_success_phases(monkeypatch, paths)

    execution.execute_capture_cycle()
    report = json.loads(paths["live_status"].read_text(encoding="utf-8"))

    assert report["cycle_id"]
    assert report["executed_at_utc"] == "2026-07-17T09:30:00+00:00"
    assert report["tracked_outputs_unchanged"] == "PASS"


def test_step_009_7_cli_commands_are_registered() -> None:
    assert "run-first-real-batch-capture-session" in COMMANDS
    assert "refresh-first-real-batch-live-progress" in COMMANDS
    assert "verify-capture-execution" in COMMANDS


def test_execution_guide_uses_semantic_filename() -> None:
    assert EXECUTION_GUIDE_PATH.name == (
        "first_batch_capture_execution_and_live_progress.md"
    )
    assert not EXECUTION_GUIDE_PATH.name.startswith("step_")


def test_current_step_009_7_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]


def test_verifier_accepts_clean_checkpoint_without_runtime_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import src.verification.capture_execution as verifier

    runtime = tmp_path / "data" / "real" / "runtime" / "first_batch_capture"
    paths = tuple(runtime / path.name for path in verifier.RUNTIME_ARTIFACT_PATHS)
    monkeypatch.setattr(verifier, "FIRST_BATCH_RUNTIME_DIRECTORY", runtime)
    monkeypatch.setattr(verifier, "RUNTIME_ARTIFACT_PATHS", paths)
    monkeypatch.setattr(verifier, "FIRST_BATCH_EXECUTION_JOURNAL_PATH", paths[-1])
    monkeypatch.setattr(verifier, "FIRST_BATCH_LIVE_STATUS_PATH", paths[2])

    assert verifier.validate_structure() == []
    assert verifier.validate_journal_schema() == []
    assert verifier.validate_current_status() == []
