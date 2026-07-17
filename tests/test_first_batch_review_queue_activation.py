from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src import activate_first_batch_review_queue as activation
from src import prepare_first_batch_manual_decisions as decisions
from src.project_cli import COMMANDS
from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    FIRST_BATCH_MANUAL_DECISION_COLUMNS,
    FIRST_BATCH_PLAN_COLUMNS,
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    SAMPLE_INTAKE_COLUMNS,
)
from src.verify_step_009_8 import build_verification_report


def sample_plan_row() -> dict[str, str]:
    return {
        "batch_id": "batch_001",
        "batch_item_id": "batch_001_001",
        "intake_id": "intake_000001",
        "staging_path": "data/real/staging/intake_000001.jpg",
        "part_group_id": "real_starter_001",
        "part_family": "electrical",
        "part_category": "starter",
        "view": "front",
        "source": "warehouse_photo",
        "match_description": "Automotive starter.",
        "partial_description": "Automotive alternator.",
        "mismatch_description": "Automotive brake disc.",
        "notes": "First balanced real-data batch plan.",
    }


def sample_queue_row(**updates: str) -> dict[str, str]:
    plan = sample_plan_row()
    row = {
        column: plan.get(column, "")
        for column in SAMPLE_INTAKE_COLUMNS
    }
    row["decision"] = "pending"
    row["rejection_reason"] = ""
    row.update(updates)
    return row


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def pass_review_report(intake: pd.DataFrame, *_args, **_kwargs) -> dict[str, object]:
    items = []
    for row in intake.to_dict(orient="records"):
        items.append(
            {
                "intake_id": row["intake_id"],
                "image_id": f"{row['part_group_id']}_{row['view']}",
                "status": "PASS",
                "metrics": {
                    "width": 640,
                    "height": 480,
                    "format": "JPEG",
                },
                "errors": [],
                "warnings": [],
            }
        )
    return {
        "status": "PASS",
        "readiness": "REVIEW_READY" if len(intake) else "EMPTY_QUEUE",
        "counts": {
            "rows": len(intake),
            "pending": len(intake),
            "approved": 0,
            "rejected": 0,
        },
        "items": items,
        "errors": [],
        "warnings": [],
    }


def configure_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    draft_rows: list[dict[str, str]] | None = None,
    live_rows: list[dict[str, str]] | None = None,
) -> dict[str, Path]:
    paths = {
        "draft": tmp_path / "processed" / "draft.csv",
        "runtime_draft": tmp_path / "runtime_capture" / "draft.csv",
        "queue": tmp_path / "annotations" / "sample_intake.csv",
        "plan": tmp_path / "annotations" / "first_batch_plan.csv",
        "groups": tmp_path / "annotations" / "part_groups.csv",
        "images": tmp_path / "annotations" / "images.csv",
        "approval": tmp_path / "processed" / "approval_log.csv",
        "manifest": tmp_path / "processed" / "manifest.csv",
        "processed_images": tmp_path / "processed" / "images",
        "status": tmp_path / "runtime_review" / "status.json",
        "summary": tmp_path / "runtime_review" / "summary.md",
    }
    write_csv(paths["draft"], SAMPLE_INTAKE_COLUMNS, draft_rows or [])
    write_csv(paths["queue"], SAMPLE_INTAKE_COLUMNS, live_rows or [])
    write_csv(paths["plan"], FIRST_BATCH_PLAN_COLUMNS, [sample_plan_row()])
    write_csv(paths["groups"], PART_GROUP_COLUMNS, [])
    write_csv(paths["images"], IMAGE_MANIFEST_COLUMNS, [])
    write_csv(paths["approval"], APPROVAL_LOG_COLUMNS, [])
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text("image_id,sha256\n", encoding="utf-8")
    paths["processed_images"].mkdir(parents=True, exist_ok=True)

    mapping = {
        "PROJECT_ROOT": tmp_path,
        "FIRST_BATCH_REVIEW_QUEUE_DRAFT_PATH": paths["draft"],
        "RUNTIME_REVIEW_QUEUE_DRAFT_PATH": paths["runtime_draft"],
        "REAL_SAMPLE_INTAKE_PATH": paths["queue"],
        "FIRST_BATCH_PLAN_PATH": paths["plan"],
        "REAL_PART_GROUPS_PATH": paths["groups"],
        "REAL_IMAGES_PATH": paths["images"],
        "REAL_APPROVAL_LOG_PATH": paths["approval"],
        "REAL_IMAGE_MANIFEST_PATH": paths["manifest"],
        "REAL_PROCESSED_IMAGES_DIRECTORY": paths["processed_images"],
        "FIRST_BATCH_REVIEW_ACTIVATION_STATUS_PATH": paths["status"],
        "FIRST_BATCH_REVIEW_ACTIVATION_SUMMARY_PATH": paths["summary"],
        "PROTECTED_LIVE_PATHS": (
            paths["groups"],
            paths["images"],
            paths["approval"],
            paths["manifest"],
        ),
    }
    for name, value in mapping.items():
        monkeypatch.setattr(activation, name, value)
    monkeypatch.setattr(activation, "build_review_report", pass_review_report)
    return paths


def empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def configure_decisions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    queue_rows: list[dict[str, str]],
) -> dict[str, Path]:
    paths = {
        "queue": tmp_path / "sample_intake.csv",
        "workbook": tmp_path / "runtime" / "workbook.csv",
        "guide": tmp_path / "runtime" / "guide.md",
        "status": tmp_path / "runtime" / "status.json",
        "summary": tmp_path / "runtime" / "summary.md",
    }
    write_csv(paths["queue"], SAMPLE_INTAKE_COLUMNS, queue_rows)
    intake = pd.DataFrame(queue_rows, columns=SAMPLE_INTAKE_COLUMNS)
    plan = pd.DataFrame([sample_plan_row()], columns=FIRST_BATCH_PLAN_COLUMNS)
    monkeypatch.setattr(decisions, "REAL_SAMPLE_INTAKE_PATH", paths["queue"])
    monkeypatch.setattr(decisions, "FIRST_BATCH_PLAN_PATH", tmp_path / "plan.csv")
    monkeypatch.setattr(
        decisions,
        "FIRST_BATCH_MANUAL_DECISION_WORKBOOK_PATH",
        paths["workbook"],
    )
    monkeypatch.setattr(
        decisions,
        "FIRST_BATCH_MANUAL_DECISION_GUIDE_PATH",
        paths["guide"],
    )
    monkeypatch.setattr(
        decisions,
        "FIRST_BATCH_MANUAL_DECISION_STATUS_PATH",
        paths["status"],
    )
    monkeypatch.setattr(
        decisions,
        "FIRST_BATCH_MANUAL_DECISION_SUMMARY_PATH",
        paths["summary"],
    )
    monkeypatch.setattr(
        decisions,
        "load_review_inputs",
        lambda: (
            intake,
            empty_frame(PART_GROUP_COLUMNS),
            empty_frame(IMAGE_MANIFEST_COLUMNS),
            empty_frame(APPROVAL_LOG_COLUMNS),
            [],
        ),
    )
    monkeypatch.setattr(
        decisions,
        "read_csv_exact",
        lambda *_args, **_kwargs: (plan, []),
    )
    monkeypatch.setattr(decisions, "build_review_report", pass_review_report)
    return paths


def test_empty_activation_plan_is_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_activation(monkeypatch, tmp_path)
    report = activation.plan_review_queue_activation()
    assert report["status"] == "PASS"
    assert report["result"] == "NO_REVIEW_READY_ITEMS"


def test_runtime_draft_is_preferred(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_activation(monkeypatch, tmp_path)
    write_csv(paths["runtime_draft"], SAMPLE_INTAKE_COLUMNS, [sample_queue_row()])
    report = activation.plan_review_queue_activation()
    assert report["draft_path"].endswith("runtime_capture/draft.csv")
    assert report["counts"]["rows_to_activate"] == 1


def test_nonpending_draft_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_activation(
        monkeypatch,
        tmp_path,
        draft_rows=[sample_queue_row(decision="approved")],
    )
    report = activation.plan_review_queue_activation()
    assert report["status"] == "FAIL"
    assert "only pending" in " ".join(report["errors"])


def test_unknown_intake_id_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_activation(
        monkeypatch,
        tmp_path,
        draft_rows=[sample_queue_row(intake_id="intake_999999")],
    )
    report = activation.plan_review_queue_activation()
    assert report["status"] == "FAIL"
    assert "not part of batch_001" in " ".join(report["errors"])


def test_plan_metadata_conflict_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_activation(
        monkeypatch,
        tmp_path,
        draft_rows=[sample_queue_row(part_category="alternator")],
    )
    report = activation.plan_review_queue_activation()
    assert report["status"] == "FAIL"
    assert "conflicts with the batch plan" in " ".join(report["errors"])


def test_exact_live_row_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    row = sample_queue_row()
    configure_activation(monkeypatch, tmp_path, draft_rows=[row], live_rows=[row])
    report = activation.plan_review_queue_activation()
    assert report["result"] == "ALREADY_ACTIVE"
    assert report["counts"]["rows_to_activate"] == 0


def test_conflicting_live_row_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_activation(
        monkeypatch,
        tmp_path,
        draft_rows=[sample_queue_row()],
        live_rows=[sample_queue_row(notes="different")],
    )
    report = activation.plan_review_queue_activation()
    assert report["status"] == "FAIL"
    assert "conflicts with the draft" in " ".join(report["errors"])


def test_review_failure_blocks_activation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_activation(monkeypatch, tmp_path, draft_rows=[sample_queue_row()])
    monkeypatch.setattr(
        activation,
        "build_review_report",
        lambda *_args, **_kwargs: {
            "status": "FAIL",
            "errors": ["bad staged image"],
            "warnings": [],
        },
    )
    report = activation.plan_review_queue_activation()
    assert report["status"] == "FAIL"
    assert "bad staged image" in report["errors"]


def test_successful_activation_writes_pending_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_activation(
        monkeypatch,
        tmp_path,
        draft_rows=[sample_queue_row()],
    )
    report = activation.activate_review_queue()
    queue = pd.read_csv(paths["queue"], dtype=str, keep_default_na=False)
    assert report["result"] == "ACTIVATED"
    assert queue["decision"].tolist() == ["pending"]


def test_activation_never_creates_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_activation(
        monkeypatch,
        tmp_path,
        draft_rows=[sample_queue_row()],
    )
    report = activation.activate_review_queue()
    assert report["counts"]["rows_activated"] == 1
    assert set(report["prospective_queue"]["decision"]) == {"pending"}


def test_activation_rollback_restores_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_activation(
        monkeypatch,
        tmp_path,
        draft_rows=[sample_queue_row()],
    )
    original = paths["queue"].read_bytes()
    calls = 0
    original_plan = activation.plan_review_queue_activation

    def fail_after_write() -> dict[str, object]:
        nonlocal calls
        calls += 1
        report = original_plan()
        if calls > 1:
            report["status"] = "FAIL"
            report["errors"] = ["forced post-write failure"]
        return report

    monkeypatch.setattr(activation, "plan_review_queue_activation", fail_after_write)
    with pytest.raises(activation.ReviewQueueActivationError):
        activation.activate_review_queue()
    assert paths["queue"].read_bytes() == original


def test_empty_manual_workbook_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_decisions(monkeypatch, tmp_path, [])
    report = decisions.prepare_manual_decisions()
    workbook = pd.read_csv(paths["workbook"], dtype=str, keep_default_na=False)
    assert report["readiness"] == "AWAITING_QUEUE_ACTIVATION"
    assert tuple(workbook.columns) == FIRST_BATCH_MANUAL_DECISION_COLUMNS


def test_pending_queue_creates_operator_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_decisions(monkeypatch, tmp_path, [sample_queue_row()])
    report = decisions.prepare_manual_decisions()
    workbook = pd.read_csv(paths["workbook"], dtype=str, keep_default_na=False)
    assert report["readiness"] == "MANUAL_DECISION_WORKBOOK_READY"
    assert workbook.loc[0, "operator_decision"] == ""
    assert workbook.loc[0, "quality_status"] == "PASS"


def test_operator_entries_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_decisions(monkeypatch, tmp_path, [sample_queue_row()])
    existing = {column: "" for column in FIRST_BATCH_MANUAL_DECISION_COLUMNS}
    existing.update(
        {
            "intake_id": "intake_000001",
            "operator_decision": "approved",
            "operator_notes": "Clear visual match.",
        }
    )
    write_csv(paths["workbook"], FIRST_BATCH_MANUAL_DECISION_COLUMNS, [existing])
    report = decisions.prepare_manual_decisions()
    workbook = pd.read_csv(paths["workbook"], dtype=str, keep_default_na=False)
    assert report["readiness"] == "READY_TO_RECORD_DECISIONS"
    assert workbook.loc[0, "operator_notes"] == "Clear visual match."


def test_rejected_entry_requires_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_decisions(monkeypatch, tmp_path, [sample_queue_row()])
    existing = {column: "" for column in FIRST_BATCH_MANUAL_DECISION_COLUMNS}
    existing.update(
        {
            "intake_id": "intake_000001",
            "operator_decision": "rejected",
        }
    )
    write_csv(paths["workbook"], FIRST_BATCH_MANUAL_DECISION_COLUMNS, [existing])
    report = decisions.prepare_manual_decisions()
    assert report["status"] == "FAIL"
    assert report["readiness"] == "MANUAL_DECISION_INPUT_INVALID"


def test_valid_rejection_is_ready(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_decisions(monkeypatch, tmp_path, [sample_queue_row()])
    existing = {column: "" for column in FIRST_BATCH_MANUAL_DECISION_COLUMNS}
    existing.update(
        {
            "intake_id": "intake_000001",
            "operator_decision": "rejected",
            "rejection_reason": "Part is obscured.",
        }
    )
    write_csv(paths["workbook"], FIRST_BATCH_MANUAL_DECISION_COLUMNS, [existing])
    report = decisions.prepare_manual_decisions()
    assert report["status"] == "PASS"
    assert report["readiness"] == "READY_TO_RECORD_DECISIONS"


def test_manual_preparation_does_not_change_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_decisions(monkeypatch, tmp_path, [sample_queue_row()])
    before = paths["queue"].read_bytes()
    report = decisions.prepare_manual_decisions()
    assert report["live_queue_unchanged"] == "PASS"
    assert paths["queue"].read_bytes() == before


def test_manual_guide_prohibits_live_queue_editing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_decisions(monkeypatch, tmp_path, [])
    decisions.prepare_manual_decisions()
    guide = paths["guide"].read_text(encoding="utf-8")
    assert "Do not edit the live" in guide
    assert "never records decisions" in guide


def test_step_009_8_cli_commands_are_registered() -> None:
    assert COMMANDS["activate-first-real-batch-review-queue"].module == (
        "src.activate_first_batch_review_queue"
    )
    assert COMMANDS["prepare-first-real-batch-manual-decisions"].module == (
        "src.prepare_first_batch_manual_decisions"
    )
    assert COMMANDS["verify-step-009-8"].module == "src.verify_step_009_8"


def test_current_step_009_8_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
