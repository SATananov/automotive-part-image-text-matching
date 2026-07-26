from __future__ import annotations

import json

import nbformat

from src.final_delivery_config import (
    BASE_CHECKPOINT_COMMIT,
    BASE_CHECKPOINT_COMMIT_COUNT,
    CANONICAL_NOTEBOOK_PATH,
    CANONICAL_NOTEBOOK_RELATIVE,
    GENERATED_ARTIFACTS,
    MANIFEST_PATH,
    READINESS,
    SOURCE_ARCHIVE_SHA256,
    STATUS_PATH,
    STEP,
)
from src.project_cli import COMMANDS
from src.verification.final_delivery import build_verification_report
from src.verification.project_verification import VERIFICATION_MODULES


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_final_delivery_commands_are_registered() -> None:
    assert COMMANDS["build-final-delivery"].module == (
        "src.build_final_delivery"
    )
    assert COMMANDS["verify-final-delivery"].module == (
        "src.verification.final_delivery"
    )


def test_final_delivery_verifier_is_in_project_verification() -> None:
    assert "src.verification.final_delivery" in VERIFICATION_MODULES


def test_final_delivery_artifacts_exist() -> None:
    assert all(path.is_file() for path in GENERATED_ARTIFACTS)


def test_final_delivery_status_locks_checkpoint_and_boundary() -> None:
    status = read_json(STATUS_PATH)

    assert status["step"] == STEP
    assert status["status"] == "PASS"
    assert status["readiness"] == READINESS
    assert status["base_checkpoint_commit"] == BASE_CHECKPOINT_COMMIT
    assert status["base_checkpoint_commit_count"] == BASE_CHECKPOINT_COMMIT_COUNT
    assert status["source_archive_sha256"] == SOURCE_ARCHIVE_SHA256
    assert status["canonical_entry_point_count"] == 1
    assert status["canonical_submission_notebook"] == (
        CANONICAL_NOTEBOOK_RELATIVE
    )
    assert status["single_prediction_set_used"] is True
    assert status["model_training_performed"] is False
    assert status["locked_test_csv_files_opened"] is False
    assert status["test_split_used"] is False
    assert status["final_test_evaluation_authorized"] is False
    assert status["production_final_model_changed"] is False
    assert status["model_selection_changed"] is False
    assert status["remote_github_render_review_required_after_push"] is True


def test_canonical_notebook_is_executed_and_static_rendered() -> None:
    status = read_json(STATUS_PATH)
    notebook = nbformat.read(CANONICAL_NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
    ]

    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.get("output_type") == "error" for output in outputs)
    assert status["notebook_execution"]["local_static_html_render"] == "PASS"
    assert status["local_static_render_verified"] is True


def test_final_delivery_manifest_is_complete() -> None:
    manifest = read_json(MANIFEST_PATH)

    assert manifest["step"] == STEP
    assert manifest["status"] == "PASS"
    assert manifest["readiness"] == READINESS
    assert manifest["canonical_submission_notebook"] == (
        CANONICAL_NOTEBOOK_RELATIVE
    )
    assert manifest["source_artifact_count"] == len(
        manifest["source_artifact_sha256"]
    )
    assert manifest["implementation_artifact_count"] == len(
        manifest["implementation_artifact_sha256"]
    )
    assert manifest["generated_artifact_count"] == len(
        manifest["generated_artifact_sha256"]
    )
    assert manifest["model_training_performed"] is False
    assert manifest["test_split_used"] is False
    assert manifest["production_final_model_changed"] is False


def test_final_delivery_verification_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["summary"]["canonical_entry_point_count"] == 1
    assert report["summary"]["validation_correct"] == 32
    assert report["summary"]["validation_errors"] == 28
    assert report["summary"]["remote_github_review_pending"] is True
