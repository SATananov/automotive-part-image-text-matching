from __future__ import annotations

import json

import nbformat

from src.build_final_exam_notebook import validate_prerequisites
from src.final_exam_notebook_config import (
    BASE_CHECKPOINT_COMMIT,
    FINAL_EXAM_NOTEBOOK_MANIFEST_PATH,
    FINAL_EXAM_NOTEBOOK_PATH,
    FINAL_EXAM_NOTEBOOK_STATUS_PATH,
    FORBIDDEN_NOTEBOOK_CODE_TOKENS,
    NOTEBOOK_READINESS,
    REFERENCE_TITLES,
    REQUIRED_NOTEBOOK_HEADINGS,
)
from src.project_cli import COMMANDS
from src.verification.final_exam_notebook import build_verification_report


def read_json(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_final_exam_notebook_cli_commands_are_registered() -> None:
    build_spec = COMMANDS["build-final-exam-notebook"]
    verify_spec = COMMANDS["verify-final-exam-notebook"]
    assert build_spec.requires_tensorflow is False
    assert verify_spec.requires_tensorflow is False


def test_final_exam_notebook_prerequisites_preserve_test_lock() -> None:
    payloads = validate_prerequisites()
    assert payloads["selection"]["decision"] == "REFERENCE_RETAINED"
    assert payloads["authorization"]["authorized"] is False
    assert payloads["locked_contract"]["test_split_used"] is False


def test_final_exam_notebook_is_complete_and_executed() -> None:
    notebook = nbformat.read(FINAL_EXAM_NOTEBOOK_PATH, as_version=4)
    markdown_text = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert all(heading in markdown_text for heading in REQUIRED_NOTEBOOK_HEADINGS)
    assert all(title in markdown_text for title in REFERENCE_TITLES)
    assert code_cells
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    assert sum(len(cell.get("outputs", [])) for cell in code_cells) >= 10


def test_final_exam_notebook_code_excludes_locked_inputs_and_training() -> None:
    notebook = nbformat.read(FINAL_EXAM_NOTEBOOK_PATH, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        lowered = cell.source.lower()
        for token in FORBIDDEN_NOTEBOOK_CODE_TOKENS:
            assert token.lower() not in lowered
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error"


def test_final_exam_notebook_status_and_manifest_are_closed() -> None:
    status = read_json(FINAL_EXAM_NOTEBOOK_STATUS_PATH)
    manifest = read_json(FINAL_EXAM_NOTEBOOK_MANIFEST_PATH)

    assert status["status"] == "PASS"
    assert status["readiness"] == NOTEBOOK_READINESS
    assert status["base_checkpoint_commit"] == BASE_CHECKPOINT_COMMIT
    assert status["locked_test_csv_files_opened"] is False
    assert status["test_split_used"] is False
    assert status["model_retraining_performed"] is False
    assert status["model_selection_changed"] is False
    assert status["final_test_evaluation_authorized"] is False

    assert manifest["status"] == "PASS"
    assert manifest["locked_test_csv_files_opened"] is False
    assert manifest["test_split_used"] is False
    assert manifest["final_test_evaluation_authorized"] is False


def test_current_final_exam_notebook_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
