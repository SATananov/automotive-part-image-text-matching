from __future__ import annotations

import json

import nbformat

from src.final_exam_notebook_config import (
    FINAL_EXAM_NOTEBOOK_EXPECTED_IMAGE_OUTPUT_COUNT,
    FINAL_EXAM_NOTEBOOK_EXPECTED_VALIDATION_ERROR_COUNT,
    FINAL_EXAM_NOTEBOOK_PATH,
    NOTEBOOK_INTEGRATION_COMMIT,
)
from src.notebook_quality_audit_config import (
    CITATION_AUDIT_PATH,
    NOTEBOOK_EXECUTION_AUDIT_PATH,
    NUMERIC_CONSISTENCY_AUDIT_PATH,
    QUALITY_AUDIT_READINESS,
    QUALITY_AUDIT_STATUS_PATH,
    VISUAL_OUTPUT_AUDIT_PATH,
)
from src.project_cli import COMMANDS
from src.verification.notebook_execution_visual_and_citation_audit import (
    build_verification_report,
)


def read_json(path):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    assert isinstance(payload, dict)
    return payload


def test_notebook_quality_audit_commands_are_registered() -> None:
    run_spec = COMMANDS["run-notebook-quality-audit"]
    verify_spec = COMMANDS["verify-notebook-quality-audit"]

    assert run_spec.requires_tensorflow is False
    assert verify_spec.requires_tensorflow is False


def test_notebook_quality_audit_status_is_closed_and_pass() -> None:
    status = read_json(QUALITY_AUDIT_STATUS_PATH)

    assert status["status"] == "PASS"
    assert status["readiness"] == QUALITY_AUDIT_READINESS
    assert status["step"] == "010.7"
    assert status["base_commit"] == "2f41d84"
    assert status["notebook_integration_commit"] == NOTEBOOK_INTEGRATION_COMMIT
    assert status["model_retraining_performed"] is False
    assert status["model_selection_changed"] is False
    assert status["locked_test_csv_files_opened"] is False
    assert status["test_split_used"] is False
    assert status["final_test_evaluation_authorized"] is False


def test_notebook_execution_audit_is_deterministic() -> None:
    execution = read_json(NOTEBOOK_EXECUTION_AUDIT_PATH)

    assert execution["status"] == "PASS"
    assert execution["checks"]["sequential_execution_counts"] is True
    assert execution["checks"]["no_error_outputs"] is True
    assert (
        execution["checks"]["transient_execution_metadata_absent"]
        is True
    )
    assert (
        execution["checks"]["deterministic_scientific_output_fingerprint"]
        is True
    )
    assert (
        execution["committed_output_fingerprint"]
        == execution["fresh_output_fingerprint"]
    )
    assert execution["checks"]["html_render"] is True
    assert execution["html_embedded_image_count"] >= 6


def test_visual_output_audit_confirms_six_readable_figures() -> None:
    visual = read_json(VISUAL_OUTPUT_AUDIT_PATH)

    assert visual["status"] == "PASS"
    assert visual["figure_count"] == FINAL_EXAM_NOTEBOOK_EXPECTED_IMAGE_OUTPUT_COUNT
    assert visual["checks"]["minimum_dimensions"] is True
    assert visual["checks"]["minimum_payload_size"] is True
    assert visual["checks"]["non_blank_figures"] is True
    assert visual["checks"]["unique_figure_payloads"] is True
    assert (
        visual["checks"]["confusion_matrix_contrast_logic_present"]
        is True
    )


def test_numeric_consistency_uses_retained_model_predictions() -> None:
    numeric = read_json(NUMERIC_CONSISTENCY_AUDIT_PATH)

    assert numeric["status"] == "PASS"
    assert numeric["validation_samples"] == 60
    assert (
        numeric["incorrect_predictions"]
        == FINAL_EXAM_NOTEBOOK_EXPECTED_VALIDATION_ERROR_COUNT
    )
    assert numeric["correct_predictions"] == 32
    assert numeric["checks"]["confusion_matches_predictions"] is True
    assert (
        numeric["checks"][
            "error_chart_derived_from_retained_predictions"
        ]
        is True
    )


def test_citation_audit_uses_numbered_primary_or_official_sources() -> None:
    citations = read_json(CITATION_AUDIT_PATH)

    assert citations["status"] == "PASS"
    assert len(citations["sources"]) == 6
    assert citations["checks"]["all_sources_pass"] is True
    assert citations["checks"]["all_urls_accounted_for"] is True
    assert citations["checks"]["primary_or_official_sources_only"] is True
    assert citations["checks"]["inline_citation_range_complete"] is True


def test_final_notebook_contains_step_010_7_metadata_and_clean_cells() -> None:
    notebook = nbformat.read(FINAL_EXAM_NOTEBOOK_PATH, as_version=4)
    metadata = notebook.metadata["project"]
    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code"
    ]

    assert metadata["notebook_integration_commit"] == NOTEBOOK_INTEGRATION_COMMIT
    assert metadata["quality_audit_step"] == "010.7"
    assert metadata["visual_qa_required"] is True
    assert metadata["citation_audit_required"] is True
    assert all(
        "execution" not in cell.get("metadata", {})
        for cell in code_cells
    )


def test_current_notebook_quality_audit_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
