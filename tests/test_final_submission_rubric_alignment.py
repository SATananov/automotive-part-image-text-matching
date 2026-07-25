from __future__ import annotations

import csv

import nbformat

from src.final_submission_config import (
    CHECKLIST_PATH,
    DEFENSE_GUIDE_PATH,
    ERROR_ANALYSIS_PATH,
    ERROR_SUMMARY_PATH,
    FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL,
    FINAL_SUBMISSION_NOTEBOOK_PATH,
    GENERATED_ARTIFACTS,
    MANIFEST_PATH,
    READINESS,
    RUBRIC_MATRIX_PATH,
    SELF_ASSESSMENT_PATH,
    STATUS_PATH,
    STEP,
    SUBMISSION_DEADLINE,
)
from src.project_cli import COMMANDS
from src.verification.final_submission_rubric_alignment import (
    build_verification_report,
    execution_counts_are_complete,
    read_json,
)
from src.verification.project_verification import VERIFICATION_MODULES


def test_final_submission_commands_are_registered() -> None:
    assert COMMANDS["build-final-submission-notebook"].module == (
        "src.build_final_submission_notebook"
    )
    assert COMMANDS["verify-final-submission"].module == (
        "src.verification.final_submission_rubric_alignment"
    )
    assert COMMANDS["run-vision-suite"].module == (
        "src.run_vision_experimental_suite"
    )
    assert COMMANDS["build-vision-notebooks"].module == (
        "src.build_vision_experiment_notebooks"
    )
    assert COMMANDS["verify-vision-suite"].module == (
        "src.verification.vision_experimental_suite"
    )


def test_final_submission_verifier_is_standalone() -> None:
    assert (
        "src.verification.final_submission_rubric_alignment"
        not in VERIFICATION_MODULES
    )


def test_final_submission_artifacts_exist() -> None:
    assert all(path.is_file() for path in GENERATED_ARTIFACTS)
    assert CHECKLIST_PATH.is_file()
    assert SELF_ASSESSMENT_PATH.is_file()
    assert DEFENSE_GUIDE_PATH.is_file()
    assert ERROR_ANALYSIS_PATH.is_file()


def test_final_submission_status_is_pass_and_test_locked() -> None:
    status = read_json(STATUS_PATH)

    assert status["step"] == STEP
    assert status["status"] == "PASS"
    assert status["readiness"] == READINESS
    assert status["submission_deadline"] == SUBMISSION_DEADLINE
    assert status["self_assessment_points"] == 98
    assert status["maximum_points"] == 100
    assert status["model_training_performed"] is False
    assert status["locked_test_csv_files_opened"] is False
    assert status["test_split_used"] is False
    assert status["final_test_evaluation_authorized"] is False
    assert status["production_final_model_changed"] is False


def test_final_submission_notebook_is_executed() -> None:
    notebook = nbformat.read(FINAL_SUBMISSION_NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
    ]
    figures = [
        output
        for output in outputs
        if output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]

    assert len(notebook.cells) >= 28
    assert len(code_cells) >= 14
    assert execution_counts_are_complete(code_cells)
    original_counts = [cell.execution_count for cell in code_cells]
    for cell, count in zip(code_cells, original_counts):
        cell.execution_count = count + 5
    assert execution_counts_are_complete(code_cells)
    assert len(outputs) >= 20
    assert len(figures) >= 7
    assert not any(output.get("output_type") == "error" for output in outputs)


def test_final_submission_notebook_has_no_test_or_training_access() -> None:
    notebook = nbformat.read(FINAL_SUBMISSION_NOTEBOOK_PATH, as_version=4)
    code = "\n".join(
        str(cell.source) for cell in notebook.cells if cell.cell_type == "code"
    ).lower()
    metadata = notebook.metadata["project"]

    assert "integrated_test.csv" not in code
    assert "external_test.csv" not in code
    assert "model.fit(" not in code
    assert "keras.fit(" not in code
    assert metadata["model_training_performed"] is False
    assert metadata["locked_test_csv_files_opened"] is False
    assert metadata["test_split_used"] is False
    assert metadata["final_test_evaluation_authorized"] is False
    assert metadata["production_final_model_changed"] is False


def test_rubric_matrix_is_complete_and_conservative() -> None:
    with RUBRIC_MATRIX_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert sum(int(row["maximum_points"]) for row in rows) == 100
    assert sum(int(row["self_assessed_points"]) for row in rows) == 98
    assert all(row["repository_evidence"] for row in rows)


def test_deep_learning_error_summary_is_grounded() -> None:
    summary = read_json(ERROR_SUMMARY_PATH)

    assert summary["status"] == "PASS"
    assert summary["validation_sample_count"] == 60
    assert summary["error_count"] == 35
    assert summary["partial_match_error_count"] == 20
    assert summary["high_confidence_error_count"] == 0
    assert summary["controlled_failure_case_count"] == 9
    assert summary["real_image_error_rate"] > summary[
        "generated_image_error_rate"
    ]
    assert summary["human_explainability_claimed"] is False
    assert summary["test_split_used"] is False


def test_teacher_facing_links_and_deadline_are_present() -> None:
    from src.real_dataset_config import PROJECT_ROOT

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8-sig")
    notebook_readme = (PROJECT_ROOT / "notebooks" / "README.md").read_text(
        encoding="utf-8-sig"
    )
    checklist = CHECKLIST_PATH.read_text(encoding="utf-8-sig")

    assert FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL in readme
    assert FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL in notebook_readme
    assert "11 August 2026" in readme
    assert SUBMISSION_DEADLINE in checklist
    assert "self_assessment.md" in readme
    assert "Deep Learning Error Analysis" in readme


def test_final_submission_manifest_is_complete() -> None:
    manifest = read_json(MANIFEST_PATH)

    assert manifest["step"] == STEP
    assert manifest["status"] == "PASS"
    assert manifest["readiness"] == READINESS
    assert manifest["source_artifact_count"] == len(
        manifest["source_artifact_sha256"]
    )
    assert manifest["generated_artifact_count"] == len(
        manifest["generated_artifact_sha256"]
    )
    assert manifest["model_training_performed"] is False
    assert manifest["test_split_used"] is False
    assert manifest["final_test_evaluation_authorized"] is False
    assert manifest["production_final_model_changed"] is False


def test_final_submission_verification_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["summary"]["rubric_criteria"] == 8
    assert report["summary"]["self_assessment_points"] == 98
