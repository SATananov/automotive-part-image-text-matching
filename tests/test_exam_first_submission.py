from __future__ import annotations

import json

import nbformat
import pandas as pd

from src.exam_first_config import (
    BASE_CHECKPOINT_COMMIT,
    BASE_CHECKPOINT_COMMIT_COUNT,
    COURSE_ALIGNMENT_PATH,
    GENERATED_ARTIFACTS,
    MANIFEST_PATH,
    NOTEBOOK_PATH,
    PRIMARY_QUESTION,
    READINESS,
    SOURCE_ARCHIVE_SHA256,
    STATUS_PATH,
    STEP,
    SUPPLEMENTARY_INDEX_PATH,
)
from src.project_cli import COMMANDS
from src.verification.exam_first_submission import build_verification_report
from src.verification.project_verification import VERIFICATION_MODULES


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_exam_first_commands_are_registered() -> None:
    assert COMMANDS["build-exam-first-submission"].module == (
        "src.build_exam_first_submission"
    )
    assert COMMANDS["verify-exam-first-submission"].module == (
        "src.verification.exam_first_submission"
    )


def test_exam_first_verifier_is_standalone() -> None:
    assert "src.verification.exam_first_submission" not in VERIFICATION_MODULES


def test_exam_first_generated_artifacts_exist() -> None:
    assert all(path.is_file() for path in GENERATED_ARTIFACTS)


def test_exam_first_status_records_checkpoint_and_boundary() -> None:
    status = read_json(STATUS_PATH)

    assert status["step"] == STEP
    assert status["status"] == "PASS"
    assert status["readiness"] == READINESS
    assert status["primary_research_question"] == PRIMARY_QUESTION
    assert status["base_checkpoint_commit"] == BASE_CHECKPOINT_COMMIT
    assert status["base_checkpoint_commit_count"] == (
        BASE_CHECKPOINT_COMMIT_COUNT
    )
    assert status["source_archive_sha256"] == SOURCE_ARCHIVE_SHA256
    assert status["model_training_performed"] is False
    assert status["locked_test_csv_files_opened"] is False
    assert status["test_split_used"] is False
    assert status["final_test_evaluation_authorized"] is False
    assert status["production_final_model_changed"] is False
    assert status["model_selection_changed"] is False


def test_focused_notebook_is_executed_and_narrow() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
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

    assert 20 <= len(notebook.cells) <= 26
    assert 8 <= len(code_cells) <= 10
    assert all(cell.execution_count is not None for cell in code_cells)
    assert len({cell.execution_count for cell in code_cells}) == len(code_cells)
    assert len(outputs) >= 14
    assert len(figures) >= 5
    assert not any(output.get("output_type") == "error" for output in outputs)


def test_focused_notebook_has_closed_test_and_training_boundary() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code = "\n".join(
        str(cell.source) for cell in notebook.cells if cell.cell_type == "code"
    ).lower()
    metadata = notebook.metadata["project"]

    assert "integrated_test.csv" not in code
    assert "external_test.csv" not in code
    assert "model.fit(" not in code
    assert "load_model(" not in code
    assert metadata["model_training_performed"] is False
    assert metadata["locked_test_csv_files_opened"] is False
    assert metadata["test_split_used"] is False
    assert metadata["final_test_evaluation_authorized"] is False
    assert metadata["production_final_model_changed"] is False


def test_train_validation_group_and_image_isolation_is_zero() -> None:
    from src.real_dataset_config import PROJECT_ROOT

    train = pd.read_csv(PROJECT_ROOT / "data/processed/integrated_train.csv")
    validation = pd.read_csv(
        PROJECT_ROOT / "data/processed/integrated_validation.csv"
    )

    assert not set(train["part_group_id"]) & set(validation["part_group_id"])
    assert not set(train["image_id"]) & set(validation["image_id"])
    assert not set(train["image_path"]) & set(validation["image_path"])
    assert len(validation) == 60
    assert validation["part_group_id"].nunique() == 20


def test_exam_first_result_is_the_retained_validation_result() -> None:
    status = read_json(STATUS_PATH)

    assert status["retained_model_slug"] == "keras_multimodal"
    assert abs(status["validation_accuracy"] - 0.5333333333) < 1e-8
    assert abs(status["validation_macro_f1"] - 0.5208271299) < 1e-6
    assert status["validation_error_count"] == 35
    assert status["partial_match_error_count"] == 20


def test_course_alignment_is_honest_about_unfinished_topics() -> None:
    text = COURSE_ALIGNMENT_PATH.read_text(encoding="utf-8-sig")

    assert "pretrained-transformer task remains gated" in text
    assert "genuine human annotation remain gated" in text
    assert "Not claimed as part of the primary experiment" in text
    assert "wait for the exact exercise" in text


def test_supplementary_index_preserves_standard_paths() -> None:
    text = SUPPLEMENTARY_INDEX_PATH.read_text(encoding="utf-8-sig")

    assert "not physically moved" in " ".join(text.split()).lower()
    assert "stable paths" in text
    assert "not the recommended starting point" in " ".join(text.split())


def test_exam_first_manifest_is_complete() -> None:
    manifest = read_json(MANIFEST_PATH)

    assert manifest["step"] == STEP
    assert manifest["status"] == "PASS"
    assert manifest["readiness"] == READINESS
    assert manifest["base_checkpoint_commit"] == BASE_CHECKPOINT_COMMIT
    assert manifest["source_archive_sha256"] == SOURCE_ARCHIVE_SHA256
    assert manifest["source_artifact_count"] == len(
        manifest["source_artifact_sha256"]
    )
    assert manifest["generated_artifact_count"] == len(
        manifest["generated_artifact_sha256"]
    )
    assert manifest["model_training_performed"] is False
    assert manifest["test_split_used"] is False
    assert manifest["production_final_model_changed"] is False


def test_exam_first_verification_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["summary"]["validation_errors"] == 35
    assert report["summary"]["multimodal_specific_wins"] >= 10
