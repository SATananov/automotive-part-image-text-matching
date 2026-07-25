from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import nbformat

from src.build_final_submission_notebook import normalized_sha256
from src.final_submission_config import (
    BASE_CHECKPOINT,
    CHECKLIST_PATH,
    DEFENSE_GUIDE_PATH,
    ERROR_ANALYSIS_PATH,
    ERROR_SUMMARY_PATH,
    FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL,
    FINAL_SUBMISSION_NOTEBOOK_PATH,
    GENERATED_ARTIFACTS,
    MANIFEST_PATH,
    READINESS,
    REFERENCE_TITLES,
    REQUIRED_NOTEBOOK_HEADINGS,
    RUBRIC_MATRIX_PATH,
    RUBRIC_MAX_POINTS,
    RUBRIC_SELF_ASSESSMENT,
    SELF_ASSESSMENT_PATH,
    SOURCE_ARTIFACTS,
    STATUS_PATH,
    STEP,
    SUBMISSION_DEADLINE,
    SUMMARY_PATH,
)
from src.real_dataset_config import PROJECT_ROOT


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _record(
    checks: dict[str, bool],
    name: str,
    passed: bool,
    errors: list[str],
    message: str,
) -> None:
    checks[name] = bool(passed)
    if not passed:
        errors.append(message)


def inspect_notebook() -> dict[str, Any]:
    notebook = nbformat.read(FINAL_SUBMISSION_NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    markdown = "\n".join(
        str(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    code = "\n".join(str(cell.source) for cell in code_cells)
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
    return {
        "notebook": notebook,
        "code_cells": code_cells,
        "markdown": markdown,
        "code": code,
        "outputs": outputs,
        "figures": figures,
    }


def execution_counts_are_complete(code_cells: list[Any]) -> bool:
    counts = [cell.execution_count for cell in code_cells]
    return (
        bool(counts)
        and all(isinstance(count, int) for count in counts)
        and all(
            current > previous
            for previous, current in zip(counts, counts[1:])
        )
    )


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    required_paths = tuple(
        dict.fromkeys(
            (
                *SOURCE_ARTIFACTS,
                *GENERATED_ARTIFACTS,
                SUMMARY_PATH,
                SELF_ASSESSMENT_PATH,
                CHECKLIST_PATH,
                DEFENSE_GUIDE_PATH,
                ERROR_ANALYSIS_PATH,
                RUBRIC_MATRIX_PATH,
                ERROR_SUMMARY_PATH,
            )
        )
    )
    _record(
        checks,
        "structure",
        all(path.is_file() for path in required_paths),
        errors,
        "Required Step 011.4 source or generated artifacts are missing.",
    )
    if not checks["structure"]:
        return {"status": "FAIL", "checks": checks, "errors": errors}

    status = read_json(STATUS_PATH)
    manifest = read_json(MANIFEST_PATH)
    error_summary = read_json(ERROR_SUMMARY_PATH)
    notebook_info = inspect_notebook()
    notebook = notebook_info["notebook"]
    code_cells = notebook_info["code_cells"]
    metadata = notebook.metadata.get("project", {})

    _record(
        checks,
        "status",
        status.get("status") == "PASS"
        and status.get("step") == STEP
        and status.get("readiness") == READINESS
        and status.get("base_checkpoint") == BASE_CHECKPOINT
        and status.get("submission_deadline") == SUBMISSION_DEADLINE
        and status.get("self_assessment_points") == 98
        and status.get("maximum_points") == 100
        and status.get("model_training_performed") is False
        and status.get("locked_test_csv_files_opened") is False
        and status.get("test_split_used") is False
        and status.get("final_test_evaluation_authorized") is False
        and status.get("production_final_model_changed") is False,
        errors,
        "Final submission status is incomplete or violates the evaluation boundary.",
    )

    _record(
        checks,
        "notebook_structure",
        all(
            heading in notebook_info["markdown"]
            for heading in REQUIRED_NOTEBOOK_HEADINGS
        )
        and len(notebook.cells) >= 28
        and len(code_cells) >= 14,
        errors,
        "Teacher-facing notebook sections or code evidence are incomplete.",
    )

    execution_counts = [cell.execution_count for cell in code_cells]
    error_output_count = sum(
        output.get("output_type") == "error"
        for output in notebook_info["outputs"]
    )
    notebook_execution_ok = (
        execution_counts_are_complete(code_cells)
        and len(notebook_info["outputs"]) >= 20
        and len(notebook_info["figures"]) >= 7
        and error_output_count == 0
    )
    _record(
        checks,
        "notebook_execution",
        notebook_execution_ok,
        errors,
        (
            "Final submission notebook execution evidence is incomplete: "
            f"execution_counts={execution_counts}, "
            f"outputs={len(notebook_info['outputs'])}, "
            f"figures={len(notebook_info['figures'])}, "
            f"error_outputs={error_output_count}."
        ),
    )

    forbidden_code_tokens = (
        "integrated_test.csv",
        "external_test.csv",
        "model.fit(",
        "keras.fit(",
        "load_model(",
    )
    _record(
        checks,
        "notebook_test_lock",
        not any(
            token.lower() in notebook_info["code"].lower()
            for token in forbidden_code_tokens
        )
        and metadata.get("test_split_used") is False
        and metadata.get("locked_test_csv_files_opened") is False
        and metadata.get("final_test_evaluation_authorized") is False
        and metadata.get("model_training_performed") is False
        and metadata.get("production_final_model_changed") is False,
        errors,
        "Notebook code or metadata violates the train/validation-only contract.",
    )

    _record(
        checks,
        "references",
        all(title in notebook_info["markdown"] for title in REFERENCE_TITLES),
        errors,
        "One or more required previous-research references are missing.",
    )

    with RUBRIC_MATRIX_PATH.open(encoding="utf-8", newline="") as handle:
        rubric_rows = list(csv.DictReader(handle))
    rubric_names = {row["criterion"] for row in rubric_rows}
    rubric_score = sum(int(row["self_assessed_points"]) for row in rubric_rows)
    rubric_maximum = sum(int(row["maximum_points"]) for row in rubric_rows)
    _record(
        checks,
        "rubric_alignment",
        len(rubric_rows) == 8
        and rubric_names == set(RUBRIC_MAX_POINTS)
        and rubric_score == sum(RUBRIC_SELF_ASSESSMENT.values()) == 98
        and rubric_maximum == sum(RUBRIC_MAX_POINTS.values()) == 100,
        errors,
        "Exam rubric evidence matrix is incomplete or numerically inconsistent.",
    )

    _record(
        checks,
        "deep_learning_error_analysis",
        error_summary.get("status") == "PASS"
        and error_summary.get("validation_sample_count") == 60
        and error_summary.get("error_count") == 35
        and error_summary.get("partial_match_error_count") == 20
        and error_summary.get("high_confidence_error_count") == 0
        and error_summary.get("controlled_failure_case_count") == 9
        and error_summary.get("real_image_error_rate")
        > error_summary.get("generated_image_error_rate")
        and error_summary.get("human_explainability_claimed") is False
        and error_summary.get("test_split_used") is False,
        errors,
        "Deep Learning error summary is incomplete, overstated, or not test locked.",
    )

    readme_text = (PROJECT_ROOT / "README.md").read_text(
        encoding="utf-8-sig"
    )
    notebook_readme = (PROJECT_ROOT / "notebooks" / "README.md").read_text(
        encoding="utf-8-sig"
    )
    _record(
        checks,
        "teacher_facing_documentation",
        FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL in readme_text
        and FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL in notebook_readme
        and "11 August 2026" in readme_text
        and "Deep Learning Error Analysis" in readme_text
        and "self_assessment.md" in readme_text
        and "03_final_exam_submission.ipynb" in notebook_readme,
        errors,
        "README or notebook catalogue does not point reviewers to the final Step 011.4 evidence.",
    )

    checklist_text = CHECKLIST_PATH.read_text(encoding="utf-8-sig")
    defense_text = DEFENSE_GUIDE_PATH.read_text(encoding="utf-8-sig")
    self_assessment_text = SELF_ASSESSMENT_PATH.read_text(
        encoding="utf-8-sig"
    )
    _record(
        checks,
        "communication_artifacts",
        SUBMISSION_DEADLINE in checklist_text
        and "PARTIAL_MATCH" in defense_text
        and "98/100" in self_assessment_text
        and "test split remains locked" in checklist_text.lower(),
        errors,
        "Submission checklist, self-assessment, or defense guide is incomplete.",
    )

    generated_hashes = manifest.get("generated_artifact_sha256", {})
    source_hashes = manifest.get("source_artifact_sha256", {})
    expected_generated = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in GENERATED_ARTIFACTS
        if path != MANIFEST_PATH
    }
    expected_sources = {
        path.relative_to(PROJECT_ROOT).as_posix() for path in SOURCE_ARTIFACTS
    }
    manifest_ok = (
        manifest.get("status") == "PASS"
        and manifest.get("step") == STEP
        and manifest.get("readiness") == READINESS
        and manifest.get("base_checkpoint") == BASE_CHECKPOINT
        and set(generated_hashes) == expected_generated
        and set(source_hashes) == expected_sources
        and manifest.get("model_training_performed") is False
        and manifest.get("test_split_used") is False
        and manifest.get("final_test_evaluation_authorized") is False
        and manifest.get("production_final_model_changed") is False
    )
    if manifest_ok:
        for relative_path, expected_hash in {
            **source_hashes,
            **generated_hashes,
        }.items():
            path = PROJECT_ROOT / relative_path
            if not path.is_file() or normalized_sha256(path) != expected_hash:
                manifest_ok = False
                errors.append(
                    f"Final submission artifact hash differs: {relative_path}."
                )
    _record(
        checks,
        "manifest",
        manifest_ok,
        errors,
        "Final submission manifest integrity failed.",
    )

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
        "summary": {
            "rubric_criteria": len(rubric_rows),
            "self_assessment_points": rubric_score,
            "notebook_cells": len(notebook.cells),
            "executed_code_cells": len(code_cells),
            "saved_outputs": len(notebook_info["outputs"]),
            "figures": len(notebook_info["figures"]),
            "deep_learning_errors": error_summary.get("error_count"),
            "controlled_failure_cases": error_summary.get(
                "controlled_failure_case_count"
            ),
        },
    }


def main() -> None:
    report = build_verification_report()
    print("Final submission rubric alignment verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["status"] != "PASS":
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
