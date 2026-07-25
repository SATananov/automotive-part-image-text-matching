from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd

from src.build_exam_first_submission import normalized_sha256
from src.exam_first_config import (
    BASE_CHECKPOINT_COMMIT,
    BASE_CHECKPOINT_COMMIT_COUNT,
    COURSE_ALIGNMENT_PATH,
    DEFENSE_NOTES_PATH,
    EXAM_README_PATH,
    FORBIDDEN_NOTEBOOK_CODE_TOKENS,
    GENERATED_ARTIFACTS,
    HISTORICAL_README_PATH,
    HISTORICAL_NOTEBOOK_CATALOGUE_PATH,
    MANIFEST_PATH,
    NOTEBOOK_PATH,
    PRIMARY_QUESTION,
    PROJECT_ROOT,
    READINESS,
    REPRODUCTION_PATH,
    REQUIRED_NOTEBOOK_HEADINGS,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_ARTIFACTS,
    STATUS_PATH,
    STEP,
    SUMMARY_PATH,
    SUPPLEMENTARY_INDEX_PATH,
    project_relative,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def record(
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
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
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


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    required = (
        NOTEBOOK_PATH,
        EXAM_README_PATH,
        REPRODUCTION_PATH,
        DEFENSE_NOTES_PATH,
        COURSE_ALIGNMENT_PATH,
        SUPPLEMENTARY_INDEX_PATH,
        STATUS_PATH,
        SUMMARY_PATH,
        MANIFEST_PATH,
        *SOURCE_ARTIFACTS,
        *GENERATED_ARTIFACTS,
    )
    record(
        checks,
        "structure",
        all(path.is_file() for path in required),
        errors,
        "Required Step 011.5 source or generated artifacts are missing.",
    )
    if not checks["structure"]:
        return {"status": "FAIL", "checks": checks, "errors": errors}

    status = read_json(STATUS_PATH)
    manifest = read_json(MANIFEST_PATH)
    notebook_info = inspect_notebook()
    notebook = notebook_info["notebook"]
    metadata = notebook.metadata.get("project", {})

    record(
        checks,
        "status",
        status.get("status") == "PASS"
        and status.get("step") == STEP
        and status.get("readiness") == READINESS
        and status.get("primary_research_question") == PRIMARY_QUESTION
        and status.get("base_checkpoint_commit")
        == BASE_CHECKPOINT_COMMIT
        and status.get("base_checkpoint_commit_count")
        == BASE_CHECKPOINT_COMMIT_COUNT
        and status.get("source_archive_sha256") == SOURCE_ARCHIVE_SHA256
        and status.get("model_training_performed") is False
        and status.get("locked_test_csv_files_opened") is False
        and status.get("test_split_used") is False
        and status.get("final_test_evaluation_authorized") is False
        and status.get("production_final_model_changed") is False
        and status.get("model_selection_changed") is False,
        errors,
        "Step 011.5 status or safety boundary is incomplete.",
    )

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8-sig")
    exam_readme = EXAM_README_PATH.read_text(encoding="utf-8-sig")
    primary_readme = readme.split("<details>", 1)[0]
    normalized_readme = " ".join(primary_readme.split())
    normalized_exam_readme = " ".join(exam_readme.split())
    record(
        checks,
        "exam_first_entry_point",
        PRIMARY_QUESTION in normalized_readme
        and "exam/01_focused_deep_learning_project.ipynb" in readme
        and "one analysis" in readme.lower()
        and "grouped" in readme.lower()
        and "concrete errors" in readme.lower()
        and "supplementary/README.md" in readme
        and len(primary_readme.split()) < 1200
        and PRIMARY_QUESTION in normalized_exam_readme,
        errors,
        "README does not present one concise, professor-focused entry point.",
    )

    code_cells = notebook_info["code_cells"]
    execution_counts = [cell.execution_count for cell in code_cells]
    record(
        checks,
        "notebook_structure",
        all(
            heading in notebook_info["markdown"]
            for heading in REQUIRED_NOTEBOOK_HEADINGS
        )
        and 20 <= len(notebook.cells) <= 26
        and 8 <= len(code_cells) <= 10,
        errors,
        "Focused notebook structure is incomplete or has become too broad.",
    )
    record(
        checks,
        "notebook_execution",
        len(execution_counts) == len(set(execution_counts))
        and all(isinstance(count, int) and count > 0 for count in execution_counts)
        and len(notebook_info["outputs"]) >= 14
        and len(notebook_info["figures"]) >= 5
        and not any(
            output.get("output_type") == "error"
            for output in notebook_info["outputs"]
        ),
        errors,
        "Focused notebook is not fully executed or lacks saved evidence.",
    )

    code_lower = notebook_info["code"].lower()
    record(
        checks,
        "notebook_boundary",
        not any(token.lower() in code_lower for token in FORBIDDEN_NOTEBOOK_CODE_TOKENS)
        and metadata.get("model_training_performed") is False
        and metadata.get("locked_test_csv_files_opened") is False
        and metadata.get("test_split_used") is False
        and metadata.get("final_test_evaluation_authorized") is False
        and metadata.get("production_final_model_changed") is False,
        errors,
        "Notebook code or metadata violates the locked evaluation boundary.",
    )

    train = pd.read_csv(PROJECT_ROOT / "data/processed/integrated_train.csv")
    validation = pd.read_csv(
        PROJECT_ROOT / "data/processed/integrated_validation.csv"
    )
    isolation = (
        not (set(train["part_group_id"]) & set(validation["part_group_id"]))
        and not (set(train["image_id"]) & set(validation["image_id"]))
        and not (set(train["image_path"]) & set(validation["image_path"]))
        and len(validation) == 60
        and validation["part_group_id"].nunique() == 20
    )
    record(
        checks,
        "grouped_split",
        isolation,
        errors,
        "Train/validation group or image isolation is not preserved.",
    )

    comparison = pd.read_csv(
        PROJECT_ROOT
        / "reports/integrated_training/validation_comparison.csv"
    ).sort_values("validation_rank")
    best = comparison.iloc[0]
    record(
        checks,
        "main_result",
        best["model_slug"] == "keras_multimodal"
        and abs(best["integrated_validation_accuracy"] - 0.5333333333)
        < 1e-8
        and abs(best["integrated_validation_macro_f1"] - 0.5208271299)
        < 1e-6
        and comparison["test_split_used"].eq(False).all(),
        errors,
        "Primary validation result is inconsistent or test contaminated.",
    )

    def predictions(slug: str, prefix: str) -> pd.DataFrame:
        frame = pd.read_csv(
            PROJECT_ROOT
            / f"reports/integrated_training/{slug}/validation_predictions.csv"
        )
        return frame[["sample_id", "is_correct"]].rename(
            columns={"is_correct": f"{prefix}_correct"}
        )

    multimodal = predictions("keras_multimodal", "multimodal")
    joined = (
        multimodal.merge(predictions("keras_text", "text"), on="sample_id")
        .merge(predictions("keras_image", "image"), on="sample_id")
    )
    multimodal_specific_wins = int(
        (
            joined["multimodal_correct"]
            & (~joined["text_correct"] | ~joined["image_correct"])
        ).sum()
    )
    record(
        checks,
        "concrete_modality_evidence",
        multimodal_specific_wins >= 10
        and "multimodal model is correct" in notebook_info["markdown"].lower(),
        errors,
        "Concrete examples do not show what multimodal input contributes.",
    )

    error_summary = read_json(
        PROJECT_ROOT
        / "reports/validation_model_improvement/validation_error_analysis.json"
    )
    error_rows = pd.read_csv(
        PROJECT_ROOT
        / "reports/validation_model_improvement/validation_error_analysis.csv"
    )
    confusion = pd.read_csv(
        PROJECT_ROOT
        / "reports/validation_model_improvement/candidates/"
        "reference_multimodal/validation_confusion_matrix.csv",
        index_col=0,
    )
    record(
        checks,
        "error_analysis",
        error_summary.get("validation_sample_count") == 60
        and error_summary.get("error_count") == 35
        and int((error_rows["true_label"] == "PARTIAL_MATCH").sum()) == 20
        and error_summary["errors_by_source"][
            "wikimedia_commons_open_license"
        ]["error_rate"]
        > error_summary["errors_by_source"][
            "generated_development"
        ]["error_rate"]
        and int(confusion["predicted_PARTIAL_MATCH"].sum()) == 1,
        errors,
        "Concrete Deep Learning error analysis is incomplete or inconsistent.",
    )

    course_alignment = COURSE_ALIGNMENT_PATH.read_text(encoding="utf-8-sig")
    supplementary = SUPPLEMENTARY_INDEX_PATH.read_text(encoding="utf-8-sig")
    notebook_catalogue = (PROJECT_ROOT / "notebooks/README.md").read_text(
        encoding="utf-8-sig"
    )
    normalized_supplementary = " ".join(supplementary.split()).lower()
    normalized_catalogue = " ".join(notebook_catalogue.split())
    record(
        checks,
        "supporting_evidence_isolation",
        "Not claimed as part of the primary experiment" in course_alignment
        and "pretrained-transformer task remains gated" in course_alignment
        and "not physically moved" in normalized_supplementary
        and "not intended to replace the focused exam narrative"
        in normalized_catalogue,
        errors,
        "Course coverage or supporting evidence is overstated or poorly separated.",
    )

    source_hashes = manifest.get("source_artifact_sha256", {})
    generated_hashes = manifest.get("generated_artifact_sha256", {})
    expected_sources = {project_relative(path) for path in SOURCE_ARTIFACTS}
    expected_generated = {
        project_relative(path)
        for path in GENERATED_ARTIFACTS
        if path != MANIFEST_PATH
    }
    manifest_ok = (
        manifest.get("status") == "PASS"
        and manifest.get("step") == STEP
        and manifest.get("readiness") == READINESS
        and manifest.get("base_checkpoint_commit") == BASE_CHECKPOINT_COMMIT
        and manifest.get("source_archive_sha256") == SOURCE_ARCHIVE_SHA256
        and set(source_hashes) == expected_sources
        and set(generated_hashes) == expected_generated
        and manifest.get("model_training_performed") is False
        and manifest.get("test_split_used") is False
        and manifest.get("final_test_evaluation_authorized") is False
        and manifest.get("production_final_model_changed") is False
        and manifest.get("model_selection_changed") is False
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
                    f"Step 011.5 artifact hash differs: {relative_path}."
                )
    record(
        checks,
        "manifest",
        manifest_ok,
        errors,
        "Step 011.5 manifest integrity failed.",
    )

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
        "summary": {
            "notebook_cells": len(notebook.cells),
            "executed_code_cells": len(code_cells),
            "saved_outputs": len(notebook_info["outputs"]),
            "figures": len(notebook_info["figures"]),
            "multimodal_specific_wins": multimodal_specific_wins,
            "validation_errors": error_summary.get("error_count"),
        },
    }


def main() -> None:
    report = build_verification_report()
    print("Exam-first submission verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["status"] != "PASS":
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
