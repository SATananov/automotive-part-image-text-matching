from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat

from src.build_final_exam_notebook import sha256_file
from src.final_exam_notebook_config import (
    FINAL_EXAM_NOTEBOOK_EXPECTED_CELL_COUNT,
    FINAL_EXAM_NOTEBOOK_EXPECTED_CODE_CELL_COUNT,
    FINAL_EXAM_NOTEBOOK_EXPECTED_IMAGE_OUTPUT_COUNT,
    FINAL_EXAM_NOTEBOOK_EXPECTED_OUTPUT_COUNT,
    FINAL_EXAM_NOTEBOOK_EXPECTED_VALIDATION_ERROR_COUNT,
    FINAL_EXAM_NOTEBOOK_PATH,
    NOTEBOOK_INTEGRATION_COMMIT,
)
from src.notebook_quality_audit_config import (
    CITATION_AUDIT_PATH,
    NOTEBOOK_EXECUTION_AUDIT_PATH,
    NUMERIC_CONSISTENCY_AUDIT_PATH,
    QUALITY_AUDIT_BASE_COMMIT,
    QUALITY_AUDIT_MANIFEST_PATH,
    QUALITY_AUDIT_READINESS,
    QUALITY_AUDIT_STATUS_PATH,
    QUALITY_AUDIT_SUMMARY_PATH,
    VISUAL_OUTPUT_AUDIT_PATH,
)
from src.real_dataset_config import PROJECT_ROOT


REQUIRED_FILES = (
    FINAL_EXAM_NOTEBOOK_PATH,
    NOTEBOOK_EXECUTION_AUDIT_PATH,
    VISUAL_OUTPUT_AUDIT_PATH,
    NUMERIC_CONSISTENCY_AUDIT_PATH,
    CITATION_AUDIT_PATH,
    QUALITY_AUDIT_STATUS_PATH,
    QUALITY_AUDIT_MANIFEST_PATH,
    QUALITY_AUDIT_SUMMARY_PATH,
    PROJECT_ROOT / "src" / "notebook_quality_audit_config.py",
    PROJECT_ROOT
    / "src"
    / "run_notebook_execution_visual_and_citation_audit.py",
    PROJECT_ROOT
    / "src"
    / "verification"
    / "notebook_execution_visual_and_citation_audit.py",
    PROJECT_ROOT / "tests" / "test_notebook_quality_audit.py",
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    checks["structure"] = all(path.is_file() for path in REQUIRED_FILES)
    if not checks["structure"]:
        errors.append("Required Step 010.7 files are missing.")

    if FINAL_EXAM_NOTEBOOK_PATH.is_file():
        notebook = nbformat.read(FINAL_EXAM_NOTEBOOK_PATH, as_version=4)
        code_cells = [
            cell for cell in notebook.cells if cell.cell_type == "code"
        ]
        output_count = sum(
            len(cell.get("outputs", [])) for cell in code_cells
        )
        image_count = sum(
            1
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type")
            in {"display_data", "execute_result"}
            and "image/png" in output.get("data", {})
        )
        metadata = notebook.metadata.get("project", {})

        checks["notebook_shape"] = (
            len(notebook.cells)
            == FINAL_EXAM_NOTEBOOK_EXPECTED_CELL_COUNT
            and len(code_cells)
            == FINAL_EXAM_NOTEBOOK_EXPECTED_CODE_CELL_COUNT
            and output_count
            == FINAL_EXAM_NOTEBOOK_EXPECTED_OUTPUT_COUNT
            and image_count
            == FINAL_EXAM_NOTEBOOK_EXPECTED_IMAGE_OUTPUT_COUNT
        )
        checks["notebook_execution"] = (
            all(
                cell.get("execution_count") == index
                for index, cell in enumerate(code_cells, start=1)
            )
            and all(
                "execution" not in cell.get("metadata", {})
                for cell in code_cells
            )
            and all(
                output.get("output_type") != "error"
                for cell in code_cells
                for output in cell.get("outputs", [])
            )
        )
        checks["notebook_quality_metadata"] = (
            metadata.get("notebook_integration_commit")
            == NOTEBOOK_INTEGRATION_COMMIT
            and metadata.get("quality_audit_step") == "010.7"
            and metadata.get("visual_qa_required") is True
            and metadata.get("citation_audit_required") is True
            and metadata.get("test_split_used") is False
            and metadata.get("final_test_evaluation_authorized")
            is False
        )
    else:
        checks["notebook_shape"] = False
        checks["notebook_execution"] = False
        checks["notebook_quality_metadata"] = False

    component_paths = {
        "execution_report": NOTEBOOK_EXECUTION_AUDIT_PATH,
        "visual_report": VISUAL_OUTPUT_AUDIT_PATH,
        "numeric_report": NUMERIC_CONSISTENCY_AUDIT_PATH,
        "citation_report": CITATION_AUDIT_PATH,
    }
    for name, path in component_paths.items():
        passed = False
        if path.is_file():
            payload = read_json(path)
            passed = payload.get("status") == "PASS"
        checks[name] = passed
        if not passed:
            errors.append(f"Step 010.7 component is not PASS: {name}.")

    if NUMERIC_CONSISTENCY_AUDIT_PATH.is_file():
        numeric = read_json(NUMERIC_CONSISTENCY_AUDIT_PATH)
        checks["retained_model_error_count"] = (
            numeric.get("incorrect_predictions")
            == FINAL_EXAM_NOTEBOOK_EXPECTED_VALIDATION_ERROR_COUNT
            and numeric.get("validation_samples") == 60
        )
    else:
        checks["retained_model_error_count"] = False

    if QUALITY_AUDIT_STATUS_PATH.is_file():
        status = read_json(QUALITY_AUDIT_STATUS_PATH)
        checks["status"] = (
            status.get("status") == "PASS"
            and status.get("readiness") == QUALITY_AUDIT_READINESS
            and status.get("step") == "010.7"
            and status.get("base_commit") == QUALITY_AUDIT_BASE_COMMIT
            and status.get("notebook_integration_commit")
            == NOTEBOOK_INTEGRATION_COMMIT
            and status.get("model_retraining_performed") is False
            and status.get("model_selection_changed") is False
            and status.get("locked_test_csv_files_opened") is False
            and status.get("test_split_used") is False
            and status.get("final_test_evaluation_authorized") is False
        )
    else:
        checks["status"] = False

    if QUALITY_AUDIT_MANIFEST_PATH.is_file():
        manifest = read_json(QUALITY_AUDIT_MANIFEST_PATH)
        artifacts = manifest.get("artifact_sha256")
        checks["manifest"] = (
            manifest.get("status") == "PASS"
            and manifest.get("step") == "010.7"
            and manifest.get("base_commit") == QUALITY_AUDIT_BASE_COMMIT
            and manifest.get("hash_normalization") == "utf-8-lf"
            and isinstance(artifacts, dict)
            and len(artifacts) == 6
            and manifest.get("notebook_sha256")
            == sha256_file(FINAL_EXAM_NOTEBOOK_PATH)
            and manifest.get("model_retraining_performed") is False
            and manifest.get("locked_test_csv_files_opened") is False
            and manifest.get("test_split_used") is False
            and manifest.get("final_test_evaluation_authorized") is False
        )
        if checks["manifest"]:
            for relative_path, expected_hash in artifacts.items():
                artifact = PROJECT_ROOT / str(relative_path)
                if (
                    not artifact.is_file()
                    or sha256_file(artifact) != expected_hash
                ):
                    checks["manifest"] = False
                    errors.append(
                        f"Step 010.7 artifact hash differs: {relative_path}."
                    )
    else:
        checks["manifest"] = False

    cli_path = PROJECT_ROOT / "src" / "project_cli.py"
    readme_path = PROJECT_ROOT / "README.md"
    notebooks_readme_path = PROJECT_ROOT / "notebooks" / "README.md"
    checks["cli_and_documentation"] = False
    if (
        cli_path.is_file()
        and readme_path.is_file()
        and notebooks_readme_path.is_file()
    ):
        cli_text = cli_path.read_text(encoding="utf-8-sig")
        readme_text = readme_path.read_text(encoding="utf-8-sig")
        notebooks_readme_text = notebooks_readme_path.read_text(
            encoding="utf-8-sig"
        )
        checks["cli_and_documentation"] = (
            "run-notebook-quality-audit" in cli_text
            and "verify-notebook-quality-audit" in cli_text
            and "Notebook Execution, Visual QA and Citation Audit"
            in readme_text
            and "Step 010.7" in notebooks_readme_text
        )

    for name, passed in checks.items():
        if not passed and not any(name in error for error in errors):
            errors.append(f"Step 010.7 verification failed: {name}.")

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()
    print("Notebook execution, visual QA and citation audit verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
