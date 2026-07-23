from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat

from src.final_exam_notebook_config import (
    BASE_CHECKPOINT_COMMIT,
    FINAL_EXAM_NOTEBOOK_MANIFEST_PATH,
    FINAL_EXAM_NOTEBOOK_PATH,
    FINAL_EXAM_NOTEBOOK_STATUS_PATH,
    FINAL_EXAM_NOTEBOOK_SUMMARY_PATH,
    FORBIDDEN_NOTEBOOK_CODE_TOKENS,
    NOTEBOOK_READINESS,
    REFERENCE_TITLES,
    REQUIRED_NOTEBOOK_HEADINGS,
)
from src.real_dataset_config import PROJECT_ROOT


REQUIRED_FILES = (
    FINAL_EXAM_NOTEBOOK_PATH,
    FINAL_EXAM_NOTEBOOK_STATUS_PATH,
    FINAL_EXAM_NOTEBOOK_MANIFEST_PATH,
    FINAL_EXAM_NOTEBOOK_SUMMARY_PATH,
    PROJECT_ROOT / "src" / "final_exam_notebook_config.py",
    PROJECT_ROOT / "src" / "build_final_exam_notebook.py",
    PROJECT_ROOT / "tests" / "test_final_exam_notebook.py",
)




def sha256_file(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    checks["structure"] = all(path.is_file() for path in REQUIRED_FILES)
    if not checks["structure"]:
        errors.append("Required Step 010.6 files are missing.")

    if FINAL_EXAM_NOTEBOOK_PATH.is_file():
        notebook = nbformat.read(FINAL_EXAM_NOTEBOOK_PATH, as_version=4)
        markdown_text = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        )
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

        checks["research_narrative"] = all(
            heading in markdown_text for heading in REQUIRED_NOTEBOOK_HEADINGS
        )
        if not checks["research_narrative"]:
            errors.append("Final notebook research sections are incomplete.")

        checks["previous_research"] = all(
            title in markdown_text for title in REFERENCE_TITLES
        )
        if not checks["previous_research"]:
            errors.append("Final notebook references are incomplete.")

        checks["executed_outputs"] = bool(code_cells) and all(
            cell.get("execution_count") is not None for cell in code_cells
        ) and sum(len(cell.get("outputs", [])) for cell in code_cells) >= 10
        if not checks["executed_outputs"]:
            errors.append("Final notebook is not fully executed with saved outputs.")

        checks["locked_csv_input_exclusion"] = True
        for cell in code_cells:
            lowered = cell.source.lower()
            for token in FORBIDDEN_NOTEBOOK_CODE_TOKENS:
                if token.lower() in lowered:
                    checks["locked_csv_input_exclusion"] = False
                    errors.append(f"Forbidden notebook code token: {token}.")
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    checks["locked_csv_input_exclusion"] = False
                    errors.append("Notebook contains an execution error output.")

        metadata = notebook.metadata.get("project", {})
        checks["notebook_metadata"] = (
            metadata.get("step") == "010.6"
            and metadata.get("base_checkpoint") == BASE_CHECKPOINT_COMMIT
            and metadata.get("test_split_used") is False
            and metadata.get("final_test_evaluation_authorized") is False
        )
        if not checks["notebook_metadata"]:
            errors.append("Notebook project metadata differs from Step 010.6.")
    else:
        for name in (
            "research_narrative",
            "previous_research",
            "executed_outputs",
            "locked_csv_input_exclusion",
            "notebook_metadata",
        ):
            checks[name] = False

    if FINAL_EXAM_NOTEBOOK_STATUS_PATH.is_file():
        status = read_json(FINAL_EXAM_NOTEBOOK_STATUS_PATH)
        checks["status"] = (
            status.get("status") == "PASS"
            and status.get("readiness") == NOTEBOOK_READINESS
            and status.get("base_checkpoint_commit") == BASE_CHECKPOINT_COMMIT
            and status.get("locked_test_csv_files_opened") is False
            and status.get("test_split_used") is False
            and status.get("model_retraining_performed") is False
            and status.get("model_selection_changed") is False
            and status.get("final_test_evaluation_authorized") is False
        )
        if not checks["status"]:
            errors.append("Final exam notebook status is not closed and PASS.")
    else:
        checks["status"] = False

    if FINAL_EXAM_NOTEBOOK_MANIFEST_PATH.is_file():
        manifest = read_json(FINAL_EXAM_NOTEBOOK_MANIFEST_PATH)
        generated = manifest.get("generated_artifact_sha256")
        checks["manifest"] = (
            manifest.get("status") == "PASS"
            and manifest.get("base_checkpoint_commit") == BASE_CHECKPOINT_COMMIT
            and manifest.get("hash_normalization") == "utf-8-lf"
            and isinstance(manifest.get("source_artifact_sha256"), dict)
            and isinstance(generated, dict)
            and len(generated) == 3
            and manifest.get("locked_test_csv_files_opened") is False
            and manifest.get("test_split_used") is False
            and manifest.get("final_test_evaluation_authorized") is False
        )
        if checks["manifest"]:
            for relative_path, expected_hash in generated.items():
                artifact = PROJECT_ROOT / str(relative_path)
                if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                    checks["manifest"] = False
                    errors.append(f"Notebook artifact hash differs: {relative_path}.")
        if not checks["manifest"] and not any(
            error.startswith("Notebook artifact hash differs") for error in errors
        ):
            errors.append("Final exam notebook manifest is incomplete.")
    else:
        checks["manifest"] = False

    checks["cli_and_documentation"] = False
    cli_path = PROJECT_ROOT / "src" / "project_cli.py"
    readme_path = PROJECT_ROOT / "README.md"
    notebooks_readme = PROJECT_ROOT / "notebooks" / "README.md"
    if cli_path.is_file() and readme_path.is_file() and notebooks_readme.is_file():
        cli_text = cli_path.read_text(encoding="utf-8-sig")
        readme_text = readme_path.read_text(encoding="utf-8-sig")
        notebooks_text = notebooks_readme.read_text(encoding="utf-8-sig")
        checks["cli_and_documentation"] = (
            "build-final-exam-notebook" in cli_text
            and "verify-final-exam-notebook" in cli_text
            and "02_final_exam_project.ipynb" in readme_text
            and "02_final_exam_project.ipynb" in notebooks_text
        )
    if not checks["cli_and_documentation"]:
        errors.append("Step 010.6 CLI or documentation is incomplete.")

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
    }
    return report


def main() -> None:
    report = build_verification_report()
    print("Final exam notebook and research narrative verification")
    for check_name, passed in report["checks"].items():
        print(f"- {check_name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
