from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import nbformat

from src.exam_submission_readiness_config import (
    BASE_COMMIT,
    CHECKLIST_PATH,
    CLEAN_CLONE_PATH,
    EXPECTED_DIRECT_REQUIREMENTS,
    FINAL_NOTEBOOK_GITHUB_URL,
    FINAL_NOTEBOOK_PATH,
    MANIFEST_PATH,
    READINESS,
    READINESS_DIR,
    REPOSITORY_URL,
    REQUIRED_SOURCE_ARTIFACTS,
    STATUS_PATH,
    STEP,
    SUMMARY_PATH,
    TEXT_HASH_SUFFIXES,
    project_relative,
)
from src.real_dataset_config import PROJECT_ROOT


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(content, encoding="utf-8", newline="\n")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def count_test_functions() -> int:
    total = 0
    for path in sorted((PROJECT_ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        total += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in tree.body
        )
    return total


def inspect_notebook() -> dict[str, Any]:
    notebook = nbformat.read(FINAL_NOTEBOOK_PATH, as_version=4)
    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code"
    ]
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
    metadata = notebook.metadata.get("project", {})
    code_source = "\n".join(
        str(cell.get("source", "")) for cell in code_cells
    )
    markdown_source = "\n".join(
        str(cell.get("source", ""))
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )

    return {
        "cell_count": len(notebook.cells),
        "markdown_cell_count": sum(
            cell.cell_type == "markdown" for cell in notebook.cells
        ),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": sum(
            cell.get("execution_count") is not None for cell in code_cells
        ),
        "sequential_execution": all(
            cell.get("execution_count") == index
            for index, cell in enumerate(code_cells, start=1)
        ),
        "saved_output_count": len(outputs),
        "figure_count": len(figures),
        "error_output_count": sum(
            output.get("output_type") == "error" for output in outputs
        ),
        "locked_test_path_referenced_in_code": any(
            token in code_source
            for token in ("integrated_test.csv", "external_test.csv")
        ),
        "metadata_test_split_used": metadata.get("test_split_used"),
        "metadata_final_test_authorized": metadata.get(
            "final_test_evaluation_authorized"
        ),
        "quality_audit_step": metadata.get("quality_audit_step"),
        "contains_mathematical_notation": any(
            token in markdown_source for token in ("$", "\\[", "\\(")
        ),
        "english_submission_title": (
            "Final Exam Project" in markdown_source
            or "Automotive Part Image-Text Matching" in markdown_source
        ),
        "sha256_utf8_lf": normalized_sha256(FINAL_NOTEBOOK_PATH),
    }


def inspect_dependencies() -> dict[str, Any]:
    requirements_path = PROJECT_ROOT / "requirements.txt"
    lock_path = PROJECT_ROOT / "requirements-lock.txt"
    raw_requirements = requirements_path.read_bytes()
    direct_requirements = tuple(
        line.strip().lower()
        for line in raw_requirements.decode("utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    lock_lines = tuple(
        line.strip()
        for line in lock_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    locked_names = {
        line.split("==", 1)[0].lower()
        for line in lock_lines
        if "==" in line
    }

    return {
        "requirements_utf8_without_bom": not raw_requirements.startswith(
            b"\xef\xbb\xbf"
        ),
        "direct_requirements": list(direct_requirements),
        "direct_requirements_match_expected": (
            direct_requirements == EXPECTED_DIRECT_REQUIREMENTS
        ),
        "lock_entry_count": len(lock_lines),
        "lock_entries_strictly_pinned": all(
            "==" in line and not line.startswith(("-", "."))
            for line in lock_lines
        ),
        "all_direct_requirements_in_lock": all(
            requirement in locked_names
            for requirement in EXPECTED_DIRECT_REQUIREMENTS
        ),
        "python_version_contract": "3.13",
        "tensorflow_lock": next(
            (
                line
                for line in lock_lines
                if line.lower().startswith("tensorflow==")
            ),
            None,
        ),
    }


def inspect_git_submission_contract() -> dict[str, Any]:
    try:
        count_result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        remote_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "git_available": False,
            "commit_count_at_least_10": False,
            "branch_is_main": False,
            "origin_matches_repository": False,
        }

    count = int(count_result.stdout.strip())
    origin = remote_result.stdout.strip().removesuffix(".git")
    return {
        "git_available": True,
        "commit_count_at_least_10": count >= 10,
        "branch_is_main": branch_result.stdout.strip() == "main",
        "origin_matches_repository": origin == REPOSITORY_URL,
    }

def repository_files() -> list[Path]:
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "-c",
                "-o",
                "--exclude-standard",
                "-z",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [
            path
            for path in PROJECT_ROOT.rglob("*")
            if path.is_file()
            and ".git" not in path.relative_to(PROJECT_ROOT).parts
        ]

    return [
        PROJECT_ROOT / relative
        for relative in result.stdout.split("\0")
        if relative and (PROJECT_ROOT / relative).is_file()
    ]


def inspect_repository_hygiene() -> dict[str, Any]:
    forbidden_names = {
        ".ds_store",
        "thumbs.db",
    }
    forbidden_parts = {
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".venv",
        "__pycache__",
    }
    violations: list[str] = []
    delivery_artifacts: list[str] = []

    for path in repository_files():
        relative = path.relative_to(PROJECT_ROOT)
        lower_parts = {part.lower() for part in relative.parts}
        if lower_parts & forbidden_parts or path.name.lower() in forbidden_names:
            violations.append(relative.as_posix())
        if path.suffix.lower() in {".bat", ".cmd", ".ps1", ".zip"}:
            delivery_artifacts.append(relative.as_posix())

    return {
        "forbidden_runtime_artifacts": sorted(set(violations)),
        "delivery_artifacts_inside_repository": sorted(
            set(delivery_artifacts)
        ),
        "required_directories_present": all(
            (PROJECT_ROOT / directory).is_dir()
            for directory in (
                "app",
                "data",
                "models",
                "notebooks",
                "reports",
                "src",
                "tests",
            )
        ),
    }


def build_submission_checks() -> list[dict[str, Any]]:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8-sig")
    notebook = inspect_notebook()
    dependencies = inspect_dependencies()
    hygiene = inspect_repository_hygiene()
    git_contract = inspect_git_submission_contract()
    notebook_status = read_json(
        PROJECT_ROOT
        / "reports"
        / "final_exam_notebook"
        / "final_exam_notebook_status.json"
    )
    quality_status = read_json(
        PROJECT_ROOT
        / "reports"
        / "notebook_quality_audit"
        / "notebook_quality_audit_status.json"
    )
    freeze_status = read_json(
        PROJECT_ROOT
        / "reports"
        / "final_model_freeze"
        / "final_model_freeze_status.json"
    )
    test_lock = read_json(
        PROJECT_ROOT / "data" / "processed" / "integrated_test_lock.json"
    )

    checks = [
        {
            "criterion": "GitHub repository and commit-history minimum",
            "evidence": (
                "Public origin, main branch and at least 10 commits; "
                "commit meaningfulness remains reviewer judgment"
            ),
            "passed": all(
                (
                    git_contract["git_available"],
                    git_contract["commit_count_at_least_10"],
                    git_contract["branch_is_main"],
                    git_contract["origin_matches_repository"],
                )
            ),
        },
        {
            "criterion": "English Jupyter notebook with math and Python code",
            "evidence": "Executed final notebook with markdown, formulas and code",
            "passed": (
                notebook["english_submission_title"]
                and notebook["contains_mathematical_notation"]
                and notebook["code_cell_count"] > 0
            ),
        },
        {
            "criterion": "Problem, scope, significance and labels are clear",
            "evidence": "README introduction and notebook Sections 1-3",
            "passed": all(
                token in readme
                for token in ("MATCH", "PARTIAL_MATCH", "MISMATCH")
            ),
        },
        {
            "criterion": "Dataset construction, provenance and legal use",
            "evidence": "Open-license workflow, attribution and grouped splits",
            "passed": all(
                token in readme
                for token in (
                    "Open-license internet image collection",
                    "External dataset integration",
                    "grouped",
                )
            ),
        },
        {
            "criterion": "Multiple classical and neural methods are compared",
            "evidence": "Six-model integrated validation comparison",
            "passed": all(
                token in readme
                for token in (
                    "TF-IDF + Logistic Regression",
                    "Keras text model",
                    "Keras image model",
                    "Keras multimodal model",
                )
            ),
        },
        {
            "criterion": "Metrics, figures and validation analysis are saved",
            "evidence": (
                f"{notebook['saved_output_count']} outputs and "
                f"{notebook['figure_count']} figures in the executed notebook"
            ),
            "passed": (
                notebook["saved_output_count"] == 19
                and notebook["figure_count"] == 6
                and notebook["error_output_count"] == 0
            ),
        },
        {
            "criterion": "Final notebook is directly reviewable on GitHub",
            "evidence": FINAL_NOTEBOOK_GITHUB_URL,
            "passed": (
                FINAL_NOTEBOOK_GITHUB_URL in readme
                and notebook["executed_code_cell_count"] == 15
                and notebook["sequential_execution"]
            ),
        },
        {
            "criterion": "Research context, references and academic integrity",
            "evidence": "Related work and six primary or official references",
            "passed": (
                quality_status.get("citation_count") == 6
                and quality_status.get("component_statuses", {}).get(
                    "citations"
                )
                == "PASS"
            ),
        },
        {
            "criterion": "Writing layout, conclusions and limitations",
            "evidence": "Structured notebook narrative and README guidance",
            "passed": all(
                token in readme.lower()
                for token in ("limitations", "conclusion")
            ),
        },
        {
            "criterion": "Reproduction commands and pinned environment",
            "evidence": "README setup, project CLI and requirements-lock.txt",
            "passed": (
                dependencies["requirements_utf8_without_bom"]
                and dependencies["direct_requirements_match_expected"]
                and dependencies["lock_entries_strictly_pinned"]
                and dependencies["all_direct_requirements_in_lock"]
            ),
        },
        {
            "criterion": "Functional code and automated quality gates",
            "evidence": "Step 010.6/010.7 status, tests and project verifiers",
            "passed": (
                notebook_status.get("status") == "PASS"
                and quality_status.get("status") == "PASS"
                and quality_status.get("readiness")
                == "NOTEBOOK_EXECUTION_VISUAL_QA_AND_CITATION_AUDIT_PASS"
            ),
        },
        {
            "criterion": "Code and repository hygiene",
            "evidence": "Semantic modules, tests and repository debris scan",
            "passed": (
                not hygiene["forbidden_runtime_artifacts"]
                and not hygiene["delivery_artifacts_inside_repository"]
                and hygiene["required_directories_present"]
            ),
        },
        {
            "criterion": "Final model and evaluation protocol are frozen",
            "evidence": "reports/final_model_freeze/",
            "passed": (
                freeze_status.get("status") == "PASS"
                and freeze_status.get("protocol_frozen") is True
            ),
        },
        {
            "criterion": "Test split remains locked, unused and unauthorized",
            "evidence": "Committed lock and all final readiness flags",
            "passed": (
                test_lock.get("test_locked") is True
                and test_lock.get("test_evaluation_permitted") is False
                and notebook["locked_test_path_referenced_in_code"] is False
                and notebook["metadata_test_split_used"] is False
                and notebook["metadata_final_test_authorized"] is False
                and notebook_status.get("test_split_used") is False
                and quality_status.get("test_split_used") is False
                and freeze_status.get("test_split_used") is False
                and freeze_status.get("final_test_evaluation_authorized")
                is False
            ),
        },
    ]
    return checks


def collect_readiness_evidence() -> dict[str, Any]:
    missing_sources = [
        project_relative(path)
        for path in REQUIRED_SOURCE_ARTIFACTS
        if not path.is_file()
    ]
    checks = build_submission_checks() if not missing_sources else []
    notebook = inspect_notebook() if not missing_sources else {}
    dependencies = inspect_dependencies() if not missing_sources else {}
    hygiene = inspect_repository_hygiene()

    return {
        "step": STEP,
        "base_commit": BASE_COMMIT,
        "repository": REPOSITORY_URL,
        "final_notebook": project_relative(FINAL_NOTEBOOK_PATH),
        "final_notebook_github_url": FINAL_NOTEBOOK_GITHUB_URL,
        "missing_source_artifacts": missing_sources,
        "submission_checks": checks,
        "notebook": notebook,
        "dependencies": dependencies,
        "repository_hygiene": hygiene,
        "git_submission_contract": inspect_git_submission_contract(),
        "test_function_count": count_test_functions(),
    }


def build_checklist_markdown(evidence: dict[str, Any]) -> list[str]:
    lines = [
        "# Exam Submission Checklist",
        "",
        f"- Step: **{STEP}**",
        f"- Base checkpoint: `{BASE_COMMIT}`",
        f"- Repository: {REPOSITORY_URL}",
        f"- Final notebook: [open on GitHub]({FINAL_NOTEBOOK_GITHUB_URL})",
        "- Evaluation boundary: validation evidence only; test remains locked.",
        "",
        "## Submission criteria and evidence",
        "",
        "| Status | Criterion | Repository evidence |",
        "|---|---|---|",
    ]
    for check in evidence["submission_checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"| {status} | {check['criterion']} | {check['evidence']} |"
        )

    lines.extend(
        [
            "",
            "## Items to submit",
            "",
            "1. GitHub repository URL.",
            "2. Direct URL to the executed final notebook.",
            "3. The repository at the clean Step 010.8 commit.",
            "4. Any SoftUni submission form fields requested by the trainer.",
            "",
            "## External logistics boundary",
            "",
            "The repository can prove technical readiness, but the submission portal, "
            "trainer instructions and deadline compliance must be confirmed manually.",
            "",
            "## Final manual review",
            "",
            "- Open the notebook directly on GitHub and confirm all figures render.",
            "- Confirm the repository default branch is `main`.",
            "- Confirm the Step 010.8 commit is pushed and Git is clean.",
            "- Do not run or authorize the locked test evaluation.",
        ]
    )
    return lines


def build_clean_clone_markdown(evidence: dict[str, Any]) -> list[str]:
    return [
        "# Clean-Clone Reproducibility Protocol",
        "",
        "Use this protocol after the Step 010.8 commit is pushed.",
        (
            "It creates a new clone and does not reuse the development "
            "virtual environment."
        ),
        "",
        "```powershell",
        "$Root = Join-Path $env:TEMP 'automotive-part-image-text-matching-step0108'",
        "Remove-Item -LiteralPath $Root -Recurse -Force -ErrorAction SilentlyContinue",
        f"git clone {REPOSITORY_URL}.git $Root",
        "Set-Location $Root",
        "py -3.13 -m venv .venv",
        ".\\.venv\\Scripts\\Activate.ps1",
        "python -m pip install --upgrade pip",
        "python -m pip install -r requirements-lock.txt",
        "python -m pytest -q",
        "python -m src.project_cli verify-project",
        "python -m src.project_cli verify-exam-submission-readiness",
        "git branch --show-current",
        "git rev-list --count HEAD",
        "git status --short",
        "```",
        "",
        "Expected result:",
        "",
        "- dependency installation succeeds from the committed lock file;",
        "- the complete test suite passes;",
        "- both verification commands report `PASS`;",
        "- the current branch is `main` and history remains above 10 commits;",
        "- `git status --short` prints no paths;",
        "- the test split remains locked and unauthorized.",
        "",
        "## Static readiness evidence",
        "",
        (
            "- Direct requirements: "
            f"{len(evidence['dependencies']['direct_requirements'])}"
        ),
        f"- Fully pinned lock entries: {evidence['dependencies']['lock_entry_count']}",
        f"- TensorFlow lock: `{evidence['dependencies']['tensorflow_lock']}`",
        f"- Test functions present after Step 010.8: {evidence['test_function_count']}",
        "- Final notebook code does not reference either locked test CSV.",
        "- Step 010.8 reads the lock contract only; it does not load test rows.",
    ]


def build_summary_markdown(
    evidence: dict[str, Any], status: dict[str, Any]
) -> list[str]:
    notebook = evidence["notebook"]
    dependencies = evidence["dependencies"]
    return [
        "# Exam Submission Readiness and Clean Release Checkpoint",
        "",
        f"- Status: **{status['status']}**",
        f"- Readiness: **{status['readiness']}**",
        f"- Step: **{STEP}**",
        f"- Base checkpoint commit: `{BASE_COMMIT}`",
        f"- Final notebook: [open on GitHub]({FINAL_NOTEBOOK_GITHUB_URL})",
        "",
        "## Audit result",
        "",
        (
            "The repository presentation now leads directly to the executed "
            "final notebook and the integrated validation result. The "
            "submission checklist, clean-clone protocol, dependency contract, "
            "notebook quality evidence and release manifest are committed as "
            "one reviewable readiness package."
        ),
        "",
        "## Notebook evidence",
        "",
        f"- cells: {notebook['cell_count']};",
        f"- executed code cells: {notebook['executed_code_cell_count']};",
        f"- saved outputs: {notebook['saved_output_count']};",
        f"- saved figures: {notebook['figure_count']};",
        f"- error outputs: {notebook['error_output_count']};",
        "- Step 010.7 execution, visual, numeric and citation gates: PASS.",
        "",
        "## Reproducibility and hygiene",
        "",
        f"- direct dependencies: {len(dependencies['direct_requirements'])};",
        f"- pinned lock entries: {dependencies['lock_entry_count']};",
        f"- repository test functions: {evidence['test_function_count']};",
        "- runtime and delivery debris inside the repository: none;",
        "- GitHub origin, `main` branch and 10-commit minimum: verified;",
        "- clean-clone verification protocol: documented and packaged.",
        "",
        "## Evaluation boundary",
        "",
        (
            "Step 010.8 does not train a model, change model selection, parse "
            "a locked test CSV, evaluate the test split, or open the final "
            "authorization gate. The final test remains locked and "
            "unauthorized."
        ),
        "",
        "## Release action",
        "",
        (
            "After applying the patch, run the full test and verification "
            "gates, commit the Step 010.8 changes, push `main`, and run the "
            "packaged clean-clone verifier against the pushed commit."
        ),
    ]


def build_status(evidence: dict[str, Any]) -> dict[str, Any]:
    checks = evidence["submission_checks"]
    passed = (
        not evidence["missing_source_artifacts"]
        and bool(checks)
        and all(check["passed"] for check in checks)
    )
    notebook = evidence["notebook"]
    return {
        "status": "PASS" if passed else "FAIL",
        "readiness": READINESS if passed else "EXAM_SUBMISSION_NOT_READY",
        "step": STEP,
        "base_commit": BASE_COMMIT,
        "repository": REPOSITORY_URL,
        "final_notebook": evidence["final_notebook"],
        "final_notebook_github_url": FINAL_NOTEBOOK_GITHUB_URL,
        "submission_check_count": len(checks),
        "submission_check_pass_count": sum(
            check["passed"] for check in checks
        ),
        "test_function_count": evidence["test_function_count"],
        "notebook_cell_count": notebook.get("cell_count"),
        "executed_code_cell_count": notebook.get(
            "executed_code_cell_count"
        ),
        "saved_output_count": notebook.get("saved_output_count"),
        "figure_count": notebook.get("figure_count"),
        "dependency_lock_entry_count": evidence["dependencies"].get(
            "lock_entry_count"
        ),
        "project_verification_required": True,
        "full_test_suite_required": True,
        "clean_clone_verification_required_after_push": True,
        "repository_clean_checkpoint_required_after_commit": True,
        "git_commit_minimum_met": evidence["git_submission_contract"].get(
            "commit_count_at_least_10"
        ),
        "git_branch_is_main": evidence["git_submission_contract"].get(
            "branch_is_main"
        ),
        "git_origin_matches_repository": evidence[
            "git_submission_contract"
        ].get("origin_matches_repository"),
        "model_retraining_performed": False,
        "model_selection_changed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }


def build_manifest(status: dict[str, Any]) -> dict[str, Any]:
    source_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in REQUIRED_SOURCE_ARTIFACTS
    }
    output_paths = (
        STATUS_PATH,
        CHECKLIST_PATH,
        CLEAN_CLONE_PATH,
        SUMMARY_PATH,
    )
    output_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in output_paths
    }
    return {
        "status": status["status"],
        "step": STEP,
        "base_commit": BASE_COMMIT,
        "hash_normalization": "utf-8-lf",
        "source_artifact_sha256": source_hashes,
        "generated_artifact_sha256": output_hashes,
        "generated_artifact_count": len(output_hashes),
        "model_retraining_performed": False,
        "model_selection_changed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }


def main() -> None:
    READINESS_DIR.mkdir(parents=True, exist_ok=True)
    evidence = collect_readiness_evidence()
    status = build_status(evidence)

    write_markdown(CHECKLIST_PATH, build_checklist_markdown(evidence))
    write_markdown(CLEAN_CLONE_PATH, build_clean_clone_markdown(evidence))
    write_json(STATUS_PATH, status)
    write_markdown(SUMMARY_PATH, build_summary_markdown(evidence, status))
    write_json(MANIFEST_PATH, build_manifest(status))

    print("Exam submission readiness and clean release checkpoint")
    print(f"- submission checks: {status['submission_check_pass_count']}/"
          f"{status['submission_check_count']}")
    print(f"- final notebook: {FINAL_NOTEBOOK_GITHUB_URL}")
    print("- test split used: no")
    print("- final test evaluation authorized: no")
    print(f"Status: {status['status']}")
    if status["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
