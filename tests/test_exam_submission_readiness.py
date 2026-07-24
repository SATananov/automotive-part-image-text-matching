from __future__ import annotations

from src.build_exam_submission_readiness import (
    collect_readiness_evidence,
    normalized_sha256,
    read_json,
)
from src.exam_submission_readiness_config import (
    BASE_COMMIT,
    FINAL_NOTEBOOK_GITHUB_URL,
    MANIFEST_PATH,
    READINESS,
    REQUIRED_OUTPUTS,
    STATUS_PATH,
)
from src.project_cli import COMMANDS
from src.verification.exam_submission_readiness import (
    build_verification_report,
)


def test_exam_submission_commands_are_registered() -> None:
    assert COMMANDS["build-exam-submission-readiness"].module == (
        "src.build_exam_submission_readiness"
    )
    assert COMMANDS["verify-exam-submission-readiness"].module == (
        "src.verification.exam_submission_readiness"
    )


def test_readiness_outputs_exist() -> None:
    assert all(path.is_file() for path in REQUIRED_OUTPUTS)


def test_readiness_status_is_pass_and_test_locked() -> None:
    status = read_json(STATUS_PATH)

    assert status["status"] == "PASS"
    assert status["readiness"] == READINESS
    assert status["base_commit"] == BASE_COMMIT
    assert status["submission_check_count"] == 14
    assert status["submission_check_pass_count"] == 14
    assert status["git_commit_minimum_met"] is True
    assert status["git_branch_is_main"] is True
    assert status["git_origin_matches_repository"] is True
    assert status["locked_test_csv_files_opened"] is False
    assert status["test_split_used"] is False
    assert status["final_test_evaluation_authorized"] is False


def test_final_notebook_is_executed_and_directly_linked() -> None:
    evidence = collect_readiness_evidence()
    notebook = evidence["notebook"]

    assert evidence["final_notebook_github_url"] == (
        FINAL_NOTEBOOK_GITHUB_URL
    )
    assert notebook["cell_count"] == 31
    assert notebook["executed_code_cell_count"] == 15
    assert notebook["sequential_execution"] is True
    assert notebook["saved_output_count"] == 19
    assert notebook["figure_count"] == 6
    assert notebook["error_output_count"] == 0
    assert notebook["locked_test_path_referenced_in_code"] is False


def test_dependency_and_repository_hygiene_pass() -> None:
    evidence = collect_readiness_evidence()
    dependencies = evidence["dependencies"]
    hygiene = evidence["repository_hygiene"]

    assert dependencies["requirements_utf8_without_bom"] is True
    assert dependencies["direct_requirements_match_expected"] is True
    assert dependencies["lock_entries_strictly_pinned"] is True
    assert dependencies["all_direct_requirements_in_lock"] is True
    assert hygiene["forbidden_runtime_artifacts"] == []
    assert hygiene["delivery_artifacts_inside_repository"] == []
    assert hygiene["required_directories_present"] is True


def test_readiness_manifest_hashes_match() -> None:
    manifest = read_json(MANIFEST_PATH)

    assert manifest["status"] == "PASS"
    assert manifest["base_commit"] == BASE_COMMIT
    for relative_path, expected_hash in {
        **manifest["source_artifact_sha256"],
        **manifest["generated_artifact_sha256"],
    }.items():
        from src.real_dataset_config import PROJECT_ROOT

        assert normalized_sha256(PROJECT_ROOT / relative_path) == (
            expected_hash
        )


def test_all_submission_checks_pass() -> None:
    evidence = collect_readiness_evidence()

    assert len(evidence["submission_checks"]) == 14
    assert all(check["passed"] for check in evidence["submission_checks"])


def test_exam_submission_verification_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS"
    assert report["errors"] == []
