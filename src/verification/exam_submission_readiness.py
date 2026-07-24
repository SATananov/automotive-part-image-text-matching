from __future__ import annotations

from typing import Any

from src.build_exam_submission_readiness import (
    build_status,
    collect_readiness_evidence,
    normalized_sha256,
    read_json,
)
from src.exam_submission_readiness_config import (
    BASE_COMMIT,
    CHECKLIST_PATH,
    CLEAN_CLONE_PATH,
    FINAL_NOTEBOOK_GITHUB_URL,
    MANIFEST_PATH,
    READINESS,
    REQUIRED_OUTPUTS,
    STATUS_PATH,
    STEP,
    SUMMARY_PATH,
)
from src.real_dataset_config import PROJECT_ROOT


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    checks["structure"] = all(path.is_file() for path in REQUIRED_OUTPUTS)
    if not checks["structure"]:
        errors.append("Required exam-submission readiness files are missing.")
        return {"status": "FAIL", "checks": checks, "errors": errors}

    evidence = collect_readiness_evidence()
    expected_status = build_status(evidence)
    status = read_json(STATUS_PATH)
    manifest = read_json(MANIFEST_PATH)

    checks["submission_criteria"] = (
        expected_status["status"] == "PASS"
        and expected_status["submission_check_count"] == 14
        and expected_status["submission_check_pass_count"] == 14
    )
    checks["status"] = (
        status == expected_status
        and status.get("readiness") == READINESS
        and status.get("step") == STEP
        and status.get("base_commit") == BASE_COMMIT
    )
    checks["notebook_and_reports"] = (
        status.get("executed_code_cell_count") == 15
        and status.get("saved_output_count") == 19
        and status.get("figure_count") == 6
        and evidence["notebook"].get("error_output_count") == 0
        and evidence["notebook"].get("quality_audit_step") == "010.7"
    )
    checks["dependency_hygiene"] = all(
        (
            evidence["dependencies"].get("requirements_utf8_without_bom"),
            evidence["dependencies"].get(
                "direct_requirements_match_expected"
            ),
            evidence["dependencies"].get("lock_entries_strictly_pinned"),
            evidence["dependencies"].get("all_direct_requirements_in_lock"),
        )
    )
    checks["git_submission"] = all(
        (
            status.get("git_commit_minimum_met"),
            status.get("git_branch_is_main"),
            status.get("git_origin_matches_repository"),
        )
    )
    checks["repository_hygiene"] = (
        not evidence["repository_hygiene"].get(
            "forbidden_runtime_artifacts"
        )
        and not evidence["repository_hygiene"].get(
            "delivery_artifacts_inside_repository"
        )
        and evidence["repository_hygiene"].get(
            "required_directories_present"
        )
        is True
    )
    checks["test_lock"] = (
        status.get("locked_test_csv_files_opened") is False
        and status.get("test_split_used") is False
        and status.get("final_test_evaluation_authorized") is False
        and evidence["notebook"].get(
            "locked_test_path_referenced_in_code"
        )
        is False
    )

    readme_text = (PROJECT_ROOT / "README.md").read_text(
        encoding="utf-8-sig"
    )
    notebook_readme_text = (
        PROJECT_ROOT / "notebooks" / "README.md"
    ).read_text(encoding="utf-8-sig")
    checks["github_presentation"] = (
        FINAL_NOTEBOOK_GITHUB_URL in readme_text
        and FINAL_NOTEBOOK_GITHUB_URL in notebook_readme_text
        and "Exam submission" in readme_text
        and "Integrated validation results" in readme_text
    )

    expected_outputs = {
        STATUS_PATH.relative_to(PROJECT_ROOT).as_posix(),
        CHECKLIST_PATH.relative_to(PROJECT_ROOT).as_posix(),
        CLEAN_CLONE_PATH.relative_to(PROJECT_ROOT).as_posix(),
        SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix(),
    }
    output_hashes = manifest.get("generated_artifact_sha256", {})
    checks["manifest"] = (
        manifest.get("status") == "PASS"
        and manifest.get("step") == STEP
        and manifest.get("base_commit") == BASE_COMMIT
        and manifest.get("hash_normalization") == "utf-8-lf"
        and set(output_hashes) == expected_outputs
        and manifest.get("locked_test_csv_files_opened") is False
        and manifest.get("test_split_used") is False
        and manifest.get("final_test_evaluation_authorized") is False
    )
    if checks["manifest"]:
        for relative_path, expected_hash in output_hashes.items():
            path = PROJECT_ROOT / relative_path
            if (
                not path.is_file()
                or normalized_sha256(path) != expected_hash
            ):
                checks["manifest"] = False
                errors.append(
                    f"Generated readiness artifact hash differs: "
                    f"{relative_path}."
                )

    source_hashes = manifest.get("source_artifact_sha256", {})
    checks["source_artifacts"] = bool(source_hashes)
    if checks["source_artifacts"]:
        for relative_path, expected_hash in source_hashes.items():
            path = PROJECT_ROOT / relative_path
            if (
                not path.is_file()
                or normalized_sha256(path) != expected_hash
            ):
                checks["source_artifacts"] = False
                errors.append(
                    f"Readiness source artifact hash differs: "
                    f"{relative_path}."
                )

    for name, passed in checks.items():
        if not passed and not any(name in error for error in errors):
            errors.append(f"Exam submission readiness check failed: {name}.")

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()
    print("Exam submission readiness verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
