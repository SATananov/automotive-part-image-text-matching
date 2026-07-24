from __future__ import annotations

import csv
import json
from collections import Counter
from typing import Any

from src.build_full_course_coverage_architecture import (
    build_readiness,
    build_registry,
    normalized_sha256,
    read_json,
    validate_source_mapping,
)
from src.course_coverage_config import (
    ALLOWED_TRAIN_SPLIT,
    ALLOWED_VALIDATION_SPLIT,
    BASE_COMMIT,
    DEFAULTS_PATH,
    EXPECTED_EXPERIMENT_COUNT,
    EXPECTED_EXPERIMENT_COUNTS,
    EXPECTED_RESOURCE_TIERS,
    GENERATED_PATHS,
    MANIFEST_PATH,
    MAPPING_PATH,
    MATRIX_PATH,
    NOTEBOOK_PLAN_PATH,
    PROHIBITED_TEST_INPUTS,
    READINESS,
    READINESS_PATH,
    REGISTRY_CSV_PATH,
    REGISTRY_JSON_PATH,
    REQUIRED_PATHS,
    RESOURCE_TIERS_PATH,
    SOURCE_PATHS,
    STEP,
    project_relative,
)
from src.exam_submission_readiness_config import (
    STATUS_PATH as EXAM_READINESS_STATUS_PATH,
)


def read_registry_csv() -> list[dict[str, str]]:
    with REGISTRY_CSV_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        return list(csv.DictReader(handle))


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    checks["structure"] = all(path.is_file() for path in REQUIRED_PATHS)
    if not checks["structure"]:
        missing = [
            project_relative(path)
            for path in REQUIRED_PATHS
            if not path.is_file()
        ]
        errors.append(f"Required Step 011.0 files are missing: {missing}.")
        return {"status": "FAIL", "checks": checks, "errors": errors}

    mapping = read_json(MAPPING_PATH)
    defaults = read_json(DEFAULTS_PATH)
    tiers = read_json(RESOURCE_TIERS_PATH)
    registry = read_json(REGISTRY_JSON_PATH)
    readiness = read_json(READINESS_PATH)
    manifest = read_json(MANIFEST_PATH)
    exam_readiness = read_json(EXAM_READINESS_STATUS_PATH)

    source_errors = validate_source_mapping(mapping)
    expected_registry = build_registry(mapping)
    expected_readiness = build_readiness(
        mapping, expected_registry, defaults, tiers
    )

    checks["source_mapping"] = not source_errors
    errors.extend(source_errors)

    checks["registry_json"] = registry == expected_registry
    checks["exercise_coverage"] = (
        registry.get("experiment_count") == EXPECTED_EXPERIMENT_COUNT
        and registry.get("course_section_counts")
        == EXPECTED_EXPERIMENT_COUNTS
    )

    experiments = registry.get("experiments", [])
    identifiers = [item.get("experiment_id") for item in experiments]
    checks["unique_ids"] = (
        len(identifiers) == EXPECTED_EXPERIMENT_COUNT
        and len(set(identifiers)) == EXPECTED_EXPERIMENT_COUNT
    )
    checks["problem_numbers"] = all(
        sorted(
            item["exercise_problem_number"]
            for item in experiments
            if item["course_section"] == section
        )
        == list(range(1, count + 1))
        for section, count in EXPECTED_EXPERIMENT_COUNTS.items()
    )

    csv_rows = read_registry_csv()
    checks["registry_csv"] = (
        len(csv_rows) == EXPECTED_EXPERIMENT_COUNT
        and [row["experiment_id"] for row in csv_rows] == identifiers
        and all(row["test_split_allowed"] == "false" for row in csv_rows)
        and all(row["test_split_path"] == "" for row in csv_rows)
        and all(
            row["final_test_evaluation_authorized"] == "false"
            for row in csv_rows
        )
    )

    checks["metrics_and_outputs"] = all(
        item.get("primary_metric")
        and item.get("secondary_metrics")
        and item.get("expected_outputs")
        and item.get("evidence_paths")
        for item in experiments
    )
    checks["resource_tiers"] = (
        set(tiers.get("tiers", {})) == EXPECTED_RESOURCE_TIERS
        and all(
            item.get("resource_tier") in tiers["tiers"]
            for item in experiments
        )
    )
    checks["planned_without_claimed_results"] = (
        Counter(item.get("execution_status") for item in experiments)
        == {"PLANNED": EXPECTED_EXPERIMENT_COUNT}
        and all(item.get("result_summary") is None for item in experiments)
    )
    checks["locked_evaluation"] = (
        all(
            item.get("train_split") == ALLOWED_TRAIN_SPLIT
            and item.get("validation_split")
            == ALLOWED_VALIDATION_SPLIT
            and item.get("test_split_allowed") is False
            and item.get("test_split_path") is None
            and item.get("final_test_evaluation_authorized") is False
            for item in experiments
        )
        and set(defaults.get("prohibited_inputs", []))
        == PROHIBITED_TEST_INPUTS
        and registry.get("test_split_used") is False
        and registry.get("final_test_evaluation_authorized") is False
    )
    checks["human_annotation_safeguard"] = any(
        item.get("experiment_id") == "VIS-007"
        and any(
            "two real independent human annotators" in prerequisite
            for prerequisite in item.get("prerequisites", [])
        )
        and any(
            "Do not simulate annotators" in note
            for note in item.get("safety_notes", [])
        )
        for item in experiments
    )

    matrix_text = MATRIX_PATH.read_text(encoding="utf-8-sig")
    notebook_plan = NOTEBOOK_PLAN_PATH.read_text(encoding="utf-8-sig")
    checks["documentation"] = (
        all(f"`{experiment_id}`" in matrix_text for experiment_id in identifiers)
        and "01_fundamentals_experiments.ipynb" in notebook_plan
        and "02_sequence_model_comparison.ipynb" in notebook_plan
        and "03_vision_model_comparison.ipynb" in notebook_plan
        and "04_scoring_ranking_explainability.ipynb" in notebook_plan
        and "05_course_coverage_synthesis.ipynb" in notebook_plan
    )

    checks["readiness"] = (
        readiness == expected_readiness
        and readiness.get("status") == "PASS"
        and readiness.get("readiness") == READINESS
        and readiness.get("step") == STEP
        and readiness.get("base_commit") == BASE_COMMIT
        and readiness.get("model_training_performed") is False
        and readiness.get("model_selection_changed") is False
        and readiness.get("locked_test_csv_files_opened") is False
        and readiness.get("test_split_used") is False
        and readiness.get("final_test_evaluation_authorized") is False
    )
    checks["previous_checkpoint_preserved"] = (
        exam_readiness.get("status") == "PASS"
        and exam_readiness.get("test_split_used") is False
        and exam_readiness.get("final_test_evaluation_authorized") is False
    )

    expected_source_paths = {
        project_relative(path) for path in SOURCE_PATHS
    }
    expected_generated_paths = {
        project_relative(path) for path in GENERATED_PATHS
    }
    source_hashes = manifest.get("source_artifact_sha256", {})
    generated_hashes = manifest.get("generated_artifact_sha256", {})
    checks["manifest"] = (
        manifest.get("status") == "PASS"
        and manifest.get("step") == STEP
        and manifest.get("base_commit") == BASE_COMMIT
        and manifest.get("hash_normalization") == "utf-8-lf"
        and set(source_hashes) == expected_source_paths
        and set(generated_hashes) == expected_generated_paths
        and manifest.get("locked_test_csv_files_opened") is False
        and manifest.get("test_split_used") is False
        and manifest.get("final_test_evaluation_authorized") is False
    )
    if checks["manifest"]:
        for relative_path, expected_hash in {
            **source_hashes,
            **generated_hashes,
        }.items():
            path = MAPPING_PATH.parents[2] / relative_path
            if (
                not path.is_file()
                or normalized_sha256(path) != expected_hash
            ):
                checks["manifest"] = False
                errors.append(
                    f"Step 011.0 artifact hash differs: {relative_path}."
                )

    for name, passed in checks.items():
        if not passed and not any(name in error for error in errors):
            errors.append(f"Full course coverage check failed: {name}.")

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()
    print("Full course coverage architecture verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
