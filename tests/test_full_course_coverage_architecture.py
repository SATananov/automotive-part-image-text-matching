from __future__ import annotations

import csv
import json
from pathlib import Path

from src.course_coverage_config import (
    EXPECTED_EXPERIMENT_COUNT,
    EXPECTED_EXPERIMENT_COUNTS,
    EXPECTED_RESOURCE_TIERS,
    MAPPING_PATH,
    MATRIX_PATH,
    NOTEBOOK_PLAN_PATH,
    READINESS,
    READINESS_PATH,
    REGISTRY_CSV_PATH,
    REGISTRY_JSON_PATH,
    RESOURCE_TIERS_PATH,
)
from src.project_cli import COMMANDS
from src.verification.full_course_coverage_architecture import (
    build_verification_report,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_all_exercise_problems_have_unique_experiment_ids() -> None:
    mapping = read_json(MAPPING_PATH)
    experiments = mapping["experiments"]
    identifiers = [item["experiment_id"] for item in experiments]

    assert len(experiments) == EXPECTED_EXPERIMENT_COUNT
    assert len(set(identifiers)) == EXPECTED_EXPERIMENT_COUNT

    for section, count in EXPECTED_EXPERIMENT_COUNTS.items():
        numbers = sorted(
            item["exercise_problem_number"]
            for item in experiments
            if item["course_section"] == section
        )
        assert numbers == list(range(1, count + 1))


def test_registry_defines_metrics_tiers_and_evidence() -> None:
    registry = read_json(REGISTRY_JSON_PATH)
    tiers = read_json(RESOURCE_TIERS_PATH)["tiers"]

    assert set(tiers) == EXPECTED_RESOURCE_TIERS
    assert registry["experiment_count"] == EXPECTED_EXPERIMENT_COUNT
    assert registry["course_section_counts"] == EXPECTED_EXPERIMENT_COUNTS

    for experiment in registry["experiments"]:
        assert experiment["primary_metric"]
        assert experiment["secondary_metrics"]
        assert experiment["resource_tier"] in tiers
        assert experiment["expected_outputs"]
        assert experiment["evidence_paths"]
        assert experiment["configuration_path"].endswith(
            f"{experiment['experiment_id']}.json"
        )


def test_every_experiment_preserves_the_locked_test_boundary() -> None:
    registry = read_json(REGISTRY_JSON_PATH)

    assert registry["test_split_used"] is False
    assert registry["final_test_evaluation_authorized"] is False

    for experiment in registry["experiments"]:
        assert experiment["test_split_allowed"] is False
        assert experiment["test_split_path"] is None
        assert experiment["final_test_evaluation_authorized"] is False
        assert experiment["execution_status"] == "PLANNED"
        assert experiment["result_summary"] is None


def test_registry_csv_matches_json_order_and_lock_flags() -> None:
    registry = read_json(REGISTRY_JSON_PATH)
    with REGISTRY_CSV_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert [row["experiment_id"] for row in rows] == [
        item["experiment_id"] for item in registry["experiments"]
    ]
    assert all(row["test_split_allowed"] == "false" for row in rows)
    assert all(row["test_split_path"] == "" for row in rows)
    assert all(
        row["final_test_evaluation_authorized"] == "false"
        for row in rows
    )


def test_annotation_agreement_requires_real_independent_humans() -> None:
    registry = read_json(REGISTRY_JSON_PATH)
    experiment = next(
        item
        for item in registry["experiments"]
        if item["experiment_id"] == "VIS-007"
    )

    assert any(
        "two real independent human annotators" in prerequisite
        for prerequisite in experiment["prerequisites"]
    )
    assert any(
        "Do not simulate annotators" in note
        for note in experiment["safety_notes"]
    )


def test_documentation_contains_all_ids_and_notebook_plan() -> None:
    registry = read_json(REGISTRY_JSON_PATH)
    matrix = MATRIX_PATH.read_text(encoding="utf-8-sig")
    notebook_plan = NOTEBOOK_PLAN_PATH.read_text(encoding="utf-8-sig")

    for experiment in registry["experiments"]:
        assert f"`{experiment['experiment_id']}`" in matrix

    for notebook_name in (
        "01_fundamentals_experiments.ipynb",
        "02_sequence_model_comparison.ipynb",
        "03_vision_model_comparison.ipynb",
        "04_scoring_ranking_explainability.ipynb",
        "05_course_coverage_synthesis.ipynb",
    ):
        assert notebook_name in notebook_plan


def test_step_011_0_readiness_is_planning_only() -> None:
    readiness = read_json(READINESS_PATH)

    assert readiness["status"] == "PASS"
    assert readiness["readiness"] == READINESS
    assert readiness["model_training_performed"] is False
    assert readiness["model_selection_changed"] is False
    assert readiness["locked_test_csv_files_opened"] is False
    assert readiness["test_split_used"] is False
    assert readiness["final_test_evaluation_authorized"] is False


def test_course_coverage_cli_commands_are_registered() -> None:
    assert (
        COMMANDS["build-course-coverage-architecture"].module
        == "src.build_full_course_coverage_architecture"
    )
    assert (
        COMMANDS["verify-course-coverage-architecture"].module
        == "src.verification.full_course_coverage_architecture"
    )
    assert not COMMANDS[
        "build-course-coverage-architecture"
    ].requires_tensorflow
    assert not COMMANDS[
        "verify-course-coverage-architecture"
    ].requires_tensorflow


def test_step_011_0_verification_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS"
    assert report["errors"] == []
