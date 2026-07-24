from __future__ import annotations

import csv
import json

import nbformat
import pytest

from src.run_vision_experimental_suite import load_split
from src.vision_suite_config import (
    AUGMENTATION_POLICIES,
    AUGMENTATION_RUNS_PATH,
    COMPATIBILITY_RUNS_PATH,
    COMPLETED_VISION_IDS,
    DEFERRED_VISION_IDS,
    EXPERIMENT_CONFIG_PATHS,
    EXPLAINABILITY_SUMMARY_PATH,
    FINE_TUNING_GATE_PATH,
    HUMAN_ANNOTATION_GATE_PATH,
    IMAGE_PROFILE_PATH,
    MANIFEST_PATH,
    PRETRAINED_BACKBONE_GATE_PATH,
    RANDOM_SEEDS,
    RANKING_METRICS_PATH,
    READINESS,
    REPRESENTATION_RUNS_PATH,
    REPRESENTATIONS,
    RESOLUTIONS,
    SCORING_NOTEBOOK_PATH,
    STATUS_PATH,
    SUITE_CONFIG_PATH,
    VISION_IDS,
    VISION_NOTEBOOK_PATH,
)
from src.verification.vision_experimental_suite import build_verification_report


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_vision_suite_verifier_passes():
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
    assert all(report["checks"].values())


def test_all_nine_vision_problem_configs_are_present_and_locked():
    configs = [read_json(path) for path in EXPERIMENT_CONFIG_PATHS]
    assert [item["experiment_id"] for item in configs] == list(VISION_IDS)
    assert all(item["test_split_allowed"] is False for item in configs)
    assert all(item["test_split_path"] is None for item in configs)
    assert all(item["final_test_evaluation_authorized"] is False for item in configs)


def test_suite_declares_six_completed_and_three_controlled_gates():
    config = read_json(SUITE_CONFIG_PATH)
    assert config["readiness"] == READINESS
    assert config["completed_problem_numbers"] == [1, 3, 4, 5, 8, 9]
    assert config["deferred_problem_numbers"] == [2, 6, 7]
    assert len(COMPLETED_VISION_IDS) == 6
    assert len(DEFERRED_VISION_IDS) == 3


def test_status_records_forty_eight_train_validation_runs_only():
    status = read_json(STATUS_PATH)
    assert status["status"] == "PASS"
    assert status["training_runs_recorded"] == 48
    assert status["model_training_performed"] is True
    assert status["test_split_used"] is False
    assert status["final_test_evaluation_authorized"] is False
    assert status["production_final_model_changed"] is False
    assert status["pretrained_weights_downloaded"] is False


def test_image_profile_covers_integrated_train_and_validation_images():
    profile = read_json(IMAGE_PROFILE_PATH)
    assert profile["unique_image_count"] == 80
    assert profile["train_unique_image_count"] == 60
    assert profile["validation_unique_image_count"] == 20
    assert profile["category_count"] == 10


def test_representation_run_matrix_is_complete():
    rows = read_csv(REPRESENTATION_RUNS_PATH)
    assert len(rows) == len(REPRESENTATIONS) * len(RESOLUTIONS) * len(RANDOM_SEEDS)
    assert {row["representation"] for row in rows} == set(REPRESENTATIONS)
    assert {int(row["resolution"]) for row in rows} == set(RESOLUTIONS)
    assert {int(row["seed"]) for row in rows} == set(RANDOM_SEEDS)


def test_augmentation_run_matrix_is_complete():
    rows = read_csv(AUGMENTATION_RUNS_PATH)
    assert len(rows) == len(AUGMENTATION_POLICIES) * len(RANDOM_SEEDS)
    assert {row["augmentation_policy"] for row in rows} == set(AUGMENTATION_POLICIES)
    assert all(row["status"] == "COMPLETED" for row in rows)


def test_compatibility_strategies_and_ranking_metrics_are_recorded():
    rows = read_csv(COMPATIBILITY_RUNS_PATH)
    ranking = read_json(RANKING_METRICS_PATH)
    assert len(rows) == 6
    assert {row["strategy"] for row in rows} == {
        "ordinal_ridge",
        "class_probability_expected_score",
    }
    assert 0 <= ranking["pairwise_ranking_accuracy"] <= 1
    assert ranking["equal_pair_threshold_fit_split"] == "train_only"
    assert ranking["scalar_score_guarantees_transitive_ordering"] is True


def test_explainability_does_not_claim_human_review():
    payload = read_json(EXPLAINABILITY_SUMMARY_PATH)
    assert payload["selected_example_count"] == 8
    assert payload["occlusion_evaluation_count"] == 72
    assert payload["manual_plausible_region_review_rate"] is None
    assert payload["manual_review_claimed"] is False


def test_pretrained_fine_tuning_and_human_gates_remain_closed():
    pretrained = read_json(PRETRAINED_BACKBONE_GATE_PATH)
    fine_tuning = read_json(FINE_TUNING_GATE_PATH)
    human = read_json(HUMAN_ANNOTATION_GATE_PATH)
    assert pretrained["network_download_attempted"] is False
    assert pretrained["pretrained_weights_downloaded"] is False
    assert fine_tuning["fine_tuning_performed"] is False
    assert human["independent_annotator_count"] == 0
    assert human["synthetic_human_agreement_reported"] is False


def test_locked_test_path_is_rejected_by_loader(tmp_path):
    with pytest.raises(RuntimeError, match="Unauthorized Step 011.3A data input"):
        load_split(tmp_path / "integrated_test.csv")


def test_notebooks_and_manifest_are_complete():
    for path in (VISION_NOTEBOOK_PATH, SCORING_NOTEBOOK_PATH):
        notebook = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert code_cells
        assert all(cell.execution_count is not None for cell in code_cells)
        assert not any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )
    manifest = read_json(MANIFEST_PATH)
    assert manifest["status"] == "PASS"
    assert manifest["training_runs_recorded"] == 48
    assert manifest["completed_problem_count"] == 6
    assert manifest["deferred_problem_count"] == 3
