from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.project_cli import COMMANDS
from src.run_integrated_training_validation import (
    load_integrated_datasets,
    locked_test_fingerprints,
    read_test_lock,
)
from src.run_validation_error_analysis_and_model_improvement import (
    build_comparison,
    build_data_diagnostics,
    build_selection_decision,
)
from src.validation_model_improvement_config import (
    CANDIDATE_METRIC_PATHS,
    CANDIDATE_PREDICTION_PATHS,
    CANDIDATE_TITLES,
    DATA_DIAGNOSTICS_JSON_PATH,
    DISAGREEMENT_CSV_PATH,
    ERROR_ANALYSIS_CSV_PATH,
    EXPERIMENT_COMPARISON_CSV_PATH,
    EXPERIMENT_SEEDS,
    SELECTION_DECISION_PATH,
    VALIDATION_IMPROVEMENT_STATUS_PATH,
)
from src.verification.validation_model_improvement import (
    build_verification_report,
)


def _synthetic_metrics(
    *,
    mean_macro_f1: float,
    aggregate_macro_f1: float,
    accuracy: float,
    worst_class_f1: float,
) -> dict[str, float | int]:
    return {
        "mean_seed_macro_f1": mean_macro_f1,
        "macro_f1": aggregate_macro_f1,
        "accuracy": accuracy,
        "mean_seed_accuracy": accuracy,
        "std_seed_accuracy": 0.01,
        "std_seed_macro_f1": 0.01,
        "worst_class_f1": worst_class_f1,
        "parameter_count": 100,
    }


def test_data_diagnostics_preserve_group_and_image_isolation() -> None:
    train_dataframe, validation_dataframe = load_integrated_datasets()
    diagnostics = build_data_diagnostics(
        train_dataframe,
        validation_dataframe,
    )

    assert diagnostics["status"] == "PASS"
    assert diagnostics["train_validation_group_overlap"] == 0
    assert diagnostics["train_validation_exact_image_hash_overlap"] == 0
    assert diagnostics["test_split_used"] is False
    assert diagnostics["test_evaluation_permitted"] is False


def test_selection_gate_accepts_only_material_stable_gain() -> None:
    metrics = {
        "reference_multimodal": _synthetic_metrics(
            mean_macro_f1=0.45,
            aggregate_macro_f1=0.46,
            accuracy=0.48,
            worst_class_f1=0.35,
        ),
        "relation_aware_multimodal": _synthetic_metrics(
            mean_macro_f1=0.48,
            aggregate_macro_f1=0.49,
            accuracy=0.50,
            worst_class_f1=0.38,
        ),
        "regularized_relation_multimodal": _synthetic_metrics(
            mean_macro_f1=0.455,
            aggregate_macro_f1=0.46,
            accuracy=0.49,
            worst_class_f1=0.36,
        ),
    }
    comparison = build_comparison(metrics)  # type: ignore[arg-type]
    decision = build_selection_decision(
        comparison,
        metrics,  # type: ignore[arg-type]
        incumbent_validation_accuracy=0.48,
        incumbent_validation_macro_f1=0.46,
    )

    assert decision["decision"] == "IMPROVEMENT_ACCEPTED"
    assert decision["selected_candidate_slug"] == (
        "relation_aware_multimodal"
    )
    assert decision["test_split_used"] is False
    assert decision["final_test_evaluation_authorized"] is False


def test_selection_gate_can_retain_reference() -> None:
    metrics = {
        slug: _synthetic_metrics(
            mean_macro_f1=0.45,
            aggregate_macro_f1=0.45,
            accuracy=0.48,
            worst_class_f1=0.35,
        )
        for slug in CANDIDATE_TITLES
    }
    comparison = build_comparison(metrics)  # type: ignore[arg-type]
    decision = build_selection_decision(
        comparison,
        metrics,  # type: ignore[arg-type]
        incumbent_validation_accuracy=0.48,
        incumbent_validation_macro_f1=0.46,
    )

    assert decision["decision"] == "REFERENCE_RETAINED"
    assert decision["selected_candidate_slug"] == "reference_multimodal"


def test_validation_improvement_cli_commands_are_registered() -> None:
    run_spec = COMMANDS[
        "run-validation-error-analysis-model-improvement"
    ]
    verify_spec = COMMANDS["verify-validation-model-improvement"]

    assert run_spec.requires_tensorflow is True
    assert verify_spec.requires_tensorflow is False


def test_locked_test_fingerprints_remain_valid() -> None:
    lock = read_test_lock()
    fingerprints = locked_test_fingerprints(lock)

    assert fingerprints["data/processed/integrated_test.csv"] == (
        lock["integrated_test_sha256"]
    )


def test_generated_candidate_metrics_are_validation_only() -> None:
    for slug, path in CANDIDATE_METRIC_PATHS.items():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        assert metrics["candidate_slug"] == slug
        assert metrics["training_sample_count"] == 180
        assert metrics["sample_count"] == 60
        assert metrics["seeds"] == list(EXPERIMENT_SEEDS)
        assert len(metrics["seed_results"]) == len(EXPERIMENT_SEEDS)
        assert metrics["test_split_used"] is False
        assert metrics["test_evaluation_permitted"] is False


def test_generated_probabilities_are_normalized() -> None:
    probability_columns = [
        "probability_match",
        "probability_partial_match",
        "probability_mismatch",
    ]
    for path in CANDIDATE_PREDICTION_PATHS.values():
        predictions = pd.read_csv(path)
        assert len(predictions) == 60
        totals = predictions[probability_columns].sum(axis=1).to_numpy()
        assert np.allclose(totals, 1.0, atol=1e-5)
        assert set(predictions["is_correct"]) <= {True, False}


def test_generated_analysis_and_decision_are_complete() -> None:
    comparison = pd.read_csv(EXPERIMENT_COMPARISON_CSV_PATH)
    errors = pd.read_csv(ERROR_ANALYSIS_CSV_PATH)
    disagreement = pd.read_csv(DISAGREEMENT_CSV_PATH)
    diagnostics = json.loads(
        DATA_DIAGNOSTICS_JSON_PATH.read_text(encoding="utf-8")
    )
    decision = json.loads(SELECTION_DECISION_PATH.read_text(encoding="utf-8"))
    status = json.loads(
        VALIDATION_IMPROVEMENT_STATUS_PATH.read_text(encoding="utf-8")
    )

    assert len(comparison) == len(CANDIDATE_TITLES)
    assert set(comparison["candidate_slug"]) == set(CANDIDATE_TITLES)
    assert len(disagreement) == 60
    assert len(errors) == status["validation_error_count"]
    assert diagnostics["train_validation_group_overlap"] == 0
    assert decision["selected_candidate_slug"] in CANDIDATE_TITLES
    assert decision["final_test_evaluation_authorized"] is False
    assert status["readiness"] == "MODEL_IMPROVEMENT_DECISION_COMPLETE"
    assert status["locked_test_fingerprints_unchanged"] is True


def test_current_validation_improvement_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
