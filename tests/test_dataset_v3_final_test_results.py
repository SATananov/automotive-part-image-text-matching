from __future__ import annotations

from src.final_test_results_v3 import (
    EXPECTED_ORIGINAL_METRICS_SHA256,
    EXPECTED_RELATION_PROTOCOL_FINGERPRINT,
    EXPECTED_RESULT_HASHES,
    build_final_test_results_manifest,
)
from src.verify_dataset_v3_final_test_results import (
    verify_final_test_results,
)


def test_final_test_results_verify() -> None:
    result = verify_final_test_results()

    assert result["status"] == (
        "PASS_DATASET_V3_FINAL_TEST_RESULTS_VERIFIED"
    )
    assert result["result_artifacts"] == 9
    assert result["correct_predictions"] == 354
    assert result["accuracy"] == 0.7375
    assert result["macro_f1"] == (
        0.7382299830250852
    )
    assert result["metadata_omission_reconciled"] is True
    assert result["evaluator_rerun"] is False
    assert result["result_artifacts_modified"] is False
    assert result["further_tuning_permitted"] is False


def test_original_result_artifacts_are_immutable() -> None:
    manifest = build_final_test_results_manifest()

    assert manifest["result_artifact_hashes"] == (
        EXPECTED_RESULT_HASHES
    )
    assert manifest["result_artifact_hashes"][
        "test_metrics.json"
    ] == EXPECTED_ORIGINAL_METRICS_SHA256
    assert manifest["result_artifacts_modified"] is False


def test_relation_fingerprint_omission_is_reconciled() -> None:
    manifest = build_final_test_results_manifest()
    reconciliation = manifest[
        "metadata_reconciliation"
    ]

    assert reconciliation["saved_fingerprint_present"] is False
    assert reconciliation[
        "derived_relation_protocol_fingerprint"
    ] == EXPECTED_RELATION_PROTOCOL_FINGERPRINT
    assert reconciliation[
        "frozen_protocol_fingerprint"
    ] == EXPECTED_RELATION_PROTOCOL_FINGERPRINT
    assert reconciliation[
        "saved_relations_match_frozen_builder"
    ] is True
    assert reconciliation["result_file_rewritten"] is False
    assert reconciliation["inference_rerun"] is False


def test_final_test_is_complete_and_not_tunable() -> None:
    manifest = build_final_test_results_manifest()
    execution = manifest["execution_audit"]

    assert execution["authorization_consumed"] is True
    assert execution[
        "authorized_evaluations_completed"
    ] == 1
    assert execution["test_evaluation_executed"] is True
    assert execution["repository_test_manifest_read"] is True
    assert execution["test_images_read"] is True
    assert manifest["selection_frozen"] is True
    assert manifest["further_tuning_permitted"] is False
    assert manifest["post_test_tuning_permitted"] is False
    assert manifest[
        "test_results_may_be_used_for_tuning"
    ] is False
