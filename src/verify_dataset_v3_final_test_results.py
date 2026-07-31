from __future__ import annotations

import json

from src.data import PROJECT_ROOT
from src.final_test_results_v3 import (
    build_final_test_results_manifest,
    canonical_json_bytes,
    file_sha256,
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_v3"
    / "dataset_v3_final_test_results.json"
)
CHECKSUM_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_v3"
    / "dataset_v3_final_test_results.sha256.txt"
)


def verify_final_test_results() -> dict[str, object]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(
            "Final-test results manifest is missing."
        )

    if not CHECKSUM_PATH.is_file():
        raise RuntimeError(
            "Final-test results checksum is missing."
        )

    saved = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8-sig"
        )
    )
    expected = build_final_test_results_manifest()

    if canonical_json_bytes(saved) != (
        canonical_json_bytes(expected)
    ):
        raise RuntimeError(
            "Final-test results manifest differs from "
            "independently recomputed evidence."
        )

    actual_checksum = file_sha256(
        MANIFEST_PATH
    )
    saved_checksum = CHECKSUM_PATH.read_text(
        encoding="utf-8-sig"
    ).split()[0]

    if saved_checksum != actual_checksum:
        raise RuntimeError(
            "Final-test results manifest checksum mismatch."
        )

    if saved["metadata_reconciliation"]["saved_fingerprint_present"]:
        raise RuntimeError(
            "The original metrics fingerprint omission "
            "is no longer preserved."
        )

    if not saved["metadata_reconciliation"][
        "saved_relations_match_frozen_builder"
    ]:
        raise RuntimeError(
            "Saved final-test relations do not match "
            "the frozen relation builder."
        )

    if saved["evaluator_rerun"]:
        raise RuntimeError(
            "Final-test results report an evaluator rerun."
        )

    if saved["result_artifacts_modified"]:
        raise RuntimeError(
            "Original final-test artifacts were modified."
        )

    if saved["execution_audit"][
        "authorized_evaluations_completed"
    ] != 1:
        raise RuntimeError(
            "Unexpected completed-evaluation count."
        )

    return {
        "status": (
            "PASS_DATASET_V3_FINAL_TEST_RESULTS_VERIFIED"
        ),
        "manifest_sha256": actual_checksum,
        "result_artifacts": saved[
            "result_artifact_count"
        ],
        "correct_predictions": saved[
            "final_reporting"
        ]["correct_predictions"],
        "accuracy": saved[
            "final_reporting"
        ]["accuracy"],
        "macro_f1": saved[
            "final_reporting"
        ]["macro_f1"],
        "relation_protocol_fingerprint": saved[
            "metadata_reconciliation"
        ]["derived_relation_protocol_fingerprint"],
        "metadata_omission_reconciled": True,
        "evaluator_rerun": False,
        "result_artifacts_modified": False,
        "further_tuning_permitted": False,
    }


def main() -> None:
    result = verify_final_test_results()

    print()
    print("=== DATASET V3 FINAL TEST RESULTS VERIFIED ===")
    print(f"Artifacts:  {result['result_artifacts']}/9")
    print(f"Correct:    {result['correct_predictions']}/480")
    print(f"Accuracy:   {result['accuracy']:.10f}")
    print(f"Macro F1:   {result['macro_f1']:.10f}")
    print("Rerun:      False")
    print("Tuning:     False")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
