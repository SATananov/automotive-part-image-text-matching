from __future__ import annotations

from src.verify_dataset_v3_final_test_authorization import (
    verify_final_test_authorization,
)


def test_dataset_v3_final_test_authorization() -> None:
    result = verify_final_test_authorization()

    assert result["status"] == (
        "PASS_DATASET_V3_FINAL_TEST_AUTHORIZATION_VERIFIED"
    )
    assert result["selected_model_slug"] == (
        "torch_multimodal_dataset_v3"
    )
    assert result["selection_frozen"] is True
    assert result["further_tuning_permitted"] is False
    assert result["post_test_tuning_permitted"] is False
    assert result["test_evaluation_authorized"] is True
    assert result["maximum_authorized_evaluations"] == 1
    assert result["authorized_evaluations_completed"] == 0
    assert result["authorization_consumed"] is False
    assert result["test_evaluation_executed"] is False
    assert result["test_manifest_read"] is False
    assert result["test_images_read"] is False
    assert result["test_results_may_be_used_for_tuning"] is False
    assert result["historical_notebook_blob"] == (
        "6eb357a2997b0a660cfa57a8c42afab7ecc3d5f9"
    )
    assert result["historical_notebook_blob_verified"] is True
    assert result["current_notebook_sha256"] == (
        "4f87a9818cf91d1011b4b09794da086b6cc8fc9469e10d612a7c6ae6efb1900a"
    )
    assert result["current_notebook_blob_sha256"] == (
        "4f87a9818cf91d1011b4b09794da086b6cc8fc9469e10d612a7c6ae6efb1900a"
    )
    assert result["current_notebook_frozen_bytes_verified"] is True
