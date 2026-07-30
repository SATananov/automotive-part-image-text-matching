from __future__ import annotations

from src.verify_dataset_v3_final_selection_lock import (
    verify_final_selection_lock,
)


def test_dataset_v3_final_selection_lock() -> None:
    result = verify_final_selection_lock()

    assert result["status"] == (
        "PASS_DATASET_V3_FINAL_SELECTION_LOCK_VERIFIED"
    )
    assert result["selected_model_slug"] == (
        "torch_multimodal_dataset_v3"
    )
    assert result["selection_frozen"] is True
    assert result["further_tuning_permitted"] is False
    assert result["test_evaluation_authorized"] is False
    assert result["test_evaluation_executed"] is False
    assert result["test_manifest_read"] is False
    assert result["test_images_read"] is False
    assert result["component_hashes_verified"] >= 20
