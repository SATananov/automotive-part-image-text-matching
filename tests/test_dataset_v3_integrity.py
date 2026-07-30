from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "verify_dataset_v3.py"

SPEC = importlib.util.spec_from_file_location(
    "verify_dataset_v3",
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dataset_v3_integrity() -> None:
    result = MODULE.verify()

    assert result["status"] == "PASS"
    assert result["images"] == 640
    assert result["split_counts"] == {
        "train": 480,
        "validation": 80,
        "test": 80,
    }
    assert result["unique_candidate_ids"] == 640
    assert result["unique_image_groups"] == 640
    assert result["unique_sha256"] == 640
    assert result["group_overlap"] == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert result["test_lock_status"] == (
        "LOCKED_NOT_AUTHORIZED_FOR_TRAINING_OR_SELECTION"
    )
    assert result["test_lock_sha256"] == (
        "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
    )
