from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "dataset_v3_final_test"
)
TEMP_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / ".dataset_v3_final_test_execution"
)
EXECUTION_STATE_PATH = (
    FINAL_RESULTS_DIR
    / "execution_state.json"
)
CONSUMPTION_PATH = (
    FINAL_RESULTS_DIR
    / "authorization_consumption.json"
)

PRE_EXECUTION_TEST_NODEID = (
    "tests/test_dataset_v3_final_test_protocol.py::"
    "test_final_results_do_not_exist_before_execution"
)

EXPECTED_RESULT_FILES = {
    "authorization_consumption.json",
    "environment.json",
    "execution_state.json",
    "test_confusion_matrix.csv",
    "test_image_manifest.csv",
    "test_metrics.json",
    "test_per_category.csv",
    "test_predictions.csv",
    "test_relations.csv",
}


def load_json(path: Path) -> dict[str, object]:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def completed_final_test_state_is_valid() -> bool:
    if not FINAL_RESULTS_DIR.is_dir():
        return False

    if TEMP_RESULTS_DIR.exists():
        return False

    actual_files = {
        path.name
        for path in FINAL_RESULTS_DIR.iterdir()
        if path.is_file()
    }

    if actual_files != EXPECTED_RESULT_FILES:
        return False

    if not EXECUTION_STATE_PATH.is_file():
        return False

    if not CONSUMPTION_PATH.is_file():
        return False

    execution = load_json(
        EXECUTION_STATE_PATH
    )
    consumption = load_json(
        CONSUMPTION_PATH
    )

    return (
        execution.get("status")
        == "FINAL_TEST_EVALUATION_COMPLETE"
        and execution.get(
            "authorization_consumed"
        )
        is True
        and execution.get(
            "authorized_evaluations_completed"
        )
        == 1
        and execution.get(
            "test_evaluation_executed"
        )
        is True
        and consumption.get("status")
        == "FINAL_TEST_AUTHORIZATION_CONSUMED"
        and consumption.get(
            "authorization_consumed"
        )
        is True
        and consumption.get(
            "authorized_evaluations_completed"
        )
        == 1
        and consumption.get(
            "test_evaluation_executed"
        )
        is True
        and consumption.get(
            "further_tuning_permitted"
        )
        is False
        and consumption.get(
            "post_test_tuning_permitted"
        )
        is False
        and consumption.get(
            "test_results_may_be_used_for_tuning"
        )
        is False
    )


def pytest_collection_modifyitems(
    items: list[pytest.Item],
) -> None:
    if not completed_final_test_state_is_valid():
        return

    marker = pytest.mark.skip(
        reason=(
            "Expected post-execution phase transition: "
            "the hash-locked protocol regression checks "
            "that final results are absent before execution. "
            "The completed state is verified by "
            "test_dataset_v3_final_test_results.py."
        )
    )

    for item in items:
        if item.nodeid == PRE_EXECUTION_TEST_NODEID:
            item.add_marker(marker)
