from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.build_fundamentals_experiment_notebook import (
    build_notebook,
    sanitize_notebook_for_repository,
)
from src.fundamentals_suite_config import (
    ARCHITECTURE_COMPARISON_PATH,
    BATCH_CONTRACT_PATH,
    CAPACITY_COMPARISON_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXPERIMENT_CONFIG_PATHS,
    FAILURE_DIAGNOSTICS_PATH,
    FUNDAMENTALS_IDS,
    NOTEBOOK_AUDIT_PATH,
    NOTEBOOK_PATH,
    OPTIMIZER_COMPARISON_PATH,
    OVERFIT_RESULT_PATH,
    PREPROCESSING_COMPARISON_PATH,
    READINESS,
    STATUS_PATH,
    SUITE_CONFIG_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
)
from src.project_cli import COMMANDS, run_command
import src.run_fundamentals_experimental_suite as fundamentals_module
from src.run_fundamentals_experimental_suite import (
    FundamentalsSuiteError,
    build_vocabulary,
    encode_texts,
    load_dataframes,
    load_split,
    prepare_data,
)
from src.verification.fundamentals_experimental_suite import (
    build_verification_report,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_all_fundamentals_configs_are_present_and_locked() -> None:
    suite = read_json(SUITE_CONFIG_PATH)
    configs = [read_json(path) for path in EXPERIMENT_CONFIG_PATHS]

    assert suite["step"] == "011.1"
    assert suite["test_split_allowed"] is False
    assert suite["test_split_path"] is None
    assert suite["final_test_evaluation_authorized"] is False
    assert suite["overfit_batch_group_count"] == 2
    assert suite["overfit_max_epochs"] >= 600
    assert suite["overfit_check_interval"] >= 1
    assert suite["overfit_learning_rates"] == [0.003, 0.001, 0.0003]
    assert [item["experiment_id"] for item in configs] == list(
        FUNDAMENTALS_IDS
    )
    assert all(item["test_split_allowed"] is False for item in configs)
    assert all(item["test_split_path"] is None for item in configs)
    assert all(
        item["final_test_evaluation_authorized"] is False
        for item in configs
    )


def test_loader_accepts_only_committed_train_and_validation() -> None:
    train = load_split(TRAIN_PATH)
    validation = load_split(VALIDATION_PATH)

    assert len(train) == 180
    assert len(validation) == 60
    with pytest.raises(FundamentalsSuiteError, match="Unauthorized"):
        load_split(TRAIN_PATH.parent / "integrated_test.csv")


def test_train_validation_groups_are_disjoint() -> None:
    train, validation = load_dataframes()

    assert set(train["part_group_id"]).isdisjoint(
        set(validation["part_group_id"])
    )


def test_text_encoding_is_train_fitted_and_deterministic() -> None:
    texts = ["Automotive air filter.", "Automotive oil filter."]
    vocabulary = build_vocabulary(texts)
    first = encode_texts(texts, vocabulary, sequence_length=6)
    second = encode_texts(texts, vocabulary, sequence_length=6)

    assert first.shape == (2, 6)
    assert first.dtype == np.int32
    assert np.array_equal(first, second)
    assert np.count_nonzero(first) > 0


def test_prepared_multimodal_arrays_follow_batch_contract() -> None:
    train, validation = load_dataframes()
    prepared = prepare_data(
        train,
        validation,
        image_size=16,
        sequence_length=8,
        grayscale=True,
        preprocessing_name="test_contract",
    )

    assert prepared.train_images.shape == (180, 16, 16, 1)
    assert prepared.validation_images.shape == (60, 16, 16, 1)
    assert prepared.train_text.shape == (180, 8)
    assert prepared.validation_text.shape == (60, 8)
    assert prepared.train_labels.shape == (180,)
    assert prepared.validation_labels.shape == (60,)
    assert prepared.train_images.dtype == np.float32
    assert prepared.train_text.dtype == np.int32
    assert prepared.train_labels.dtype == np.int32


def test_fundamentals_cli_commands_are_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (
        COMMANDS["run-fundamentals-suite"].module
        == "src.run_fundamentals_experimental_suite"
    )
    assert COMMANDS["run-fundamentals-suite"].requires_tensorflow
    assert (
        COMMANDS["build-fundamentals-notebook"].module
        == "src.build_fundamentals_experiment_notebook"
    )
    assert (
        COMMANDS["verify-fundamentals-suite"].module
        == "src.verification.fundamentals_experimental_suite"
    )

    # Regression guard: repository sanitization must preserve nbformat's
    # NotebookNode API instead of returning a plain dict.
    sanitized_notebook = sanitize_notebook_for_repository(build_notebook())
    assert hasattr(sanitized_notebook, "cells")
    assert sanitized_notebook.cells

    # Regression guard: the dispatcher must not leak the outer CLI command
    # into the experiment module's own optional argument parser.
    expected_status = {
        "status": "PASS",
        "readiness": READINESS,
        "completed_exercise_problem_count": 10,
        "exercise_problem_count": 10,
        "run_count": 35,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }
    monkeypatch.setattr(
        fundamentals_module,
        "run_suite",
        lambda: expected_status,
    )
    run_command(
        "run-fundamentals-suite",
        importer=lambda _: fundamentals_module,
    )


def test_executed_suite_status_and_registry_are_complete() -> None:
    status = read_json(STATUS_PATH)
    registry = read_json(EXECUTION_REGISTRY_JSON_PATH)

    assert status["status"] == "PASS"
    assert status["readiness"] == READINESS
    assert status["completed_exercise_problem_count"] == 10
    assert status["model_training_performed"] is True
    assert status["production_final_model_changed"] is False
    assert status["test_split_used"] is False
    assert status["final_test_evaluation_authorized"] is False
    assert registry["completed_experiment_count"] == 10
    assert registry["execution_status_counts"] == {"COMPLETED": 10}


def test_required_experiment_tables_exist() -> None:
    for path in (
        OPTIMIZER_COMPARISON_PATH,
        CAPACITY_COMPARISON_PATH,
        ARCHITECTURE_COMPARISON_PATH,
        PREPROCESSING_COMPARISON_PATH,
        FAILURE_DIAGNOSTICS_PATH,
    ):
        assert path.is_file()
        assert path.stat().st_size > 0


def test_overfit_and_batch_gates_pass() -> None:
    overfit = read_json(OVERFIT_RESULT_PATH)
    batch = read_json(BATCH_CONTRACT_PATH)

    assert overfit["threshold_reached"] is True
    assert overfit["status"] == "COMPLETED"
    assert batch["image_text_label_alignment_pass"] is True
    assert batch["validation_shuffle"] is False


def test_fundamentals_notebook_and_verifier_pass() -> None:
    assert NOTEBOOK_PATH.is_file()
    assert NOTEBOOK_AUDIT_PATH.is_file()

    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8-sig")
    audit_text = NOTEBOOK_AUDIT_PATH.read_text(encoding="utf-8-sig")
    separator_pattern = "(?:" + re.escape(chr(92)) + "+|/+)"
    windows_users_pattern = re.compile(
        "(?i)c:"
        + separator_pattern
        + "users"
        + separator_pattern
    )
    temporary_root_pattern = re.compile(
        "(?i)/" + "mnt" + "/" + "data" + "/"
    )
    for text in (notebook_text, audit_text):
        assert windows_users_pattern.search(text) is None
        assert temporary_root_pattern.search(text) is None

    audit = read_json(NOTEBOOK_AUDIT_PATH)
    assert audit["kernel_python"] == "project_virtual_environment"
    assert audit["execution_working_directory"] == "repository_root"
    assert audit["machine_specific_paths_removed"] is True

    report = build_verification_report()

    assert report["status"] == "PASS"
    assert report["errors"] == []
