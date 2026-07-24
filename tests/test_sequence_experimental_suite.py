from __future__ import annotations

import csv
import json

import nbformat

from src.sequence_suite_config import (
    ATTENTION_EVIDENCE_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXPERIMENT_CONFIG_PATHS,
    MANIFEST_PATH,
    MODEL_COMPARISON_CSV_PATH,
    NOTEBOOK_PATH,
    PRETRAINED_GATE_PATH,
    READINESS,
    SEQUENCE_IDS,
    STATUS_PATH,
    SUITE_CONFIG_PATH,
    TOKENIZATION_SUMMARY_PATH,
    TRAINING_RUNS_PATH,
)
from src.verification.sequence_experimental_suite import build_verification_report


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_sequence_suite_verifier_passes():
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
    assert all(report["checks"].values())


def test_all_ten_sequence_problem_configs_are_present_and_locked():
    configs = [read_json(path) for path in EXPERIMENT_CONFIG_PATHS]
    assert [item["experiment_id"] for item in configs] == list(SEQUENCE_IDS)
    assert all(item["test_split_allowed"] is False for item in configs)
    assert all(item["test_split_path"] is None for item in configs)
    assert all(item["final_test_evaluation_authorized"] is False for item in configs)


def test_suite_configuration_declares_core_completion_and_pretrained_gate():
    config = read_json(SUITE_CONFIG_PATH)
    assert config["readiness"] == READINESS
    assert config["completed_problem_numbers"] == list(range(1, 10))
    assert config["deferred_problem_numbers"] == [10]
    assert config["pretrained_download_authorized"] is False


def test_status_records_twenty_one_runs_without_production_change():
    status = read_json(STATUS_PATH)
    assert status["status"] == "PASS"
    assert status["training_runs_recorded"] == 21
    assert status["production_final_model_changed"] is False
    assert status["test_split_used"] is False
    assert status["final_test_evaluation_authorized"] is False


def test_training_runs_cover_six_model_families():
    rows = read_csv(TRAINING_RUNS_PATH)
    assert len(rows) == 21
    assert {row["family"] for row in rows} == {
        "embedding_average",
        "tfidf_logistic",
        "textcnn",
        "gru",
        "lstm",
        "transformer",
    }
    assert all(row["status"] == "COMPLETED" for row in rows)


def test_model_comparison_has_one_selected_run_per_family():
    rows = read_csv(MODEL_COMPARISON_CSV_PATH)
    assert len(rows) == 6
    assert all(row["selected_run_id"] for row in rows)
    assert all(0 <= float(row["validation_macro_f1"]) <= 1 for row in rows)
    assert all(int(row["run_count"]) >= 3 for row in rows)


def test_tokenization_is_train_only_and_padded():
    payload = read_json(TOKENIZATION_SUMMARY_PATH)
    assert payload["vocabulary_fit_split"] == "train_only"
    assert payload["train_shape"] == [180, 12]
    assert payload["validation_shape"] == [60, 12]
    assert payload["padding_index"] == 0
    assert payload["unknown_index"] == 1


def test_attention_evidence_contains_two_examples_and_two_heads():
    payload = read_json(ATTENTION_EVIDENCE_PATH)
    assert set(payload["selection"]) == {"correct", "incorrect"}
    assert payload["head_count"] == 2
    assert all(len(payload["selection"][name]["heads"]) == 2 for name in payload["selection"])


def test_pretrained_transformer_is_explicitly_deferred_without_download():
    payload = read_json(PRETRAINED_GATE_PATH)
    assert payload["status"] == "DEFERRED_EXPLICIT_APPROVAL_REQUIRED"
    assert payload["approval_received"] is False
    assert payload["network_download_attempted"] is False
    assert payload["pretrained_weights_downloaded"] is False
    assert payload["model_identifier"] is None


def test_notebook_and_manifest_are_complete():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 8
    assert all(cell.execution_count is not None for cell in code_cells)
    manifest = read_json(MANIFEST_PATH)
    registry = read_json(EXECUTION_REGISTRY_JSON_PATH)
    assert manifest["status"] == "PASS"
    assert manifest["training_runs_recorded"] == 21
    assert registry["total_training_runs"] == 21
