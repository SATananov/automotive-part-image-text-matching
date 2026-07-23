from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.final_model_freeze_config import (
    FINAL_EVALUATION_METRICS,
    FINAL_EVALUATION_PROTOCOL_PATH,
    FINAL_MODEL_FAMILY,
    FINAL_MODEL_FREEZE_MANIFEST_PATH,
    FINAL_MODEL_FREEZE_STATUS_PATH,
    FINAL_MODEL_SLUG,
    FINAL_MODEL_SPECIFICATION_PATH,
    FINAL_TEST_AUTHORIZATION_PATH,
    FREEZE_READINESS,
    FROZEN_LABEL_ORDER,
    FROZEN_LOCKED_TEST_PATHS,
    LOCKED_TEST_CONTRACT_PATH,
)
from src.freeze_final_model_and_evaluation_protocol import (
    FinalModelFreezeError,
    assert_not_locked_path,
    validate_prerequisites,
)
from src.project_cli import COMMANDS
from src.verification.final_model_and_evaluation_protocol import (
    build_verification_report,
)


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_final_model_freeze_cli_commands_are_registered() -> None:
    freeze_spec = COMMANDS["freeze-final-model-evaluation-protocol"]
    verify_spec = COMMANDS["verify-final-model-freeze"]
    assert freeze_spec.requires_tensorflow is False
    assert verify_spec.requires_tensorflow is False


def test_locked_test_paths_are_rejected_by_freeze_reader() -> None:
    for path in FROZEN_LOCKED_TEST_PATHS:
        with pytest.raises(FinalModelFreezeError):
            assert_not_locked_path(path)


def test_freeze_prerequisites_remain_valid() -> None:
    payloads = validate_prerequisites()
    assert payloads["decision"]["decision"] == "REFERENCE_RETAINED"
    assert payloads["decision"]["final_test_evaluation_authorized"] is False
    assert payloads["improvement_status"]["test_split_used"] is False


def test_final_model_specification_is_frozen_without_false_weight_claim() -> None:
    model = read_json(FINAL_MODEL_SPECIFICATION_PATH)
    assert model["freeze_state"] == "FROZEN"
    assert model["final_model_slug"] == FINAL_MODEL_SLUG
    assert model["final_model_family"] == FINAL_MODEL_FAMILY
    assert model["serialized_weights_committed"] is False
    assert model["input_contract"]["label_order"] == list(FROZEN_LABEL_ORDER)
    assert model["test_split_used"] is False
    assert model["final_test_evaluation_authorized"] is False


def test_final_evaluation_protocol_is_predeclared_and_one_shot() -> None:
    protocol = read_json(FINAL_EVALUATION_PROTOCOL_PATH)
    assert protocol["protocol_state"] == "FROZEN_NOT_AUTHORIZED"
    assert protocol["evaluation_mode"] == "single_authorized_run"
    assert protocol["required_metrics"] == list(FINAL_EVALUATION_METRICS)
    assert protocol["label_order"] == list(FROZEN_LABEL_ORDER)
    assert protocol["test_split_used"] is False
    assert protocol["final_test_evaluation_authorized"] is False


def test_locked_contract_and_authorization_remain_closed() -> None:
    contract = read_json(LOCKED_TEST_CONTRACT_PATH)
    authorization = read_json(FINAL_TEST_AUTHORIZATION_PATH)
    assert contract["contract_state"] == "LOCKED_AFTER_PROTOCOL_FREEZE"
    assert contract["step0105_test_csv_opened"] is False
    assert contract["test_evaluation_permitted"] is False
    assert authorization["status"] == "LOCKED"
    assert authorization["authorized"] is False
    assert authorization["required_next_transition"] == (
        "SEPARATE_CONTROLLED_STEP_010_6"
    )


def test_freeze_manifest_and_status_are_complete() -> None:
    manifest = read_json(FINAL_MODEL_FREEZE_MANIFEST_PATH)
    status = read_json(FINAL_MODEL_FREEZE_STATUS_PATH)
    assert manifest["status"] == "PASS"
    assert manifest["locked_test_csv_files_opened"] is False
    assert len(manifest["generated_artifact_sha256"]) == 5
    assert status["status"] == "PASS"
    assert status["readiness"] == FREEZE_READINESS
    assert status["test_lock_preserved"] is True
    assert status["final_test_evaluation_authorized"] is False


def test_current_final_model_freeze_verifier_passes() -> None:
    report = build_verification_report()
    assert report["status"] == "PASS", report["errors"]
