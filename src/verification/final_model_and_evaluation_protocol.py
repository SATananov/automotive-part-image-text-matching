from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.final_model_freeze_config import (
    EXPECTED_SELECTION_DECISION,
    FINAL_EVALUATION_METRICS,
    FINAL_EVALUATION_PROTOCOL_PATH,
    FINAL_MODEL_FAMILY,
    FINAL_MODEL_FREEZE_MANIFEST_PATH,
    FINAL_MODEL_FREEZE_ROOT,
    FINAL_MODEL_FREEZE_STATUS_PATH,
    FINAL_MODEL_FREEZE_SUMMARY_PATH,
    FINAL_MODEL_SLUG,
    FINAL_MODEL_SPECIFICATION_PATH,
    FINAL_TEST_AUTHORIZATION_PATH,
    FREEZE_READINESS,
    FROZEN_LABEL_ORDER,
    FROZEN_LOCKED_TEST_PATHS,
    LOCKED_TEST_CONTRACT_PATH,
    SELECTION_CHECKPOINT_COMMIT,
)
from src.integrated_training_config import INTEGRATED_TEST_LOCK_PATH
from src.project_cli import COMMANDS
from src.real_dataset_config import PROJECT_ROOT
from src.validate_external_training_readiness import project_relative_path
from src.validation_model_improvement_config import (
    SELECTION_DECISION_PATH,
    VALIDATION_IMPROVEMENT_STATUS_PATH,
)

README_PATH = PROJECT_ROOT / "README.md"
WORKFLOW_PATH = PROJECT_ROOT / "src" / "freeze_final_model_and_evaluation_protocol.py"
CONFIG_PATH = PROJECT_ROOT / "src" / "final_model_freeze_config.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_final_model_freeze.py"


def read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"Missing freeze artifact: {project_relative_path(path)}.")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"Cannot read {project_relative_path(path)}: {error}.")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"Expected JSON object: {project_relative_path(path)}.")
        return {}
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_structure() -> list[str]:
    required = (
        CONFIG_PATH,
        WORKFLOW_PATH,
        TEST_PATH,
        FINAL_MODEL_SPECIFICATION_PATH,
        FINAL_EVALUATION_PROTOCOL_PATH,
        LOCKED_TEST_CONTRACT_PATH,
        FINAL_TEST_AUTHORIZATION_PATH,
        FINAL_MODEL_FREEZE_MANIFEST_PATH,
        FINAL_MODEL_FREEZE_SUMMARY_PATH,
        FINAL_MODEL_FREEZE_STATUS_PATH,
    )
    return [
        f"Missing final-model freeze file: {project_relative_path(path)}."
        for path in required
        if not path.is_file()
    ]


def validate_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    freeze_spec = COMMANDS.get("freeze-final-model-evaluation-protocol")
    verify_spec = COMMANDS.get("verify-final-model-freeze")
    if freeze_spec is None:
        errors.append("Final-model freeze command is missing.")
    elif freeze_spec.requires_tensorflow:
        errors.append("Metadata-only freeze command is incorrectly TensorFlow-gated.")
    if verify_spec is None:
        errors.append("Final-model freeze verification command is missing.")
    combined = "\n".join(
        path.read_text(encoding="utf-8-sig")
        for path in (README_PATH, FINAL_MODEL_FREEZE_SUMMARY_PATH)
        if path.is_file()
    )
    for fragment in (
        "freeze-final-model-evaluation-protocol",
        "verify-final-model-freeze",
        "REFERENCE_RETAINED",
        "locked test CSV files were not opened",
        "Step 010.6",
        "authorization remains `false`",
    ):
        if fragment not in combined:
            errors.append(f"Freeze documentation is missing '{fragment}'.")
    return errors


def validate_source_safeguards() -> list[str]:
    if not WORKFLOW_PATH.is_file():
        return ["Final-model freeze workflow is missing."]
    source = WORKFLOW_PATH.read_text(encoding="utf-8-sig")
    required = (
        "assert_not_locked_path",
        "locked_test_csv_files_opened",
        "final_test_evaluation_authorized",
        "SEPARATE_CONTROLLED_STEP_010_6",
        "single_authorized_run",
        "serialized_weights_committed",
    )
    errors = [
        f"Final-model freeze safeguard is missing: {fragment}."
        for fragment in required
        if fragment not in source
    ]
    for forbidden in (
        "import pandas",
        "pd.read_csv",
        "read_integrated_split(INTEGRATED_TEST_PATH",
        "locked_test_fingerprints(",
        "model.fit(",
        "model.predict(",
    ):
        if forbidden in source:
            errors.append(f"Freeze source contains forbidden operation: {forbidden}.")
    return errors


def validate_prerequisite_decision() -> list[str]:
    errors: list[str] = []
    decision = read_json(SELECTION_DECISION_PATH, errors)
    status = read_json(VALIDATION_IMPROVEMENT_STATUS_PATH, errors)
    if decision:
        if decision.get("decision") != EXPECTED_SELECTION_DECISION:
            errors.append("Step 010.4 decision no longer retains the reference.")
        if decision.get("selected_candidate_slug") != FINAL_MODEL_FAMILY:
            errors.append("Step 010.4 selected family differs from the freeze.")
        if decision.get("final_test_evaluation_authorized") is not False:
            errors.append("Step 010.4 decision authorizes final test use.")
    if status:
        if status.get("status") != "PASS":
            errors.append("Step 010.4 status is not PASS.")
        if status.get("locked_test_fingerprints_unchanged") is not True:
            errors.append("Step 010.4 test fingerprints are not preserved.")
    return errors


def validate_locked_contract() -> list[str]:
    errors: list[str] = []
    lock = read_json(INTEGRATED_TEST_LOCK_PATH, errors)
    contract = read_json(LOCKED_TEST_CONTRACT_PATH, errors)
    authorization = read_json(FINAL_TEST_AUTHORIZATION_PATH, errors)
    if lock:
        if lock.get("test_locked") is not True:
            errors.append("Integrated test lock is open.")
        if lock.get("test_evaluation_permitted") is not False:
            errors.append("Integrated test lock permits evaluation.")
    if contract:
        if contract.get("contract_state") != "LOCKED_AFTER_PROTOCOL_FREEZE":
            errors.append("Frozen test contract state differs.")
        if contract.get("step0105_test_csv_opened") is not False:
            errors.append("Step 010.5 contract reports opening test CSV files.")
        if contract.get("test_split_used") is not False:
            errors.append("Step 010.5 contract reports test use.")
        if contract.get("final_test_evaluation_authorized") is not False:
            errors.append("Step 010.5 contract authorizes test evaluation.")
        if lock and contract.get("locked_test_paths") != lock.get("locked_test_paths"):
            errors.append("Frozen locked-test paths differ from the committed lock.")
        expected_fingerprints = {
            str(lock.get("locked_test_paths", ["", ""])[0]): str(
                lock.get("external_test_sha256", "")
            ),
            str(lock.get("locked_test_paths", ["", ""])[1]): str(
                lock.get("integrated_test_sha256", "")
            ),
        }
        if lock and contract.get("locked_test_fingerprints") != expected_fingerprints:
            errors.append("Frozen fingerprints differ from the committed lock.")
    if authorization:
        if authorization.get("status") != "LOCKED":
            errors.append("Final test authorization status is not LOCKED.")
        if authorization.get("authorized") is not False:
            errors.append("Final test authorization is true.")
        if authorization.get("required_next_transition") != (
            "SEPARATE_CONTROLLED_STEP_010_6"
        ):
            errors.append("Final test authorization transition differs.")
    return errors


def validate_model_and_protocol() -> list[str]:
    errors: list[str] = []
    model = read_json(FINAL_MODEL_SPECIFICATION_PATH, errors)
    protocol = read_json(FINAL_EVALUATION_PROTOCOL_PATH, errors)
    if model:
        if model.get("freeze_state") != "FROZEN":
            errors.append("Final model specification is not frozen.")
        if model.get("final_model_slug") != FINAL_MODEL_SLUG:
            errors.append("Final model slug differs.")
        if model.get("final_model_family") != FINAL_MODEL_FAMILY:
            errors.append("Final model family differs.")
        if model.get("selection_checkpoint_commit") != SELECTION_CHECKPOINT_COMMIT:
            errors.append("Selection checkpoint commit differs.")
        if model.get("serialized_weights_committed") is not False:
            errors.append("Freeze incorrectly claims committed serialized weights.")
        if model.get("test_split_used") is not False:
            errors.append("Final model specification reports test use.")
        if model.get("final_test_evaluation_authorized") is not False:
            errors.append("Final model specification authorizes test use.")
        if model.get("input_contract", {}).get("label_order") != list(
            FROZEN_LABEL_ORDER
        ):
            errors.append("Frozen label order differs.")
    if protocol:
        if protocol.get("protocol_state") != "FROZEN_NOT_AUTHORIZED":
            errors.append("Final evaluation protocol state differs.")
        if protocol.get("evaluation_mode") != "single_authorized_run":
            errors.append("Final evaluation is not one-shot.")
        if protocol.get("required_metrics") != list(FINAL_EVALUATION_METRICS):
            errors.append("Frozen final metrics differ.")
        if protocol.get("label_order") != list(FROZEN_LABEL_ORDER):
            errors.append("Protocol label order differs.")
        if protocol.get("test_split_used") is not False:
            errors.append("Final evaluation protocol reports test use.")
        if protocol.get("final_test_evaluation_authorized") is not False:
            errors.append("Final evaluation protocol authorizes test use.")
    return errors


def validate_manifest_and_status() -> list[str]:
    errors: list[str] = []
    manifest = read_json(FINAL_MODEL_FREEZE_MANIFEST_PATH, errors)
    status = read_json(FINAL_MODEL_FREEZE_STATUS_PATH, errors)
    if manifest:
        if manifest.get("status") != "PASS":
            errors.append("Freeze manifest is not PASS.")
        if manifest.get("locked_test_csv_files_opened") is not False:
            errors.append("Freeze manifest reports opening test CSV files.")
        artifact_hashes = manifest.get("generated_artifact_sha256")
        if not isinstance(artifact_hashes, dict):
            errors.append("Freeze manifest generated hashes are missing.")
        else:
            for relative_path, expected_hash in artifact_hashes.items():
                path = PROJECT_ROOT / str(relative_path)
                if not path.is_file():
                    errors.append(f"Manifest artifact is missing: {relative_path}.")
                    continue
                if sha256_file(path) != expected_hash:
                    errors.append(f"Manifest artifact hash differs: {relative_path}.")
    if status:
        if status.get("status") != "PASS":
            errors.append("Final-model freeze status is not PASS.")
        if status.get("readiness") != FREEZE_READINESS:
            errors.append("Final-model freeze readiness differs.")
        if status.get("protocol_frozen") is not True:
            errors.append("Final evaluation protocol is not frozen.")
        if status.get("test_lock_preserved") is not True:
            errors.append("Final-model freeze did not preserve the test lock.")
        if status.get("locked_test_csv_files_opened") is not False:
            errors.append("Final-model freeze status reports opening test CSV files.")
        if status.get("final_test_evaluation_authorized") is not False:
            errors.append("Final-model freeze status authorizes final test use.")
    return errors


def validate_semantic_names() -> list[str]:
    errors: list[str] = []
    if FINAL_MODEL_FREEZE_ROOT.exists():
        for path in FINAL_MODEL_FREEZE_ROOT.rglob("*"):
            if path.is_file() and path.name.lower().startswith("step_"):
                errors.append(
                    "Final-model freeze artifact uses a technical step filename: "
                    f"{project_relative_path(path)}."
                )
    return errors


def validate_locked_csv_not_referenced_as_input() -> list[str]:
    errors: list[str] = []
    model = read_json(FINAL_MODEL_SPECIFICATION_PATH, errors)
    if not model:
        return errors
    training = model.get("training_contract", {})
    frozen_inputs = {
        str(training.get("training_input", "")),
        str(training.get("validation_input", "")),
    }
    locked = {project_relative_path(path) for path in FROZEN_LOCKED_TEST_PATHS}
    if frozen_inputs & locked:
        errors.append("A locked test CSV is frozen as a training input.")
    return errors


def build_verification_report() -> dict[str, Any]:
    checks = {
        "structure": validate_structure(),
        "cli_and_documentation": validate_cli_and_documentation(),
        "source_safeguards": validate_source_safeguards(),
        "prerequisite_decision": validate_prerequisite_decision(),
        "locked_test_contract": validate_locked_contract(),
        "model_and_protocol": validate_model_and_protocol(),
        "manifest_and_status": validate_manifest_and_status(),
        "semantic_filenames": validate_semantic_names(),
        "locked_csv_input_exclusion": validate_locked_csv_not_referenced_as_input(),
    }
    errors = [error for group in checks.values() for error in group]
    return {
        "status": "PASS" if not errors else "FAIL",
        "checks": {
            name: "PASS" if not check_errors else "FAIL"
            for name, check_errors in checks.items()
        },
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()
    print("Final model and locked-test evaluation protocol verification")
    for name, status in report["checks"].items():
        print(f"- {name}: {status}")
    print(f"Status: {report['status']}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
