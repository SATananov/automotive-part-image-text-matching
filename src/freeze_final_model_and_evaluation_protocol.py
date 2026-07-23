from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from src.final_model_freeze_config import (
    EXPECTED_SELECTION_DECISION,
    FINAL_EVALUATION_METRICS,
    FINAL_EVALUATION_PROTOCOL_PATH,
    FINAL_MODEL_ARCHITECTURE_PATH,
    FINAL_MODEL_FAMILY,
    FINAL_MODEL_FREEZE_MANIFEST_PATH,
    FINAL_MODEL_FREEZE_ROOT,
    FINAL_MODEL_FREEZE_STATUS_PATH,
    FINAL_MODEL_FREEZE_SUMMARY_PATH,
    FINAL_MODEL_SLUG,
    FINAL_MODEL_SPECIFICATION_PATH,
    FINAL_MODEL_TITLE,
    FINAL_MODEL_VALIDATION_METRICS_PATH,
    FINAL_TEST_AUTHORIZATION_PATH,
    FREEZE_READINESS,
    FROZEN_BATCH_SIZE,
    FROZEN_EARLY_STOPPING_PATIENCE,
    FROZEN_IMAGE_HEIGHT,
    FROZEN_IMAGE_WIDTH,
    FROZEN_LABEL_ORDER,
    FROZEN_LOCKED_TEST_PATHS,
    FROZEN_MAX_EPOCHS,
    FROZEN_RANDOM_STATE,
    FROZEN_TEXT_EMBEDDING_DIMENSION,
    FROZEN_TEXT_SEQUENCE_LENGTH,
    FROZEN_TRAINING_INPUTS,
    LOCKED_TEST_CONTRACT_PATH,
    REQUIREMENTS_LOCK_PATH,
    SELECTION_CHECKPOINT_COMMIT,
)
from src.integrated_training_config import (
    INTEGRATED_RUN_STATUS_PATH,
    INTEGRATED_TEST_LOCK_PATH,
)
from src.real_dataset_config import PROJECT_ROOT
from src.validate_external_training_readiness import project_relative_path
from src.validation_model_improvement_config import (
    EXPERIMENT_REGISTRY_PATH,
    SELECTION_DECISION_PATH,
    VALIDATION_IMPROVEMENT_STATUS_PATH,
)


class FinalModelFreezeError(RuntimeError):
    """Raised when the final-model freeze contract cannot be established."""


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def assert_not_locked_path(path: Path) -> None:
    resolved = path.resolve()
    locked = {candidate.resolve() for candidate in FROZEN_LOCKED_TEST_PATHS}
    if resolved in locked:
        raise FinalModelFreezeError(
            "Step 010.5 cannot open a locked test CSV artifact."
        )


def read_safe_text(path: Path) -> str:
    assert_not_locked_path(path)
    if not path.is_file():
        raise FinalModelFreezeError(
            f"Required freeze input is missing: {project_relative_path(path)}."
        )
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise FinalModelFreezeError(
            f"Cannot read freeze input {project_relative_path(path)}: {error}."
        ) from error


def read_safe_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(read_safe_text(path))
    except json.JSONDecodeError as error:
        raise FinalModelFreezeError(
            f"Invalid JSON freeze input {project_relative_path(path)}: {error}."
        ) from error
    if not isinstance(payload, dict):
        raise FinalModelFreezeError(
            f"Freeze input is not a JSON object: {project_relative_path(path)}."
        )
    return payload


def sha256_safe_file(path: Path) -> str:
    assert_not_locked_path(path)
    if not path.is_file():
        raise FinalModelFreezeError(
            f"Cannot fingerprint missing input: {project_relative_path(path)}."
        )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FinalModelFreezeError(
            f"Cannot fingerprint {project_relative_path(path)}: {error}."
        ) from error
    return digest.hexdigest()


def validate_prerequisites() -> dict[str, dict[str, Any]]:
    integrated_status = read_safe_json(INTEGRATED_RUN_STATUS_PATH)
    decision = read_safe_json(SELECTION_DECISION_PATH)
    improvement_status = read_safe_json(VALIDATION_IMPROVEMENT_STATUS_PATH)
    experiment_registry = read_safe_json(EXPERIMENT_REGISTRY_PATH)
    test_lock = read_safe_json(INTEGRATED_TEST_LOCK_PATH)
    validation_metrics = read_safe_json(FINAL_MODEL_VALIDATION_METRICS_PATH)

    if integrated_status.get("status") != "PASS":
        raise FinalModelFreezeError("Step 010.3 status is not PASS.")
    if integrated_status.get("readiness") != "VALIDATION_COMPARISON_COMPLETE":
        raise FinalModelFreezeError("Step 010.3 validation comparison is incomplete.")
    if integrated_status.get("best_model_slug") != FINAL_MODEL_SLUG:
        raise FinalModelFreezeError("Step 010.3 best model differs from the freeze target.")
    if integrated_status.get("test_split_used") is not False:
        raise FinalModelFreezeError("Step 010.3 reports locked-test use.")

    if decision.get("status") != "PASS":
        raise FinalModelFreezeError("Step 010.4 model decision is not PASS.")
    if decision.get("decision") != EXPECTED_SELECTION_DECISION:
        raise FinalModelFreezeError("Step 010.4 did not retain the reference model.")
    if decision.get("selected_candidate_slug") != FINAL_MODEL_FAMILY:
        raise FinalModelFreezeError("Step 010.4 selected family differs from the freeze target.")
    if decision.get("final_test_evaluation_authorized") is not False:
        raise FinalModelFreezeError("Step 010.4 already authorizes final test use.")

    if improvement_status.get("status") != "PASS":
        raise FinalModelFreezeError("Step 010.4 status is not PASS.")
    if improvement_status.get("readiness") != "MODEL_IMPROVEMENT_DECISION_COMPLETE":
        raise FinalModelFreezeError("Step 010.4 readiness is incomplete.")
    if improvement_status.get("locked_test_fingerprints_unchanged") is not True:
        raise FinalModelFreezeError("Step 010.4 did not preserve test fingerprints.")
    if improvement_status.get("test_split_used") is not False:
        raise FinalModelFreezeError("Step 010.4 reports locked-test use.")

    if experiment_registry.get("status") != "PASS":
        raise FinalModelFreezeError("Step 010.4 experiment registry is not PASS.")
    if experiment_registry.get("protocol") != "fixed_candidates_fixed_seeds_validation_only":
        raise FinalModelFreezeError("Step 010.4 experiment protocol differs.")

    if test_lock.get("test_locked") is not True:
        raise FinalModelFreezeError("The integrated test lock is open.")
    if test_lock.get("test_evaluation_permitted") is not False:
        raise FinalModelFreezeError("The integrated test lock permits evaluation.")
    expected_training_inputs = {
        project_relative_path(path) for path in FROZEN_TRAINING_INPUTS
    }
    if set(test_lock.get("training_inputs", [])) != expected_training_inputs:
        raise FinalModelFreezeError("The test lock authorizes unexpected training inputs.")

    status_fingerprints = improvement_status.get("locked_test_fingerprints")
    if not isinstance(status_fingerprints, dict):
        raise FinalModelFreezeError("Step 010.4 locked-test fingerprints are missing.")
    lock_fingerprints = {
        str(test_lock["locked_test_paths"][0]): str(test_lock["external_test_sha256"]),
        str(test_lock["locked_test_paths"][1]): str(test_lock["integrated_test_sha256"]),
    }
    if status_fingerprints != lock_fingerprints:
        raise FinalModelFreezeError("Step 010.4 fingerprints differ from the test lock.")

    if validation_metrics.get("model_slug") != FINAL_MODEL_SLUG:
        raise FinalModelFreezeError("Final validation metrics identify another model.")
    if validation_metrics.get("evaluation_split") != "integrated_validation":
        raise FinalModelFreezeError("Final validation evidence is not validation-only.")
    if validation_metrics.get("test_split_used") is not False:
        raise FinalModelFreezeError("Final validation metrics report test use.")

    return {
        "integrated_status": integrated_status,
        "decision": decision,
        "improvement_status": improvement_status,
        "experiment_registry": experiment_registry,
        "test_lock": test_lock,
        "validation_metrics": validation_metrics,
    }


def build_freeze_documents(
    prerequisites: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    integrated_status = prerequisites["integrated_status"]
    decision = prerequisites["decision"]
    improvement_status = prerequisites["improvement_status"]
    experiment_registry = prerequisites["experiment_registry"]
    test_lock = prerequisites["test_lock"]
    validation_metrics = prerequisites["validation_metrics"]

    source_paths = (
        INTEGRATED_RUN_STATUS_PATH,
        SELECTION_DECISION_PATH,
        VALIDATION_IMPROVEMENT_STATUS_PATH,
        EXPERIMENT_REGISTRY_PATH,
        INTEGRATED_TEST_LOCK_PATH,
        FINAL_MODEL_ARCHITECTURE_PATH,
        FINAL_MODEL_VALIDATION_METRICS_PATH,
        REQUIREMENTS_LOCK_PATH,
    )
    source_fingerprints = {
        project_relative_path(path): sha256_safe_file(path) for path in source_paths
    }

    model_specification = {
        "status": "PASS",
        "freeze_state": "FROZEN",
        "selection_checkpoint_commit": SELECTION_CHECKPOINT_COMMIT,
        "final_model_slug": FINAL_MODEL_SLUG,
        "final_model": FINAL_MODEL_TITLE,
        "final_model_family": FINAL_MODEL_FAMILY,
        "selection_decision": EXPECTED_SELECTION_DECISION,
        "selection_evidence": {
            "step0103_best_model_slug": integrated_status["best_model_slug"],
            "step0103_validation_accuracy": integrated_status[
                "best_validation_accuracy"
            ],
            "step0103_validation_macro_f1": integrated_status[
                "best_validation_macro_f1"
            ],
            "step0104_selected_candidate_slug": decision[
                "selected_candidate_slug"
            ],
            "step0104_decision": decision["decision"],
            "step0104_improvements_accepted": False,
        },
        "frozen_object": (
            "architecture_preprocessing_training_and_checkpoint_selection_recipe"
        ),
        "serialized_weights_committed": False,
        "weights_policy": (
            "Step 010.3 did not commit serialized weights. A later separately "
            "authorized final-evaluation step may reconstruct the model exactly "
            "once from this frozen recipe before the one-shot test evaluation."
        ),
        "input_contract": {
            "image_shape": [FROZEN_IMAGE_HEIGHT, FROZEN_IMAGE_WIDTH, 3],
            "image_mode": "RGB",
            "image_resize": "PIL_bilinear",
            "model_rescaling": "1/255",
            "text_token_pattern": "[a-z0-9]+",
            "text_lowercase": True,
            "text_sequence_length": FROZEN_TEXT_SEQUENCE_LENGTH,
            "text_embedding_dimension": FROZEN_TEXT_EMBEDDING_DIMENSION,
            "text_padding_index": 0,
            "text_oov_index": 1,
            "vocabulary_order": "frequency_descending_then_token_ascending",
            "label_order": list(FROZEN_LABEL_ORDER),
        },
        "architecture_contract": {
            "model_name": "keras_multimodal_classifier",
            "text_branch": [
                "Embedding(vocabulary_size,16)",
                "GlobalAveragePooling1D",
                "Dense(32,relu)",
            ],
            "image_branch": [
                "Rescaling(1/255)",
                "Flatten",
                "Dense(64,relu)",
                "Dense(32,relu)",
            ],
            "fusion": [
                "Concatenate",
                "Dense(64,relu)",
                "Dropout(0.15)",
                "Dense(3,softmax)",
            ],
            "trainable_parameter_count": validation_metrics["training"][
                "parameter_count"
            ],
            "architecture_source": project_relative_path(
                FINAL_MODEL_ARCHITECTURE_PATH
            ),
            "architecture_sha256": source_fingerprints[
                project_relative_path(FINAL_MODEL_ARCHITECTURE_PATH)
            ],
        },
        "training_contract": {
            "training_input": project_relative_path(FROZEN_TRAINING_INPUTS[0]),
            "validation_input": project_relative_path(FROZEN_TRAINING_INPUTS[1]),
            "random_state": FROZEN_RANDOM_STATE,
            "deterministic_operations": "enabled_where_supported",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "SparseCategoricalCrossentropy",
            "metric": "SparseCategoricalAccuracy",
            "batch_size": FROZEN_BATCH_SIZE,
            "maximum_epochs": FROZEN_MAX_EPOCHS,
            "shuffle": True,
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": FROZEN_EARLY_STOPPING_PATIENCE,
            "restore_best_weights": True,
            "checkpoint_selection_rule": "minimum_validation_loss",
        },
        "validation_evidence": {
            "metrics_source": project_relative_path(
                FINAL_MODEL_VALIDATION_METRICS_PATH
            ),
            "metrics_sha256": source_fingerprints[
                project_relative_path(FINAL_MODEL_VALIDATION_METRICS_PATH)
            ],
            "accuracy": validation_metrics["accuracy"],
            "macro_f1": validation_metrics["macro_f1"],
            "best_epoch": validation_metrics["training"]["best_epoch"],
            "best_validation_loss": validation_metrics["training"][
                "best_validation_loss"
            ],
        },
        "environment_contract": {
            "requirements_lock": project_relative_path(REQUIREMENTS_LOCK_PATH),
            "requirements_lock_sha256": source_fingerprints[
                project_relative_path(REQUIREMENTS_LOCK_PATH)
            ],
            "keras_backend": integrated_status["keras_backend"],
        },
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "final_test_evaluation_authorized": False,
    }

    evaluation_protocol = {
        "status": "PASS",
        "protocol_state": "FROZEN_NOT_AUTHORIZED",
        "primary_final_evaluation_dataset": "data/processed/integrated_test.csv",
        "secondary_locked_component_dataset": (
            "data/external/integrated/external_test.csv"
        ),
        "secondary_dataset_policy": (
            "Do not evaluate it separately. Source-specific results must be "
            "derived only as a predefined slice of the integrated test ledger."
        ),
        "evaluation_mode": "single_authorized_run",
        "model_reconstruction_policy": (
            "Reconstruct exactly once from the frozen model recipe using only "
            "the frozen train and validation inputs, then evaluate once."
        ),
        "required_metrics": list(FINAL_EVALUATION_METRICS),
        "label_order": list(FROZEN_LABEL_ORDER),
        "reporting_rules": {
            "zero_division": 0,
            "macro_average": "unweighted_across_all_three_labels",
            "confusion_matrix_order": list(FROZEN_LABEL_ORDER),
            "probability_columns_required": True,
            "sample_identifiers_required": True,
            "category_and_source_slices_predeclared": True,
        },
        "one_shot_rules": [
            "No architecture, preprocessing, seed, optimizer, epoch, threshold, or label change after test access.",
            "No candidate comparison or model selection on test metrics.",
            "No repeated run unless a documented technical failure invalidates all outputs before interpretation.",
            "Publish all predefined metrics, including weak classes and slices.",
            "Fingerprint locked artifacts before and after the authorized run.",
        ],
        "future_authorization_requirements": [
            "Step 010.5 freeze artifacts committed and unchanged.",
            "Clean synchronized Git checkpoint.",
            "Separate explicit Step 010.6 authorization artifact.",
            "Locked-test fingerprints equal the committed contract.",
            "No new validation-guided model change after this freeze.",
        ],
        "future_output_contract": {
            "metrics_json": "reports/final_test_evaluation/final_test_metrics.json",
            "confusion_matrix_csv": (
                "reports/final_test_evaluation/final_test_confusion_matrix.csv"
            ),
            "prediction_ledger_csv": (
                "reports/final_test_evaluation/final_test_predictions.csv"
            ),
            "category_metrics_csv": (
                "reports/final_test_evaluation/final_test_category_metrics.csv"
            ),
            "source_metrics_csv": (
                "reports/final_test_evaluation/final_test_source_metrics.csv"
            ),
            "summary_markdown": (
                "reports/final_test_evaluation/final_test_evaluation_summary.md"
            ),
        },
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "final_test_evaluation_authorized": False,
    }

    locked_test_contract = {
        "status": "PASS",
        "contract_state": "LOCKED_AFTER_PROTOCOL_FREEZE",
        "locked_test_paths": list(test_lock["locked_test_paths"]),
        "locked_test_fingerprints": improvement_status[
            "locked_test_fingerprints"
        ],
        "hash_normalization": test_lock["hash_normalization"],
        "integrated_test_rows": test_lock["integrated_test_rows"],
        "integrated_test_groups": test_lock["integrated_test_groups"],
        "external_test_rows": test_lock["external_test_rows"],
        "external_test_groups": test_lock["external_test_groups"],
        "authorized_current_inputs": list(test_lock["training_inputs"]),
        "step0105_test_csv_opened": False,
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "final_test_evaluation_authorized": False,
        "unlock_policy": (
            "Protocol freeze alone does not unlock test data. A separate future "
            "authorization transition is mandatory."
        ),
    }

    authorization = {
        "status": "LOCKED",
        "authorization_version": 1,
        "authorized": False,
        "authorized_step": None,
        "authorized_at": None,
        "authorization_reason": (
            "Step 010.5 freezes the model and evaluation protocol but does not "
            "authorize access to the locked test split."
        ),
        "required_next_transition": "SEPARATE_CONTROLLED_STEP_010_6",
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "final_test_evaluation_authorized": False,
    }

    manifest = {
        "status": "PASS",
        "selection_checkpoint_commit": SELECTION_CHECKPOINT_COMMIT,
        "source_artifact_sha256": source_fingerprints,
        "source_artifact_count": len(source_fingerprints),
        "locked_test_csv_fingerprints_copied_from_committed_lock": True,
        "locked_test_csv_files_opened": False,
        "generated_artifacts": [
            project_relative_path(FINAL_MODEL_SPECIFICATION_PATH),
            project_relative_path(FINAL_EVALUATION_PROTOCOL_PATH),
            project_relative_path(LOCKED_TEST_CONTRACT_PATH),
            project_relative_path(FINAL_TEST_AUTHORIZATION_PATH),
            project_relative_path(FINAL_MODEL_FREEZE_SUMMARY_PATH),
            project_relative_path(FINAL_MODEL_FREEZE_STATUS_PATH),
        ],
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }

    return {
        "model_specification": model_specification,
        "evaluation_protocol": evaluation_protocol,
        "locked_test_contract": locked_test_contract,
        "authorization": authorization,
        "manifest": manifest,
    }


def render_summary(documents: dict[str, dict[str, Any]]) -> str:
    model = documents["model_specification"]
    protocol = documents["evaluation_protocol"]
    contract = documents["locked_test_contract"]
    lines = [
        "# Final Model and Locked-Test Evaluation Protocol Freeze",
        "",
        "- Status: **PASS**",
        f"- Readiness: **{FREEZE_READINESS}**",
        f"- Final model: **{model['final_model']}** (`{model['final_model_slug']}`)",
        f"- Model family: `{model['final_model_family']}`",
        f"- Step 010.4 decision: `{model['selection_decision']}`",
        "- The frozen object is the architecture, preprocessing, training, and checkpoint-selection recipe.",
        "- No serialized weights are claimed because Step 010.3 did not commit a model file.",
        "- The locked test CSV files were not opened, parsed, trained on, predicted on, or evaluated in Step 010.5.",
        "- Final test authorization remains `false`.",
        "",
        "## Frozen validation evidence",
        "",
        f"- Validation accuracy: `{model['validation_evidence']['accuracy']:.4f}`",
        f"- Validation macro F1: `{model['validation_evidence']['macro_f1']:.4f}`",
        f"- Best epoch: `{model['validation_evidence']['best_epoch']}`",
        f"- Selection checkpoint: `{model['selection_checkpoint_commit']}`",
        "",
        "## One-shot future evaluation",
        "",
        f"- Primary dataset: `{protocol['primary_final_evaluation_dataset']}`",
        "- The external locked test file is not a second model-selection benchmark.",
        "- Every metric and slice is fixed before test access.",
        "- A separate controlled Step 010.6 authorization is mandatory.",
        "",
        "## Locked-test contract",
        "",
    ]
    for path, fingerprint in contract["locked_test_fingerprints"].items():
        lines.append(f"- `{path}`: `{fingerprint}`")
    lines.extend(
        [
            "",
            "## Frozen metrics",
            "",
        ]
    )
    for metric in protocol["required_metrics"]:
        lines.append(f"- `{metric}`")
    lines.extend(
        [
            "",
            "Step 010.5 changes no test authorization state. The final evaluation "
            "cannot run until a later step verifies this freeze and creates a "
            "separate explicit authorization artifact.",
            "",
        ]
    )
    return "\n".join(lines)


def write_freeze_outputs(documents: dict[str, dict[str, Any]]) -> None:
    FINAL_MODEL_FREEZE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        FINAL_MODEL_SPECIFICATION_PATH, documents["model_specification"]
    )
    atomic_write_json(
        FINAL_EVALUATION_PROTOCOL_PATH, documents["evaluation_protocol"]
    )
    atomic_write_json(
        LOCKED_TEST_CONTRACT_PATH, documents["locked_test_contract"]
    )
    atomic_write_json(FINAL_TEST_AUTHORIZATION_PATH, documents["authorization"])
    atomic_write_text(FINAL_MODEL_FREEZE_SUMMARY_PATH, render_summary(documents))

    generated_for_hash = (
        FINAL_MODEL_SPECIFICATION_PATH,
        FINAL_EVALUATION_PROTOCOL_PATH,
        LOCKED_TEST_CONTRACT_PATH,
        FINAL_TEST_AUTHORIZATION_PATH,
        FINAL_MODEL_FREEZE_SUMMARY_PATH,
    )
    artifact_hashes = {
        project_relative_path(path): sha256_safe_file(path)
        for path in generated_for_hash
    }
    manifest = dict(documents["manifest"])
    manifest["generated_artifact_sha256"] = artifact_hashes
    atomic_write_json(FINAL_MODEL_FREEZE_MANIFEST_PATH, manifest)

    status = {
        "status": "PASS",
        "readiness": FREEZE_READINESS,
        "final_model_slug": FINAL_MODEL_SLUG,
        "final_model": FINAL_MODEL_TITLE,
        "final_model_family": FINAL_MODEL_FAMILY,
        "selection_decision": EXPECTED_SELECTION_DECISION,
        "selection_checkpoint_commit": SELECTION_CHECKPOINT_COMMIT,
        "serialized_weights_committed": False,
        "protocol_frozen": True,
        "test_lock_preserved": True,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "final_test_evaluation_authorized": False,
        "manifest": project_relative_path(FINAL_MODEL_FREEZE_MANIFEST_PATH),
        "summary": project_relative_path(FINAL_MODEL_FREEZE_SUMMARY_PATH),
    }
    atomic_write_json(FINAL_MODEL_FREEZE_STATUS_PATH, status)


def main() -> None:
    prerequisites = validate_prerequisites()
    documents = build_freeze_documents(prerequisites)
    write_freeze_outputs(documents)
    print("Final model and locked-test evaluation protocol freeze")
    print(f"- final model: {FINAL_MODEL_SLUG}")
    print(f"- readiness: {FREEZE_READINESS}")
    print("- locked test CSV files opened: false")
    print("- final test authorization: false")
    print("Status: PASS")


if __name__ == "__main__":
    main()
