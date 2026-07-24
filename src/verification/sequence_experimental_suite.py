from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat
from PIL import Image

from src.sequence_suite_config import (
    ATTENTION_EVIDENCE_PATH,
    ATTENTION_TOKEN_SUMMARY_PATH,
    COMPLETED_SEQUENCE_IDS,
    CONFUSION_MATRICES_PATH,
    DEFERRED_SEQUENCE_IDS,
    DOCUMENTATION_PATH,
    ERROR_ANALYSIS_PATH,
    EXECUTION_REGISTRY_CSV_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXPERIMENT_CONFIG_PATHS,
    FIGURE_PATHS,
    GENERATED_PATHS,
    LOADER_CONTRACT_PATH,
    MANIFEST_PATH,
    MODEL_COMPARISON_CSV_PATH,
    MODEL_COMPARISON_JSON_PATH,
    NOTEBOOK_AUDIT_PATH,
    NOTEBOOK_PATH,
    PRETRAINED_GATE_PATH,
    PROJECT_ROOT,
    READINESS,
    ROC_CURVES_PATH,
    SEQUENCE_IDS,
    STATUS_PATH,
    STEP,
    SUITE_CONFIG_PATH,
    SUMMARY_PATH,
    TEXT_HASH_SUFFIXES,
    TEXT_PROFILE_PATH,
    TOKENIZATION_SUMMARY_PATH,
    TRAINING_RUNS_PATH,
    VALIDATION_PREDICTIONS_PATH,
    VOCABULARY_PATH,
    project_relative,
)
from src.verification.full_course_coverage_architecture import (
    build_verification_report as build_architecture_verification,
)
from src.verification.fundamentals_experimental_suite import (
    build_verification_report as build_fundamentals_verification,
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []
    static_paths = (
        SUITE_CONFIG_PATH,
        *EXPERIMENT_CONFIG_PATHS,
        PROJECT_ROOT / "src" / "sequence_suite_config.py",
        PROJECT_ROOT / "src" / "run_sequence_experimental_suite.py",
        PROJECT_ROOT / "src" / "build_sequence_experiment_notebook.py",
        PROJECT_ROOT / "src" / "verification" / "sequence_experimental_suite.py",
        DOCUMENTATION_PATH,
    )
    required_paths = (*static_paths, *GENERATED_PATHS, MANIFEST_PATH)
    checks["structure"] = all(path.is_file() for path in required_paths)
    if not checks["structure"]:
        missing = [project_relative(path) for path in required_paths if not path.is_file()]
        errors.append(f"Missing Step 011.2 artifacts: {missing}.")
        return {"status": "FAIL", "checks": checks, "errors": errors}

    suite_config = read_json(SUITE_CONFIG_PATH)
    experiment_configs = [read_json(path) for path in EXPERIMENT_CONFIG_PATHS]
    status = read_json(STATUS_PATH)
    registry = read_json(EXECUTION_REGISTRY_JSON_PATH)
    notebook_audit = read_json(NOTEBOOK_AUDIT_PATH)
    manifest = read_json(MANIFEST_PATH)
    profile = read_json(TEXT_PROFILE_PATH)
    loader = read_json(LOADER_CONTRACT_PATH)
    tokenization = read_json(TOKENIZATION_SUMMARY_PATH)
    vocabulary = read_json(VOCABULARY_PATH)
    pretrained_gate = read_json(PRETRAINED_GATE_PATH)
    attention = read_json(ATTENTION_EVIDENCE_PATH)
    roc_payload = read_json(ROC_CURVES_PATH)
    comparison_payload = read_json(MODEL_COMPARISON_JSON_PATH)

    checks["configuration"] = (
        suite_config.get("step") == STEP
        and suite_config.get("readiness") == READINESS
        and suite_config.get("test_split_allowed") is False
        and suite_config.get("test_split_path") is None
        and suite_config.get("final_test_evaluation_authorized") is False
        and suite_config.get("pretrained_download_authorized") is False
        and suite_config.get("pretrained_weights_downloaded") is False
        and suite_config.get("completed_problem_numbers") == list(range(1, 10))
        and suite_config.get("deferred_problem_numbers") == [10]
        and [item.get("experiment_id") for item in experiment_configs] == list(SEQUENCE_IDS)
        and all(item.get("test_split_allowed") is False for item in experiment_configs)
        and all(item.get("test_split_path") is None for item in experiment_configs)
        and all(item.get("final_test_evaluation_authorized") is False for item in experiment_configs)
    )

    checks["status_and_locked_boundary"] = (
        status.get("status") == "PASS"
        and status.get("readiness") == READINESS
        and status.get("completed_problem_ids") == list(COMPLETED_SEQUENCE_IDS)
        and status.get("deferred_problem_ids") == list(DEFERRED_SEQUENCE_IDS)
        and status.get("completed_core_problem_count") == 9
        and status.get("training_runs_recorded") == 21
        and status.get("production_final_model_changed") is False
        and status.get("locked_test_csv_files_opened") is False
        and status.get("test_split_used") is False
        and status.get("final_test_evaluation_authorized") is False
        and status.get("pretrained_weights_downloaded") is False
        and status.get("pretrained_extension_status") == "DEFERRED_EXPLICIT_APPROVAL_REQUIRED"
    )

    registry_rows = read_csv(EXECUTION_REGISTRY_CSV_PATH)
    registry_experiments = registry.get("experiments", [])
    checks["execution_registry"] = (
        registry.get("status") == "PASS"
        and registry.get("readiness") == READINESS
        and registry.get("total_training_runs") == 21
        and len(registry_experiments) == 10
        and len(registry_rows) == 10
        and [item.get("experiment_id") for item in registry_experiments] == list(SEQUENCE_IDS)
        and all(item.get("test_split_allowed") is False for item in registry_experiments)
        and all(item.get("final_test_evaluation_authorized") is False for item in registry_experiments)
        and registry_experiments[-1].get("status") == "DEFERRED_EXPLICIT_APPROVAL_REQUIRED"
    )

    checks["loader_and_tokenization"] = (
        profile.get("status") == "PASS"
        and profile.get("train", {}).get("rows") == 180
        and profile.get("validation", {}).get("rows") == 60
        and profile.get("group_overlap") == 0
        and loader.get("status") == "PASS"
        and loader.get("authorized_inputs")
        == ["data/processed/integrated_train.csv", "data/processed/integrated_validation.csv"]
        and loader.get("test_split_used") is False
        and tokenization.get("status") == "PASS"
        and tokenization.get("vocabulary_fit_split") == "train_only"
        and tokenization.get("padding_index") == 0
        and tokenization.get("unknown_index") == 1
        and tokenization.get("train_shape") == [180, suite_config.get("sequence_length")]
        and tokenization.get("validation_shape") == [60, suite_config.get("sequence_length")]
        and vocabulary.get("token_to_index", {}).get("<PAD>") == 0
        and vocabulary.get("token_to_index", {}).get("<UNK>") == 1
    )

    training_rows = read_csv(TRAINING_RUNS_PATH)
    prediction_rows = read_csv(VALIDATION_PREDICTIONS_PATH)
    comparison_rows = read_csv(MODEL_COMPARISON_CSV_PATH)
    families = {row.get("family") for row in comparison_rows}
    checks["models_metrics_and_runs"] = (
        len(training_rows) == 21
        and len(prediction_rows) == 21 * 60
        and len(comparison_rows) == 6
        and families == {"embedding_average", "tfidf_logistic", "textcnn", "gru", "lstm", "transformer"}
        and all(row.get("status") == "COMPLETED" for row in training_rows)
        and all(0.0 <= float(row["validation_accuracy"]) <= 1.0 for row in training_rows)
        and all(0.0 <= float(row["validation_macro_f1"]) <= 1.0 for row in training_rows)
        and all(int(row["run_count"]) >= 3 for row in comparison_rows)
        and comparison_payload.get("status") == "PASS"
        and len(comparison_payload.get("models", [])) == 6
        and len(read_json(CONFUSION_MATRICES_PATH)) == 21
        and roc_payload.get("status") == "PASS"
        and set(roc_payload.get("models", {})) == families
    )

    attention_rows = read_csv(ATTENTION_TOKEN_SUMMARY_PATH)
    checks["attention_evidence"] = (
        attention.get("status") == "PASS"
        and attention.get("head_count") == 2
        and set(attention.get("selection", {})) == {"correct", "incorrect"}
        and all(len(attention["selection"][name].get("heads", [])) == 2 for name in ("correct", "incorrect"))
        and len(attention_rows) >= 8
        and all(int(row["head"]) in {1, 2} for row in attention_rows)
    )

    checks["pretrained_gate"] = (
        pretrained_gate.get("experiment_id") == "SEQ-010"
        and pretrained_gate.get("status") == "DEFERRED_EXPLICIT_APPROVAL_REQUIRED"
        and pretrained_gate.get("approval_received") is False
        and pretrained_gate.get("network_download_attempted") is False
        and pretrained_gate.get("pretrained_weights_downloaded") is False
        and pretrained_gate.get("pretrained_model_loaded") is False
        and pretrained_gate.get("model_identifier") is None
        and pretrained_gate.get("model_revision") is None
        and pretrained_gate.get("model_license") is None
    )

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    checks["executed_notebook"] = (
        notebook_audit.get("status") == "PASS"
        and notebook_audit.get("code_cell_count") == 8
        and notebook_audit.get("executed_code_cell_count") == 8
        and notebook_audit.get("error_output_count") == 0
        and all(cell.get("execution_count") is not None for cell in code_cells)
        and not error_outputs
        and notebook_audit.get("test_split_used") is False
        and notebook_audit.get("pretrained_weights_downloaded") is False
    )

    figure_ok = True
    for path in FIGURE_PATHS.values():
        try:
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            if width < 500 or height < 350 or path.stat().st_size < 5000:
                figure_ok = False
        except OSError:
            figure_ok = False
    checks["visual_evidence"] = figure_ok

    summary_text = SUMMARY_PATH.read_text(encoding="utf-8-sig")
    docs_text = DOCUMENTATION_PATH.read_text(encoding="utf-8-sig")
    checks["documentation"] = (
        all(sequence_id in docs_text for sequence_id in SEQUENCE_IDS)
        and READINESS in summary_text
        and "Test split used: **false**" in summary_text
        and "Pretrained weights downloaded: **false**" in summary_text
        and "explicit approval" in docs_text.lower()
    )

    architecture_report = build_architecture_verification()
    fundamentals_report = build_fundamentals_verification()
    checks["prior_steps_preserved"] = (
        architecture_report.get("status") == "PASS"
        and fundamentals_report.get("status") == "PASS"
    )

    artifacts = manifest.get("artifacts", [])
    expected_manifest_paths = {
        project_relative(path)
        for path in (
            SUITE_CONFIG_PATH,
            *EXPERIMENT_CONFIG_PATHS,
            DOCUMENTATION_PATH,
            *GENERATED_PATHS,
        )
        if path != MANIFEST_PATH
    }
    actual_manifest_paths = {item.get("path") for item in artifacts}
    checks["manifest"] = (
        manifest.get("status") == "PASS"
        and manifest.get("step") == STEP
        and manifest.get("readiness") == READINESS
        and manifest.get("training_runs_recorded") == 21
        and manifest.get("completed_core_problems") == 9
        and manifest.get("deferred_pretrained_problems") == 1
        and manifest.get("test_split_used") is False
        and manifest.get("pretrained_weights_downloaded") is False
        and actual_manifest_paths == expected_manifest_paths
        and manifest.get("artifact_count") == len(artifacts)
    )
    if checks["manifest"]:
        for item in artifacts:
            path = PROJECT_ROOT / item["path"]
            if not path.is_file() or normalized_sha256(path) != item.get("sha256"):
                checks["manifest"] = False
                errors.append(f"Artifact hash differs: {item['path']}.")

    error_rows = read_csv(ERROR_ANALYSIS_PATH)
    checks["error_analysis"] = all(
        row.get("true_label") != row.get("predicted_label") for row in error_rows
    )

    for name, passed in checks.items():
        if not passed and not any(name in error for error in errors):
            errors.append(f"Sequence suite check failed: {name}.")

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()
    print("Transformers & Sequence Modelling Experimental Suite verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
