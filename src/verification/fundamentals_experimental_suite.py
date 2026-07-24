from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import nbformat
from PIL import Image

from src.fundamentals_suite_config import (
    ARCHITECTURE_COMPARISON_PATH,
    BASELINE_DIAGNOSTIC_PATH,
    BATCH_CONTRACT_PATH,
    CAPACITY_COMPARISON_PATH,
    EXECUTION_REGISTRY_CSV_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXPERIMENT_COMPARISON_CSV_PATH,
    EXPERIMENT_CONFIG_PATHS,
    FAILURE_DIAGNOSTICS_PATH,
    FIGURE_PATHS,
    FUNDAMENTALS_IDS,
    GENERATED_PATHS,
    MANIFEST_PATH,
    NOTEBOOK_AUDIT_PATH,
    NOTEBOOK_PATH,
    OPTIMIZER_COMPARISON_PATH,
    OVERFIT_RESULT_PATH,
    PREPROCESSING_COMPARISON_PATH,
    PROJECT_ROOT,
    READINESS,
    STATUS_PATH,
    STEP,
    SUITE_CONFIG_PATH,
    SUMMARY_PATH,
    TEXT_HASH_SUFFIXES,
    TRAINING_LOOP_AUDIT_PATH,
    project_relative,
)
from src.verification.full_course_coverage_architecture import (
    build_verification_report as build_architecture_verification,
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
    import hashlib

    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    static_paths = (
        SUITE_CONFIG_PATH,
        *EXPERIMENT_CONFIG_PATHS,
        PROJECT_ROOT / "src" / "fundamentals_suite_config.py",
        PROJECT_ROOT / "src" / "run_fundamentals_experimental_suite.py",
        PROJECT_ROOT / "src" / "build_fundamentals_experiment_notebook.py",
        PROJECT_ROOT / "docs" / "course_coverage" / "fundamentals_experimental_suite.md",
    )
    required_paths = (*static_paths, *GENERATED_PATHS, MANIFEST_PATH)
    checks["structure"] = all(path.is_file() for path in required_paths)
    if not checks["structure"]:
        missing = [
            project_relative(path) for path in required_paths if not path.is_file()
        ]
        errors.append(f"Missing Step 011.1 artifacts: {missing}.")
        return {"status": "FAIL", "checks": checks, "errors": errors}

    suite_config = read_json(SUITE_CONFIG_PATH)
    experiment_configs = [read_json(path) for path in EXPERIMENT_CONFIG_PATHS]
    status = read_json(STATUS_PATH)
    registry = read_json(EXECUTION_REGISTRY_JSON_PATH)
    notebook_audit = read_json(NOTEBOOK_AUDIT_PATH)
    manifest = read_json(MANIFEST_PATH)
    baseline = read_json(BASELINE_DIAGNOSTIC_PATH)
    overfit = read_json(OVERFIT_RESULT_PATH)
    batch_contract = read_json(BATCH_CONTRACT_PATH)
    loop_audit = read_json(TRAINING_LOOP_AUDIT_PATH)

    checks["configuration"] = (
        suite_config.get("step") == STEP
        and suite_config.get("test_split_allowed") is False
        and suite_config.get("test_split_path") is None
        and suite_config.get("final_test_evaluation_authorized") is False
        and [item.get("experiment_id") for item in experiment_configs]
        == list(FUNDAMENTALS_IDS)
        and all(item.get("test_split_allowed") is False for item in experiment_configs)
        and all(item.get("test_split_path") is None for item in experiment_configs)
        and all(
            item.get("final_test_evaluation_authorized") is False
            for item in experiment_configs
        )
    )

    checks["readiness"] = (
        status.get("status") == "PASS"
        and status.get("readiness") == READINESS
        and status.get("step") == STEP
        and status.get("exercise_problem_count") == 10
        and status.get("completed_exercise_problem_count") == 10
        and status.get("model_training_performed") is True
        and status.get("model_selection_changed") is False
        and status.get("production_final_model_changed") is False
        and status.get("locked_test_csv_files_opened") is False
        and status.get("test_split_used") is False
        and status.get("final_test_evaluation_authorized") is False
        and status.get("notebook_executed") is True
        and status.get("notebook_error_outputs") == 0
    )

    registry_entries = registry.get("experiments", [])
    registry_csv = read_csv(EXECUTION_REGISTRY_CSV_PATH)
    checks["execution_registry"] = (
        registry.get("experiment_count") == 10
        and registry.get("completed_experiment_count") == 10
        and registry.get("execution_status_counts") == {"COMPLETED": 10}
        and [item.get("experiment_id") for item in registry_entries]
        == list(FUNDAMENTALS_IDS)
        and all(item.get("execution_status") == "COMPLETED" for item in registry_entries)
        and all(item.get("test_split_allowed") is False for item in registry_entries)
        and all(item.get("test_split_path") is None for item in registry_entries)
        and all(
            item.get("final_test_evaluation_authorized") is False
            for item in registry_entries
        )
        and [row.get("experiment_id") for row in registry_csv]
        == list(FUNDAMENTALS_IDS)
        and all(row.get("test_split_allowed") == "false" for row in registry_csv)
        and all(row.get("test_split_path") == "" for row in registry_csv)
    )

    checks["data_and_batch_contract"] = (
        batch_contract.get("status") == "COMPLETED"
        and batch_contract.get("image_text_label_alignment_pass") is True
        and batch_contract.get("train_shuffle") is True
        and batch_contract.get("validation_shuffle") is False
        and batch_contract.get("train_shuffle_reproducible") is True
        and batch_contract.get("validation_order_stable") is True
    )
    checks["gradient_and_overfit"] = (
        baseline.get("status") == "COMPLETED"
        and baseline.get("loss_reduced") is True
        and baseline.get("gradient_finite") is True
        and float(baseline.get("weight_change_norm", 0.0)) > 0
        and overfit.get("status") == "COMPLETED"
        and overfit.get("threshold_reached") is True
        and float(overfit.get("final_batch_accuracy", 0.0))
        >= float(overfit.get("accuracy_threshold", 1.0))
    )
    checks["training_loop"] = (
        loop_audit.get("status") == "COMPLETED"
        and loop_audit.get("validation_weights_unchanged_during_evaluate") is True
        and loop_audit.get("train_shuffle") is True
        and loop_audit.get("validation_shuffle") is False
    )

    optimizer_rows = read_csv(OPTIMIZER_COMPARISON_PATH)
    optimizer_names = {row.get("optimizer") for row in optimizer_rows}
    checks["optimizer_and_lr_coverage"] = (
        {"SGD", "RMSprop", "Adam", "AdamW"}.issubset(optimizer_names)
        and any(row.get("schedule") == "exponential_decay" for row in optimizer_rows)
        and len({row.get("learning_rate") for row in optimizer_rows}) >= 4
    )

    capacity_rows = read_csv(CAPACITY_COMPARISON_PATH)
    checks["capacity_coverage"] = (
        {row.get("variant") for row in capacity_rows}
        == {"small", "medium", "large"}
        and all(int(row.get("parameter_count", "0")) > 0 for row in capacity_rows)
    )

    architecture_rows = read_csv(ARCHITECTURE_COMPARISON_PATH)
    expected_architectures = {
        "baseline",
        "l2_regularization",
        "dropout_regularization",
        "batch_normalization",
        "residual_fusion",
        "learning_rate_schedule",
        "cnn_image_branch",
        "pretrained_mobilenet_v2",
    }
    checks["architecture_coverage"] = (
        {row.get("variant") for row in architecture_rows}
        == expected_architectures
        and all(
            row.get("status")
            in {"COMPLETED", "SKIPPED_RESOURCE_UNAVAILABLE"}
            for row in architecture_rows
        )
    )

    preprocessing_rows = read_csv(PREPROCESSING_COMPARISON_PATH)
    checks["preprocessing_coverage"] = {
        row.get("variant") for row in preprocessing_rows
    } == {
        "compact_16_seq8",
        "baseline_24_seq12",
        "larger_32_seq20",
        "grayscale_24_seq12",
    }

    failure_rows = read_csv(FAILURE_DIAGNOSTICS_PATH)
    checks["failure_diagnostics"] = (
        len(failure_rows) == 9
        and {
            "unscaled_images",
            "excessive_learning_rate",
            "tiny_learning_rate",
            "excessive_dropout",
            "misaligned_train_labels",
            "sigmoid_activation",
            "deep_sigmoid_gradient_probe",
            "missing_optimizer_step_probe",
            "validation_training_blocked",
        }
        == {row.get("case") for row in failure_rows}
        and all(row.get("observation") for row in failure_rows)
        and all(row.get("prevention") for row in failure_rows)
    )

    comparison_rows = read_csv(EXPERIMENT_COMPARISON_CSV_PATH)
    checks["metrics_and_ranges"] = (
        len(comparison_rows) >= 30
        and all(row.get("experiment_id") in FUNDAMENTALS_IDS for row in comparison_rows)
        and all(
            row.get("status")
            in {
                "COMPLETED",
                "FAILED_DIAGNOSTIC",
                "SKIPPED_RESOURCE_UNAVAILABLE",
            }
            for row in comparison_rows
        )
        and all(
            not row.get("validation_macro_f1")
            or 0.0 <= float(row["validation_macro_f1"]) <= 1.0
            for row in comparison_rows
        )
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
        and notebook_audit.get("error_output_count") == 0
        and notebook_audit.get("executed_code_cell_count")
        == notebook_audit.get("code_cell_count")
        and len(code_cells) >= 8
        and all(cell.get("execution_count") is not None for cell in code_cells)
        and not error_outputs
    )

    figure_ok = True
    for path in FIGURE_PATHS.values():
        try:
            with Image.open(path) as image:
                image.verify()
        except OSError:
            figure_ok = False
    checks["visual_evidence"] = figure_ok
    summary_text = SUMMARY_PATH.read_text(encoding="utf-8-sig")
    checks["documentation"] = (
        all(experiment_id in summary_text for experiment_id in FUNDAMENTALS_IDS)
        and "Test split used: **false**" in summary_text
        and "Frozen exam model changed: **false**" in summary_text
    )

    architecture_verification = build_architecture_verification()
    checks["step_011_0_preserved"] = (
        architecture_verification.get("status") == "PASS"
        and architecture_verification.get("errors") == []
    )

    source_hashes = manifest.get("source_artifact_sha256", {})
    generated_hashes = manifest.get("generated_artifact_sha256", {})
    checks["manifest"] = (
        manifest.get("status") == "PASS"
        and manifest.get("step") == STEP
        and manifest.get("readiness") == READINESS
        and manifest.get("hash_normalization") == "utf-8-lf"
        and manifest.get("locked_test_csv_files_opened") is False
        and manifest.get("test_split_used") is False
        and manifest.get("final_test_evaluation_authorized") is False
        and set(generated_hashes)
        == {
            project_relative(path)
            for path in GENERATED_PATHS
            if path != MANIFEST_PATH
        }
    )
    if checks["manifest"]:
        for relative_path, expected_hash in {
            **source_hashes,
            **generated_hashes,
        }.items():
            path = PROJECT_ROOT / relative_path
            if not path.is_file() or normalized_sha256(path) != expected_hash:
                checks["manifest"] = False
                errors.append(f"Artifact hash differs: {relative_path}.")

    for name, passed in checks.items():
        if not passed and not any(name in error for error in errors):
            errors.append(f"Fundamentals suite check failed: {name}.")

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()
    print("Deep Learning Fundamentals Experimental Suite verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
