from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat

from src.final_delivery_config import (
    BASE_CHECKPOINT_COMMIT,
    BASE_CHECKPOINT_COMMIT_COUNT,
    CANONICAL_NOTEBOOK_PATH,
    CANONICAL_NOTEBOOK_RELATIVE,
    CHECKLIST_PATH,
    ENTRY_POINT_MAP_PATH,
    EXAM_FIRST_STATUS_PATH,
    EXAM_README_PATH,
    GENERATED_ARTIFACTS,
    IMPLEMENTATION_ARTIFACTS,
    MANIFEST_PATH,
    NOTEBOOK_CATALOGUE_PATH,
    PRIMARY_PREDICTIONS_PATH,
    READINESS,
    ROOT_README_PATH,
    SOURCE_ARCHIVE,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_ARTIFACTS,
    STATUS_PATH,
    STEP,
    TEXT_HASH_SUFFIXES,
    project_relative,
)


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def active_section(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").split(
        "<details>", maxsplit=1
    )[0]


def record(
    checks: dict[str, bool],
    name: str,
    passed: bool,
    errors: list[str],
    message: str,
) -> None:
    checks[name] = bool(passed)
    if not passed:
        errors.append(message)


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    required = set(SOURCE_ARTIFACTS) | set(IMPLEMENTATION_ARTIFACTS) | set(
        GENERATED_ARTIFACTS
    )
    record(
        checks,
        "structure",
        all(path.is_file() for path in required),
        errors,
        "One or more Step 011.6 source, implementation, or delivery artifacts are missing.",
    )

    status = read_json(STATUS_PATH)
    entry_map = read_json(ENTRY_POINT_MAP_PATH)
    manifest = read_json(MANIFEST_PATH)
    exam_status = read_json(EXAM_FIRST_STATUS_PATH)

    record(
        checks,
        "status",
        status.get("step") == STEP
        and status.get("status") == "PASS"
        and status.get("readiness") == READINESS
        and status.get("base_checkpoint_commit") == BASE_CHECKPOINT_COMMIT
        and status.get("base_checkpoint_commit_count")
        == BASE_CHECKPOINT_COMMIT_COUNT
        and status.get("source_archive") == SOURCE_ARCHIVE
        and status.get("source_archive_sha256") == SOURCE_ARCHIVE_SHA256,
        errors,
        "Step 011.6 status does not match the official base checkpoint.",
    )

    root_active = active_section(ROOT_README_PATH)
    notebook_active = active_section(NOTEBOOK_CATALOGUE_PATH)
    exam_readme = EXAM_README_PATH.read_text(encoding="utf-8-sig")
    current_checklist = CHECKLIST_PATH.read_text(encoding="utf-8-sig")
    single_entry_ok = (
        status.get("canonical_entry_point_count") == 1
        and entry_map.get("canonical_entry_point_count") == 1
        and status.get("canonical_submission_notebook")
        == CANONICAL_NOTEBOOK_RELATIVE
        and entry_map.get("canonical_submission_notebook")
        == CANONICAL_NOTEBOOK_RELATIVE
        and CANONICAL_NOTEBOOK_RELATIVE in root_active
        and CANONICAL_NOTEBOOK_RELATIVE in notebook_active
        and CANONICAL_NOTEBOOK_PATH.name in exam_readme
        and CANONICAL_NOTEBOOK_RELATIVE in current_checklist
        and "exactly one canonical submission notebook" in current_checklist.lower()
    )
    record(
        checks,
        "single_entry_point",
        single_entry_ok,
        errors,
        "Current submission documents do not agree on one canonical notebook.",
    )

    historical = entry_map.get("historical_documents", {})
    historical_ok = (
        len(historical) >= 4
        and all(
            item.get("superseded_for_current_submission") is True
            for item in historical.values()
        )
        and status.get("historical_entry_points_are_supplementary") is True
        and "historical supporting evidence" in root_active.lower()
        and "supporting notebooks" in notebook_active.lower()
    )
    record(
        checks,
        "historical_entry_point_isolation",
        historical_ok,
        errors,
        "Historical notebooks or checklists are not clearly isolated from the current entry point.",
    )

    notebook = nbformat.read(CANONICAL_NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
    ]
    figures = [
        output
        for output in outputs
        if output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]
    notebook_status = status.get("notebook_execution", {})
    notebook_ok = (
        20 <= len(notebook.cells) <= 26
        and 8 <= len(code_cells) <= 10
        and all(cell.execution_count is not None for cell in code_cells)
        and len({cell.execution_count for cell in code_cells})
        == len(code_cells)
        and not any(output.get("output_type") == "error" for output in outputs)
        and len(outputs) >= 14
        and len(figures) >= 5
        and notebook_status.get("local_static_html_render") == "PASS"
        and status.get("local_static_render_verified") is True
    )
    record(
        checks,
        "notebook_delivery_render",
        notebook_ok,
        errors,
        "Canonical notebook execution or static-render evidence is incomplete.",
    )

    primary_prediction_hash = normalized_sha256(PRIMARY_PREDICTIONS_PATH)
    model_boundary_ok = (
        status.get("retained_model_slug") == "keras_multimodal"
        and status.get("primary_prediction_artifact")
        == project_relative(PRIMARY_PREDICTIONS_PATH)
        and status.get("primary_prediction_artifact_sha256")
        == primary_prediction_hash
        and status.get("single_prediction_set_used") is True
        and status.get("validation_sample_count") == 60
        and status.get("validation_part_group_count") == 20
        and status.get("validation_correct_count") == 32
        and status.get("validation_error_count") == 28
        and abs(status.get("validation_accuracy", 0.0) - 0.5333333333)
        < 1e-8
        and abs(status.get("validation_macro_f1", 0.0) - 0.5208271299)
        < 1e-6
        and exam_status.get("single_prediction_set_used") is True
    )
    record(
        checks,
        "single_model_evidence",
        model_boundary_ok,
        errors,
        "Delivery status is not bound to the frozen single-model evidence.",
    )

    delivery_code = (
        (Path(__file__).resolve().parents[1] / "build_final_delivery.py")
        .read_text(encoding="utf-8-sig")
        .lower()
    )
    forbidden = (
        "integrated_test.csv",
        "external_test.csv",
        "model.fit(",
        "keras.fit(",
        "load_model(",
    )
    boundary_ok = (
        not any(token in delivery_code for token in forbidden)
        and status.get("model_training_performed") is False
        and status.get("locked_test_csv_files_opened") is False
        and status.get("test_split_used") is False
        and status.get("final_test_evaluation_authorized") is False
        and status.get("production_final_model_changed") is False
        and status.get("model_selection_changed") is False
    )
    record(
        checks,
        "locked_evaluation_boundary",
        boundary_ok,
        errors,
        "Step 011.6 violates the no-training or locked-test boundary.",
    )

    source_hashes = manifest.get("source_artifact_sha256", {})
    implementation_hashes = manifest.get(
        "implementation_artifact_sha256", {}
    )
    generated_hashes = manifest.get("generated_artifact_sha256", {})
    expected_sources = {project_relative(path) for path in SOURCE_ARTIFACTS}
    expected_implementation = {
        project_relative(path) for path in IMPLEMENTATION_ARTIFACTS
    }
    expected_generated = {
        project_relative(path)
        for path in GENERATED_ARTIFACTS
        if path != MANIFEST_PATH
    }
    manifest_ok = (
        manifest.get("step") == STEP
        and manifest.get("status") == "PASS"
        and manifest.get("readiness") == READINESS
        and manifest.get("base_checkpoint_commit") == BASE_CHECKPOINT_COMMIT
        and manifest.get("source_archive_sha256") == SOURCE_ARCHIVE_SHA256
        and manifest.get("canonical_submission_notebook")
        == CANONICAL_NOTEBOOK_RELATIVE
        and manifest.get("canonical_submission_notebook_sha256")
        == normalized_sha256(CANONICAL_NOTEBOOK_PATH)
        and set(source_hashes) == expected_sources
        and set(implementation_hashes) == expected_implementation
        and set(generated_hashes) == expected_generated
        and manifest.get("model_training_performed") is False
        and manifest.get("test_split_used") is False
        and manifest.get("production_final_model_changed") is False
    )
    if manifest_ok:
        for relative_path, expected_hash in {
            **source_hashes,
            **implementation_hashes,
            **generated_hashes,
        }.items():
            path = CANONICAL_NOTEBOOK_PATH.parents[1] / relative_path
            if not path.is_file() or normalized_sha256(path) != expected_hash:
                manifest_ok = False
                errors.append(
                    f"Step {STEP} artifact hash differs: {relative_path}."
                )
    record(
        checks,
        "manifest",
        manifest_ok,
        errors,
        "Step 011.6 manifest integrity failed.",
    )

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
        "summary": {
            "canonical_entry_point_count": status.get(
                "canonical_entry_point_count"
            ),
            "notebook_cells": len(notebook.cells),
            "executed_code_cells": len(code_cells),
            "saved_outputs": len(outputs),
            "saved_figures": len(figures),
            "validation_correct": status.get("validation_correct_count"),
            "validation_errors": status.get("validation_error_count"),
            "remote_github_review_pending": status.get(
                "remote_github_render_review_required_after_push"
            ),
        },
    }


def main() -> None:
    report = build_verification_report()
    print("Final submission lock and delivery readiness verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["status"] != "PASS":
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
