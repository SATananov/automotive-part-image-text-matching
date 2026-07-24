from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat

from src.vision_suite_config import (
    ANNOTATION_REVIEW_PATH,
    AUGMENTATION_COMPARISON_PATH,
    AUGMENTATION_POLICIES,
    AUGMENTATION_RUNS_PATH,
    COMPATIBILITY_COMPARISON_PATH,
    COMPATIBILITY_PREDICTIONS_PATH,
    COMPATIBILITY_RUNS_PATH,
    COMPLETED_VISION_IDS,
    DEFERRED_VISION_IDS,
    EQUAL_PAIR_EVALUATION_PATH,
    EXECUTION_REGISTRY_JSON_PATH,
    EXPERIMENT_CONFIG_PATHS,
    EXPLAINABILITY_SUMMARY_PATH,
    FAILURE_AUGMENTATION_MATRIX_PATH,
    FINE_TUNING_GATE_PATH,
    HUMAN_ANNOTATION_GATE_PATH,
    IMAGE_INVENTORY_PATH,
    IMAGE_PROFILE_PATH,
    LOCK_FLAGS,
    MANIFEST_PATH,
    NOTEBOOK_AUDIT_PATH,
    OCCLUSION_FIGURE_PATHS,
    OCCLUSION_RESULTS_PATH,
    PRETRAINED_BACKBONE_GATE_PATH,
    PROJECT_ROOT,
    RANDOM_SEEDS,
    RANKING_METRICS_PATH,
    RANKING_TRIPLETS_PATH,
    READINESS,
    REGION_PERTURBATION_PATH,
    REPRESENTATION_COMPARISON_PATH,
    REPRESENTATION_RUNS_PATH,
    REPRESENTATIONS,
    REPORT_ROOT,
    RESOLUTIONS,
    SCORING_NOTEBOOK_PATH,
    STATUS_PATH,
    SUITE_CONFIG_PATH,
    TEXT_HASH_SUFFIXES,
    VISION_IDS,
    VISION_NOTEBOOK_PATH,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalized_sha256(path: Path) -> str:
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = path.read_text(encoding="utf-8-sig")
        payload = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    else:
        payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _pass(checks: dict[str, bool], name: str, condition: bool, errors: list[str], message: str) -> None:
    checks[name] = bool(condition)
    if not condition:
        errors.append(message)


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    required_paths = [
        SUITE_CONFIG_PATH,
        *EXPERIMENT_CONFIG_PATHS,
        STATUS_PATH,
        MANIFEST_PATH,
        IMAGE_PROFILE_PATH,
        IMAGE_INVENTORY_PATH,
        ANNOTATION_REVIEW_PATH,
        REPRESENTATION_RUNS_PATH,
        REPRESENTATION_COMPARISON_PATH,
        AUGMENTATION_RUNS_PATH,
        AUGMENTATION_COMPARISON_PATH,
        FAILURE_AUGMENTATION_MATRIX_PATH,
        COMPATIBILITY_RUNS_PATH,
        COMPATIBILITY_COMPARISON_PATH,
        COMPATIBILITY_PREDICTIONS_PATH,
        RANKING_METRICS_PATH,
        RANKING_TRIPLETS_PATH,
        EQUAL_PAIR_EVALUATION_PATH,
        OCCLUSION_RESULTS_PATH,
        REGION_PERTURBATION_PATH,
        EXPLAINABILITY_SUMMARY_PATH,
        PRETRAINED_BACKBONE_GATE_PATH,
        FINE_TUNING_GATE_PATH,
        HUMAN_ANNOTATION_GATE_PATH,
        NOTEBOOK_AUDIT_PATH,
        EXECUTION_REGISTRY_JSON_PATH,
        VISION_NOTEBOOK_PATH,
        SCORING_NOTEBOOK_PATH,
        *OCCLUSION_FIGURE_PATHS,
    ]
    missing = [path for path in required_paths if not path.is_file()]
    _pass(
        checks,
        "structure",
        not missing,
        errors,
        "Missing vision artifacts: " + ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in missing),
    )
    if missing:
        return {"status": "FAIL", "checks": checks, "errors": errors}

    suite = read_json(SUITE_CONFIG_PATH)
    configs = [read_json(path) for path in EXPERIMENT_CONFIG_PATHS]
    status = read_json(STATUS_PATH)
    profile = read_json(IMAGE_PROFILE_PATH)
    ranking = read_json(RANKING_METRICS_PATH)
    explainability = read_json(EXPLAINABILITY_SUMMARY_PATH)
    pretrained_gate = read_json(PRETRAINED_BACKBONE_GATE_PATH)
    fine_tuning_gate = read_json(FINE_TUNING_GATE_PATH)
    human_gate = read_json(HUMAN_ANNOTATION_GATE_PATH)
    registry = read_json(EXECUTION_REGISTRY_JSON_PATH)
    notebook_audit = read_json(NOTEBOOK_AUDIT_PATH)

    _pass(
        checks,
        "configuration",
        suite.get("readiness") == READINESS
        and [item.get("experiment_id") for item in configs] == list(VISION_IDS)
        and suite.get("completed_problem_numbers") == [1, 3, 4, 5, 8, 9]
        and suite.get("deferred_problem_numbers") == [2, 6, 7],
        errors,
        "Vision suite configuration or experiment IDs are inconsistent.",
    )

    config_lock_ok = all(
        item.get("test_split_allowed") is False
        and item.get("test_split_path") is None
        and item.get("final_test_evaluation_authorized") is False
        for item in configs
    )
    _pass(
        checks,
        "config_test_lock",
        config_lock_ok,
        errors,
        "One or more vision experiment configs violate the locked test boundary.",
    )

    _pass(
        checks,
        "status",
        status.get("status") == "PASS"
        and status.get("readiness") == READINESS
        and status.get("completed_problem_count") == len(COMPLETED_VISION_IDS)
        and status.get("deferred_problem_count") == len(DEFERRED_VISION_IDS)
        and status.get("training_runs_recorded") == 48
        and status.get("model_training_performed") is True,
        errors,
        "Vision status does not record six completed problems, three gates, and 48 runs.",
    )

    lock_ok = all(status.get(key) is value for key, value in LOCK_FLAGS.items())
    _pass(
        checks,
        "status_test_lock",
        lock_ok,
        errors,
        "Vision status lock flags differ from the closed evaluation policy.",
    )

    inventory_rows = read_csv(IMAGE_INVENTORY_PATH)
    review_rows = read_csv(ANNOTATION_REVIEW_PATH)
    _pass(
        checks,
        "image_profile",
        profile.get("status") == "PASS"
        and profile.get("unique_image_count") == 80
        and profile.get("train_unique_image_count") == 60
        and profile.get("validation_unique_image_count") == 20
        and profile.get("category_count") == 10
        and len(inventory_rows) == 80
        and len(review_rows) == 80,
        errors,
        "VIS-001 image profile does not cover 80 unique train/validation images and 10 categories.",
    )

    representation_rows = read_csv(REPRESENTATION_RUNS_PATH)
    representation_comparison = read_csv(REPRESENTATION_COMPARISON_PATH)
    _pass(
        checks,
        "representation_runs",
        len(representation_rows) == len(REPRESENTATIONS) * len(RESOLUTIONS) * len(RANDOM_SEEDS)
        and {row["representation"] for row in representation_rows} == set(REPRESENTATIONS)
        and {int(row["resolution"]) for row in representation_rows} == set(RESOLUTIONS)
        and all(row["status"] == "COMPLETED" for row in representation_rows)
        and len(representation_comparison) == len(REPRESENTATIONS) * len(RESOLUTIONS)
        and sum(row["selected_configuration"].lower() == "true" for row in representation_comparison) == 1,
        errors,
        "VIS-004 representation/resolution run matrix is incomplete.",
    )

    augmentation_rows = read_csv(AUGMENTATION_RUNS_PATH)
    augmentation_comparison = read_csv(AUGMENTATION_COMPARISON_PATH)
    failure_matrix = read_csv(FAILURE_AUGMENTATION_MATRIX_PATH)
    _pass(
        checks,
        "augmentation_runs",
        len(augmentation_rows) == len(AUGMENTATION_POLICIES) * len(RANDOM_SEEDS)
        and {row["augmentation_policy"] for row in augmentation_rows} == set(AUGMENTATION_POLICIES)
        and len(augmentation_comparison) == len(AUGMENTATION_POLICIES)
        and len(failure_matrix) == 10
        and sum(row["selected_configuration"].lower() == "true" for row in augmentation_comparison) == 1,
        errors,
        "VIS-008 augmentation ablation or failure matrix is incomplete.",
    )

    compatibility_rows = read_csv(COMPATIBILITY_RUNS_PATH)
    compatibility_comparison = read_csv(COMPATIBILITY_COMPARISON_PATH)
    predictions = read_csv(COMPATIBILITY_PREDICTIONS_PATH)
    _pass(
        checks,
        "compatibility_runs",
        len(compatibility_rows) == 6
        and len(compatibility_comparison) == 2
        and len(predictions) == 60
        and {row["strategy"] for row in compatibility_rows}
        == {"ordinal_ridge", "class_probability_expected_score"}
        and sum(row["selected_strategy"].lower() == "true" for row in compatibility_comparison) == 1,
        errors,
        "VIS-003 compatibility evidence is incomplete.",
    )

    _pass(
        checks,
        "ranking",
        ranking.get("status") == "PASS"
        and 0.0 <= float(ranking.get("pairwise_ranking_accuracy", -1)) <= 1.0
        and 0.0 <= float(ranking.get("three_way_ordering_accuracy", -1)) <= 1.0
        and ranking.get("scalar_score_guarantees_antisymmetric_pair_differences") is True
        and ranking.get("scalar_score_guarantees_transitive_ordering") is True
        and ranking.get("equal_pair_threshold_fit_split") == "train_only"
        and len(read_csv(RANKING_TRIPLETS_PATH)) == 20
        and len(read_csv(EQUAL_PAIR_EVALUATION_PATH)) == 30,
        errors,
        "VIS-009 ranking metrics or tables are invalid.",
    )

    _pass(
        checks,
        "explainability",
        explainability.get("status") == "PASS"
        and explainability.get("selected_example_count") == 8
        and explainability.get("occlusion_evaluation_count") == 72
        and explainability.get("manual_plausible_region_review_rate") is None
        and explainability.get("manual_review_claimed") is False
        and len(read_csv(OCCLUSION_RESULTS_PATH)) == 72
        and len(read_csv(REGION_PERTURBATION_PATH)) == 8,
        errors,
        "VIS-005 occlusion evidence is incomplete or claims unsupported human review.",
    )

    _pass(
        checks,
        "controlled_gates",
        pretrained_gate.get("status") == "DEFERRED_EXPLICIT_APPROVAL_REQUIRED"
        and pretrained_gate.get("network_download_attempted") is False
        and pretrained_gate.get("pretrained_weights_downloaded") is False
        and fine_tuning_gate.get("fine_tuning_performed") is False
        and human_gate.get("independent_annotator_count") == 0
        and human_gate.get("human_agreement_computed") is False
        and human_gate.get("synthetic_human_agreement_reported") is False,
        errors,
        "VIS-002, VIS-006, or VIS-007 controlled gate is not closed.",
    )

    registry_entries = registry.get("entries", [])
    _pass(
        checks,
        "execution_registry",
        registry.get("status") == "PASS"
        and registry.get("total_training_runs") == 48
        and len(registry_entries) == 9
        and {entry["experiment_id"] for entry in registry_entries} == set(VISION_IDS),
        errors,
        "Vision execution registry is inconsistent.",
    )

    evidence_indexes = [REPORT_ROOT / experiment_id / "README.md" for experiment_id in VISION_IDS]
    _pass(
        checks,
        "evidence_indexes",
        all(path.is_file() and path.stat().st_size > 100 for path in evidence_indexes),
        errors,
        "One or more VIS-001 through VIS-009 evidence indexes are missing.",
    )

    notebook_paths = [VISION_NOTEBOOK_PATH, SCORING_NOTEBOOK_PATH]
    notebooks_ok = True
    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if not code_cells or any(cell.execution_count is None for cell in code_cells):
            notebooks_ok = False
        if any(
            output.get("output_type") == "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        ):
            notebooks_ok = False
    _pass(
        checks,
        "notebooks",
        notebooks_ok
        and notebook_audit.get("status") == "PASS"
        and len(notebook_audit.get("notebooks", [])) == 2,
        errors,
        "Vision notebooks are not fully executed or contain error outputs.",
    )

    manifest = read_json(MANIFEST_PATH)
    manifest_errors: list[str] = []
    artifacts = manifest.get("artifacts", [])
    for artifact in artifacts:
        path = PROJECT_ROOT / artifact["path"]
        if not path.is_file():
            manifest_errors.append(f"missing:{artifact['path']}")
        elif normalized_sha256(path) != artifact["sha256"]:
            manifest_errors.append(f"hash:{artifact['path']}")
    _pass(
        checks,
        "manifest",
        manifest.get("status") == "PASS"
        and manifest.get("readiness") == READINESS
        and manifest.get("training_runs_recorded") == 48
        and manifest.get("artifact_count") == len(artifacts)
        and not manifest_errors,
        errors,
        "Vision manifest integrity failed: " + ", ".join(manifest_errors),
    )

    status_value = "PASS" if all(checks.values()) else "FAIL"
    return {
        "status": status_value,
        "checks": checks,
        "errors": errors,
        "summary": {
            "completed_problems": len(COMPLETED_VISION_IDS),
            "deferred_gates": len(DEFERRED_VISION_IDS),
            "training_runs": 48,
            "evidence_indexes": 9,
            "manifest_artifacts": len(artifacts),
        },
    }


def main() -> None:
    report = build_verification_report()
    print("Vision experimental suite verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Status: {report['status']}")
    if report["status"] != "PASS":
        for error in report["errors"]:
            print(f"- {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
