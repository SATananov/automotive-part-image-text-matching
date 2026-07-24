from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.course_coverage_config import (
    ALLOWED_EXECUTION_STATUSES,
    ALLOWED_TRAIN_SPLIT,
    ALLOWED_VALIDATION_SPLIT,
    BASE_COMMIT,
    DEFAULTS_PATH,
    EXECUTION_POLICY_PATH,
    EXPECTED_EXPERIMENT_COUNT,
    EXPECTED_EXPERIMENT_COUNTS,
    EXPECTED_RESOURCE_TIERS,
    GENERATED_PATHS,
    LOCKED_PLAN_PATH,
    MANIFEST_PATH,
    MAPPING_PATH,
    MATRIX_PATH,
    NOTEBOOK_PLAN_PATH,
    PROHIBITED_TEST_INPUTS,
    READINESS,
    READINESS_PATH,
    REGISTRY_CSV_PATH,
    REGISTRY_JSON_PATH,
    RESOURCE_TIERS_PATH,
    SOURCE_PATHS,
    STEP,
    SUMMARY_PATH,
    TEXT_HASH_SUFFIXES,
    project_relative,
)
from src.real_dataset_config import PROJECT_ROOT


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def flatten_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


def write_registry_csv(experiments: list[dict[str, Any]]) -> None:
    REGISTRY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(experiments[0])
    with REGISTRY_CSV_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in experiments:
            writer.writerow(
                {
                    key: flatten_csv_value(experiment.get(key))
                    for key in fieldnames
                }
            )


def validate_source_mapping(mapping: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    experiments = mapping.get("experiments", [])
    if not isinstance(experiments, list):
        return ["Mapping experiments must be a list."]

    identifiers = [item.get("experiment_id") for item in experiments]
    if len(experiments) != EXPECTED_EXPERIMENT_COUNT:
        errors.append(
            "Expected "
            f"{EXPECTED_EXPERIMENT_COUNT} experiments, found "
            f"{len(experiments)}."
        )
    if len(set(identifiers)) != len(identifiers):
        errors.append("Experiment identifiers are not unique.")

    counts = Counter(item.get("course_section") for item in experiments)
    if dict(counts) != EXPECTED_EXPERIMENT_COUNTS:
        errors.append(
            f"Course-section counts differ: {dict(counts)}."
        )

    required_fields = {
        "experiment_id",
        "course_section",
        "exercise_problem_number",
        "exercise_problem_title",
        "exercise_requirement",
        "research_question",
        "hypothesis",
        "model_family",
        "input_modalities",
        "dataset_version",
        "train_split",
        "validation_split",
        "test_split_allowed",
        "test_split_path",
        "final_test_evaluation_authorized",
        "configuration_path",
        "primary_metric",
        "secondary_metrics",
        "resource_tier",
        "expected_outputs",
        "execution_status",
        "selection_eligible",
        "result_summary",
        "evidence_paths",
        "prerequisites",
        "safety_notes",
    }
    for experiment in experiments:
        experiment_id = experiment.get("experiment_id", "<missing>")
        missing = sorted(required_fields - set(experiment))
        if missing:
            errors.append(f"{experiment_id} missing fields: {missing}.")
        if experiment.get("train_split") != ALLOWED_TRAIN_SPLIT:
            errors.append(f"{experiment_id} has an unexpected train split.")
        if experiment.get("validation_split") != ALLOWED_VALIDATION_SPLIT:
            errors.append(
                f"{experiment_id} has an unexpected validation split."
            )
        if experiment.get("test_split_allowed") is not False:
            errors.append(f"{experiment_id} allows test access.")
        if experiment.get("test_split_path") is not None:
            errors.append(f"{experiment_id} declares a test path.")
        if experiment.get("final_test_evaluation_authorized") is not False:
            errors.append(f"{experiment_id} authorizes final test use.")
        if experiment.get("execution_status") != "PLANNED":
            errors.append(f"{experiment_id} claims execution results.")
        if experiment.get("result_summary") is not None:
            errors.append(f"{experiment_id} contains a result summary.")
        if experiment.get("resource_tier") not in EXPECTED_RESOURCE_TIERS:
            errors.append(f"{experiment_id} has an unknown resource tier.")
        if experiment.get("execution_status") not in ALLOWED_EXECUTION_STATUSES:
            errors.append(f"{experiment_id} has an invalid status.")
        if not experiment.get("primary_metric"):
            errors.append(f"{experiment_id} lacks a primary metric.")
        if not experiment.get("expected_outputs"):
            errors.append(f"{experiment_id} lacks expected outputs.")
        if not experiment.get("evidence_paths"):
            errors.append(f"{experiment_id} lacks evidence paths.")

        serialized = json.dumps(experiment, ensure_ascii=False)
        for test_path in PROHIBITED_TEST_INPUTS:
            if test_path in serialized:
                errors.append(
                    f"{experiment_id} embeds prohibited test input {test_path}."
                )

    vis_007 = next(
        (
            item
            for item in experiments
            if item.get("experiment_id") == "VIS-007"
        ),
        {},
    )
    annotation_text = json.dumps(vis_007, ensure_ascii=False).lower()
    if "two real independent human annotators" not in annotation_text:
        errors.append("VIS-007 lacks the independent-annotator prerequisite.")
    if "do not simulate annotators" not in annotation_text:
        errors.append("VIS-007 lacks the no-simulated-agreement safeguard.")

    lock = mapping.get("global_test_lock", {})
    if lock.get("test_split_used") is not False:
        errors.append("Global test lock reports test use.")
    if lock.get("final_test_evaluation_authorized") is not False:
        errors.append("Global test lock authorizes final evaluation.")
    if set(lock.get("prohibited_test_inputs", [])) != PROHIBITED_TEST_INPUTS:
        errors.append("Global prohibited test inputs differ from policy.")

    return errors


def build_registry(mapping: dict[str, Any]) -> dict[str, Any]:
    experiments = mapping["experiments"]
    counts = Counter(item["course_section"] for item in experiments)
    return {
        "schema_version": "1.0",
        "step": STEP,
        "base_commit": BASE_COMMIT,
        "title": mapping["title"],
        "readiness": READINESS,
        "exercise_sources": mapping["exercise_sources"],
        "experiment_count": len(experiments),
        "course_section_counts": dict(counts),
        "execution_status_counts": dict(
            Counter(item["execution_status"] for item in experiments)
        ),
        "selection_eligible_count": sum(
            bool(item["selection_eligible"]) for item in experiments
        ),
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "experiments": experiments,
    }


def build_matrix_lines(registry: dict[str, Any]) -> list[str]:
    lines = [
        "# Full Course Exercise Coverage Matrix",
        "",
        f"- Step: **{STEP}**",
        f"- Base checkpoint: `{BASE_COMMIT}`",
        f"- Exercise tasks mapped: **{registry['experiment_count']}**",
        f"- Readiness: `{READINESS}`",
        "- This checkpoint defines experiments; it does not claim that the new experiments have run.",
        "- The locked test split remains unavailable for training, tuning, comparison, and model selection.",
        "",
        "## Exercise sources",
        "",
    ]
    for source in registry["exercise_sources"]:
        lines.extend(
            [
                f"- **{source['course_section']}** — "
                f"`{source['notebook_name']}`; "
                f"{source['problem_count']} problems; "
                f"notebook SHA-256 `{source['notebook_sha256']}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## Requirement-to-experiment mapping",
            "",
            "| ID | Exercise problem | Requirement | Current evidence | Planned evidence | Tier | Selection eligible |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for item in registry["experiments"]:
        planned = ", ".join(item["evidence_paths"])
        requirement = item["exercise_requirement"].replace("|", "/")
        lines.append(
            f"| `{item['experiment_id']}` | "
            f"{item['course_section']} P{item['exercise_problem_number']}: "
            f"{item['exercise_problem_title']} | {requirement} | "
            f"{item['baseline_coverage']} | {planned} | "
            f"{item['resource_tier']} | "
            f"{'yes' if item['selection_eligible'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`EXISTING_EVIDENCE_EXTEND` means the current project already contains relevant evidence but Step 011 will make the exercise requirement explicit.",
            "`PARTIAL_EXTENSION` means the current project covers part of the requirement and needs a controlled comparison or additional report.",
            "`NEW_EXPERIMENT` means the requirement needs a new validation-only experiment.",
            "`NEW_EXPERIMENT_REQUIRES_HUMAN_DATA` cannot be completed until genuine independent annotations exist.",
        ]
    )
    return lines


def build_locked_plan_lines() -> list[str]:
    return [
        "# Locked Evaluation Plan",
        "",
        f"- Step: **{STEP}**",
        f"- Base checkpoint: `{BASE_COMMIT}`",
        f"- Readiness target: `{READINESS}`",
        "",
        "## Data boundary",
        "",
        "Training input is limited to `data/processed/integrated_train.csv`.",
        "Model comparison and selection are limited to `data/processed/integrated_validation.csv`.",
        "The following files are prohibited experiment inputs:",
        "",
        "- `data/processed/integrated_test.csv`",
        "- `data/external/integrated/external_test.csv`",
        "",
        "The registry therefore records `test_split_allowed: false`, `test_split_path: null`, and `final_test_evaluation_authorized: false` for every experiment.",
        "",
        "## Selection policy",
        "",
        "Validation Macro F1 is the default selection metric. Ties are resolved by seed stability, parameter count, inference time, and architectural simplicity in that order.",
        "Accuracy, class-wise precision/recall/F1, confusion matrices, ROC AUC where meaningful, training time, inference time, and parameter counts remain supporting evidence.",
        "A result is not selection-eligible when it is a failure diagnostic, explainability-only analysis, annotation protocol, or synthesis report.",
        "",
        "## Fair-comparison rules",
        "",
        "1. Use the committed grouped train and validation split without regrouping after seeing results.",
        "2. Fit vocabularies, scalers, feature extractors, and thresholds on train data only unless a threshold is explicitly selected on validation and documented.",
        "3. Keep split, preprocessing, metrics, and seed policy fixed when comparing architectures.",
        "4. Record resolved configuration, software versions, parameter counts, timing, and saved evidence.",
        "5. Do not promote a model from a single favorable seed when the experiment is marked as a retained comparison.",
        "",
        "## Closed final-test gate",
        "",
        "Step 011.0 does not authorize final test evaluation. Any future one-shot evaluation requires a separate explicit checkpoint after the model recipe, preprocessing, thresholds, metrics, and reporting procedure are frozen. Until that checkpoint, no test metrics may be generated, inspected, or reported.",
    ]


def build_execution_policy_lines(
    registry: dict[str, Any], tiers: dict[str, Any]
) -> list[str]:
    lines = [
        "# Experiment Execution Policy",
        "",
        f"- Step: **{STEP}**",
        f"- Planned experiments: **{registry['experiment_count']}**",
        "- Current execution status: all entries are `PLANNED`.",
        "",
        "## Lifecycle",
        "",
        "`PLANNED → READY → RUNNING → COMPLETED → RETAINED or REJECTED`",
        "",
        "`FAILED_DIAGNOSTIC` is used when a deliberately small or broken configuration provides useful failure evidence but must never participate in model selection.",
        "",
        "## Resource gates",
        "",
    ]
    for name, payload in tiers["tiers"].items():
        approval = "yes" if payload["operator_approval_required"] else "no"
        lines.append(
            f"- **{name}** — {payload['purpose']} Operator approval: {approval}."
        )
    lines.extend(
        [
            "",
            "Tier 3 and Tier 4 runs may begin only after cheaper structural and diagnostic gates pass. Pretrained assets require a recorded source, license, architecture name, and exact revision.",
            "",
            "## Evidence contract",
            "",
            "Every executed experiment must save its resolved configuration, metrics, timing, parameter counts, test-lock assertion, and the report or notebook cells listed in the registry.",
            "No result may be entered in the registry before its artifacts exist and pass verification.",
            "",
            "## Human annotation safeguard",
            "",
            "VIS-007 requires at least two genuine independent human annotators. Confidence must be recorded before adjudication. Synthetic annotators, inferred agreement, or duplicated single-author labels must not be presented as human agreement evidence.",
            "",
            "## Failure experiments",
            "",
            "Intentional faults use copied train-only data in isolated configurations. They must not overwrite canonical metadata, validation artifacts, model-selection tables, or any locked-test file.",
        ]
    )
    return lines


def build_notebook_plan_lines() -> list[str]:
    return [
        "# Course Coverage Notebooks",
        "",
        "Step 011.0 reserves a modular notebook architecture. The notebooks are created and executed in later Step 011 substeps; no new experimental result is claimed here.",
        "",
        "| Planned notebook | Scope |",
        "|---|---|",
        "| `01_fundamentals_experiments.ipynb` | FND-001 through FND-010 |",
        "| `02_sequence_model_comparison.ipynb` | SEQ-001 through SEQ-010 |",
        "| `03_vision_model_comparison.ipynb` | VIS-001, VIS-002, VIS-004, VIS-006, VIS-008 |",
        "| `04_scoring_ranking_explainability.ipynb` | VIS-003, VIS-005, VIS-007, VIS-009 |",
        "| `05_course_coverage_synthesis.ipynb` | Cross-model comparison and course-coverage conclusions |",
        "",
        "All notebooks must use train and validation evidence only. The final test gate remains closed.",
    ]


def build_readiness(
    mapping: dict[str, Any],
    registry: dict[str, Any],
    defaults: dict[str, Any],
    tiers: dict[str, Any],
) -> dict[str, Any]:
    errors = validate_source_mapping(mapping)
    checks = {
        "source_mapping": not errors,
        "all_exercise_tasks_mapped": (
            registry["experiment_count"] == EXPECTED_EXPERIMENT_COUNT
            and registry["course_section_counts"]
            == EXPECTED_EXPERIMENT_COUNTS
        ),
        "unique_experiment_ids": len(
            {
                item["experiment_id"]
                for item in registry["experiments"]
            }
        )
        == EXPECTED_EXPERIMENT_COUNT,
        "planned_status_only": registry["execution_status_counts"]
        == {"PLANNED": EXPECTED_EXPERIMENT_COUNT},
        "metrics_and_evidence_defined": all(
            item["primary_metric"]
            and item["secondary_metrics"]
            and item["expected_outputs"]
            and item["evidence_paths"]
            for item in registry["experiments"]
        ),
        "resource_tiers_complete": (
            set(tiers["tiers"]) == EXPECTED_RESOURCE_TIERS
            and all(
                item["resource_tier"] in tiers["tiers"]
                for item in registry["experiments"]
            )
        ),
        "test_split_locked": all(
            item["test_split_allowed"] is False
            and item["test_split_path"] is None
            and item["final_test_evaluation_authorized"] is False
            for item in registry["experiments"]
        ),
        "selection_uses_validation_only": (
            defaults["allowed_training_inputs"] == [ALLOWED_TRAIN_SPLIT]
            and defaults["allowed_selection_inputs"]
            == [ALLOWED_VALIDATION_SPLIT]
            and set(defaults["prohibited_inputs"])
            == PROHIBITED_TEST_INPUTS
        ),
        "no_results_fabricated": all(
            item["execution_status"] == "PLANNED"
            and item["result_summary"] is None
            for item in registry["experiments"]
        ),
        "human_annotation_safeguard": any(
            item["experiment_id"] == "VIS-007"
            and any(
                "two real independent human annotators" in prerequisite
                for prerequisite in item["prerequisites"]
            )
            and any(
                "Do not simulate annotators" in note
                for note in item["safety_notes"]
            )
            for item in registry["experiments"]
        ),
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "readiness": READINESS if passed else "COURSE_COVERAGE_NOT_READY",
        "step": STEP,
        "base_commit": BASE_COMMIT,
        "experiment_count": registry["experiment_count"],
        "course_section_counts": registry["course_section_counts"],
        "selection_eligible_count": registry[
            "selection_eligible_count"
        ],
        "checks": checks,
        "errors": errors,
        "model_training_performed": False,
        "model_selection_changed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }


def build_summary_lines(readiness: dict[str, Any]) -> list[str]:
    return [
        "# Step 011.0 Summary",
        "",
        f"- Status: **{readiness['status']}**",
        f"- Readiness: `{readiness['readiness']}`",
        f"- Exercise tasks mapped: **{readiness['experiment_count']}**",
        f"- Selection-eligible planned comparisons: **{readiness['selection_eligible_count']}**",
        "- Model training performed: **no**",
        "- Model selection changed: **no**",
        "- Locked test CSV files opened: **no**",
        "- Test split used: **no**",
        "- Final test evaluation authorized: **no**",
        "",
        "This checkpoint establishes the complete architecture, experiment registry, evidence contract, resource gates, and validation-only selection policy for applying all tasks from the three supplied course exercises to the automotive image-text matching project.",
    ]


def build_manifest() -> dict[str, Any]:
    source_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in SOURCE_PATHS
    }
    output_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in GENERATED_PATHS
    }
    return {
        "status": "PASS",
        "step": STEP,
        "base_commit": BASE_COMMIT,
        "hash_normalization": "utf-8-lf",
        "source_artifact_sha256": source_hashes,
        "generated_artifact_sha256": output_hashes,
        "source_artifact_count": len(source_hashes),
        "generated_artifact_count": len(output_hashes),
        "model_training_performed": False,
        "model_selection_changed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }


def main() -> None:
    mapping = read_json(MAPPING_PATH)
    defaults = read_json(DEFAULTS_PATH)
    tiers = read_json(RESOURCE_TIERS_PATH)

    source_errors = validate_source_mapping(mapping)
    if source_errors:
        for error in source_errors:
            print(f"- {error}")
        raise SystemExit(1)

    registry = build_registry(mapping)
    write_json(REGISTRY_JSON_PATH, registry)
    write_registry_csv(registry["experiments"])
    write_markdown(MATRIX_PATH, build_matrix_lines(registry))
    write_markdown(LOCKED_PLAN_PATH, build_locked_plan_lines())
    write_markdown(
        EXECUTION_POLICY_PATH,
        build_execution_policy_lines(registry, tiers),
    )
    write_markdown(NOTEBOOK_PLAN_PATH, build_notebook_plan_lines())

    readiness = build_readiness(mapping, registry, defaults, tiers)
    write_json(READINESS_PATH, readiness)
    write_markdown(SUMMARY_PATH, build_summary_lines(readiness))
    write_json(MANIFEST_PATH, build_manifest())

    print("Full course coverage architecture and locked evaluation plan")
    print(
        f"- mapped exercise tasks: {readiness['experiment_count']}/"
        f"{EXPECTED_EXPERIMENT_COUNT}"
    )
    print("- model training performed: no")
    print("- test split used: no")
    print("- final test evaluation authorized: no")
    print(f"Status: {readiness['status']}")
    if readiness["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
