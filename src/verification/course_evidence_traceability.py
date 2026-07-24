from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = (
    PROJECT_ROOT
    / "configs"
    / "course_coverage"
    / "course_exercise_mapping.json"
)
STATUS_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "course_coverage"
    / "CURRENT_STATUS.md"
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def completed_course_ids() -> set[str]:
    fundamentals = read_json(
        PROJECT_ROOT
        / "data"
        / "experiment_registry"
        / "fundamentals_execution_registry.json"
    )
    sequence = read_json(
        PROJECT_ROOT
        / "data"
        / "experiment_registry"
        / "sequence_execution_registry.json"
    )
    return {
        item["experiment_id"] for item in fundamentals["experiments"]
    } | {
        item["experiment_id"] for item in sequence["experiments"]
    }


def repo_paths_from_index(index_path: Path) -> list[str]:
    text = index_path.read_text(encoding="utf-8-sig")
    candidates = re.findall(r"`([^`]+)`", text)
    return [
        candidate
        for candidate in candidates
        if candidate.startswith(
            ("configs/", "data/", "notebooks/", "reports/")
        )
    ]


def build_verification_report() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    mapping = read_json(MAPPING_PATH)
    by_id = {
        item["experiment_id"]: item
        for item in mapping.get("experiments", [])
    }
    completed_ids = completed_course_ids()
    expected_ids = {
        *(f"FND-{number:03d}" for number in range(1, 11)),
        *(f"SEQ-{number:03d}" for number in range(1, 11)),
    }

    checks["completed_suite_scope"] = completed_ids == expected_ids
    if not checks["completed_suite_scope"]:
        errors.append("Completed suite scope differs from FND/SEQ 001-010.")

    index_count = 0
    referenced_count = 0
    indexes_ok = True
    references_ok = True
    for experiment_id in sorted(completed_ids):
        experiment = by_id.get(experiment_id)
        if not experiment:
            indexes_ok = False
            errors.append(f"Mapping entry missing: {experiment_id}.")
            continue

        directory_paths = [
            path
            for path in experiment.get("evidence_paths", [])
            if path.endswith("/")
        ]
        if len(directory_paths) != 1:
            indexes_ok = False
            errors.append(
                f"Expected one evidence directory for {experiment_id}."
            )
            continue

        index_path = PROJECT_ROOT / directory_paths[0] / "README.md"
        if not index_path.is_file():
            indexes_ok = False
            errors.append(f"Evidence index missing: {experiment_id}.")
            continue

        index_count += 1
        referenced_paths = repo_paths_from_index(index_path)
        if not referenced_paths:
            references_ok = False
            errors.append(
                f"Evidence index has no repository paths: {experiment_id}."
            )
            continue

        for relative_path in referenced_paths:
            referenced_count += 1
            if not (PROJECT_ROOT / relative_path).exists():
                references_ok = False
                errors.append(
                    f"Referenced evidence missing for {experiment_id}: "
                    f"{relative_path}."
                )

    checks["evidence_indexes"] = indexes_ok and index_count == 20
    checks["evidence_references"] = references_ok and referenced_count > 0

    status_text = STATUS_PATH.read_text(encoding="utf-8-sig")
    checks["notebook_status"] = all(
        token in status_text
        for token in (
            "COMPLETED AND EXECUTED — Step 011.1",
            "CORE COMPLETED AND EXECUTED — Step 011.2",
            "SEQ-010 pretrained-transformer download remains behind explicit approval",
            "The locked test split remains unopened",
        )
    )
    if not checks["notebook_status"]:
        errors.append("Current notebook status is incomplete or stale.")

    checks["test_lock_statement"] = all(
        phrase in status_text
        for phrase in (
            "train and validation evidence only",
            "final test evaluation remains unauthorized",
        )
    )
    if not checks["test_lock_statement"]:
        errors.append("Notebook status does not preserve the test-lock contract.")

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "errors": errors,
        "evidence_index_count": index_count,
        "referenced_evidence_path_count": referenced_count,
        "model_training_performed": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
    }


def main() -> None:
    report = build_verification_report()
    print("Course evidence traceability verification")
    for name, passed in report["checks"].items():
        print(f"- {name}: {'PASS' if passed else 'FAIL'}")
    print(f"- evidence_index_count: {report['evidence_index_count']}")
    print(
        "- referenced_evidence_path_count: "
        f"{report['referenced_evidence_path_count']}"
    )
    print(f"Status: {report['status']}")
    for error in report["errors"]:
        print(f"- {error}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
