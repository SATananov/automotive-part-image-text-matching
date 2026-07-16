from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from src.apply_real_sample_intake import build_apply_plan
from src.prepare_first_real_batch import (
    build_preparation_report,
    plan_row_to_intake,
    read_plan,
)
from src.real_dataset_config import (
    APPROVAL_LOG_COLUMNS,
    FIRST_BATCH_EXPECTED_IMAGES,
    FIRST_BATCH_ID,
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    PROJECT_ROOT,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
    REAL_IMAGES_PATH,
    REAL_PART_GROUPS_PATH,
    REAL_PROCESSED_IMAGES_DIRECTORY,
    REAL_SAMPLE_INTAKE_PATH,
    SAMPLE_INTAKE_COLUMNS,
)
from src.review_real_sample_intake import (
    build_review_report,
    read_csv_exact,
)


JSON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_dry_run.json"
)

MARKDOWN_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "first_batch_dry_run.md"
)

LIVE_DATA_FILES = (
    REAL_PART_GROUPS_PATH,
    REAL_IMAGES_PATH,
    REAL_SAMPLE_INTAKE_PATH,
    REAL_APPROVAL_LOG_PATH,
    REAL_IMAGE_MANIFEST_PATH,
)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fingerprint_live_state() -> dict[str, str]:
    fingerprint: dict[str, str] = {}

    for path in LIVE_DATA_FILES:
        key = str(path.relative_to(PROJECT_ROOT))
        fingerprint[key] = (
            sha256_bytes(path.read_bytes()) if path.is_file() else "MISSING"
        )

    if REAL_PROCESSED_IMAGES_DIRECTORY.is_dir():
        for path in sorted(REAL_PROCESSED_IMAGES_DIRECTORY.rglob("*")):
            if path.is_file():
                key = str(path.relative_to(PROJECT_ROOT))
                fingerprint[key] = sha256_bytes(path.read_bytes())

    return fingerprint


def read_supporting_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:
    part_groups, group_errors = read_csv_exact(
        REAL_PART_GROUPS_PATH,
        PART_GROUP_COLUMNS,
        "real part groups",
    )
    images, image_errors = read_csv_exact(
        REAL_IMAGES_PATH,
        IMAGE_MANIFEST_COLUMNS,
        "real images",
    )
    approval_log, log_errors = read_csv_exact(
        REAL_APPROVAL_LOG_PATH,
        APPROVAL_LOG_COLUMNS,
        "approval log",
    )
    return (
        part_groups,
        images,
        approval_log,
        group_errors + image_errors + log_errors,
    )


def build_dry_run_report() -> dict[str, object]:
    plan, plan_read_errors = read_plan()
    preparation, preview = build_preparation_report(
        plan,
        initial_errors=plan_read_errors,
    )
    errors = list(preparation["errors"])
    warnings = list(preparation["warnings"])

    if errors:
        return {
            "status": "FAIL",
            "result": "PREPARATION_BLOCKED",
            "batch_id": FIRST_BATCH_ID,
            "simulation_decision": "approved",
            "counts": {
                "planned_images": int(len(plan)),
                "captured_files": int(
                    (preview.get("file_present", pd.Series(dtype=str)) == "yes")
                    .sum()
                ),
                "simulated_approved": 0,
                "prospective_groups": 0,
                "prospective_images": 0,
            },
            "immutability": "NOT_CHECKED",
            "items": [],
            "errors": errors,
            "warnings": warnings,
        }

    processed_ids = set()
    if REAL_APPROVAL_LOG_PATH.is_file():
        approval_log_for_ids = pd.read_csv(
            REAL_APPROVAL_LOG_PATH,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        if "intake_id" in approval_log_for_ids.columns:
            processed_ids = set(approval_log_for_ids["intake_id"])

    captured_rows: list[dict[str, str]] = []
    for _, row in plan.iterrows():
        planned = plan_row_to_intake(row)
        if planned["intake_id"] in processed_ids:
            continue
        if not (PROJECT_ROOT / planned["staging_path"]).is_file():
            continue
        planned["decision"] = "approved"
        captured_rows.append(planned)

    if not captured_rows:
        return {
            "status": "PASS",
            "result": "AWAITING_CAPTURE",
            "batch_id": FIRST_BATCH_ID,
            "simulation_decision": "approved",
            "counts": {
                "planned_images": int(len(plan)),
                "captured_files": 0,
                "simulated_approved": 0,
                "prospective_groups": 0,
                "prospective_images": 0,
            },
            "immutability": "PASS",
            "items": [],
            "errors": [],
            "warnings": warnings,
        }

    part_groups, images, approval_log, read_errors = (
        read_supporting_tables()
    )
    errors.extend(read_errors)
    simulated_intake = pd.DataFrame(
        captured_rows,
        columns=SAMPLE_INTAKE_COLUMNS,
    )
    review = build_review_report(
        simulated_intake,
        part_groups,
        images,
        approval_log,
        initial_errors=errors,
    )
    errors = list(review["errors"])
    warnings.extend(review["warnings"])

    if review["status"] != "PASS":
        return {
            "status": "FAIL",
            "result": "REVIEW_BLOCKED",
            "batch_id": FIRST_BATCH_ID,
            "simulation_decision": "approved",
            "counts": {
                "planned_images": int(len(plan)),
                "captured_files": len(captured_rows),
                "simulated_approved": 0,
                "prospective_groups": int(len(part_groups)),
                "prospective_images": int(len(images)),
            },
            "immutability": "NOT_CHECKED",
            "items": review["items"],
            "errors": errors,
            "warnings": warnings,
        }

    before = fingerprint_live_state()
    with tempfile.TemporaryDirectory(
        prefix="automotive_step009_2_dry_run_"
    ) as temporary_name:
        (
            prospective_groups,
            prospective_images,
            _remaining_intake,
            _prospective_log,
            staged_moves,
            applied_items,
            plan_errors,
        ) = build_apply_plan(
            simulated_intake,
            part_groups,
            images,
            approval_log,
            review,
            Path(temporary_name),
            timestamp_factory=lambda: "DRY_RUN",
        )

        normalized_files = [source for source, _ in staged_moves]
        missing_normalized = [
            str(path) for path in normalized_files if not path.is_file()
        ]
        if missing_normalized:
            plan_errors.append(
                "Dry-run normalization did not create expected temporary "
                f"files: {missing_normalized}."
            )

    after = fingerprint_live_state()
    immutability = "PASS" if before == after else "FAIL"
    if immutability != "PASS":
        plan_errors.append(
            "Controlled dry run changed live annotation or processed-image "
            "state."
        )

    errors.extend(plan_errors)
    captured_count = len(captured_rows)
    result = (
        "FULL_BATCH_SIMULATED"
        if captured_count == FIRST_BATCH_EXPECTED_IMAGES
        else "PARTIAL_BATCH_SIMULATED"
    )

    return {
        "status": "PASS" if not errors else "FAIL",
        "result": result if not errors else "SIMULATION_BLOCKED",
        "batch_id": FIRST_BATCH_ID,
        "simulation_decision": "approved",
        "counts": {
            "planned_images": int(len(plan)),
            "captured_files": captured_count,
            "simulated_approved": len(applied_items),
            "prospective_groups": int(len(prospective_groups)),
            "prospective_images": int(len(prospective_images)),
        },
        "immutability": immutability,
        "items": applied_items,
        "errors": errors,
        "warnings": warnings,
    }


def write_outputs(report: dict[str, object]) -> None:
    JSON_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    counts = report["counts"]
    lines = [
        "# First Real Sample Batch Controlled Dry Run",
        "",
        f"**Status:** {report['status']}",
        f"**Result:** {report['result']}",
        f"**Batch:** {report['batch_id']}",
        f"**Simulation decision:** {report['simulation_decision']}",
        f"**Live-state immutability:** {report['immutability']}",
        "",
        "## Counts",
        "",
        f"- Planned images: {counts['planned_images']}",
        f"- Captured staging files: {counts['captured_files']}",
        f"- Simulated approvals: {counts['simulated_approved']}",
        f"- Prospective groups: {counts['prospective_groups']}",
        f"- Prospective images: {counts['prospective_images']}",
        "",
        "## Safety",
        "",
        "- The simulation treats captured candidates as approved only "
        "inside a temporary directory.",
        "- It does not update `sample_intake.csv`, annotations, the "
        "approval log, the manifest, or processed images.",
        "- Real approval still requires the Step 009.1 review and apply "
        "commands.",
        "",
        "## Errors",
        "",
    ]
    if report["errors"]:
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("- No dry-run errors found.")

    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No dry-run warnings found.")

    MARKDOWN_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_dry_run() -> dict[str, object]:
    report = build_dry_run_report()
    write_outputs(report)
    return report


def main() -> None:
    report = run_dry_run()
    counts = report["counts"]

    print("First real sample batch controlled dry run")
    print(f"- Status: {report['status']}")
    print(f"- Result: {report['result']}")
    print(f"- Captured files: {counts['captured_files']}")
    print(f"- Simulated approvals: {counts['simulated_approved']}")
    print(f"- Live-state immutability: {report['immutability']}")
    print(
        "- Report: "
        f"{MARKDOWN_REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
