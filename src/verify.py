from __future__ import annotations

import json

import nbformat
import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from sklearn.metrics import accuracy_score, f1_score

from src.data import DATA_DIR, PROJECT_ROOT, RESULTS_DIR
from src.evaluation import grouped_paired_randomization

CANONICAL_TORCH_VERSION = "2.10.0"
MAIN_MODEL_SLUG = "torch_multimodal_real_only"
NON_NEURAL_MULTIMODAL_SLUG = "image_text_logistic_regression"
SYNTHETIC_MODEL_SLUG = "torch_multimodal_real_plus_synthetic"
RESULT_SUMMARY_PATH = RESULTS_DIR / "result_summary.md"
VERIFICATION_SUMMARY_PATH = RESULTS_DIR / "verification_summary.json"


def base_torch_version(version: str) -> str:
    return version.split("+", 1)[0]


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _notebook_status() -> dict[str, object]:
    notebook = nbformat.read(PROJECT_ROOT / "project.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    execution_counts = [cell.get("execution_count") for cell in code_cells]
    output_count = sum(len(cell.get("outputs", [])) for cell in code_cells)
    markdown_text = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    stale_literals = [
        value
        for value in ("30 paired rows from 10", "0.4667", "0.4407", "0.424")
        if value in markdown_text
    ]
    return {
        "code_cells": len(code_cells),
        "executed_code_cells": sum(value is not None for value in execution_counts),
        "execution_counts": execution_counts,
        "errors": len(error_outputs),
        "outputs": output_count,
        "stale_hard_coded_result_literals": stale_literals,
    }


def _load_and_validate_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    predictions = pd.read_csv(RESULTS_DIR / "validation_predictions.csv")
    paired = pd.read_csv(RESULTS_DIR / "paired_comparisons.csv")

    expected_order = table.sort_values(
        ["validation_macro_f1", "validation_accuracy", "model"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    assert_frame_equal(table.reset_index(drop=True), expected_order, check_dtype=False)

    table_indexed = table.set_index("model_slug")
    if set(predictions["model_slug"]) != set(table_indexed.index):
        raise AssertionError("Prediction models and comparison-table models differ")

    sample_sets = []
    for slug, group in predictions.groupby("model_slug", sort=False):
        row = table_indexed.loc[slug]
        accuracy = float(accuracy_score(group["true_label"], group["predicted_label"]))
        macro_f1 = float(
            f1_score(
                group["true_label"],
                group["predicted_label"],
                average="macro",
                zero_division=0,
            )
        )
        if abs(accuracy - float(row["validation_accuracy"])) > 1e-12:
            raise AssertionError(f"Accuracy mismatch for {slug}")
        if abs(macro_f1 - float(row["validation_macro_f1"])) > 1e-12:
            raise AssertionError(f"Macro-F1 mismatch for {slug}")
        if int(group["is_correct"].sum()) != int(row["correct_predictions"]):
            raise AssertionError(f"Correct-prediction mismatch for {slug}")
        sample_sets.append(set(group["sample_id"]))
    if any(sample_set != sample_sets[0] for sample_set in sample_sets[1:]):
        raise AssertionError("Models do not cover identical validation samples")

    main_predictions = predictions[predictions["model_slug"].eq(MAIN_MODEL_SLUG)].reset_index(
        drop=True
    )
    saved_main = pd.read_csv(RESULTS_DIR / "multimodal_validation_predictions.csv")
    assert_frame_equal(saved_main, main_predictions, check_dtype=False)

    ablation = pd.read_csv(RESULTS_DIR / "synthetic_ablation.csv").reset_index(drop=True)
    expected_ablation = table[
        table["model_slug"].isin([MAIN_MODEL_SLUG, SYNTHETIC_MODEL_SLUG])
    ].reset_index(drop=True)
    assert_frame_equal(ablation, expected_ablation, check_dtype=False)

    expected_paired = pd.DataFrame(
        [
            grouped_paired_randomization(
                predictions, MAIN_MODEL_SLUG, NON_NEURAL_MULTIMODAL_SLUG
            ),
            grouped_paired_randomization(
                predictions, MAIN_MODEL_SLUG, SYNTHETIC_MODEL_SLUG
            ),
        ]
    )
    assert_frame_equal(paired, expected_paired, check_dtype=False)
    return table, predictions, paired


def render_result_summary() -> str:
    table, _, paired = _load_and_validate_results()
    run_info = _read_json(RESULTS_DIR / "run_info.json")
    indexed = table.set_index("model_slug")
    main = indexed.loc[MAIN_MODEL_SLUG]
    baseline = indexed.loc[NON_NEURAL_MULTIMODAL_SLUG]
    synthetic = indexed.loc[SYNTHETIC_MODEL_SLUG]
    paired_baseline = paired[
        paired["right_model_slug"].eq(NON_NEURAL_MULTIMODAL_SLUG)
    ].iloc[0]

    rows = [
        "# Validation Result Summary",
        "",
        "This summary is generated from the saved Dataset V2 predictions and metrics.",
        "",
        f"- Environment: Python `{run_info['python_version']}`, PyTorch `{run_info['torch_version']}`",
        (
            f"- Evaluation: {int(run_info['validation_rows'])} paired rows from "
            f"{int(run_info['validation_images'])} independent real validation images"
        ),
        f"- Training: {int(run_info['training_real_images'])} real and {int(run_info['training_synthetic_images'])} synthetic images",
        "- Locked test split used: no",
        "",
        "| Model | Accuracy | Macro F1 | Correct |",
        "|---|---:|---:|---:|",
        (
            f"| Real-only multimodal CNN | {float(main['validation_accuracy']):.4f} | "
            f"{float(main['validation_macro_f1']):.4f} | "
            f"{int(main['correct_predictions'])}/{int(main['total_predictions'])} |"
        ),
        (
            f"| Image + text Logistic Regression | {float(baseline['validation_accuracy']):.4f} | "
            f"{float(baseline['validation_macro_f1']):.4f} | "
            f"{int(baseline['correct_predictions'])}/{int(baseline['total_predictions'])} |"
        ),
        (
            f"| Real + synthetic multimodal CNN | {float(synthetic['validation_accuracy']):.4f} | "
            f"{float(synthetic['validation_macro_f1']):.4f} | "
            f"{int(synthetic['correct_predictions'])}/{int(synthetic['total_predictions'])} |"
        ),
        "",
        (
            f"The image-group paired comparison between the real-only neural model and "
            f"the image + text Logistic Regression baseline gives p = "
            f"`{float(paired_baseline['grouped_two_sided_p_value']):.6f}` over "
            f"{int(paired_baseline['independent_groups'])} independent images."
        ),
        (
            "The real-only model's grouped-bootstrap 95% accuracy interval is "
            f"`[{float(main['accuracy_ci_low']):.4f}, {float(main['accuracy_ci_high']):.4f}]`."
        ),
        "",
        "This is a validation result from a small development set, not proof that one model is always better. The final test images remain locked.",
        "",
    ]
    return "\n".join(rows)


def write_result_summary() -> None:
    RESULT_SUMMARY_PATH.write_text(render_result_summary(), encoding="utf-8", newline="\n")


def write_pending_verification_summary() -> None:
    pending = {
        "status": "PENDING_NOTEBOOK_REEXECUTION",
        "reason": (
            "Dataset V2 training artifacts were regenerated. Rebuild the report, "
            "Execute project.ipynb, then run python -m src.verify before submission."
        ),
    }
    VERIFICATION_SUMMARY_PATH.write_text(
        json.dumps(pending, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def build_verification_summary() -> dict[str, object]:
    table, _, paired = _load_and_validate_results()
    audit = _read_json(RESULTS_DIR / "data_audit.json")
    run_info = _read_json(RESULTS_DIR / "run_info.json")
    test_lock = _read_json(DATA_DIR / "test_lock.json")
    manifest = pd.read_csv(DATA_DIR / "image_manifest.csv")
    notebook = _notebook_status()
    environment_lock = _read_json(RESULTS_DIR / "environment_lock.json")

    if str(run_info.get("dataset_version")) != "2.0":
        raise AssertionError("Saved results are not Dataset V2 artifacts")
    if base_torch_version(str(run_info["torch_version"])) != CANONICAL_TORCH_VERSION:
        raise AssertionError("Saved results were not produced with the canonical PyTorch version")
    if environment_lock["python_version"] != run_info["python_version"]:
        raise AssertionError("Environment lock and training Python version differ")
    if environment_lock["packages"].get("torch", "").split("+", 1)[0] != CANONICAL_TORCH_VERSION:
        raise AssertionError("Environment lock does not contain the canonical PyTorch version")
    if notebook["executed_code_cells"] != notebook["code_cells"]:
        raise AssertionError("Notebook has unexecuted code cells")
    if notebook["errors"] != 0:
        raise AssertionError("Notebook contains error outputs")
    if notebook["stale_hard_coded_result_literals"]:
        raise AssertionError("Notebook Markdown contains stale hard-coded result numbers")

    critical_audit_checks = {
        "identity_overlap_zero": all(value == 0 for value in audit["overlap"].values()),
        "exact_image_hash_overlap_zero": audit["exact_cross_split_image_hash_overlap"] == 0,
        "exact_caption_overlap_zero": audit["exact_description_overlap"] == 0,
        "suspicious_real_pairs_zero": audit["suspicious_real_image_pairs"] == 0,
        "license_hashes_match": audit["licenses"]["all_license_hashes_match"] is True,
        "test_lock_matches": audit["test_lock"]["sha256_matches"] is True,
        "test_split_unused": audit["test_split_used"] is False,
    }
    if not all(critical_audit_checks.values()):
        raise AssertionError("One or more critical data-audit checks failed")

    expected_summary = render_result_summary()
    saved_summary = RESULT_SUMMARY_PATH.read_text(encoding="utf-8")
    if saved_summary != expected_summary:
        raise AssertionError("Generated Markdown result summary is stale")

    indexed = table.set_index("model_slug")
    main = indexed.loc[MAIN_MODEL_SLUG]
    synthetic = indexed.loc[SYNTHETIC_MODEL_SLUG]
    baseline_pair = paired[
        paired["right_model_slug"].eq(NON_NEURAL_MULTIMODAL_SLUG)
    ].iloc[0]
    source_counts = manifest.groupby("source").size().to_dict()

    return {
        "status": "PASS",
        "dataset_version": "2.0",
        "canonical_environment": {
            "required_torch_version": CANONICAL_TORCH_VERSION,
            "saved_torch_version": run_info["torch_version"],
            "saved_python_version": run_info["python_version"],
            "environment_lock_sha256": environment_lock["text_lock_sha256"],
            "locked_packages": environment_lock["packages"],
        },
        "dataset": {
            "total_images": int(len(manifest)),
            "real_images": int(manifest["source"].ne("generated").sum()),
            "synthetic_images": int(manifest["source"].eq("generated").sum()),
            "images_by_source": {key: int(value) for key, value in source_counts.items()},
            "train_rows": int(audit["train_samples"]),
            "validation_rows": int(audit["validation_samples"]),
            "validation_independent_images": int(audit["validation_images"]),
            "test_rows_locked": int(test_lock["test_rows"]),
            "test_images_locked": int(test_lock["test_images"]),
        },
        "data_audit": {
            **critical_audit_checks,
            "license_rows": int(audit["licenses"]["license_rows"]),
            "unique_train_descriptions": int(audit["unique_descriptions_train"]),
            "unique_validation_descriptions": int(audit["unique_descriptions_validation"]),
        },
        "training": {
            "framework": run_info["deep_learning_framework"],
            "saved_models": int(len(table)),
            "main_model_slug": MAIN_MODEL_SLUG,
            "main_validation_accuracy": float(main["validation_accuracy"]),
            "main_validation_macro_f1": float(main["validation_macro_f1"]),
            "main_correct_predictions": int(main["correct_predictions"]),
            "main_accuracy_ci": [
                float(main["accuracy_ci_low"]),
                float(main["accuracy_ci_high"]),
            ],
            "grouped_paired_baseline_p_value": float(
                baseline_pair["grouped_two_sided_p_value"]
            ),
            "paired_comparison_method": str(baseline_pair["method"]),
            "paired_independent_groups": int(baseline_pair["independent_groups"]),
            "synthetic_validation_accuracy": float(synthetic["validation_accuracy"]),
            "synthetic_validation_macro_f1": float(synthetic["validation_macro_f1"]),
        },
        "artifact_consistency": {
            "saved_metrics_match_predictions": True,
            "main_prediction_file_matches": True,
            "synthetic_ablation_matches": True,
            "paired_comparisons_match": True,
            "generated_markdown_summary_matches": True,
        },
        "notebook": notebook,
    }


def write_verification_summary(*, require_current_environment: bool = True) -> dict[str, object]:
    if require_current_environment and base_torch_version(torch.__version__) != CANONICAL_TORCH_VERSION:
        raise AssertionError(
            f"Current PyTorch is {torch.__version__}; expected {CANONICAL_TORCH_VERSION}"
        )
    summary = build_verification_summary()
    VERIFICATION_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main() -> None:
    summary = write_verification_summary()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
