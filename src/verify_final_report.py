"""Verify the executed final Dataset V3 reporting notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "project_final.ipynb"
FROZEN_NOTEBOOK_PATH = PROJECT_ROOT / "project_v3.ipynb"
README_PATH = PROJECT_ROOT / "README.md"
DEV_RESULTS = PROJECT_ROOT / "results" / "dataset_v3"
FINAL_RESULTS = PROJECT_ROOT / "results" / "dataset_v3_final_test"
MANIFESTS = PROJECT_ROOT / "data" / "manifests" / "dataset_v3"

EXPECTED_FROZEN_NOTEBOOK_SHA256 = (
    "4f87a9818cf91d1011b4b09794da086b6cc8fc9469e10d612a7c6ae6efb1900a"
)
EXPECTED_ACCURACY = 0.7375
EXPECTED_MACRO_F1 = 0.7382299830250852
EXPECTED_CORRECT = 354
EXPECTED_ROWS = 480
EXPECTED_IMAGES = 80

DISALLOWED_REPORT_CODE = (
    "evaluate_dataset_v3_final_test",
    "train_dataset_v3",
    "load_frozen_multimodal_model",
    "predict_in_batches",
    "torch.load",
    "data/locked_test",
    "data\\\\locked_test",
    "image.open(",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_final_report() -> dict[str, object]:
    required_paths = [
        REPORT_PATH,
        FROZEN_NOTEBOOK_PATH,
        README_PATH,
        DEV_RESULTS / "model_comparison.csv",
        FINAL_RESULTS / "test_metrics.json",
        FINAL_RESULTS / "test_predictions.csv",
        FINAL_RESULTS / "test_per_category.csv",
        FINAL_RESULTS / "test_confusion_matrix.csv",
        FINAL_RESULTS / "authorization_consumption.json",
        MANIFESTS / "dataset_v3_final_selection_lock.json",
        MANIFESTS / "dataset_v3_final_test_results.json",
    ]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final-report inputs: {missing}")

    frozen_sha = file_sha256(FROZEN_NOTEBOOK_PATH)
    if frozen_sha != EXPECTED_FROZEN_NOTEBOOK_SHA256:
        raise AssertionError(
            "Frozen project_v3.ipynb changed. "
            f"Expected {EXPECTED_FROZEN_NOTEBOOK_SHA256}, got {frozen_sha}."
        )

    selection_lock = load_json(
        MANIFESTS / "dataset_v3_final_selection_lock.json"
    )
    if selection_lock["selected_notebook_sha256"] != frozen_sha:
        raise AssertionError("Selection lock does not match frozen notebook.")

    metrics = load_json(FINAL_RESULTS / "test_metrics.json")
    consumption = load_json(
        FINAL_RESULTS / "authorization_consumption.json"
    )
    results_manifest = load_json(
        MANIFESTS / "dataset_v3_final_test_results.json"
    )
    predictions = pd.read_csv(FINAL_RESULTS / "test_predictions.csv")

    if metrics["status"] != "PASS_DATASET_V3_FINAL_TEST_EVALUATION_COMPLETE":
        raise AssertionError("Unexpected final-test metric status.")
    if metrics["model_slug"] != "torch_multimodal_dataset_v3":
        raise AssertionError("Unexpected final model.")
    if metrics["correct_predictions"] != EXPECTED_CORRECT:
        raise AssertionError("Unexpected correct-prediction count.")
    if metrics["test_rows"] != EXPECTED_ROWS:
        raise AssertionError("Unexpected final-test row count.")
    if metrics["test_images"] != EXPECTED_IMAGES:
        raise AssertionError("Unexpected independent-image count.")
    if not np.isclose(metrics["accuracy"], EXPECTED_ACCURACY, atol=1e-12):
        raise AssertionError("Unexpected final-test accuracy.")
    if not np.isclose(metrics["macro_f1"], EXPECTED_MACRO_F1, atol=1e-12):
        raise AssertionError("Unexpected final-test macro F1.")
    if len(predictions) != EXPECTED_ROWS:
        raise AssertionError("Saved prediction row count changed.")
    if predictions["image_id"].nunique() != EXPECTED_IMAGES:
        raise AssertionError("Saved independent-image count changed.")
    recomputed_correct = int(
        predictions["true_label"]
        .eq(predictions["predicted_label"])
        .sum()
    )
    if recomputed_correct != EXPECTED_CORRECT:
        raise AssertionError("Saved predictions disagree with final metrics.")

    if consumption["authorization_consumed"] is not True:
        raise AssertionError("Final-test authorization is not consumed.")
    if consumption["authorized_evaluations_completed"] != 1:
        raise AssertionError("Expected exactly one completed evaluation.")
    if consumption["further_tuning_permitted"] is not False:
        raise AssertionError("Further tuning must remain prohibited.")
    if metrics["post_test_tuning_permitted"] is not False:
        raise AssertionError("Post-test tuning must remain prohibited.")
    if metrics["test_results_may_be_used_for_tuning"] is not False:
        raise AssertionError("Final results must not be used for tuning.")
    if results_manifest["status"] != (
        "PASS_DATASET_V3_FINAL_TEST_RESULTS_RECORDED"
    ):
        raise AssertionError("Unexpected final-results manifest status.")

    notebook = nbformat.read(REPORT_PATH, as_version=4)
    if not notebook.cells:
        raise AssertionError("Final report notebook is empty.")

    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code"
    ]
    if not code_cells:
        raise AssertionError("Final report contains no code cells.")

    report_code = "\n".join(
        "".join(cell.source) for cell in code_cells
    ).lower()
    for token in DISALLOWED_REPORT_CODE:
        if token in report_code:
            raise AssertionError(
                f"Final report contains prohibited execution token: {token}"
            )

    errors = []
    unexecuted = []
    for index, cell in enumerate(code_cells):
        if cell.source.strip() and cell.execution_count is None:
            unexecuted.append(index)
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    {
                        "cell": index,
                        "ename": output.get("ename"),
                        "evalue": output.get("evalue"),
                    }
                )
    if unexecuted:
        raise AssertionError(
            f"Final report has unexecuted code cells: {unexecuted}"
        )
    if errors:
        raise AssertionError(f"Final report contains errors: {errors}")

    serialized = json.dumps(notebook, ensure_ascii=False)
    required_text = (
        "Final Deep Learning Exam Report",
        "354/480",
        "0.7375",
        "0.7382",
        "Final report integrity checks: PASS",
        "no post-test tuning",
    )
    missing_text = [text for text in required_text if text not in serialized]
    if missing_text:
        raise AssertionError(
            f"Final report is missing expected content: {missing_text}"
        )

    readme = README_PATH.read_text(encoding="utf-8")
    if "[project_final.ipynb](project_final.ipynb)" not in readme:
        raise AssertionError("README does not point to the final report.")
    if "project_v3.ipynb" not in readme or "frozen" not in readme.lower():
        raise AssertionError("README does not explain the frozen notebook.")

    return {
        "status": "PASS_FINAL_DATASET_V3_REPORT_VERIFIED",
        "report": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "report_sha256": file_sha256(REPORT_PATH),
        "frozen_notebook_sha256": frozen_sha,
        "code_cells": len(code_cells),
        "errors": 0,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "correct_predictions": metrics["correct_predictions"],
        "test_rows": metrics["test_rows"],
        "test_images": metrics["test_images"],
        "training_or_inference_executed": False,
    }


def main() -> None:
    summary = verify_final_report()
    print("Final Dataset V3 report verification")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
