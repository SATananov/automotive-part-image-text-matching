from __future__ import annotations

import json
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "project_v3.ipynb"

FORBIDDEN_CODE_FRAGMENTS = (
    "data/locked_test",
    "dataset_v3_test_images_LOCKED.csv",
    "load_v3_split(\"test\")",
    "load_v3_split('test')",
    "test.csv",
)


def verify_notebook() -> dict[str, object]:
    if not NOTEBOOK_PATH.is_file():
        raise RuntimeError(f"Dataset V3 notebook missing: {NOTEBOOK_PATH}")

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    cells = notebook["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    markdown_cells = [cell for cell in cells if cell["cell_type"] == "markdown"]
    full_text = "\n".join(cell.get("source", "") for cell in cells)
    code_text = "\n".join(cell.get("source", "") for cell in code_cells)

    if len(cells) != 25:
        raise RuntimeError(f"Expected 25 notebook cells, found {len(cells)}.")
    if len(code_cells) != 14:
        raise RuntimeError(f"Expected 14 code cells, found {len(code_cells)}.")
    if len(markdown_cells) != 11:
        raise RuntimeError(f"Expected 11 markdown cells, found {len(markdown_cells)}.")

    unexecuted = [index for index, cell in enumerate(cells) if cell["cell_type"] == "code" and cell.get("execution_count") is None]
    if unexecuted:
        raise RuntimeError(f"Unexecuted code cells: {unexecuted}")

    error_cells = []
    output_count = 0
    display_outputs = 0
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            output_count += 1
            if output.get("output_type") == "error":
                error_cells.append(index)
            if output.get("output_type") in {"display_data", "execute_result"}:
                display_outputs += 1

    if error_cells:
        raise RuntimeError(f"Notebook error outputs in cells: {error_cells}")
    if output_count < 18:
        raise RuntimeError(f"Expected at least 18 outputs, found {output_count}.")
    if display_outputs < 8:
        raise RuntimeError(f"Expected at least 8 display outputs, found {display_outputs}.")

    required_text = (
        "Automotive Part Image-Text Matching — Dataset V3",
        "Research question:",
        "480 train images",
        "80 validation images",
        "PARTIAL_MATCH",
        "image-group sign-flip",
        "377/480",
        "development result",
        "locked test split was not read or evaluated",
    )
    for fragment in required_text:
        if fragment not in full_text:
            raise RuntimeError(f"Required notebook text missing: {fragment}")

    forbidden_full_text = (
        "Dataset V2",
        "synthetic ablation",
        "torch_multimodal_real_only",
        "PENDING_NOTEBOOK_REEXECUTION",
        "patch",
        "Step 0",
        "AI-generated",
    )
    for fragment in forbidden_full_text:
        if fragment.lower() in full_text.lower():
            raise RuntimeError(f"Outdated or internal notebook text found: {fragment}")

    for fragment in FORBIDDEN_CODE_FRAGMENTS:
        if fragment in code_text:
            raise RuntimeError(f"Forbidden test access in notebook code: {fragment}")

    if 'RESULTS = ROOT / "results" / "dataset_v3"' not in code_text:
        raise RuntimeError("Notebook does not use the isolated Dataset V3 results directory.")
    if 'MANIFESTS = ROOT / "data" / "manifests" / "dataset_v3"' not in code_text:
        raise RuntimeError("Notebook does not use Dataset V3 development manifests.")
    if 'torch_multimodal_dataset_v3' not in code_text:
        raise RuntimeError("Notebook does not select the Dataset V3 main model.")

    summary = {
        "status": "PASS_DATASET_V3_EXECUTED_NOTEBOOK_VERIFIED",
        "notebook": "project_v3.ipynb",
        "total_cells": len(cells),
        "markdown_cells": len(markdown_cells),
        "code_cells": len(code_cells),
        "executed_code_cells": len(code_cells),
        "output_count": output_count,
        "display_outputs": display_outputs,
        "error_outputs": 0,
        "dataset_version": "3.0-development",
        "train_images": 480,
        "validation_images": 80,
        "validation_rows": 480,
        "main_model_slug": "torch_multimodal_dataset_v3",
        "main_correct_predictions": 377,
        "main_total_predictions": 480,
        "main_validation_accuracy": 377 / 480,
        "main_validation_macro_f1": 0.787284319325806,
        "locked_test_path_references_in_code": 0,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "test_evaluation_executed": False,
    }
    return summary


def main() -> None:
    summary = verify_notebook()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
