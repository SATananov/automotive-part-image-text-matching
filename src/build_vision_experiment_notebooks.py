from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

from src.vision_suite_config import (
    NOTEBOOK_AUDIT_PATH,
    PROJECT_ROOT,
    SCORING_NOTEBOOK_PATH,
    VISION_NOTEBOOK_PATH,
)


def _markdown(text: str) -> Any:
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def _code(source: str) -> Any:
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def _execute_and_write(notebook: Any, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    try:
        os.chdir(PROJECT_ROOT)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
        )
        executed = client.execute()
    finally:
        os.chdir(old_cwd)

    nbformat.write(executed, path)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "cell_count": len(executed.cells),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": sum(
            cell.get("execution_count") is not None for cell in code_cells
        ),
        "output_count": sum(len(cell.get("outputs", [])) for cell in code_cells),
        "error_output_count": len(error_outputs),
    }


def build_vision_notebook() -> Any:
    cells = [
        _markdown(
            """
# Step 011.3A — Vision Core Experimental Suite

This executed notebook presents train/validation-only evidence for VIS-001, VIS-004, VIS-005, and VIS-008. It uses deterministic local fixed-convolutional representations and does not download pretrained weights, inspect the locked test split, or alter the production final model.
"""
        ),
        _code(
            """
import json
from pathlib import Path
import pandas as pd
from IPython.display import display
ROOT = Path.cwd()
VISION = ROOT / "reports" / "course_coverage" / "vision"
profile = json.loads((VISION / "image_profile.json").read_text(encoding="utf-8-sig"))
profile
"""
        ),
        _markdown("## VIS-001 — Image inventory and review flags"),
        _code(
            """
inventory = pd.read_csv(VISION / "image_inventory.csv")
review = pd.read_csv(VISION / "annotation_review.csv")
display(inventory.head(10))
display(review[review["review_required"] == True].head(10))
"""
        ),
        _markdown("## VIS-004 — Representation and resolution comparison"),
        _code(
            """
representation = pd.read_csv(VISION / "representation_resolution_comparison.csv")
display(representation.sort_values(["validation_macro_f1_mean", "validation_accuracy_mean"], ascending=False))
"""
        ),
        _code(
            """
from PIL import Image
for name in ["representation_macro_f1.png", "image_dimension_scatter.png", "representative_image_gallery.png"]:
    display(Image.open(VISION / "figures" / name))
"""
        ),
        _markdown("## VIS-008 — Failure-driven augmentation ablation"),
        _code(
            """
augmentation = pd.read_csv(VISION / "augmentation_comparison.csv")
failure_matrix = pd.read_csv(VISION / "failure_to_augmentation_matrix.csv")
display(augmentation.sort_values(["validation_macro_f1_mean", "validation_accuracy_mean"], ascending=False))
display(failure_matrix)
"""
        ),
        _markdown("## VIS-005 — Model-independent occlusion evidence"),
        _code(
            """
explainability = json.loads((VISION / "explainability_summary.json").read_text(encoding="utf-8-sig"))
regions = pd.read_csv(VISION / "region_perturbation_summary.csv")
explainability, display(regions.head(12))
"""
        ),
        _markdown(
            """
### Interpretation boundary

The occlusion analysis reports an automated foreground-proxy alignment measure. It does **not** claim a human plausible-region review rate. Human agreement remains deferred to VIS-007 until genuine independent annotations are available.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    return notebook


def build_scoring_notebook() -> Any:
    cells = [
        _markdown(
            """
# Step 011.3A — Compatibility Scoring, Ranking and Controlled Gates

This executed notebook presents VIS-003 and VIS-009 validation evidence plus the explicit VIS-002, VIS-006, and VIS-007 gates. All scores are derived from scalar compatibility functions, preserving pair reversal algebraically. The locked test split remains unused.
"""
        ),
        _code(
            """
import json
from pathlib import Path
import pandas as pd
from IPython.display import display
ROOT = Path.cwd()
VISION = ROOT / "reports" / "course_coverage" / "vision"
comparison = pd.read_csv(VISION / "compatibility_strategy_comparison.csv")
display(comparison)
"""
        ),
        _markdown("## VIS-003 — Ordered compatibility scores"),
        _code(
            """
predictions = pd.read_csv(VISION / "compatibility_validation_predictions.csv")
display(predictions.head(12))
display(predictions.groupby("true_label")["compatibility_score"].agg(["count", "mean", "std", "min", "max"]))
"""
        ),
        _code(
            """
from PIL import Image
for name in ["compatibility_score_distribution.png", "ranking_margin_distribution.png"]:
    display(Image.open(VISION / "figures" / name))
"""
        ),
        _markdown("## VIS-009 — Ranking consistency"),
        _code(
            """
ranking = json.loads((VISION / "ranking_metrics.json").read_text(encoding="utf-8-sig"))
triplets = pd.read_csv(VISION / "ranking_triplets.csv")
equal_pairs = pd.read_csv(VISION / "equal_pair_evaluation.csv")
ranking, display(triplets.head(12)), display(equal_pairs.head(12))
"""
        ),
        _markdown("## Controlled gates"),
        _code(
            """
gates = {
    name: json.loads((VISION / name).read_text(encoding="utf-8-sig"))
    for name in [
        "pretrained_backbone_gate.json",
        "fine_tuning_gate.json",
        "human_annotation_gate.json",
    ]
}
gates
"""
        ),
        _markdown(
            """
### Gate conclusion

VIS-002 and VIS-006 remain closed because pretrained model downloads and Tier 4 fine-tuning have not been authorized. VIS-007 remains closed because genuine independent annotators and pre-adjudication confidence labels do not yet exist. No synthetic annotator agreement is reported.
"""
        ),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    return notebook


def build_notebooks() -> dict[str, Any]:
    audits = [
        _execute_and_write(build_vision_notebook(), VISION_NOTEBOOK_PATH),
        _execute_and_write(build_scoring_notebook(), SCORING_NOTEBOOK_PATH),
    ]
    payload = {
        "step": "011.3A",
        "status": "PASS" if all(item["error_output_count"] == 0 for item in audits) else "FAIL",
        "notebooks": audits,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
    }
    NOTEBOOK_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_AUDIT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    payload = build_notebooks()
    print("Vision notebooks")
    for item in payload["notebooks"]:
        print(
            f"- {item['path']}: {item['executed_code_cell_count']}/"
            f"{item['code_cell_count']} code cells executed, "
            f"errors={item['error_output_count']}"
        )
    print(f"Status: {payload['status']}")
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
