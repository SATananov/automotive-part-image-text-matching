from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient

from src.fundamentals_suite_config import (
    NOTEBOOK_AUDIT_PATH,
    NOTEBOOK_PATH,
    PROJECT_ROOT,
)


def code(source: str) -> dict[str, Any]:
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str) -> dict[str, Any]:
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


def build_notebook() -> dict[str, Any]:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Step 011.1 project environment",
            "language": "python",
            "name": "step0111",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    notebook.cells = [
        markdown(
            """
# Deep Learning Fundamentals Experimental Suite

This notebook presents the executed evidence for **Step 011.1**. It covers
Problems 1–10 from the Deep Learning Fundamentals exercise using the
project's integrated automotive-part image–text dataset.

The experiments use only the committed train and validation splits. The
locked test split is not loaded, inspected, scored, or used for selection.
"""
        ),
        markdown(
            r"""
## Mathematical setup

For an image–text pair \((I, T)\), the model estimates a three-class
probability vector

\[
p_\theta(y\mid I,T)=\operatorname{softmax}(f_\theta(I,T)),
\]

and minimizes sparse categorical cross-entropy

\[
\mathcal{L}(\theta)=-\frac{1}{N}\sum_{i=1}^{N}
\log p_\theta(y_i\mid I_i,T_i).
\]

The primary validation metric is Macro F1 so every relation class contributes
equally to the comparison.
"""
        ),
        code(
            """
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image

PROJECT_ROOT = Path.cwd()
REPORT_ROOT = PROJECT_ROOT / "reports" / "course_coverage" / "fundamentals"
print("Runtime: project virtual environment")
print("Working directory: repository root")
"""
        ),
        markdown("## FND-001 and FND-002 — Data inspection and batch contract"),
        code(
            """
profile = json.loads((REPORT_ROOT / "dataset_profile.json").read_text())
batch = json.loads((REPORT_ROOT / "batch_contract.json").read_text())
display(pd.DataFrame([profile["train"], profile["validation"]], index=["train", "validation"]))
print(json.dumps(batch, indent=2))
display(Image(filename=str(REPORT_ROOT / "figures" / "eda_label_distribution.png")))
display(Image(filename=str(REPORT_ROOT / "figures" / "eda_text_length_distribution.png")))
"""
        ),
        markdown("## FND-003 and FND-004 — Gradient signal and one-batch overfit"),
        code(
            """
baseline = json.loads((REPORT_ROOT / "baseline_gradient_diagnostic.json").read_text())
overfit = json.loads((REPORT_ROOT / "overfit_result.json").read_text())
print("Baseline diagnostic:", json.dumps(baseline, indent=2))
print("Overfit diagnostic:", json.dumps(overfit, indent=2))
display(Image(filename=str(REPORT_ROOT / "figures" / "overfit_learning_curves.png")))
"""
        ),
        markdown("## FND-005 and FND-006 — Training loop, optimizers, and learning rates"),
        code(
            """
loop_audit = json.loads((REPORT_ROOT / "training_loop_audit.json").read_text())
optimizers = pd.read_csv(REPORT_ROOT / "optimizer_comparison.csv")
print(json.dumps(loop_audit, indent=2))
display(optimizers.sort_values(["validation_macro_f1", "validation_accuracy"], ascending=False))
display(Image(filename=str(REPORT_ROOT / "figures" / "optimizer_macro_f1.png")))
"""
        ),
        markdown("## FND-007 — Model capacity"),
        code(
            """
capacity = pd.read_csv(REPORT_ROOT / "capacity_comparison.csv")
display(capacity.sort_values("parameter_count"))
display(Image(filename=str(REPORT_ROOT / "figures" / "capacity_tradeoff.png")))
"""
        ),
        markdown("## FND-008 — Architecture ablation"),
        code(
            """
architecture = pd.read_csv(REPORT_ROOT / "architecture_comparison.csv")
display(architecture.sort_values("validation_macro_f1", ascending=False))
display(Image(filename=str(REPORT_ROOT / "figures" / "architecture_comparison.png")))
"""
        ),
        markdown("## FND-009 — Preprocessing alternatives"),
        code(
            """
preprocessing = pd.read_csv(REPORT_ROOT / "preprocessing_comparison.csv")
display(preprocessing.sort_values("validation_macro_f1", ascending=False))
display(Image(filename=str(REPORT_ROOT / "figures" / "preprocessing_comparison.png")))
"""
        ),
        markdown("## FND-010 — Controlled failure diagnostics"),
        code(
            """
failures = pd.read_csv(REPORT_ROOT / "failure_diagnostics.csv")
display(failures)
display(Image(filename=str(REPORT_ROOT / "figures" / "failure_signatures.png")))
"""
        ),
        markdown("## Synthesis and locked evaluation status"),
        code(
            """
status = json.loads((REPORT_ROOT / "fundamentals_suite_status.json").read_text())
comparison = pd.read_csv(REPORT_ROOT / "experiment_comparison.csv")
print(json.dumps(status, indent=2))
cols = ["experiment_id", "variant", "validation_accuracy", "validation_macro_f1", "parameter_count", "training_time_seconds"]
display(comparison[cols].sort_values(["validation_macro_f1", "validation_accuracy"], ascending=False).head(15))
assert status["test_split_used"] is False
assert status["final_test_evaluation_authorized"] is False
"""
        ),
        markdown(
            """
## Conclusion

The suite demonstrates data contracts, gradient flow, deliberate one-batch
overfitting, a correct train/validation loop, optimizer and learning-rate
comparisons, capacity control, regularization and architectural ablations,
preprocessing alternatives, and safe failure diagnostics. These experiments
are educational comparisons; they do not replace or modify the frozen final
exam model from Step 010.8.
"""
        ),
    ]
    return notebook


def sanitize_notebook_for_repository(
    notebook: dict[str, Any],
) -> dict[str, Any]:
    """Remove machine-specific absolute paths from executed evidence."""

    replacements = {
        str(PROJECT_ROOT): "<PROJECT_ROOT>",
        PROJECT_ROOT.as_posix(): "<PROJECT_ROOT>",
        str(Path(sys.executable)): "<PROJECT_PYTHON>",
        Path(sys.executable).as_posix(): "<PROJECT_PYTHON>",
    }

    def sanitize(value: Any) -> Any:
        if isinstance(value, str):
            result = value
            for original, replacement in sorted(
                replacements.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            ):
                if original:
                    result = result.replace(original, replacement)
            return result
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        return value

    # ``sanitize`` intentionally traverses mappings and therefore produces
    # plain dictionaries. Convert the sanitized structure back to nbformat's
    # NotebookNode hierarchy before ``nbformat.write`` and NotebookClient use.
    return nbformat.from_dict(sanitize(notebook))


def execute_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    temporary_root = Path(tempfile.mkdtemp(prefix="step0111_kernel_"))
    kernel_directory = temporary_root / "kernels" / "step0111"
    kernel_directory.mkdir(parents=True)
    kernel_payload = {
        "argv": [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Step 011.1 project environment",
        "language": "python",
    }
    (kernel_directory / "kernel.json").write_text(
        json.dumps(kernel_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    previous_jupyter_path = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(temporary_root)
    try:
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="step0111",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
        )
        executed = client.execute()
    finally:
        if previous_jupyter_path is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = previous_jupyter_path
        shutil.rmtree(temporary_root, ignore_errors=True)
    return executed


def build_and_execute_notebook() -> dict[str, Any]:
    notebook = execute_notebook(build_notebook())
    notebook = sanitize_notebook_for_repository(notebook)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    execution_counts = [cell.get("execution_count") for cell in code_cells]
    audit = {
        "status": "PASS" if not errors else "FAIL",
        "notebook": NOTEBOOK_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "cell_count": len(notebook.cells),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": sum(count is not None for count in execution_counts),
        "output_count": sum(len(cell.get("outputs", [])) for cell in code_cells),
        "error_output_count": len(errors),
        "execution_counts": execution_counts,
        "kernel_python": "project_virtual_environment",
        "execution_working_directory": "repository_root",
        "machine_specific_paths_removed": True,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }
    NOTEBOOK_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_AUDIT_PATH.write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )
    if audit["status"] != "PASS":
        raise RuntimeError("Fundamentals evidence notebook execution failed.")
    return audit


def main() -> None:
    audit = build_and_execute_notebook()
    print("Fundamentals notebook execution")
    for key, value in audit.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
