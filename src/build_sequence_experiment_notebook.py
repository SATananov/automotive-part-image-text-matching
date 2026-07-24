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

from src.sequence_suite_config import NOTEBOOK_AUDIT_PATH, NOTEBOOK_PATH, PROJECT_ROOT


def code(source: str) -> dict[str, Any]:
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def markdown(source: str) -> dict[str, Any]:
    return nbformat.v4.new_markdown_cell(source.strip() + "\n")


def build_notebook() -> dict[str, Any]:
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Step 011.2 project environment",
            "language": "python",
            "name": "step0112",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    nb.cells = [
        markdown(
            """
# Transformers & Sequence Modelling Experimental Suite

This executed notebook presents the committed evidence for **Step 011.2**.
It compares deterministic tokenization, a dense embedding baseline, TF-IDF,
TextCNN, GRU, LSTM, and a small Transformer encoder on the automotive-part
text descriptions.

Only the committed **train** and **validation** splits are used. The locked
test split is not opened or scored. The optional pretrained-transformer task
remains gated because pretrained downloads require explicit authorization.
"""
        ),
        markdown(
            r"""
## Sequence modelling setup

A description is tokenized to a fixed-length integer sequence
\(x=(x_1,\ldots,x_L)\). Neural models estimate

\[
p_\theta(y\mid x)=\operatorname{softmax}(f_\theta(x)),
\]

and are compared using validation accuracy and Macro F1. The Transformer uses
multi-head self-attention

\[
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]
"""
        ),
        code(
            """
from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image

PROJECT_ROOT = Path.cwd()
REPORT_ROOT = PROJECT_ROOT / "reports" / "course_coverage" / "sequence"
print("Runtime: project virtual environment")
print("Working directory: repository root")
"""
        ),
        markdown("## SEQ-001 and SEQ-002 — Text audit and deterministic loader"),
        code(
            """
profile = json.loads((REPORT_ROOT / "text_profile.json").read_text())
loader = json.loads((REPORT_ROOT / "text_loader_contract.json").read_text())
examples = pd.read_csv(REPORT_ROOT / "representative_examples.csv")
print(json.dumps(profile, indent=2))
print(json.dumps(loader, indent=2))
display(examples)
display(Image(filename=str(REPORT_ROOT / "figures" / "text_length_distribution.png")))
display(Image(filename=str(REPORT_ROOT / "figures" / "label_distribution.png")))
"""
        ),
        markdown("## SEQ-003 — Tokenization and padded integer sequences"),
        code(
            """
tokenization = json.loads((REPORT_ROOT / "tokenization_summary.json").read_text())
token_examples = pd.read_csv(REPORT_ROOT / "tokenization_examples.csv")
print(json.dumps(tokenization, indent=2))
display(token_examples.head(12))
"""
        ),
        markdown("## SEQ-004 to SEQ-007 — Baselines and sequence architectures"),
        code(
            """
comparison = pd.read_csv(REPORT_ROOT / "model_comparison.csv")
runs = pd.read_csv(REPORT_ROOT / "training_runs.csv")
display(comparison.sort_values(["validation_macro_f1", "validation_accuracy"], ascending=False))
print("Recorded training runs:", len(runs))
display(Image(filename=str(REPORT_ROOT / "figures" / "model_macro_f1.png")))
display(Image(filename=str(REPORT_ROOT / "figures" / "complexity_tradeoff.png")))
"""
        ),
        markdown("## SEQ-008 — Validation comparison, ROC evidence, and errors"),
        code(
            """
errors = pd.read_csv(REPORT_ROOT / "validation_error_analysis.csv")
print("Validation errors recorded:", len(errors))
display(errors.head(15))
display(Image(filename=str(REPORT_ROOT / "figures" / "roc_curves.png")))
"""
        ),
        markdown("## SEQ-009 — Multi-head attention inspection"),
        code(
            """
attention = json.loads((REPORT_ROOT / "attention_evidence.json").read_text())
attention_summary = pd.read_csv(REPORT_ROOT / "attention_token_summary.csv")
print(json.dumps(attention["selection"], indent=2))
display(attention_summary)
for name in [
    "attention_correct_head_1.png", "attention_correct_head_2.png",
    "attention_incorrect_head_1.png", "attention_incorrect_head_2.png",
]:
    display(Image(filename=str(REPORT_ROOT / "figures" / name)))
"""
        ),
        markdown("## SEQ-010 — Explicit pretrained-transformer gate"),
        code(
            """
gate = json.loads((REPORT_ROOT / "pretrained_transformer_gate.json").read_text())
print(json.dumps(gate, indent=2))
assert gate["status"] == "DEFERRED_EXPLICIT_APPROVAL_REQUIRED"
assert gate["pretrained_weights_downloaded"] is False
"""
        ),
        markdown("## Final locked state"),
        code(
            """
status = json.loads((REPORT_ROOT / "sequence_suite_status.json").read_text())
print(json.dumps(status, indent=2))
assert status["status"] == "PASS"
assert status["test_split_used"] is False
assert status["final_test_evaluation_authorized"] is False
assert status["production_final_model_changed"] is False
assert status["pretrained_weights_downloaded"] is False
"""
        ),
        markdown(
            """
## Conclusion

The core sequence-modelling suite is complete and reproducible: deterministic
loading and tokenization, six model families, validation-only comparison,
error analysis, and attention-head evidence. It is an educational experimental
extension and does not alter the frozen production/final model. The pretrained
extension remains intentionally gated rather than silently downloading weights.
"""
        ),
    ]
    return nb


def sanitize_notebook_for_repository(notebook: dict[str, Any]) -> dict[str, Any]:
    replacements = {
        str(PROJECT_ROOT): "<PROJECT_ROOT>",
        PROJECT_ROOT.as_posix(): "<PROJECT_ROOT>",
        str(Path(sys.executable)): "<PROJECT_PYTHON>",
        Path(sys.executable).as_posix(): "<PROJECT_PYTHON>",
    }

    def sanitize(value: Any) -> Any:
        if isinstance(value, str):
            result = value
            for original, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
                if original:
                    result = result.replace(original, replacement)
            return result
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        return value

    return nbformat.from_dict(sanitize(notebook))


def execute_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    temporary_root = Path(tempfile.mkdtemp(prefix="step0112_kernel_"))
    kernel_directory = temporary_root / "kernels" / "step0112"
    kernel_directory.mkdir(parents=True)
    kernel_payload = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Step 011.2 project environment",
        "language": "python",
    }
    (kernel_directory / "kernel.json").write_text(json.dumps(kernel_payload, indent=2) + "\n", encoding="utf-8")
    previous_jupyter_path = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(temporary_root)
    try:
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="step0112",
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
    notebook = sanitize_notebook_for_repository(execute_notebook(build_notebook()))
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, NOTEBOOK_PATH)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    counts = [cell.get("execution_count") for cell in code_cells]
    audit = {
        "status": "PASS" if not errors else "FAIL",
        "notebook": NOTEBOOK_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "cell_count": len(notebook.cells),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": sum(count is not None for count in counts),
        "output_count": sum(len(cell.get("outputs", [])) for cell in code_cells),
        "error_output_count": len(errors),
        "execution_counts": counts,
        "kernel_python": "project_virtual_environment",
        "execution_working_directory": "repository_root",
        "machine_specific_paths_removed": True,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "pretrained_weights_downloaded": False,
    }
    NOTEBOOK_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8", newline="\n")
    if audit["status"] != "PASS":
        raise RuntimeError("Sequence evidence notebook execution failed.")
    return audit


def main() -> None:
    audit = build_and_execute_notebook()
    print("Sequence notebook execution")
    for key, value in audit.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
