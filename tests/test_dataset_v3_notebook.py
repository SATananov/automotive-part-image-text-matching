from __future__ import annotations

from pathlib import Path

from src.build_notebook_v3 import NOTEBOOK_PATH, build_notebook
from src.verify_notebook_v3 import verify_notebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_v3_notebook_builder_has_expected_structure():
    notebook = build_notebook()
    assert len(notebook["cells"]) == 25
    assert sum(cell["cell_type"] == "code" for cell in notebook["cells"]) == 14
    assert sum(cell["cell_type"] == "markdown" for cell in notebook["cells"]) == 11


def test_dataset_v3_notebook_is_executed_and_verified():
    assert NOTEBOOK_PATH == PROJECT_ROOT / "project_v3.ipynb"
    summary = verify_notebook()
    assert summary["status"] == "PASS_DATASET_V3_EXECUTED_NOTEBOOK_VERIFIED"
    assert summary["executed_code_cells"] == 14
    assert summary["error_outputs"] == 0


def test_dataset_v3_notebook_does_not_replace_dataset_v2_notebook():
    assert (PROJECT_ROOT / "project.ipynb").is_file()
    assert (PROJECT_ROOT / "project_v3.ipynb").is_file()
    assert (PROJECT_ROOT / "project.ipynb") != NOTEBOOK_PATH


def test_dataset_v3_notebook_code_has_no_locked_test_path():
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "data/locked_test" not in text
    assert "dataset_v3_test_images_LOCKED.csv" not in text
    assert 'load_v3_split(\\"test\\")' not in text
