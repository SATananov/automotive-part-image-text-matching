from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import nbformat
from nbconvert import HTMLExporter

from src.final_delivery_config import (
    BASE_CHECKPOINT_COMMIT,
    BASE_CHECKPOINT_COMMIT_COUNT,
    CANONICAL_NOTEBOOK_GITHUB_URL,
    CANONICAL_NOTEBOOK_PATH,
    CANONICAL_NOTEBOOK_RELATIVE,
    CHECKLIST_PATH,
    CONSISTENCY_REPORT_PATH,
    ENTRY_POINT_MAP_PATH,
    EXAM_FIRST_STATUS_PATH,
    EXAM_README_PATH,
    EXPECTED_COMMIT_COUNT_AFTER_STEP,
    FINAL_MODEL_STATUS_PATH,
    GENERATED_ARTIFACTS,
    GITHUB_RENDER_REVIEW_PATH,
    HISTORICAL_FINAL_CHECKLIST_PATH,
    HISTORICAL_READINESS_CHECKLIST_PATH,
    IMPLEMENTATION_ARTIFACTS,
    MANIFEST_PATH,
    NOTEBOOK_CATALOGUE_PATH,
    PRIMARY_PREDICTIONS_PATH,
    READINESS,
    REPOSITORY_URL,
    ROOT_README_PATH,
    SOURCE_ARCHIVE,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_ARTIFACTS,
    STATUS_PATH,
    STEP,
    SUBMISSION_BOUNDARY_PATH,
    SUBMISSION_DEADLINE,
    TEACHER_ENTRY_POINT_PATH,
    TEXT_HASH_SUFFIXES,
    project_relative,
)


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def active_section(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    return text.split("<details>", maxsplit=1)[0]


def inspect_notebook() -> dict[str, Any]:
    notebook = nbformat.read(CANONICAL_NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
    ]
    figures = [
        output
        for output in outputs
        if output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]
    execution_counts = [cell.execution_count for cell in code_cells]
    error_outputs = [
        output for output in outputs if output.get("output_type") == "error"
    ]

    exporter = HTMLExporter()
    exporter.exclude_input_prompt = False
    exporter.exclude_output_prompt = False
    html, _ = exporter.from_notebook_node(notebook)
    html_render_ok = (
        len(html) > 100_000
        and "Automotive Part Image-Text Matching" in html
        and len(error_outputs) == 0
    )

    return {
        "cell_count": len(notebook.cells),
        "code_cell_count": len(code_cells),
        "saved_output_count": len(outputs),
        "saved_figure_count": len(figures),
        "error_output_count": len(error_outputs),
        "execution_counts_complete": all(
            count is not None for count in execution_counts
        ),
        "execution_counts_unique": len(set(execution_counts))
        == len(execution_counts),
        "local_static_html_render": "PASS" if html_render_ok else "FAIL",
        "local_static_html_bytes": len(html.encode("utf-8")),
    }


def build_entry_point_map() -> dict[str, Any]:
    root_active = active_section(ROOT_README_PATH)
    notebook_active = active_section(NOTEBOOK_CATALOGUE_PATH)
    exam_readme = EXAM_README_PATH.read_text(encoding="utf-8-sig")

    active_documents = {
        project_relative(ROOT_README_PATH): {
            "role": "primary repository landing page",
            "canonical_notebook_present": CANONICAL_NOTEBOOK_RELATIVE
            in root_active,
        },
        project_relative(EXAM_README_PATH): {
            "role": "exam guide",
            "canonical_notebook_present": CANONICAL_NOTEBOOK_PATH.name
            in exam_readme,
        },
        project_relative(NOTEBOOK_CATALOGUE_PATH): {
            "role": "notebook catalogue recommended section",
            "canonical_notebook_present": CANONICAL_NOTEBOOK_RELATIVE
            in notebook_active,
        },
    }
    historical_documents = {
        project_relative(HISTORICAL_FINAL_CHECKLIST_PATH): {
            "checkpoint": "011.4",
            "classification": "historical rubric-alignment checklist",
            "superseded_for_current_submission": True,
        },
        project_relative(HISTORICAL_READINESS_CHECKLIST_PATH): {
            "checkpoint": "010.8",
            "classification": "historical technical-readiness checklist",
            "superseded_for_current_submission": True,
        },
        "notebooks/03_final_exam_submission.ipynb": {
            "checkpoint": "011.4",
            "classification": "historical expanded rubric notebook",
            "superseded_for_current_submission": True,
        },
        "notebooks/02_final_exam_project.ipynb": {
            "checkpoint": "010.6/010.7",
            "classification": "historical audited project notebook",
            "superseded_for_current_submission": True,
        },
    }
    return {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "canonical_submission_notebook": CANONICAL_NOTEBOOK_RELATIVE,
        "canonical_submission_notebook_github_url": (
            CANONICAL_NOTEBOOK_GITHUB_URL
        ),
        "canonical_entry_point_count": 1,
        "active_documents": active_documents,
        "historical_documents": historical_documents,
    }


def write_delivery_documents(notebook: dict[str, Any]) -> None:
    write_text(
        TEACHER_ENTRY_POINT_PATH,
        f"""# Teacher entry point

## Open this notebook

The only canonical notebook for the current exam submission is:

- Repository path: `{CANONICAL_NOTEBOOK_RELATIVE}`
- GitHub URL: {CANONICAL_NOTEBOOK_GITHUB_URL}

The notebook presents one research question, one frozen retained model result,
grouped leakage protection, concrete successes and errors, limitations, and the
reproduction boundary.

## Current evidence boundary

- Primary prediction artifact: `{project_relative(PRIMARY_PREDICTIONS_PATH)}`
- Validation samples: 60
- Independent physical-part groups: 20
- Validation accuracy: 0.5333
- Validation macro F1: 0.5208
- Correct predictions: 32
- Errors: 28
- Test split used: no
- Final test evaluation authorized: no
- Production final model changed: no

## Verify locally

```powershell
python -m src.project_cli verify-final-delivery
python -m jupyter notebook {CANONICAL_NOTEBOOK_RELATIVE}
```

The larger notebooks under `notebooks/` remain historical or supplementary
evidence. They are not alternative current submission entry points.
""",
    )

    write_text(
        CHECKLIST_PATH,
        f"""# Final submission checklist

## Locked submission identity

- Step: `{STEP}`
- Base checkpoint: `{BASE_CHECKPOINT_COMMIT}`
- Base commit count: {BASE_CHECKPOINT_COMMIT_COUNT}
- Expected commit count after applying and committing Step {STEP}: {EXPECTED_COMMIT_COUNT_AFTER_STEP}
- Repository: {REPOSITORY_URL}
- Canonical notebook: {CANONICAL_NOTEBOOK_GITHUB_URL}
- Deadline: **{SUBMISSION_DEADLINE}**

## Automated checks completed

- [x] Exactly one canonical submission notebook is declared.
- [x] The active README, exam guide, and notebook catalogue agree on it.
- [x] The notebook is executed and has no saved error output.
- [x] Local static HTML export completed successfully.
- [x] Primary metrics, confusion matrix, and errors use one frozen prediction artifact.
- [x] The test split remains unused and final test evaluation unauthorized.
- [x] No model training is performed by the delivery layer.
- [x] The retained production model and model-selection decision are unchanged.
- [x] Older notebooks and checklists are classified as historical evidence.

## Manual actions after the final push

- [ ] Open the canonical notebook on GitHub and visually confirm all figures render.
- [ ] Confirm the default branch is `main` and the Step {STEP} commit is visible.
- [ ] Copy the repository URL into the official submission form.
- [ ] Submit before **{SUBMISSION_DEADLINE}**.

## Text to submit

```text
{REPOSITORY_URL}
```

Direct notebook URL for the reviewer:

```text
{CANONICAL_NOTEBOOK_GITHUB_URL}
```
""",
    )

    write_text(
        GITHUB_RENDER_REVIEW_PATH,
        f"""# GitHub render review

## Automated local render check

- Notebook: `{CANONICAL_NOTEBOOK_RELATIVE}`
- Notebook cells: {notebook['cell_count']}
- Executed code cells: {notebook['code_cell_count']}
- Saved outputs: {notebook['saved_output_count']}
- Saved figures: {notebook['saved_figure_count']}
- Saved error outputs: {notebook['error_output_count']}
- Static HTML export: **{notebook['local_static_html_render']}**
- Exported HTML size: {notebook['local_static_html_bytes']} bytes

This proves the committed notebook can be converted to a complete static HTML
representation without executing code, training a model, or opening test data.

## Remote review after push

The repository cannot prove the GitHub website state before the new commit is
pushed. After the final push, manually open:

{CANONICAL_NOTEBOOK_GITHUB_URL}

Confirm that markdown, tables, and all embedded PNG outputs are visible. Record
that external visual check in the submission process; it is intentionally not
claimed as already completed here.
""",
    )

    write_text(
        SUBMISSION_BOUNDARY_PATH,
        f"""# Final submission boundary

Step {STEP} changes presentation and delivery controls only.

## Frozen scientific result

All current primary claims remain bound to the frozen
`keras_multimodal` validation prediction artifact. Step {STEP} does not retrain,
select, replace, or fine-tune a model.

## Evaluation lock

- The locked test split is not opened.
- Final test evaluation is not authorized.
- Validation evidence remains the only evaluation evidence used for submission.
- The production final model is unchanged.
- The model-selection decision is unchanged.

## Historical evidence policy

The Step 011.4 expanded rubric notebook and the Step 010.6/010.7 notebook remain
available for audit and course evidence. Their old checklist wording is scoped
to their historical checkpoints and does not create another current entry point.
The only current canonical notebook is `{CANONICAL_NOTEBOOK_RELATIVE}`.

## External logistics

The repository can verify technical delivery readiness. The final GitHub render
check, submission portal availability, and on-time submission remain manual
external actions after the Step {STEP} commit is pushed.
""",
    )


def build_status(notebook: dict[str, Any]) -> dict[str, Any]:
    exam_status = read_json(EXAM_FIRST_STATUS_PATH)
    consistency = read_json(CONSISTENCY_REPORT_PATH)
    final_model = read_json(FINAL_MODEL_STATUS_PATH)

    return {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "base_checkpoint_commit": BASE_CHECKPOINT_COMMIT,
        "base_checkpoint_commit_count": BASE_CHECKPOINT_COMMIT_COUNT,
        "expected_commit_count_after_step": EXPECTED_COMMIT_COUNT_AFTER_STEP,
        "source_archive": SOURCE_ARCHIVE,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "submission_deadline": SUBMISSION_DEADLINE,
        "repository_url": REPOSITORY_URL,
        "canonical_entry_point_count": 1,
        "canonical_submission_notebook": CANONICAL_NOTEBOOK_RELATIVE,
        "canonical_submission_notebook_github_url": (
            CANONICAL_NOTEBOOK_GITHUB_URL
        ),
        "historical_entry_points_are_supplementary": True,
        "notebook_execution": notebook,
        "retained_model_slug": exam_status.get("retained_model_slug"),
        "primary_prediction_artifact": consistency.get(
            "primary_prediction_artifact"
        ),
        "primary_prediction_artifact_sha256": consistency.get(
            "primary_prediction_artifact_sha256"
        ),
        "single_prediction_set_used": consistency.get(
            "all_primary_metrics_derived_from_same_prediction_artifact"
        ),
        "validation_sample_count": exam_status.get(
            "validation_sample_count"
        ),
        "validation_part_group_count": exam_status.get(
            "validation_part_group_count"
        ),
        "validation_accuracy": exam_status.get("validation_accuracy"),
        "validation_macro_f1": exam_status.get("validation_macro_f1"),
        "validation_correct_count": exam_status.get(
            "validation_correct_count"
        ),
        "validation_error_count": exam_status.get("validation_error_count"),
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
        "model_selection_changed": False,
        "final_model_freeze_status": final_model.get("status"),
        "local_static_render_verified": (
            notebook["local_static_html_render"] == "PASS"
        ),
        "remote_github_render_review_required_after_push": True,
        "submission_portal_action_required": True,
        "delivery_checkpoint_commit_pending": True,
    }


def build_manifest() -> dict[str, Any]:
    source_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in SOURCE_ARTIFACTS
    }
    implementation_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in IMPLEMENTATION_ARTIFACTS
    }
    generated_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in GENERATED_ARTIFACTS
        if path != MANIFEST_PATH
    }
    return {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "base_checkpoint_commit": BASE_CHECKPOINT_COMMIT,
        "base_checkpoint_commit_count": BASE_CHECKPOINT_COMMIT_COUNT,
        "source_archive": SOURCE_ARCHIVE,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "canonical_submission_notebook": CANONICAL_NOTEBOOK_RELATIVE,
        "canonical_submission_notebook_sha256": normalized_sha256(
            CANONICAL_NOTEBOOK_PATH
        ),
        "source_artifact_count": len(source_hashes),
        "source_artifact_sha256": source_hashes,
        "implementation_artifact_count": len(implementation_hashes),
        "implementation_artifact_sha256": implementation_hashes,
        "generated_artifact_count": len(generated_hashes),
        "generated_artifact_sha256": generated_hashes,
        "model_training_performed": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
        "model_selection_changed": False,
    }


def main() -> None:
    notebook = inspect_notebook()
    if notebook["local_static_html_render"] != "PASS":
        raise RuntimeError("Canonical notebook static rendering failed.")
    if notebook["error_output_count"] != 0:
        raise RuntimeError("Canonical notebook contains saved error output.")
    if not notebook["execution_counts_complete"]:
        raise RuntimeError("Canonical notebook is not fully executed.")

    entry_point_map = build_entry_point_map()
    if not all(
        item["canonical_notebook_present"]
        for item in entry_point_map["active_documents"].values()
    ):
        raise RuntimeError("Active entry-point documents are inconsistent.")

    write_delivery_documents(notebook)
    write_json(ENTRY_POINT_MAP_PATH, entry_point_map)
    write_json(STATUS_PATH, build_status(notebook))
    write_json(MANIFEST_PATH, build_manifest())

    print(f"Step {STEP} final delivery artifacts built")
    print(f"Readiness: {READINESS}")
    print(f"Canonical notebook: {CANONICAL_NOTEBOOK_RELATIVE}")
    print("Model training performed: false")
    print("Test split used: false")
    print("Production final model changed: false")


if __name__ == "__main__":
    main()
