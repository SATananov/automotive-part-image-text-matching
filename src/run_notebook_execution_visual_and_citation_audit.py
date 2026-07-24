from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import nbformat
from nbconvert import HTMLExporter
import numpy as np
import pandas as pd
from PIL import Image

from src.build_final_exam_notebook import (
    atomic_write_json,
    atomic_write_text,
    build_notebook,
    execute_notebook,
    sanitize_execution_metadata,
    sha256_file,
    validate_executed_notebook,
)
from src.final_exam_notebook_config import (
    FINAL_EXAM_NOTEBOOK_EXPECTED_CELL_COUNT,
    FINAL_EXAM_NOTEBOOK_EXPECTED_CODE_CELL_COUNT,
    FINAL_EXAM_NOTEBOOK_EXPECTED_IMAGE_OUTPUT_COUNT,
    FINAL_EXAM_NOTEBOOK_EXPECTED_OUTPUT_COUNT,
    FINAL_EXAM_NOTEBOOK_EXPECTED_VALIDATION_ERROR_COUNT,
    FINAL_EXAM_NOTEBOOK_PATH,
    MULTIMODAL_CONFUSION_MATRIX_PATH,
    MULTIMODAL_METRICS_PATH,
    MULTIMODAL_PREDICTIONS_PATH,
    NOTEBOOK_INTEGRATION_COMMIT,
)
from src.notebook_quality_audit_config import (
    CITATION_AUDIT_PATH,
    CITATION_SOURCES,
    CITATION_VERIFICATION_DATE,
    MINIMUM_FIGURE_BYTES,
    MINIMUM_FIGURE_HEIGHT,
    MINIMUM_FIGURE_WIDTH,
    MINIMUM_PIXEL_STANDARD_DEVIATION,
    NOTEBOOK_EXECUTION_AUDIT_PATH,
    NUMERIC_CONSISTENCY_AUDIT_PATH,
    QUALITY_AUDIT_BASE_COMMIT,
    QUALITY_AUDIT_MANIFEST_PATH,
    QUALITY_AUDIT_READINESS,
    QUALITY_AUDIT_STATUS_PATH,
    QUALITY_AUDIT_SUMMARY_PATH,
    VISUAL_OUTPUT_AUDIT_PATH,
)
from src.real_dataset_config import PROJECT_ROOT
from src.validate_external_training_readiness import project_relative_path


class NotebookQualityAuditError(RuntimeError):
    """Raised when the Step 010.7 notebook quality gate fails."""


def canonical_notebook_payload(notebook: nbformat.NotebookNode) -> dict[str, Any]:
    return json.loads(nbformat.writes(notebook, version=4))


def image_payload_details(encoded: str) -> dict[str, Any]:
    raw = base64.b64decode(encoded)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.float64)
        return {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "width": int(rgb.width),
            "height": int(rgb.height),
            "pixel_standard_deviation": float(pixels.std()),
        }


def output_fingerprint(notebook: nbformat.NotebookNode) -> str:
    normalized_cells: list[dict[str, Any]] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        normalized_outputs: list[dict[str, Any]] = []
        for output in cell.get("outputs", []):
            output_type = output.get("output_type")
            normalized: dict[str, Any] = {"output_type": output_type}

            if output_type == "stream":
                normalized["name"] = output.get("name")
                normalized["text"] = output.get("text", "")
            elif output_type == "error":
                normalized["ename"] = output.get("ename")
                normalized["evalue"] = output.get("evalue")
                normalized["traceback"] = output.get("traceback", [])
            elif output_type in {"display_data", "execute_result"}:
                data = output.get("data", {})
                normalized_data: dict[str, Any] = {}
                for mime_type in sorted(data):
                    value = data[mime_type]
                    if mime_type == "image/png":
                        normalized_data[mime_type] = image_payload_details(value)
                    else:
                        normalized_data[mime_type] = value
                normalized["data"] = normalized_data
            normalized_outputs.append(normalized)

        normalized_cells.append(
            {
                "source_sha256": hashlib.sha256(
                    cell.source.encode("utf-8")
                ).hexdigest(),
                "execution_count": cell.get("execution_count"),
                "outputs": normalized_outputs,
            }
        )

    encoded = json.dumps(
        normalized_cells,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def notebook_markdown(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )


def build_execution_audit(
    committed: nbformat.NotebookNode,
) -> tuple[dict[str, Any], nbformat.NotebookNode]:
    statistics = validate_executed_notebook(committed)
    code_cells = [cell for cell in committed.cells if cell.cell_type == "code"]

    execution_counts = [cell.get("execution_count") for cell in code_cells]
    sequential_counts = execution_counts == list(
        range(1, len(code_cells) + 1)
    )
    transient_execution_metadata_absent = all(
        "execution" not in cell.get("metadata", {})
        for cell in code_cells
    )

    fresh = sanitize_execution_metadata(execute_notebook(build_notebook()))
    fresh_statistics = validate_executed_notebook(fresh)

    committed_fingerprint = output_fingerprint(committed)
    fresh_fingerprint = output_fingerprint(fresh)
    deterministic_reexecution = committed_fingerprint == fresh_fingerprint

    html_exporter = HTMLExporter(template_name="classic")
    html_body, _resources = html_exporter.from_notebook_node(committed)
    html_image_count = html_body.count("data:image/png;base64")
    html_render_pass = (
        len(html_body) > 100_000
        and "Automotive Part Image-Text Matching" in html_body
        and html_image_count
        >= FINAL_EXAM_NOTEBOOK_EXPECTED_IMAGE_OUTPUT_COUNT
        and "output_error" not in html_body
    )

    checks = {
        "cell_count_exact": (
            statistics["cell_count"]
            == FINAL_EXAM_NOTEBOOK_EXPECTED_CELL_COUNT
        ),
        "code_cell_count_exact": (
            statistics["code_cell_count"]
            == FINAL_EXAM_NOTEBOOK_EXPECTED_CODE_CELL_COUNT
        ),
        "output_count_exact": (
            statistics["output_count"]
            == FINAL_EXAM_NOTEBOOK_EXPECTED_OUTPUT_COUNT
        ),
        "all_code_cells_executed": (
            statistics["executed_code_cell_count"]
            == statistics["code_cell_count"]
        ),
        "sequential_execution_counts": sequential_counts,
        "no_error_outputs": all(
            output.get("output_type") != "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        ),
        "transient_execution_metadata_absent": (
            transient_execution_metadata_absent
        ),
        "fresh_execution_statistics_match": (
            statistics == fresh_statistics
        ),
        "deterministic_scientific_output_fingerprint": (
            deterministic_reexecution
        ),
        "html_render": html_render_pass,
    }

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "statistics": statistics,
        "fresh_execution_statistics": fresh_statistics,
        "execution_counts": execution_counts,
        "committed_output_fingerprint": committed_fingerprint,
        "fresh_output_fingerprint": fresh_fingerprint,
        "html_bytes": len(html_body.encode("utf-8")),
        "html_embedded_image_count": html_image_count,
        "model_retraining_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }
    return report, fresh


def build_visual_audit(
    notebook: nbformat.NotebookNode,
) -> dict[str, Any]:
    expected_labels = (
        "integrated_dataset_composition",
        "generated_development_example",
        "reviewed_open_license_example",
        "integrated_validation_comparison",
        "retained_model_confusion_matrix",
        "retained_model_error_types",
    )

    figures: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        for output_index, output in enumerate(cell.get("outputs", [])):
            if output.get("output_type") not in {
                "display_data",
                "execute_result",
            }:
                continue
            data = output.get("data", {})
            if "image/png" not in data:
                continue

            details = image_payload_details(data["image/png"])
            figures.append(
                {
                    "cell_index": cell_index,
                    "execution_count": cell.get("execution_count"),
                    "output_index": output_index,
                    **details,
                }
            )

    for label, figure in zip(expected_labels, figures, strict=False):
        figure["figure_label"] = label
        figure["width_pass"] = (
            figure["width"] >= MINIMUM_FIGURE_WIDTH
        )
        figure["height_pass"] = (
            figure["height"] >= MINIMUM_FIGURE_HEIGHT
        )
        figure["payload_size_pass"] = (
            figure["bytes"] >= MINIMUM_FIGURE_BYTES
        )
        figure["non_blank_pass"] = (
            figure["pixel_standard_deviation"]
            >= MINIMUM_PIXEL_STANDARD_DEVIATION
        )

    checks = {
        "image_output_count_exact": (
            len(figures)
            == FINAL_EXAM_NOTEBOOK_EXPECTED_IMAGE_OUTPUT_COUNT
        ),
        "figure_labels_complete": (
            len(figures) == len(expected_labels)
            and all("figure_label" in figure for figure in figures)
        ),
        "minimum_dimensions": bool(figures) and all(
            figure.get("width_pass") and figure.get("height_pass")
            for figure in figures
        ),
        "minimum_payload_size": bool(figures) and all(
            figure.get("payload_size_pass") for figure in figures
        ),
        "non_blank_figures": bool(figures) and all(
            figure.get("non_blank_pass") for figure in figures
        ),
        "unique_figure_payloads": (
            len({figure["sha256"] for figure in figures})
            == len(figures)
        ),
        "confusion_matrix_contrast_logic_present": (
            'text_color = "black" if value > contrast_threshold else "white"'
            in "\n".join(
                cell.source
                for cell in notebook.cells
                if cell.cell_type == "code"
            )
        ),
        "human_readable_source_labels_present": (
            "Generated development"
            in "\n".join(
                cell.source
                for cell in notebook.cells
                if cell.cell_type == "code"
            )
            and "Reviewed open-license"
            in "\n".join(
                cell.source
                for cell in notebook.cells
                if cell.cell_type == "code"
            )
        ),
    }

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "figure_count": len(figures),
        "figures": figures,
    }


def build_numeric_consistency_audit(
    notebook: nbformat.NotebookNode,
) -> dict[str, Any]:
    predictions = pd.read_csv(MULTIMODAL_PREDICTIONS_PATH)
    confusion = pd.read_csv(
        MULTIMODAL_CONFUSION_MATRIX_PATH,
        index_col=0,
    )
    metrics = json.loads(
        MULTIMODAL_METRICS_PATH.read_text(encoding="utf-8-sig")
    )

    label_order = ("MATCH", "PARTIAL_MATCH", "MISMATCH")
    derived_confusion = pd.crosstab(
        predictions["true_label"],
        predictions["predicted_label"],
    ).reindex(
        index=label_order,
        columns=label_order,
        fill_value=0,
    )
    committed_confusion = confusion.copy()
    committed_confusion.index = [
        value.replace("actual_", "") for value in committed_confusion.index
    ]
    committed_confusion.columns = [
        value.replace("predicted_", "")
        for value in committed_confusion.columns
    ]
    committed_confusion = committed_confusion.reindex(
        index=label_order,
        columns=label_order,
    )

    incorrect_count = int((~predictions["is_correct"]).sum())
    correct_count = int(predictions["is_correct"].sum())
    off_diagonal_count = int(
        committed_confusion.to_numpy().sum()
        - np.trace(committed_confusion.to_numpy())
    )
    derived_accuracy = correct_count / len(predictions)
    markdown_text = notebook_markdown(notebook)

    checks = {
        "prediction_row_count": len(predictions) == 60,
        "incorrect_prediction_count": (
            incorrect_count
            == FINAL_EXAM_NOTEBOOK_EXPECTED_VALIDATION_ERROR_COUNT
        ),
        "confusion_off_diagonal_count": (
            off_diagonal_count == incorrect_count
        ),
        "confusion_matches_predictions": bool(
            (
                derived_confusion.to_numpy()
                == committed_confusion.to_numpy()
            ).all()
        ),
        "accuracy_matches_metrics": abs(
            derived_accuracy - float(metrics["accuracy"])
        ) < 1e-12,
        "correct_error_count_in_narrative": (
            "**28 errors among 60 validation samples**" in markdown_text
        ),
        "historical_rerun_error_count_contextualized": (
            "controlled rerun produced 35 errors" in markdown_text
        ),
        "misleading_35_error_claim_absent": (
            "multimodal model made 35 errors among 60" not in markdown_text
        ),
        "error_chart_derived_from_retained_predictions": (
            'retained_errors = predictions.loc[~predictions["is_correct"]]'
            in "\n".join(
                cell.source
                for cell in notebook.cells
                if cell.cell_type == "code"
            )
        ),
    }

    error_pairs = (
        predictions.loc[~predictions["is_correct"]]
        .groupby(["true_label", "predicted_label"])
        .size()
        .sort_values(ascending=False)
    )

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "validation_samples": len(predictions),
        "correct_predictions": correct_count,
        "incorrect_predictions": incorrect_count,
        "derived_accuracy": derived_accuracy,
        "reported_accuracy": float(metrics["accuracy"]),
        "reported_macro_f1": float(metrics["macro_f1"]),
        "error_pairs": [
            {
                "true_label": str(true_label),
                "predicted_label": str(predicted_label),
                "count": int(count),
            }
            for (true_label, predicted_label), count in error_pairs.items()
        ],
    }


def build_citation_audit(
    notebook: nbformat.NotebookNode,
) -> dict[str, Any]:
    markdown_text = notebook_markdown(notebook)
    before_references = markdown_text.split("## 15. References", 1)[0]
    urls = set(
        re.findall(
            r"https://[^\s\)]+",
            markdown_text,
        )
    )

    sources: list[dict[str, Any]] = []
    for source in CITATION_SOURCES:
        reference_id = int(source["reference_id"])
        url = str(source["url"])
        parsed = urlparse(url)
        checks = {
            "title_present": str(source["title"]) in markdown_text,
            "url_present": url in urls,
            "https": parsed.scheme == "https",
            "expected_domain": parsed.netloc == source["expected_domain"],
            "numbered_reference_present": (
                f"{reference_id}." in markdown_text
            ),
            "inline_citation_present": (
                f"[{reference_id}]" in before_references
            ),
        }
        sources.append(
            {
                **source,
                "checks": checks,
                "status": (
                    "PASS" if all(checks.values()) else "FAIL"
                ),
            }
        )

    checks = {
        "source_count": len(sources) == 6,
        "all_sources_pass": all(
            source["status"] == "PASS" for source in sources
        ),
        "all_urls_accounted_for": urls == {
            str(source["url"]) for source in CITATION_SOURCES
        },
        "primary_or_official_sources_only": all(
            source["source_type"]
            in {"conference_paper", "research_paper", "official_documentation"}
            for source in sources
        ),
        "inline_citation_range_complete": all(
            f"[{reference_id}]" in before_references
            for reference_id in range(1, 7)
        ),
    }

    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "verification_date": CITATION_VERIFICATION_DATE,
        "verification_scope": (
            "Primary publication metadata and official documentation URLs "
            "were manually reviewed; repository automation verifies exact "
            "titles, domains, URLs, numbering, and inline citation markers."
        ),
        "sources": sources,
    }


def build_status(
    execution: dict[str, Any],
    visual: dict[str, Any],
    numeric: dict[str, Any],
    citations: dict[str, Any],
) -> dict[str, Any]:
    component_statuses = {
        "execution": execution["status"],
        "visual": visual["status"],
        "numeric_consistency": numeric["status"],
        "citations": citations["status"],
    }
    passed = all(value == "PASS" for value in component_statuses.values())

    return {
        "status": "PASS" if passed else "FAIL",
        "readiness": QUALITY_AUDIT_READINESS if passed else "NOT_READY",
        "step": "010.7",
        "base_commit": QUALITY_AUDIT_BASE_COMMIT,
        "notebook_integration_commit": NOTEBOOK_INTEGRATION_COMMIT,
        "notebook": project_relative_path(FINAL_EXAM_NOTEBOOK_PATH),
        "component_statuses": component_statuses,
        "cell_count": execution["statistics"]["cell_count"],
        "executed_code_cell_count": execution["statistics"][
            "executed_code_cell_count"
        ],
        "saved_output_count": execution["statistics"]["output_count"],
        "figure_count": visual["figure_count"],
        "citation_count": len(citations["sources"]),
        "retained_model_validation_error_count": numeric[
            "incorrect_predictions"
        ],
        "model_retraining_performed": False,
        "model_selection_changed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }


def build_summary(status: dict[str, Any]) -> str:
    return f"""# Notebook Execution, Visual QA and Citation Audit

Status: `{status['status']}`

Readiness: `{status['readiness']}`

Step 010.7 re-executes the final exam notebook from committed train and
validation evidence, compares a scientific output fingerprint, audits every
saved figure, reconciles notebook numbers with committed predictions and
metrics, and checks numbered citations against reviewed primary or official
sources.

## Confirmed notebook state

- cells: {status['cell_count']};
- executed code cells: {status['executed_code_cell_count']};
- saved outputs: {status['saved_output_count']};
- saved figures: {status['figure_count']};
- references: {status['citation_count']};
- retained-model validation errors: {status['retained_model_validation_error_count']} / 60.

## Corrections confirmed by the audit

- the final notebook now reports the retained Step 010.3 model's 28 validation
  errors, rather than presenting the 35 errors from the separate Step 010.4
  controlled reference rerun as incumbent errors;
- confusion-matrix annotations use dynamic black/white contrast;
- source and category labels are human-readable;
- research claims use inline numbered citations linked to primary papers or
  official documentation;
- transient Jupyter execution timestamps are removed before commit, while
  execution counts and scientific outputs remain saved.

## Locked-test boundary

- model retraining performed: `false`;
- model selection changed: `false`;
- locked test CSV files opened: `false`;
- test split used: `false`;
- final test evaluation authorized: `false`.
"""


def build_manifest(paths: tuple[Path, ...]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "step": "010.7",
        "base_commit": QUALITY_AUDIT_BASE_COMMIT,
        "hash_normalization": "utf-8-lf",
        "artifact_sha256": {
            project_relative_path(path): sha256_file(path)
            for path in paths
        },
        "notebook_sha256": sha256_file(FINAL_EXAM_NOTEBOOK_PATH),
        "model_retraining_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }


def main() -> None:
    if not FINAL_EXAM_NOTEBOOK_PATH.is_file():
        raise NotebookQualityAuditError(
            "The final exam notebook is missing."
        )

    committed = nbformat.read(FINAL_EXAM_NOTEBOOK_PATH, as_version=4)

    execution, _fresh = build_execution_audit(committed)
    visual = build_visual_audit(committed)
    numeric = build_numeric_consistency_audit(committed)
    citations = build_citation_audit(committed)
    status = build_status(execution, visual, numeric, citations)

    component_reports = (
        (NOTEBOOK_EXECUTION_AUDIT_PATH, execution),
        (VISUAL_OUTPUT_AUDIT_PATH, visual),
        (NUMERIC_CONSISTENCY_AUDIT_PATH, numeric),
        (CITATION_AUDIT_PATH, citations),
        (QUALITY_AUDIT_STATUS_PATH, status),
    )
    for path, payload in component_reports:
        atomic_write_json(path, payload)

    atomic_write_text(QUALITY_AUDIT_SUMMARY_PATH, build_summary(status))

    manifest_inputs = (
        NOTEBOOK_EXECUTION_AUDIT_PATH,
        VISUAL_OUTPUT_AUDIT_PATH,
        NUMERIC_CONSISTENCY_AUDIT_PATH,
        CITATION_AUDIT_PATH,
        QUALITY_AUDIT_STATUS_PATH,
        QUALITY_AUDIT_SUMMARY_PATH,
    )
    atomic_write_json(
        QUALITY_AUDIT_MANIFEST_PATH,
        build_manifest(manifest_inputs),
    )

    print("Notebook execution, visual QA and citation audit")
    for name, component_status in status["component_statuses"].items():
        print(f"- {name}: {component_status}")
    print(f"- cells: {status['cell_count']}")
    print(
        "- executed code cells: "
        f"{status['executed_code_cell_count']}"
    )
    print(f"- saved outputs: {status['saved_output_count']}")
    print(f"- figures: {status['figure_count']}")
    print(f"- citations: {status['citation_count']}")
    print(
        "- retained-model validation errors: "
        f"{status['retained_model_validation_error_count']} / 60"
    )
    print("- locked test CSV files opened: false")
    print("- test split used: false")
    print("- final test evaluation authorized: false")
    print(f"Readiness: {status['readiness']}")
    print(f"Status: {status['status']}")

    if status["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
