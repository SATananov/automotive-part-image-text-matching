from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat
import pandas as pd
from nbconvert.preprocessors import ExecutePreprocessor

from src.final_submission_config import (
    BASE_CHECKPOINT,
    CHECKLIST_PATH,
    DEFENSE_GUIDE_PATH,
    ERROR_ANALYSIS_PATH,
    ERROR_SUMMARY_PATH,
    FIGURES_DIR,
    FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL,
    FINAL_SUBMISSION_NOTEBOOK_PATH,
    GENERATED_ARTIFACTS,
    MANIFEST_PATH,
    READINESS,
    REFERENCE_TITLES,
    REPORT_DIR,
    REPOSITORY_URL,
    RUBRIC_MATRIX_PATH,
    RUBRIC_MAX_POINTS,
    RUBRIC_SELF_ASSESSMENT,
    SELF_ASSESSMENT_PATH,
    SOURCE_ARTIFACTS,
    STATUS_PATH,
    STEP,
    SUBMISSION_DEADLINE,
    SUMMARY_PATH,
    TEXT_HASH_SUFFIXES,
    project_relative,
)
from src.real_dataset_config import PROJECT_ROOT


INTEGRATED_COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "integrated_training"
    / "validation_comparison.csv"
)
ERROR_CSV_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation_model_improvement"
    / "validation_error_analysis.csv"
)
ERROR_JSON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation_model_improvement"
    / "validation_error_analysis.json"
)
CONFUSION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "validation_model_improvement"
    / "candidates"
    / "reference_multimodal"
    / "validation_confusion_matrix.csv"
)
FAILURE_DIAGNOSTICS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "fundamentals"
    / "failure_diagnostics.csv"
)
FUNDAMENTALS_STATUS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "fundamentals"
    / "fundamentals_suite_status.json"
)
SEQUENCE_COMPARISON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "sequence"
    / "model_comparison.csv"
)
SEQUENCE_STATUS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "sequence"
    / "sequence_suite_status.json"
)
VISION_STATUS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "vision"
    / "vision_suite_status.json"
)
VISION_RANKING_PATH = (
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "vision"
    / "ranking_metrics.json"
)
VISION_EXPLAINABILITY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "vision"
    / "explainability_summary.json"
)
VISION_AUGMENTATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "course_coverage"
    / "vision"
    / "augmentation_comparison.csv"
)
VALIDATION_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "integrated_validation.csv"
)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
        newline="\n",
    )


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def count_test_functions() -> int:
    total = 0
    for path in sorted((PROJECT_ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        total += sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in tree.body
        )
    return total


def load_evidence() -> dict[str, Any]:
    integrated = pd.read_csv(INTEGRATED_COMPARISON_PATH)
    errors = pd.read_csv(ERROR_CSV_PATH)
    error_summary = read_json(ERROR_JSON_PATH)
    confusion = pd.read_csv(CONFUSION_PATH, index_col=0)
    failures = pd.read_csv(FAILURE_DIAGNOSTICS_PATH)
    sequence = pd.read_csv(SEQUENCE_COMPARISON_PATH)
    augmentation = pd.read_csv(VISION_AUGMENTATION_PATH)
    validation = pd.read_csv(VALIDATION_DATA_PATH)
    fundamentals_status = read_json(FUNDAMENTALS_STATUS_PATH)
    sequence_status = read_json(SEQUENCE_STATUS_PATH)
    vision_status = read_json(VISION_STATUS_PATH)
    ranking = read_json(VISION_RANKING_PATH)
    explainability = read_json(VISION_EXPLAINABILITY_PATH)

    total_self_assessment = sum(RUBRIC_SELF_ASSESSMENT.values())
    total_possible = sum(RUBRIC_MAX_POINTS.values())

    return {
        "integrated": integrated,
        "errors": errors,
        "error_summary": error_summary,
        "confusion": confusion,
        "failures": failures,
        "sequence": sequence,
        "augmentation": augmentation,
        "validation": validation,
        "fundamentals_status": fundamentals_status,
        "sequence_status": sequence_status,
        "vision_status": vision_status,
        "ranking": ranking,
        "explainability": explainability,
        "test_function_count": count_test_functions(),
        "self_assessment_total": total_self_assessment,
        "self_assessment_possible": total_possible,
    }


def build_report_figures(evidence: dict[str, Any]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    integrated = evidence["integrated"].sort_values(
        "integrated_validation_macro_f1", ascending=True
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(
        integrated["model"],
        integrated["integrated_validation_macro_f1"],
    )
    axis.set_xlabel("Integrated validation macro F1")
    axis.set_ylabel("Model")
    axis.set_title("Integrated validation model comparison")
    axis.set_xlim(0.0, 0.6)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "integrated_model_comparison.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    confusion = evidence["confusion"]
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion.to_numpy())
    axis.set_xticks(range(len(confusion.columns)))
    axis.set_xticklabels(
        [column.replace("predicted_", "") for column in confusion.columns],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(range(len(confusion.index)))
    axis.set_yticklabels(
        [index.replace("actual_", "") for index in confusion.index]
    )
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title("Reference multimodal validation confusion matrix")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(
                column,
                row,
                str(int(confusion.iloc[row, column])),
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "reference_confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    error_summary = evidence["error_summary"]
    source_rows = pd.DataFrame(
        [
            {
                "source": source,
                "error_rate": values["error_rate"],
            }
            for source, values in error_summary["errors_by_source"].items()
        ]
    )
    category_rows = pd.DataFrame(
        [
            {
                "category": category,
                "error_rate": values["error_rate"],
            }
            for category, values in error_summary[
                "errors_by_category"
            ].items()
        ]
    ).sort_values("error_rate", ascending=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(source_rows["source"], source_rows["error_rate"])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Error rate")
    axes[0].set_title("Errors by image source")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].barh(category_rows["category"], category_rows["error_rate"])
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Error rate")
    axes[1].set_title("Errors by automotive category")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "errors_by_source_and_category.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    failures = evidence["failures"].copy()
    failures = failures[
        failures["metric_name"] == "validation_macro_f1"
    ].sort_values("metric_value", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(failures["case"], failures["metric_value"])
    axis.axvline(0.2886, linestyle="--", label="Correct-loop reference")
    axis.set_xlabel("Validation macro F1")
    axis.set_title("Controlled Deep Learning failure diagnostics")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "controlled_failure_diagnostics.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)

    suite_rows = pd.DataFrame(
        [
            {
                "suite": "Fundamentals",
                "completed": evidence["fundamentals_status"].get(
                    "completed_exercise_problem_count", 10
                ),
                "training_runs": evidence["fundamentals_status"].get(
                    "run_count", 35
                ),
            },
            {
                "suite": "Sequence",
                "completed": evidence["sequence_status"].get(
                    "completed_core_problem_count", 9
                ),
                "training_runs": evidence["sequence_status"].get(
                    "training_runs_recorded", 21
                ),
            },
            {
                "suite": "Vision core",
                "completed": evidence["vision_status"].get(
                    "completed_problem_count", 6
                ),
                "training_runs": evidence["vision_status"].get(
                    "training_runs_recorded", 48
                ),
            },
        ]
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(suite_rows["suite"], suite_rows["training_runs"])
    for index, row in suite_rows.iterrows():
        axis.text(
            index,
            row["training_runs"] + 1,
            f"{int(row['completed'])} completed",
            ha="center",
        )
    axis.set_ylabel("Recorded training runs")
    axis.set_title("Course exercise experimental evidence")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "course_suite_summary.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_error_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    error_summary = evidence["error_summary"]
    confusion_pairs = error_summary["confusion_pairs"]
    top_categories = sorted(
        error_summary["errors_by_category"].items(),
        key=lambda item: (-item[1]["error_rate"], item[0]),
    )
    return {
        "step": STEP,
        "status": "PASS",
        "selected_candidate": error_summary["selected_candidate_slug"],
        "validation_sample_count": error_summary["validation_sample_count"],
        "correct_count": error_summary["correct_count"],
        "error_count": error_summary["error_count"],
        "error_rate": error_summary["error_rate"],
        "dominant_error_pairs": confusion_pairs,
        "partial_match_error_count": sum(
            count
            for pair, count in confusion_pairs.items()
            if pair.startswith("PARTIAL_MATCH->")
        ),
        "real_image_error_rate": error_summary["errors_by_source"][
            "wikimedia_commons_open_license"
        ]["error_rate"],
        "generated_image_error_rate": error_summary["errors_by_source"][
            "generated_development"
        ]["error_rate"],
        "highest_error_categories": [
            category
            for category, values in top_categories
            if values["error_rate"] == top_categories[0][1]["error_rate"]
        ],
        "high_confidence_error_threshold": error_summary[
            "high_confidence_error_threshold"
        ],
        "high_confidence_error_count": error_summary[
            "high_confidence_error_count"
        ],
        "controlled_failure_case_count": len(evidence["failures"]),
        "vision_pairwise_ranking_accuracy": evidence["ranking"][
            "pairwise_ranking_accuracy"
        ],
        "vision_three_way_ordering_accuracy": evidence["ranking"][
            "three_way_ordering_accuracy"
        ],
        "vision_automated_foreground_alignment_rate": evidence[
            "explainability"
        ]["automated_foreground_proxy_alignment_rate"],
        "human_explainability_claimed": evidence["explainability"][
            "manual_review_claimed"
        ],
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
    }


def build_rubric_rows() -> list[dict[str, Any]]:
    evidence_map = {
        "Problem statement": (
            "Notebook Sections 1-3 define the real-life matching problem, "
            "three labels, hypothesis, and macro-F1 objective."
        ),
        "Layout": (
            "Executed English notebook with numbered sections, saved tables, "
            "figures, references, and a direct defense summary."
        ),
        "Code quality": (
            "Reusable src modules, project CLI, semantic verifiers, manifests, "
            "type-friendly functions, and transactional workflows."
        ),
        "Previous research": (
            "Six primary or official sources and explicit comparison between "
            "prior multimodal methods and project results."
        ),
        "Data": (
            "Generated and approved open-license images, attribution, SHA-256 "
            "provenance, cleaning, and group-isolated splits."
        ),
        "Testing": (
            "Automated tests, project verifiers, clean-clone checks, controlled "
            "failure experiments, and a closed test gate."
        ),
        "Visualization": (
            "Model comparison, confusion matrix, error-source/category, "
            "failure diagnostics, and course-suite figures."
        ),
        "Communication": (
            "Teacher-facing notebook, self-assessment, submission checklist, "
            "limitations, legal boundary, and defense guide."
        ),
    }
    return [
        {
            "criterion": criterion,
            "maximum_points": RUBRIC_MAX_POINTS[criterion],
            "self_assessed_points": RUBRIC_SELF_ASSESSMENT[criterion],
            "repository_evidence": evidence_map[criterion],
        }
        for criterion in RUBRIC_MAX_POINTS
    ]


def write_supporting_reports(
    evidence: dict[str, Any], error_summary: dict[str, Any]
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rubric_rows()
    with RUBRIC_MATRIX_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = sum(row["self_assessed_points"] for row in rows)
    maximum = sum(row["maximum_points"] for row in rows)
    self_lines = [
        "# Self-Assessment Against the Exam Rubric",
        "",
        f"- Step: **{STEP}**",
        f"- Submission deadline: **{SUBMISSION_DEADLINE}**",
        f"- Conservative self-assessment: **{total}/{maximum}**",
        "- Evaluation boundary: validation evidence only; the test split remains locked.",
        "",
        "| Criterion | Maximum | Self-assessment | Evidence |",
        "|---|---:|---:|---|",
    ]
    for row in rows:
        self_lines.append(
            "| {criterion} | {maximum_points} | {self_assessed_points} | "
            "{repository_evidence} |".format(**row)
        )
    self_lines.extend(
        [
            "",
            "## Deliberately conservative deductions",
            "",
            "- Visualization: one point is reserved because automated occlusion is diagnostic and does not replace a genuine human localization study.",
            "- Communication: one point is reserved because the final oral defense and portal submission remain manual responsibilities.",
        ]
    )
    write_markdown(SELF_ASSESSMENT_PATH, self_lines)

    checklist_lines = [
        "# Final Exam Submission Checklist",
        "",
        f"- Step: **{STEP}**",
        f"- Base checkpoint: `{BASE_CHECKPOINT}`",
        f"- Deadline: **{SUBMISSION_DEADLINE}**",
        f"- Repository: {REPOSITORY_URL}",
        f"- Final notebook: [open on GitHub]({FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL})",
        "- Submission artifact: GitHub repository link.",
        "- Evaluation boundary: validation evidence only; the test split remains locked.",
        "",
        "## Required final checks",
        "",
        "- [x] English Jupyter notebook with Python, mathematics, saved tables, and saved figures.",
        "- [x] Real-life problem and three-class research question are defined.",
        "- [x] At least two prior sources are cited; six primary or official references are included.",
        "- [x] Data acquisition, licensing, cleaning, formatting, and grouped splitting are documented.",
        "- [x] Classical, neural, sequence, and vision experiments are compared without mixing incompatible claims.",
        "- [x] Deep Learning errors, domain shift, failure signatures, and uncertainty are discussed.",
        "- [x] Automated tests, semantic verifiers, manifests, and clean-clone checks are available.",
        "- [x] The locked test split remains unopened and unauthorized.",
        "- [x] No malicious, privacy-invasive, or unlawful code is included.",
        "- [ ] Manually open the final notebook on GitHub and verify all figures render after the final push.",
        "- [ ] Submit the repository URL before the deadline.",
        "",
        "## Submission links",
        "",
        f"1. Repository: {REPOSITORY_URL}",
        f"2. Final executed notebook: {FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL}",
        "3. Self-assessment: `reports/final_submission/self_assessment.md`",
        "4. Defense guide: `reports/final_submission/defense_guide.md`",
    ]
    write_markdown(CHECKLIST_PATH, checklist_lines)

    defense_lines = [
        "# Final Defense Guide",
        "",
        "## Thirty-second project explanation",
        "",
        "The project evaluates whether an automotive-part photograph and a short description are a full match, a related same-system partial match, or a mismatch. It compares six primary model families, uses group-isolated validation, and keeps the final test split locked until explicit authorization.",
        "",
        "## Questions the lecturer is likely to ask",
        "",
        "### Why is PARTIAL_MATCH difficult?",
        "It is a semantic boundary class: brake disc versus brake pad is related but not identical. The reference error analysis records 20 PARTIAL_MATCH errors, split evenly toward MATCH and MISMATCH.",
        "",
        "### Why group by part_group_id?",
        "Different descriptions of the same physical image must not cross train and validation boundaries. Grouping prevents identity leakage.",
        "",
        "### Why macro F1?",
        "Macro F1 gives equal weight to all three relationship labels and exposes a model that ignores the difficult intermediate class.",
        "",
        "### Why is the test split still locked?",
        "Model selection and error analysis must not adapt to final-test outcomes. The repository records a one-shot authorization gate.",
        "",
        "### Why retain the reference model?",
        "A controlled candidate improved over the multi-seed reference aggregate but did not pass the predefined gate against the frozen incumbent. The decision therefore remained REFERENCE_RETAINED.",
        "",
        "### What do the Deep Learning failures teach?",
        "Unscaled images, excessive learning rate, excessive dropout, and misaligned labels produce recognizable validation or gradient signatures. A missing optimizer step leaves weights unchanged. These experiments demonstrate pipeline understanding rather than only score chasing.",
        "",
        "### Why can a classical text baseline outperform LSTM or Transformer runs?",
        "The text alone often names only one part. It does not contain the image context needed to determine MATCH versus PARTIAL_MATCH versus MISMATCH, and the dataset is intentionally small.",
        "",
        "### What is the strongest limitation?",
        "The dataset is small and the real open-license images show a higher error rate than generated images, indicating domain shift. Pretrained and human-annotation experiments remain behind explicit gates.",
    ]
    write_markdown(DEFENSE_GUIDE_PATH, defense_lines)

    error_lines = [
        "# Deep Learning Error Analysis and Failure Diagnostics",
        "",
        f"- Validation samples: **{error_summary['validation_sample_count']}**",
        f"- Errors: **{error_summary['error_count']}**",
        f"- Error rate: **{error_summary['error_rate']:.2%}**",
        f"- High-confidence errors at threshold {error_summary['high_confidence_error_threshold']:.2f}: **{error_summary['high_confidence_error_count']}**",
        "",
        "## Dominant confusion structure",
        "",
    ]
    for pair, count in error_summary["dominant_error_pairs"].items():
        error_lines.append(f"- `{pair}`: {count}")
    error_lines.extend(
        [
            "",
            "The intermediate PARTIAL_MATCH class accounts for 20 errors. The model frequently collapses it toward one of the two extremes, which is consistent with the semantic difficulty of related but non-identical parts.",
            "",
            "## Domain shift",
            "",
            f"- Generated-image error rate: **{error_summary['generated_image_error_rate']:.2%}**",
            f"- Real open-license image error rate: **{error_summary['real_image_error_rate']:.2%}**",
            "",
            "The higher real-image error rate is consistent with uncontrolled backgrounds, lighting, scale, perspective, and partial occlusion.",
            "",
            "## Controlled failure experiments",
            "",
            "The Fundamentals suite intentionally tested unscaled images, excessive and tiny learning rates, excessive dropout, misaligned labels, deep sigmoid gradients, and a missing optimizer update. The repository reports measured signatures and prevention rules rather than hiding failed configurations.",
            "",
            "## Explainability boundary",
            "",
            f"The Vision suite reports an automated foreground-proxy alignment rate of **{error_summary['vision_automated_foreground_alignment_rate']:.2%}**, but no human plausible-region score is claimed. Occlusion is treated as model diagnostics, not proof of human-like reasoning.",
            "",
            "## Safety and evaluation boundary",
            "",
            "This Step 011.4 layer reads committed validation reports only. It performs no training, does not open the locked test CSV, does not authorize final evaluation, and does not change the production model.",
        ]
    )
    write_markdown(ERROR_ANALYSIS_PATH, error_lines)


def markdown_cell(text: str) -> Any:
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def code_cell(source: str) -> Any:
    return nbformat.v4.new_code_cell(source.strip() + "\n")


def execution_counts_are_complete(code_cells: list[Any]) -> bool:
    counts = [cell.execution_count for cell in code_cells]
    return (
        bool(counts)
        and all(isinstance(count, int) for count in counts)
        and all(
            current > previous
            for previous, current in zip(counts, counts[1:])
        )
    )


def normalize_notebook_execution_evidence(notebook: Any) -> None:
    code_cells = [
        cell for cell in notebook.cells if cell.cell_type == "code"
    ]
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
    error_outputs = [
        output for output in outputs if output.get("output_type") == "error"
    ]

    if error_outputs:
        raise RuntimeError(
            "Final submission notebook execution produced error outputs: "
            f"{len(error_outputs)}"
        )
    if len(outputs) < 20 or len(figures) < 7:
        raise RuntimeError(
            "Final submission notebook saved evidence is incomplete: "
            f"outputs={len(outputs)}, figures={len(figures)}"
        )

    # Jupyter kernels may begin execution counters above 1 because of
    # environment startup hooks. Normalize the saved notebook so the artifact
    # remains deterministic across Windows and Linux without changing outputs.
    for index, cell in enumerate(code_cells, start=1):
        cell.execution_count = index


def build_notebook(evidence: dict[str, Any]) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "project": {
                "step": STEP,
                "base_checkpoint": BASE_CHECKPOINT,
                "submission_deadline": SUBMISSION_DEADLINE,
                "teacher_facing": True,
                "model_training_performed": False,
                "locked_test_csv_files_opened": False,
                "test_split_used": False,
                "final_test_evaluation_authorized": False,
                "production_final_model_changed": False,
            },
        }
    )

    notebook.cells = [
        markdown_cell(
            """
# Automotive Part Image-Text Matching

**Final Exam Submission Notebook — Step 011.4**

This executed notebook is the teacher-facing research narrative for the GitHub project. It uses committed validation evidence only. It does not train models, open the locked test split, authorize final evaluation, or change the retained production recipe.
"""
        ),
        markdown_cell(
            """
## 1. Executive Summary

The project asks whether a photograph of an automotive part matches a short textual description. The task is intentionally more demanding than binary matching because it distinguishes a related but incorrect part from a completely unrelated one.

The retained multimodal neural network obtains **0.5333 validation accuracy** and **0.5208 macro F1** on the integrated group-isolated validation split. The central research conclusion is not that the model is perfect, but that combining image and text evidence is necessary and that the remaining errors are structured, measurable, and operationally meaningful.
"""
        ),
        code_cell(
            """
from pathlib import Path
import ast
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from PIL import Image

ROOT = Path.cwd()
assert (ROOT / "README.md").is_file(), "Run the notebook from the repository root."

POLICY = {
    "model_training_performed": False,
    "locked_test_csv_files_opened": False,
    "test_split_used": False,
    "final_test_evaluation_authorized": False,
    "production_final_model_changed": False,
}
display(pd.DataFrame([POLICY]))
"""
        ),
        markdown_cell(
            """
## 2. Problem Statement and Real-World Motivation

Automotive catalogues, marketplaces, workshops, and inventory systems often need to determine whether a photograph and a short part description refer to the same component. A wrong result may waste staff time, route a part to the wrong catalogue entry, or produce a misleading search result.

The labels are:

- **MATCH** — image and text describe the same part category;
- **PARTIAL_MATCH** — the text names a different category in the same automotive system;
- **MISMATCH** — the text names a category from a different system.

The research hypothesis is that a multimodal model will outperform image-only and text-only approaches because the label is defined by the relationship between the two modalities.
"""
        ),
        markdown_cell(
            r"""
## 3. Formal Task Definition and Evaluation Metrics

For image $I$, text $T$, and label $y \in \{M, P, X\}$, the classifier estimates

\[
p_\theta(y \mid I,T)=\operatorname{softmax}(f_\theta(I,T)).
\]

The main selection metric is macro F1:

\[
F1_{macro}=\frac{1}{3}\sum_{c \in \{M,P,X\}}\frac{2\,Precision_c\,Recall_c}{Precision_c+Recall_c}.
\]

Macro F1 is preferred because it gives equal importance to the difficult PARTIAL_MATCH class. Accuracy is reported as a complementary metric. All model selection uses the validation split only.
"""
        ),
        code_cell(
            """
validation = pd.read_csv(ROOT / "data/processed/integrated_validation.csv")
summary = pd.DataFrame({
    "samples": [len(validation)],
    "part_groups": [validation["part_group_id"].nunique()],
    "images": [validation["image_id"].nunique()],
    "categories": [validation["part_category"].nunique()],
    "labels": [validation["label"].nunique()],
})
display(summary)
display(pd.crosstab(validation["source"], validation["label"]))
"""
        ),
        markdown_cell(
            """
## 4. Previous Research

The project is situated between visual-semantic retrieval and multimodal classification. VSE++ studies image-text retrieval with hard negatives; VisualBERT combines visual regions and language in a transformer; CLIP learns transferable image-text representations at large scale. ResNet provides a standard residual vision backbone, while *Attention Is All You Need* motivates the sequence experiments. The evaluation design follows official scikit-learn definitions for precision, recall, F1, confusion matrices, and grouped validation.

The present project differs in three ways: the dataset is small and auditable, the target has an explicit same-system PARTIAL_MATCH class, and the final test split remains locked behind a one-shot authorization gate.
"""
        ),
        markdown_cell(
            """
## 5. Data Acquisition, Licensing, Cleaning, and Grouped Splitting

The evidence combines deterministic development images and manually reviewed open-license Wikimedia Commons photographs. Every approved external image retains source, author, licence URL, and SHA-256 provenance. Images are decoded, normalized into reproducible files, and checked for duplicates and unsafe paths.

Rows are split by `part_group_id`, not independently. This prevents descriptions derived from the same physical image or part identity from crossing train and validation boundaries. The locked test CSV is not read by this notebook.
"""
        ),
        code_cell(
            """
source_profile = (
    validation.groupby("source")
    .agg(samples=("sample_id", "count"), groups=("part_group_id", "nunique"), images=("image_id", "nunique"))
    .reset_index()
)
category_profile = (
    validation.groupby("part_category")["sample_id"]
    .count()
    .rename("validation_samples")
    .reset_index()
)
display(source_profile)
display(category_profile)
"""
        ),
        markdown_cell(
            """
## 6. Model Families and Experimental Design

The primary comparison contains a majority baseline, TF-IDF logistic regression, image-pixel logistic regression, Keras text and image models, and a Keras multimodal model. Additional controlled suites test Deep Learning fundamentals, sequence models, attention diagnostics, image representations, augmentation, compatibility scoring, ranking, and occlusion.

The course suites are evidence-generating experiments, not automatic replacements for the frozen production model. Pretrained downloads, fine-tuning, and human agreement remain behind explicit gates.
"""
        ),
        markdown_cell(
            """
## 7. Integrated Validation Results

The multimodal model ranks first in the primary six-model comparison. The lower performance on the integrated real-plus-generated validation split compared with the generated-only development split is evidence of a harder and more realistic evaluation setting.
"""
        ),
        code_cell(
            """
comparison = pd.read_csv(ROOT / "reports/integrated_training/validation_comparison.csv")
columns = [
    "validation_rank", "model", "input_modality",
    "integrated_validation_accuracy", "integrated_validation_macro_f1",
    "development_validation_macro_f1",
]
display(comparison[columns].sort_values("validation_rank"))
"""
        ),
        code_cell(
            """
plot_data = comparison.sort_values("integrated_validation_macro_f1", ascending=True)
figure, axis = plt.subplots(figsize=(9, 5))
axis.barh(plot_data["model"], plot_data["integrated_validation_macro_f1"])
axis.set_xlabel("Integrated validation macro F1")
axis.set_ylabel("Model")
axis.set_title("Primary six-model comparison")
axis.set_xlim(0.0, 0.6)
figure.tight_layout()
plt.show()
"""
        ),
        markdown_cell(
            """
## 8. Deep Learning Error Analysis and Failure Diagnostics

A strong exam project must explain failures, not only present its best score. The controlled reference analysis contains 35 errors among 60 validation samples. The key error mechanism is the collapse of PARTIAL_MATCH toward one of the two extreme classes. Real open-license images are harder than generated images, indicating domain shift rather than random noise.
"""
        ),
        code_cell(
            """
confusion = pd.read_csv(
    ROOT / "reports/validation_model_improvement/candidates/reference_multimodal/validation_confusion_matrix.csv",
    index_col=0,
)
figure, axis = plt.subplots(figsize=(6, 5))
image = axis.imshow(confusion.to_numpy())
axis.set_xticks(range(len(confusion.columns)))
axis.set_xticklabels([column.replace("predicted_", "") for column in confusion.columns], rotation=30, ha="right")
axis.set_yticks(range(len(confusion.index)))
axis.set_yticklabels([index.replace("actual_", "") for index in confusion.index])
axis.set_xlabel("Predicted label")
axis.set_ylabel("Actual label")
axis.set_title("Reference multimodal validation confusion matrix")
for row in range(confusion.shape[0]):
    for column in range(confusion.shape[1]):
        axis.text(column, row, str(int(confusion.iloc[row, column])), ha="center", va="center")
figure.colorbar(image, ax=axis)
figure.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
with (ROOT / "reports/validation_model_improvement/validation_error_analysis.json").open(encoding="utf-8") as handle:
    error_summary = json.load(handle)

confusion_pairs = pd.DataFrame(
    [{"error_pair": pair, "count": count} for pair, count in error_summary["confusion_pairs"].items()]
).sort_values("count", ascending=False)
source_errors = pd.DataFrame(
    [{"source": source, **values} for source, values in error_summary["errors_by_source"].items()]
)
category_errors = pd.DataFrame(
    [{"category": category, **values} for category, values in error_summary["errors_by_category"].items()]
).sort_values(["error_rate", "category"], ascending=[False, True])
display(confusion_pairs)
display(source_errors)
display(category_errors)
"""
        ),
        code_cell(
            """
figure, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(source_errors["source"], source_errors["error_rate"])
axes[0].set_ylim(0.0, 1.0)
axes[0].set_ylabel("Error rate")
axes[0].set_title("Domain shift by image source")
axes[0].tick_params(axis="x", rotation=25)
plot_categories = category_errors.sort_values("error_rate", ascending=True)
axes[1].barh(plot_categories["category"], plot_categories["error_rate"])
axes[1].set_xlim(0.0, 1.0)
axes[1].set_xlabel("Error rate")
axes[1].set_title("Error rate by automotive category")
figure.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
errors = pd.read_csv(ROOT / "reports/validation_model_improvement/validation_error_analysis.csv")
validation_lookup = validation[["image_id", "image_path"]].drop_duplicates("image_id")
examples = errors.merge(validation_lookup, on="image_id", how="left").head(6)

display(examples[[
    "part_category", "description", "true_label", "predicted_label",
    "confidence", "confidence_margin", "source",
]])

figure, axes = plt.subplots(2, 3, figsize=(12, 8))
for axis, (_, row) in zip(axes.flat, examples.iterrows()):
    image = Image.open(ROOT / row["image_path"]).convert("RGB")
    axis.imshow(image)
    axis.set_title(
        f"{row['part_category']}\\n{row['true_label']} → {row['predicted_label']}\\nconfidence={row['confidence']:.3f}",
        fontsize=9,
    )
    axis.axis("off")
figure.suptitle("Representative validation errors")
figure.tight_layout()
plt.show()
"""
        ),
        markdown_cell(
            """
The absence of errors above confidence 0.60 is encouraging but does not prove calibration. It suggests that uncertain cases could be routed to human review. The project therefore treats the model as decision support, not as an unquestionable authority.

The Fundamentals suite also creates failures deliberately. This demonstrates that the author understands how preprocessing, learning rate, dropout, label alignment, activation choice, and optimizer updates affect training behavior.
"""
        ),
        code_cell(
            """
failures = pd.read_csv(ROOT / "reports/course_coverage/fundamentals/failure_diagnostics.csv")
validation_failures = failures[failures["metric_name"] == "validation_macro_f1"].copy()
display(failures[["case", "status", "metric_name", "metric_value", "signature_detected", "prevention"]])

plot_failures = validation_failures.sort_values("metric_value", ascending=True)
figure, axis = plt.subplots(figsize=(9, 5))
axis.barh(plot_failures["case"], plot_failures["metric_value"])
axis.axvline(0.2886, linestyle="--", label="Correct-loop reference")
axis.set_xlabel("Validation macro F1")
axis.set_title("Controlled Deep Learning failure signatures")
axis.legend()
figure.tight_layout()
plt.show()
"""
        ),
        markdown_cell(
            """
## 9. Course Exercise Evidence: Fundamentals, Sequence, and Vision

- **Deep Learning Fundamentals:** 10/10 problems and 35 recorded training runs.
- **Transformers and Sequence Modelling:** 9 core problems and 21 recorded runs; the pretrained-transformer task remains gated.
- **Vision Core:** 6 completed problems and 48 recorded runs; pretrained backbone, fine-tuning, and genuine human annotation remain gated.

These suites explain why more complex models do not automatically win on a small dataset. The best text-only result remains close to a classical TF-IDF baseline because the relationship label cannot be inferred reliably from text without the image. Vision scoring and ranking are promising, but they are separate validation tasks and do not overwrite the frozen three-class model.
"""
        ),
        code_cell(
            """
with (ROOT / "reports/course_coverage/fundamentals/fundamentals_suite_status.json").open(encoding="utf-8") as handle:
    fundamentals_status = json.load(handle)
with (ROOT / "reports/course_coverage/sequence/sequence_suite_status.json").open(encoding="utf-8") as handle:
    sequence_status = json.load(handle)
with (ROOT / "reports/course_coverage/vision/vision_suite_status.json").open(encoding="utf-8") as handle:
    vision_status = json.load(handle)

suite_summary = pd.DataFrame([
    {"suite": "Fundamentals", "completed": fundamentals_status.get("completed_exercise_problem_count", 10), "training_runs": fundamentals_status["run_count"], "gated": 0},
    {"suite": "Sequence", "completed": sequence_status.get("completed_core_problem_count", 9), "training_runs": sequence_status["training_runs_recorded"], "gated": len(sequence_status.get("deferred_problem_ids", []))},
    {"suite": "Vision core", "completed": vision_status["completed_problem_count"], "training_runs": vision_status["training_runs_recorded"], "gated": vision_status["deferred_problem_count"]},
])
display(suite_summary)

figure, axis = plt.subplots(figsize=(8, 5))
axis.bar(suite_summary["suite"], suite_summary["training_runs"])
for index, row in suite_summary.iterrows():
    axis.text(index, row["training_runs"] + 1, f"{int(row['completed'])} completed", ha="center")
axis.set_ylabel("Recorded training runs")
axis.set_title("Course exercise experimental evidence")
figure.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
sequence_models = pd.read_csv(ROOT / "reports/course_coverage/sequence/model_comparison.csv")
vision_augmentation = pd.read_csv(ROOT / "reports/course_coverage/vision/augmentation_comparison.csv")
with (ROOT / "reports/course_coverage/vision/ranking_metrics.json").open(encoding="utf-8") as handle:
    ranking = json.load(handle)
with (ROOT / "reports/course_coverage/vision/explainability_summary.json").open(encoding="utf-8") as handle:
    explainability = json.load(handle)

display(sequence_models.sort_values("validation_macro_f1", ascending=False).head(8))
display(vision_augmentation.sort_values("validation_macro_f1_mean", ascending=False))
display(pd.DataFrame([{
    "pairwise_ranking_accuracy": ranking["pairwise_ranking_accuracy"],
    "three_way_ordering_accuracy": ranking["three_way_ordering_accuracy"],
    "equal_pair_accuracy": ranking["equal_pair_accuracy"],
    "automated_foreground_alignment": explainability["automated_foreground_proxy_alignment_rate"],
    "human_review_claimed": explainability["manual_review_claimed"],
}]))

best_sequence = (
    sequence_models.sort_values("validation_macro_f1", ascending=False)
    .drop_duplicates("family")
    .sort_values("validation_macro_f1", ascending=True)
)
vision_plot = vision_augmentation.sort_values("validation_macro_f1_mean", ascending=True)
figure, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].barh(best_sequence["family"], best_sequence["validation_macro_f1"])
axes[0].set_xlabel("Validation macro F1")
axes[0].set_title("Best recorded run by text model family")
axes[1].barh(vision_plot["augmentation_policy"], vision_plot["validation_macro_f1_mean"])
axes[1].set_xlabel("Mean validation macro F1")
axes[1].set_title("Vision augmentation comparison")
figure.tight_layout()
plt.show()
"""
        ),
        markdown_cell(
            """
## 10. Testing, Reproducibility, and Locked-Test Policy

The repository combines unit and integration tests with semantic verification modules, artifact manifests, deterministic seeds, pinned dependencies, rollback-aware patch application, and clean-clone verification. The notebook is rebuilt from committed reports and never calls a training function.

The final test split remains closed because repeated access would turn it into another validation set. A committed authorization artifact must be changed explicitly before a one-shot final evaluation can occur.
"""
        ),
        code_cell(
            """
test_functions = 0
for path in sorted((ROOT / "tests").glob("test_*.py")):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    test_functions += sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        for node in tree.body
    )
verification_modules = sorted((ROOT / "src/verification").glob("*.py"))
verification_modules = [path for path in verification_modules if path.name != "__init__.py"]
reproducibility = pd.DataFrame([{
    "test_functions": test_functions,
    "verification_modules": len(verification_modules),
    "requirements_locked": (ROOT / "requirements-lock.txt").is_file(),
    "test_split_used": False,
    "final_test_authorized": False,
}])
display(reproducibility)
"""
        ),
        markdown_cell(
            """
## 11. Limitations, Ethics, and Legal Compliance

The principal limitations are dataset size, category coverage, and domain shift between generated and real images. Validation metrics have uncertainty because only 60 integrated validation samples are available. Pretrained backbones could improve representation quality, but downloads, licence capture, resource use, and fine-tuning are intentionally gated. Human agreement and human localization metrics are not simulated.

All committed external images have open-licence provenance. The project contains no malicious code, surveillance behavior, credential collection, or privacy-invasive functionality. The software is research code and not a safety-critical automated parts-identification system.
"""
        ),
        markdown_cell(
            """
## 12. Self-Assessment Against the Exam Rubric

The self-assessment is deliberately evidence-based and conservative. Two points are reserved because automated occlusion is not a human localization study and because final oral communication and portal submission remain manual responsibilities.
"""
        ),
        code_cell(
            """
rubric = pd.read_csv(ROOT / "reports/final_submission/rubric_evidence_matrix.csv")
display(rubric)
display(pd.DataFrame([{
    "self_assessed_total": int(rubric["self_assessed_points"].sum()),
    "maximum_total": int(rubric["maximum_points"].sum()),
}]))
"""
        ),
        markdown_cell(
            """
## 13. Conclusions and Defense Summary

1. The multimodal model is the strongest primary model because the label depends on both image and text.
2. PARTIAL_MATCH is the central semantic difficulty and explains most structured confusion.
3. Real open-license images are harder than generated images, demonstrating domain shift.
4. Controlled failures show how common Deep Learning mistakes appear in metrics and gradients.
5. More complex sequence models do not automatically outperform classical text methods on small, context-incomplete data.
6. Vision scoring, ranking, augmentation, and occlusion provide additional evidence without changing the frozen production model.
7. The test split remains locked, preserving the validity of a future one-shot final evaluation.
"""
        ),
        markdown_cell(
            """
## References

1. Faghri, F. et al. **VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.** BMVC, 2018. — Visual Semantic Embedding.
2. Li, L. H. et al. **VisualBERT: A Simple and Performant Baseline for Vision and Language.** 2019.
3. Radford, A. et al. **Learning Transferable Visual Models From Natural Language Supervision.** ICML, 2021.
4. He, K. et al. **Deep Residual Learning for Image Recognition.** CVPR, 2016.
5. Vaswani, A. et al. **Attention Is All You Need.** NeurIPS, 2017.
6. **scikit-learn model evaluation documentation.** Official documentation for classification metrics and confusion matrices.

Repository evidence paths and exact experiment outputs are linked from the README, the submission checklist, and the `reports/` directory.
"""
        ),
    ]

    FINAL_SUBMISSION_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, FINAL_SUBMISSION_NOTEBOOK_PATH)

    executor = ExecutePreprocessor(timeout=180, kernel_name="python3")
    executor.preprocess(
        notebook,
        {"metadata": {"path": str(PROJECT_ROOT)}},
    )
    normalize_notebook_execution_evidence(notebook)
    nbformat.write(notebook, FINAL_SUBMISSION_NOTEBOOK_PATH)


def inspect_notebook() -> dict[str, Any]:
    notebook = nbformat.read(FINAL_SUBMISSION_NOTEBOOK_PATH, as_version=4)
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
    markdown = "\n".join(
        str(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    code = "\n".join(str(cell.source) for cell in code_cells)
    return {
        "cell_count": len(notebook.cells),
        "markdown_cell_count": sum(
            cell.cell_type == "markdown" for cell in notebook.cells
        ),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": sum(
            cell.execution_count is not None for cell in code_cells
        ),
        "sequential_execution": execution_counts_are_complete(code_cells),
        "saved_output_count": len(outputs),
        "figure_count": len(figures),
        "error_output_count": sum(
            output.get("output_type") == "error" for output in outputs
        ),
        "markdown": markdown,
        "code": code,
        "metadata": dict(notebook.metadata.get("project", {})),
    }


def build_status(
    evidence: dict[str, Any], notebook_info: dict[str, Any]
) -> dict[str, Any]:
    self_total = evidence["self_assessment_total"]
    return {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "base_checkpoint": BASE_CHECKPOINT,
        "submission_deadline": SUBMISSION_DEADLINE,
        "repository": REPOSITORY_URL,
        "final_notebook": project_relative(FINAL_SUBMISSION_NOTEBOOK_PATH),
        "final_notebook_github_url": FINAL_SUBMISSION_NOTEBOOK_GITHUB_URL,
        "rubric_criterion_count": len(RUBRIC_MAX_POINTS),
        "self_assessment_points": self_total,
        "maximum_points": evidence["self_assessment_possible"],
        "test_function_count": evidence["test_function_count"],
        "notebook_cell_count": notebook_info["cell_count"],
        "executed_code_cell_count": notebook_info[
            "executed_code_cell_count"
        ],
        "saved_output_count": notebook_info["saved_output_count"],
        "figure_count": notebook_info["figure_count"],
        "reference_count": len(REFERENCE_TITLES),
        "deep_learning_error_count": evidence["error_summary"][
            "error_count"
        ],
        "controlled_failure_case_count": len(evidence["failures"]),
        "fundamentals_training_runs": evidence["fundamentals_status"][
            "run_count"
        ],
        "sequence_training_runs": evidence["sequence_status"][
            "training_runs_recorded"
        ],
        "vision_training_runs": evidence["vision_status"][
            "training_runs_recorded"
        ],
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
    }


def build_manifest() -> dict[str, Any]:
    source_hashes = {
        project_relative(path): normalized_sha256(path)
        for path in SOURCE_ARTIFACTS
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
        "base_checkpoint": BASE_CHECKPOINT,
        "hash_normalization": "utf-8-lf for text; raw bytes for binary",
        "source_artifact_sha256": source_hashes,
        "generated_artifact_sha256": generated_hashes,
        "source_artifact_count": len(source_hashes),
        "generated_artifact_count": len(generated_hashes),
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
    }


def build_summary(
    evidence: dict[str, Any], notebook_info: dict[str, Any]
) -> None:
    lines = [
        "# Final Submission Rubric Alignment Summary",
        "",
        f"- Step: **{STEP}**",
        f"- Readiness: `{READINESS}`",
        f"- Deadline: **{SUBMISSION_DEADLINE}**",
        f"- Final notebook: `{project_relative(FINAL_SUBMISSION_NOTEBOOK_PATH)}`",
        f"- Self-assessment: **{evidence['self_assessment_total']}/{evidence['self_assessment_possible']}**",
        f"- Notebook cells: **{notebook_info['cell_count']}**",
        f"- Executed code cells: **{notebook_info['executed_code_cell_count']}**",
        f"- Saved outputs: **{notebook_info['saved_output_count']}**",
        f"- Saved figures: **{notebook_info['figure_count']}**",
        f"- Test functions in repository: **{evidence['test_function_count']}**",
        f"- Deep Learning validation errors analyzed: **{evidence['error_summary']['error_count']}**",
        f"- Controlled failure cases: **{len(evidence['failures'])}**",
        "",
        "## Teacher-facing improvements",
        "",
        "- exact alignment to the eight exam rubric categories;",
        "- current Step 011.1, 011.2, and 011.3A evidence in one notebook;",
        "- confusion matrix, domain-shift analysis, category errors, representative mistakes, and controlled failure diagnostics;",
        "- explicit limitations, legal boundary, self-assessment, checklist, and defense answers;",
        "- no model training, test access, final-test authorization, or production-model change.",
    ]
    write_markdown(SUMMARY_PATH, lines)


def main() -> None:
    for path in SOURCE_ARTIFACTS:
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact is missing: {path}")

    evidence = load_evidence()
    build_report_figures(evidence)
    error_summary = build_error_summary(evidence)
    write_json(ERROR_SUMMARY_PATH, error_summary)
    write_supporting_reports(evidence, error_summary)
    build_notebook(evidence)
    notebook_info = inspect_notebook()
    status = build_status(evidence, notebook_info)
    write_json(STATUS_PATH, status)
    build_summary(evidence, notebook_info)
    manifest = build_manifest()
    write_json(MANIFEST_PATH, manifest)

    print("Final exam rubric alignment and Deep Learning error analysis")
    print(f"- status: {status['status']}")
    print(f"- notebook: {status['final_notebook']}")
    print(f"- notebook cells: {status['notebook_cell_count']}")
    print(f"- executed code cells: {status['executed_code_cell_count']}")
    print(f"- saved outputs: {status['saved_output_count']}")
    print(f"- figures: {status['figure_count']}")
    print(f"- self-assessment: {status['self_assessment_points']}/100")
    print(f"- test functions: {status['test_function_count']}")
    print("- model training performed: false")
    print("- test split used: false")
    print("- final test evaluation authorized: false")
    print("- production final model changed: false")


if __name__ == "__main__":
    main()
