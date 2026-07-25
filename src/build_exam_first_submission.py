from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from src.exam_first_config import (
    BASE_CHECKPOINT_COMMIT,
    BASE_CHECKPOINT_COMMIT_COUNT,
    CONSISTENCY_REPORT_PATH,
    COURSE_ALIGNMENT_PATH,
    DEFENSE_NOTES_PATH,
    EXAM_DIR,
    EXAM_README_PATH,
    FIGURES_DIR,
    FOCUSED_CONFUSION_PATH,
    FOCUSED_ERROR_ROWS_PATH,
    FOCUSED_ERROR_SUMMARY_PATH,
    GENERATED_ARTIFACTS,
    HISTORICAL_NOTEBOOK_CATALOGUE_PATH,
    HISTORICAL_README_PATH,
    MANIFEST_PATH,
    NOTEBOOK_CATALOGUE_PATH,
    NOTEBOOK_PATH,
    PRIMARY_QUESTION,
    PROJECT_ROOT,
    READINESS,
    REPRODUCTION_PATH,
    REPORT_DIR,
    RETAINED_CONFUSION_PATH,
    RETAINED_METRICS_PATH,
    RETAINED_PREDICTIONS_PATH,
    ROOT_README_PATH,
    SOURCE_ARCHIVE,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_ARTIFACTS,
    STATUS_PATH,
    STEP,
    SUMMARY_PATH,
    SUPPLEMENTARY_INDEX_PATH,
    TEXT_HASH_SUFFIXES,
    project_relative,
)

LABELS = ("MATCH", "PARTIAL_MATCH", "MISMATCH")


def normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_HASH_SUFFIXES:
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace(
            "\r", "\n"
        )
        raw = text.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classification_summary(frame: pd.DataFrame) -> dict[str, Any]:
    confusion_raw = pd.crosstab(
        frame["true_label"], frame["predicted_label"]
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = int(confusion_raw.loc[label, label])
        predicted_positive = int(confusion_raw[label].sum())
        actual_positive = int(confusion_raw.loc[label].sum())
        precision = (
            true_positive / predicted_positive if predicted_positive else 0.0
        )
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[label] = {
            "support": actual_positive,
            "correct": true_positive,
            "errors": actual_positive - true_positive,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    confusion = confusion_raw.copy()
    confusion.index = [f"actual_{label}" for label in LABELS]
    confusion.columns = [f"predicted_{label}" for label in LABELS]

    correct_count = int(frame["is_correct"].sum())
    sample_count = int(len(frame))
    return {
        "sample_count": sample_count,
        "correct_count": correct_count,
        "error_count": sample_count - correct_count,
        "accuracy": correct_count / sample_count,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion": confusion,
    }


def load_evidence() -> dict[str, Any]:
    train = pd.read_csv(PROJECT_ROOT / "data/processed/integrated_train.csv")
    validation = pd.read_csv(
        PROJECT_ROOT / "data/processed/integrated_validation.csv"
    )
    comparison = pd.read_csv(
        PROJECT_ROOT
        / "reports/integrated_training/validation_comparison.csv"
    )

    predictions: dict[str, pd.DataFrame] = {}
    for slug in ("keras_text", "keras_image", "keras_multimodal"):
        predictions[slug] = pd.read_csv(
            PROJECT_ROOT
            / f"reports/integrated_training/{slug}/validation_predictions.csv"
        )

    retained = predictions["keras_multimodal"].copy()
    retained_metrics = read_json(RETAINED_METRICS_PATH)
    source_confusion = pd.read_csv(RETAINED_CONFUSION_PATH, index_col=0)
    derived = classification_summary(retained)

    if not derived["confusion"].equals(source_confusion):
        raise RuntimeError(
            "Frozen predictions and committed multimodal confusion matrix differ."
        )
    if abs(derived["accuracy"] - float(retained_metrics["accuracy"])) > 1e-12:
        raise RuntimeError(
            "Frozen predictions and committed multimodal accuracy differ."
        )
    if abs(derived["macro_f1"] - float(retained_metrics["macro_f1"])) > 1e-12:
        raise RuntimeError(
            "Frozen predictions and committed multimodal macro F1 differ."
        )

    validation_paths = validation[["sample_id", "image_path"]].drop_duplicates(
        "sample_id"
    )
    errors = retained.loc[~retained["is_correct"]].copy()
    errors["error_pair"] = (
        errors["true_label"] + " -> " + errors["predicted_label"]
    )
    errors = errors.merge(validation_paths, on="sample_id", how="left")
    if errors["image_path"].isna().any():
        raise RuntimeError("A focused validation error has no image path.")

    errors_by_source: dict[str, dict[str, float | int]] = {}
    for source, source_rows in retained.groupby("source", sort=True):
        source_errors = int((~source_rows["is_correct"]).sum())
        source_count = int(len(source_rows))
        errors_by_source[str(source)] = {
            "samples": source_count,
            "errors": source_errors,
            "error_rate": source_errors / source_count,
        }

    partial_recovered = int(
        (
            (retained["true_label"] == "PARTIAL_MATCH")
            & (retained["predicted_label"] == "PARTIAL_MATCH")
        ).sum()
    )
    partial_errors = int(
        (
            (retained["true_label"] == "PARTIAL_MATCH")
            & (~retained["is_correct"])
        ).sum()
    )
    predicted_partial = int(
        (retained["predicted_label"] == "PARTIAL_MATCH").sum()
    )

    error_summary = {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "retained_model_slug": "keras_multimodal",
        "prediction_artifact": project_relative(RETAINED_PREDICTIONS_PATH),
        "prediction_artifact_sha256": normalized_sha256(
            RETAINED_PREDICTIONS_PATH
        ),
        "metrics_artifact": project_relative(RETAINED_METRICS_PATH),
        "confusion_artifact": project_relative(RETAINED_CONFUSION_PATH),
        "validation_sample_count": derived["sample_count"],
        "correct_count": derived["correct_count"],
        "error_count": derived["error_count"],
        "accuracy": derived["accuracy"],
        "macro_f1": derived["macro_f1"],
        "predicted_partial_match_count": predicted_partial,
        "true_partial_match_recovered": partial_recovered,
        "partial_match_error_count": partial_errors,
        "errors_by_source": errors_by_source,
        "class_performance": derived["per_class"],
        "single_prediction_set_used": True,
        "historical_controlled_retraining_is_primary": False,
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "production_final_model_changed": False,
    }

    consistency_report = {
        "schema_version": "1.0",
        "step": STEP,
        "status": "PASS",
        "primary_prediction_artifact": project_relative(
            RETAINED_PREDICTIONS_PATH
        ),
        "primary_prediction_artifact_sha256": normalized_sha256(
            RETAINED_PREDICTIONS_PATH
        ),
        "all_primary_metrics_derived_from_same_prediction_artifact": True,
        "prediction_rows": derived["sample_count"],
        "correct_rows": derived["correct_count"],
        "error_rows": derived["error_count"],
        "confusion_matrix_matches_predictions": True,
        "accuracy_matches_metrics_artifact": True,
        "macro_f1_matches_metrics_artifact": True,
        "historical_step_010_4_reference_analysis": (
            "supporting evidence from a separate controlled retraining; "
            "not used for the focused primary result"
        ),
        "model_training_performed": False,
        "test_split_used": False,
        "production_final_model_changed": False,
    }

    return {
        "train": train,
        "validation": validation,
        "comparison": comparison,
        "errors": errors,
        "error_summary": error_summary,
        "consistency_report": consistency_report,
        "confusion": derived["confusion"],
        "retained_metrics": retained_metrics,
        "predictions": predictions,
    }


def write_focused_evidence(evidence: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    evidence["errors"].to_csv(FOCUSED_ERROR_ROWS_PATH, index=False)
    evidence["confusion"].to_csv(FOCUSED_CONFUSION_PATH)
    FOCUSED_ERROR_SUMMARY_PATH.write_text(
        json.dumps(evidence["error_summary"], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    CONSISTENCY_REPORT_PATH.write_text(
        json.dumps(
            evidence["consistency_report"], indent=2, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )


def build_figures(evidence: dict[str, Any]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    comparison = evidence["comparison"].copy()
    confusion = evidence["confusion"].copy()
    errors = evidence["errors"].copy()
    predictions = evidence["predictions"]

    plot_data = comparison.sort_values(
        "integrated_validation_macro_f1", ascending=True
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(
        plot_data["model"], plot_data["integrated_validation_macro_f1"]
    )
    axis.set_xlabel("Validation macro F1")
    axis.set_title("Primary model comparison")
    axis.set_xlim(0.0, 0.6)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion.to_numpy(), aspect="auto")
    axis.set_xticks(range(3), ["MATCH", "PARTIAL", "MISMATCH"])
    axis.set_yticks(range(3), ["MATCH", "PARTIAL", "MISMATCH"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Frozen keras_multimodal validation confusion matrix")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(
                column,
                row,
                int(confusion.iloc[row, column]),
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    by_source = (
        errors.groupby("source").size().rename("errors").reset_index()
    )
    totals = (
        evidence["validation"]
        .groupby("source")
        .size()
        .rename("samples")
        .reset_index()
    )
    by_source = by_source.merge(totals, on="source")
    by_source["error_rate"] = by_source["errors"] / by_source["samples"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(by_source["source"], by_source["error_rate"])
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Validation error rate")
    axis.set_title("Frozen model: generated and real-image error rates")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "error_rates.png", dpi=160)
    plt.close(figure)

    contribution = pd.DataFrame(
        [
            {
                "model": label,
                "correct": int(predictions[slug]["is_correct"].sum()),
            }
            for slug, label in (
                ("keras_text", "Text-only"),
                ("keras_image", "Image-only"),
                ("keras_multimodal", "Multimodal"),
            )
        ]
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(contribution["model"], contribution["correct"])
    axis.set_ylim(0, 60)
    axis.set_ylabel("Correct validation samples")
    axis.set_title("What each neural input setup resolves")
    for index, value in enumerate(contribution["correct"]):
        axis.text(index, int(value) + 1, str(int(value)), ha="center")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "multimodal_contribution.png", dpi=160)
    plt.close(figure)


def clarify_historical_readme() -> None:
    if not HISTORICAL_README_PATH.is_file():
        return
    text = HISTORICAL_README_PATH.read_text(encoding="utf-8-sig")
    old = (
        "The teacher-facing submission does not hide model failures. The "
        "controlled reference analysis contains **35 errors among 60 validation "
        "samples**. The intermediate `PARTIAL_MATCH` class accounts for 20 "
        "errors and is split evenly toward `MATCH` and `MISMATCH`. Real "
        "open-license images have a higher error rate (`66.7%`) than generated "
        "validation images (`50.0%`), which is consistent with domain shift "
        "caused by background, lighting, scale and perspective."
    )
    new = (
        "The historical Step 010.4 controlled retraining analysis contains "
        "**35 errors among 60 validation samples**. It is preserved as a "
        "separate stability experiment and is not the prediction set used for "
        "the focused primary result. Within that historical run, the "
        "intermediate `PARTIAL_MATCH` class accounts for 20 errors and real "
        "open-license images have a higher error rate (`66.7%`) than generated "
        "validation images (`50.0%`)."
    )
    if old in text:
        write_text(HISTORICAL_README_PATH, text.replace(old, new))


def build_root_readme() -> str:
    return f"""# Automotive Part Image-Text Matching

## Start here

**Research question:** {PRIMARY_QUESTION}

Open the focused, executed exam notebook:

- [Focused Deep Learning exam notebook](exam/01_focused_deep_learning_project.ipynb)
- [Exam project guide](exam/README.md)
- [Reproduction instructions](exam/reproduction.md)
- [Oral defense notes](exam/defense_notes.md)

The project is deliberately presented as one analysis rather than as a
portfolio of unrelated techniques.

## Main result

The frozen Keras multimodal run ranks first on the grouped integrated
validation split.

| Model | Input | Validation accuracy | Macro F1 |
|---|---|---:|---:|
| Majority baseline | none | 0.3333 | 0.1667 |
| TF-IDF + Logistic Regression | text | 0.4167 | 0.3300 |
| Image pixels + Logistic Regression | image | 0.3333 | 0.1667 |
| Keras text model | text | 0.4167 | 0.3300 |
| Keras image model | image | 0.3333 | 0.1667 |
| **Keras multimodal model** | **image + text** | **0.5333** | **0.5208** |

Every teacher-facing metric and error claim now comes from the same frozen
`keras_multimodal` prediction artifact. It contains **32 correct predictions
and 28 errors** among 60 validation samples. The model recovers **12 of 20
`PARTIAL_MATCH`** cases; its weakest class recall is `MATCH` at 0.30. Real
open-license images remain harder than generated images (53.3% versus 40.0%
error rate).

## Leakage protection

All samples for one physical automotive part share a `part_group_id` and stay
in one split. Train and validation have no group, image-ID, or image-path
overlap. The locked test split has not been used, and final test evaluation has
not been authorized.

## What the reviewer sees

The focused notebook shows:

1. one precise Deep Learning question;
2. the three labels and the data sources;
3. grouped splitting and leakage checks;
4. baselines and neural models on the same validation split;
5. concrete cases where multimodal input helps;
6. concrete errors derived from that same frozen prediction set;
7. limitations and an exact reproduction boundary.

## Reproduce the exam-facing evidence

```powershell
python -m src.project_cli build-exam-first-submission
python -m src.project_cli verify-exam-first-submission
python -m jupyter notebook exam/01_focused_deep_learning_project.ipynb
```

These commands do not train a model, open locked test CSV files, or change the
retained production model.

## Supporting evidence

The wider course exercises, historical notebooks, engineering tests, manifests,
and audits are indexed in [supplementary/README.md](supplementary/README.md).
The historical Step 010.4 analysis with 35 errors is explicitly treated as a
separate controlled retraining experiment and is not mixed with the primary
frozen-model result.

For compatibility with the Step 011.4 rubric checkpoint, the earlier full
teacher-facing notebook remains available as historical supporting evidence:

- [Open the Step 011.4 notebook on GitHub](https://github.com/SATananov/automotive-part-image-text-matching/blob/main/notebooks/03_final_exam_submission.ipynb)
- [Historical Deep Learning Error Analysis](reports/final_submission/deep_learning_error_analysis.md)
- [98/100 self-assessment](reports/final_submission/self_assessment.md)
- [Submission checklist](reports/final_submission/submission_checklist.md)

Final exam submission deadline: **11 August 2026, 16:00 Europe/Sofia**.

## Standard repository structure

- `exam/` — recommended teacher-facing entry point;
- `src/` — reusable implementation and command-line workflows;
- `data/` — development, real, external, and grouped split artifacts;
- `reports/` — committed metrics, predictions, manifests, and audits;
- `notebooks/` — historical and course-exercise notebooks;
- `tests/` — unit and integration tests;
- `supplementary/` — index to supporting evidence.

The standard technical directories were not moved because their stable paths
are part of reproducibility and historical manifests.
"""


def build_notebook_catalogue() -> str:
    return """# Notebook catalogue

## Recommended exam entry point

The primary teacher-facing notebook is now:

- [`exam/01_focused_deep_learning_project.ipynb`](../exam/01_focused_deep_learning_project.ipynb)

It presents one research question, grouped-split protection, the primary model
comparison, concrete successes, errors from one frozen prediction artifact,
limitations, and reproduction.

## Supporting notebooks

- `notebooks/01_development_experiment.ipynb` — early development evidence;
- `notebooks/02_final_exam_project.ipynb` — historical full project notebook;
- `notebooks/03_final_exam_submission.ipynb` — Step 011.4 rubric-alignment and historical controlled-retraining evidence;
- `notebooks/course_coverage/` — Fundamentals, sequence, vision, ranking, and controlled experiments.

The historical notebooks are supporting evidence and are not intended to
replace or redefine the focused primary metrics.
"""


def build_exam_readme(evidence: dict[str, Any]) -> str:
    best = evidence["comparison"].sort_values("validation_rank").iloc[0]
    summary = evidence["error_summary"]
    generated = summary["errors_by_source"]["generated_development"]
    real = summary["errors_by_source"]["wikimedia_commons_open_license"]
    return f"""# Exam project: start here

## One question

**{PRIMARY_QUESTION}**

The main submission is one focused, executed notebook:

- [`01_focused_deep_learning_project.ipynb`](01_focused_deep_learning_project.ipynb)

It presents the data, grouped split, model comparison, concrete successes,
concrete errors, limitations, and the exact reproduction boundary.

## Main result

The frozen multimodal model reaches **{best['integrated_validation_accuracy']:.4f} accuracy**
and **{best['integrated_validation_macro_f1']:.4f} macro F1** on 60 validation samples
from 20 independent physical-part groups. It ranks above the text-only and
image-only neural baselines.

All primary metrics and errors are derived from one exact prediction artifact:
`{summary['prediction_artifact']}`.

The model makes **{summary['error_count']} errors** and correctly recovers
**{summary['true_partial_match_recovered']}/20 `PARTIAL_MATCH` cases**. Its
weakest class recall is `MATCH` at
**{summary['class_performance']['MATCH']['recall']:.2f}**. Real open-license
images have a **{real['error_rate']:.1%}** error rate, compared with
**{generated['error_rate']:.1%}** for generated images.

## Scientific boundary

- Split isolation is by `part_group_id`, not by individual row.
- Train and validation have no physical-part group overlap.
- Selection and analysis use committed validation evidence only.
- The locked test split has not been used.
- No model training is performed by this exam-facing layer.
- The retained production model and selection decision are unchanged.
- The historical 35-error Step 010.4 retraining is supporting evidence only.

## Supporting pages

- [Reproduce the evidence](reproduction.md)
- [Prepare for the oral defense](defense_notes.md)
- [See honest course-topic alignment](course_alignment.md)
- [Browse supplementary engineering evidence](../supplementary/README.md)

The previous full rubric notebook remains available at
[`notebooks/03_final_exam_submission.ipynb`](../notebooks/03_final_exam_submission.ipynb),
but it is historical supporting evidence rather than the source of the focused
primary metrics.
"""


def build_reproduction() -> str:
    return f"""# Reproduction

Run commands from the repository root.

## Rebuild the focused exam layer

```powershell
python -m src.project_cli build-exam-first-submission
python -m src.project_cli verify-exam-first-submission
```

These commands read committed train and validation artifacts, derive every
focused metric and error table from the frozen `keras_multimodal` prediction
file, rebuild the notebook and reports, and verify their hashes. They do not
train a model, open a locked test CSV, authorize final test evaluation, or
change the retained model.

## Open the notebook

```powershell
python -m jupyter notebook exam/01_focused_deep_learning_project.ipynb
```

## Verify the wider repository

```powershell
python -m src.project_cli verify-project
python -m pytest -q
```

TensorFlow-dependent tests require the locked environment in
`requirements-lock.txt`. Archive copies without `.git` cannot independently
prove branch, push, or commit-count state; the Step {STEP} status records the
declared source checkpoint and exact source-archive SHA-256.
"""


def build_defense_notes() -> str:
    return f"""# Oral defense notes

## State the question first

{PRIMARY_QUESTION}

## Explain the split before discussing accuracy

All rows that belong to one physical part share a `part_group_id` and remain in
one split. This prevents the model from seeing the same physical object during
training and validation under a different image-text pairing.

## Explain what the comparison shows

The multimodal model is compared with majority, classical text, classical
image, neural text-only, and neural image-only baselines on the same grouped
validation split. Its advantage is evidence consistent with a benefit from
combining modalities, not proof that every individual prediction causally uses
both inputs.

## Show concrete examples

The notebook contains examples where the multimodal model is correct while one
or both unimodal models fail. The error examples, confusion matrix, source
rates, and aggregate score all come from the same frozen
`keras_multimodal/validation_predictions.csv` artifact.

## Be direct about weaknesses

The frozen model makes 28 errors on 60 validation samples. It recovers 12 of 20
`PARTIAL_MATCH` cases, so that class is not a total failure. The weakest recall
is `MATCH` at 0.30: ten true matches are predicted as `MISMATCH`. Real images
produce a 53.3% error rate versus 40.0% for generated images. The results do
not establish deployment readiness or calibrated probabilities.

## Keep the historical experiment separate

The historical Step 010.4 controlled retraining produced 35 errors and a very
different confusion matrix. It remains useful as stability evidence, but it is
not the prediction set used for the primary 0.5333 / 0.5208 result.

## Evaluation boundary

The test split remains locked. The exam-facing build does not train models and
does not change the retained production recipe.
"""


def build_course_alignment() -> str:
    return """# Course-topic alignment

The main notebook stays focused on one multimodal classification question.
Course topics are evidence, not separate competing project stories.

| Course topic | Evidence in the repository | Status |
|---|---|---|
| Deep Learning Fundamentals | batches, preprocessing, gradients, optimizers, learning rates, capacity, regularization, controlled failures | Complete supplied exercise: 10/10 |
| Transformers and Sequence Modelling | tokenization, embeddings, TextCNN, GRU, LSTM, Transformer encoder, attention evidence | Core complete; pretrained-transformer task remains gated |
| Vision Models | image profiling, representations, resolutions, augmentation, compatibility scoring, ranking, occlusion | Core complete; pretrained backbone, fine-tuning, and genuine human annotation remain gated |
| Language Models | embeddings, attention, and Transformer are demonstrated in the sequence suite | Lecture topics covered |
| Deep Learning Research | LLM API authentication, streaming, structured outputs, retries, caching, sanitization, and cost tracking | Not claimed as part of the primary experiment; wait for the exact exercise before adding an auxiliary API module |

The project deliberately avoids claiming full coverage where the exact exercise
has not been supplied or where an external resource or genuine human annotation
is required.
"""


def build_supplementary_index() -> str:
    return """# Supplementary evidence

The folders below support the main exam story but are not the recommended
starting point for assessment.

## Course exercises

- `notebooks/course_coverage/`
- `docs/course_coverage/`
- `reports/course_coverage/fundamentals/`
- `reports/course_coverage/sequence/`
- `reports/course_coverage/vision/`

## Historical research notebooks

- `notebooks/01_development_experiment.ipynb`
- `notebooks/02_final_exam_project.ipynb`
- `notebooks/03_final_exam_submission.ipynb`

The historical Step 010.4 / Step 011.4 error analysis reports 35 errors from a
separate controlled retraining. It is preserved as stability and audit
evidence. It must not be combined with the primary frozen `keras_multimodal`
run, which has 32 correct predictions and 28 errors.

## Engineering and audit evidence

- `tests/`
- `src/verification/`
- `reports/project_quality/`
- `reports/notebook_quality_audit/`
- `reports/exam_submission_readiness/`
- `reports/final_submission/`
- `reports/final_model_freeze/`

## Dataset provenance

- `data/external/`
- `data/real/`
- `reports/external_dataset/`
- `reports/real_dataset/`

The standard `src/`, `data/`, `reports/`, and `tests/` directories remain in
place so the project stays reproducible and conventional. They were not
physically moved, because their stable paths protect imports, manifests,
notebook links, and historical evidence.
"""


def markdown_cell(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.rstrip())


def code_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source.rstrip())


def build_notebook(evidence: dict[str, Any]) -> nbformat.NotebookNode:
    summary = evidence["error_summary"]
    generated_rate = summary["errors_by_source"]["generated_development"][
        "error_rate"
    ]
    real_rate = summary["errors_by_source"][
        "wikimedia_commons_open_license"
    ]["error_rate"]
    match_recall = summary["class_performance"]["MATCH"]["recall"]

    cells = [
        markdown_cell(
            f"""# Automotive Part Image-Text Matching

**Focused Deep Learning exam notebook — Step {STEP}**

### The one question

> **{PRIMARY_QUESTION}**

This notebook is intentionally narrower than the full repository. It shows one
experiment, one frozen prediction source, the evidence needed to judge it,
concrete examples, concrete failures, and a reproducible boundary. The wider
course exercises and audit material remain supporting evidence."""
        ),
        markdown_cell(
            """## 1. One research question

The input is a photograph of an automotive part and a short description. The
output is one of three relationships:

- `MATCH`: the same part category;
- `PARTIAL_MATCH`: a different category from the same automotive system;
- `MISMATCH`: a category from a different automotive system.

The central comparison is multimodal versus text-only and image-only models on
the same group-isolated validation set."""
        ),
        code_cell(
            """from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from PIL import Image

ROOT = Path.cwd()
if not (ROOT / "reports").is_dir():
    ROOT = ROOT.parent

train = pd.read_csv(ROOT / "data/processed/integrated_train.csv")
validation = pd.read_csv(ROOT / "data/processed/integrated_validation.csv")
comparison = pd.read_csv(
    ROOT / "reports/integrated_training/validation_comparison.csv"
)
retained_predictions = pd.read_csv(
    ROOT
    / "reports/integrated_training/keras_multimodal/validation_predictions.csv"
)
errors = pd.read_csv(
    ROOT
    / "reports/exam_first_submission/retained_model_validation_errors.csv"
)
confusion = pd.read_csv(
    ROOT
    / "reports/exam_first_submission/retained_model_confusion_matrix.csv",
    index_col=0,
)
with (
    ROOT
    / "reports/exam_first_submission/retained_model_error_summary.json"
).open(encoding="utf-8") as handle:
    error_summary = json.load(handle)

print("Primary prediction evidence:", error_summary["prediction_artifact"])
print("All primary metrics and errors use this one frozen prediction set.")
print("No model training is performed in this notebook.")"""
        ),
        markdown_cell(
            """## 2. Data and labels

The final comparison uses generated development images together with manually
reviewed open-license photographs. Each physical part appears in three
image-text relationships, but all relationships for that physical part remain
in the same split."""
        ),
        code_cell(
            """dataset_summary = pd.DataFrame(
    {
        "split": ["train", "validation"],
        "samples": [len(train), len(validation)],
        "physical_part_groups": [
            train["part_group_id"].nunique(),
            validation["part_group_id"].nunique(),
        ],
        "images": [train["image_id"].nunique(), validation["image_id"].nunique()],
    }
)
display(dataset_summary)

display(
    validation.groupby(["source", "label"])
    .size()
    .rename("samples")
    .reset_index()
)"""
        ),
        markdown_cell(
            """## 3. Leakage protection

A row-level random split would be invalid here. The same physical part could
appear with different descriptions on both sides of the split, letting the
model benefit from object identity rather than learn the intended relationship.
The split therefore uses `part_group_id`."""
        ),
        code_cell(
            """overlap = pd.DataFrame(
    {
        "check": [
            "part_group_id overlap",
            "image_id overlap",
            "image_path overlap",
        ],
        "count": [
            len(set(train["part_group_id"]) & set(validation["part_group_id"])),
            len(set(train["image_id"]) & set(validation["image_id"])),
            len(set(train["image_path"]) & set(validation["image_path"])),
        ],
    }
)
display(overlap)
assert overlap["count"].eq(0).all()
print("Group, image ID, and image path isolation: PASS")"""
        ),
        markdown_cell(
            """## 4. Models compared

The same validation split is used for six models: a majority baseline,
classical text and image baselines, neural text-only and image-only models, and
the frozen multimodal network. The main metric is macro F1 because all three
classes matter."""
        ),
        code_cell(
            """result_columns = [
    "validation_rank",
    "model",
    "input_modality",
    "integrated_validation_accuracy",
    "integrated_validation_macro_f1",
]
results = comparison[result_columns].sort_values("validation_rank")
display(results)

plot_data = results.sort_values(
    "integrated_validation_macro_f1", ascending=True
)
figure, axis = plt.subplots(figsize=(9, 5))
axis.barh(plot_data["model"], plot_data["integrated_validation_macro_f1"])
axis.set_xlabel("Validation macro F1")
axis.set_xlim(0.0, 0.6)
axis.set_title("Primary grouped-validation comparison")
figure.tight_layout()
plt.show()"""
        ),
        markdown_cell(
            f"""## 5. Main validation result

The frozen multimodal model ranks first with
`{summary['accuracy']:.4f}` accuracy and `{summary['macro_f1']:.4f}` macro F1.
It has {summary['correct_count']} correct predictions and
{summary['error_count']} errors. This is better than both neural unimodal
baselines, but the score is modest and must be read together with the error
analysis. A higher result on the earlier generated-only development set is not
used as the final claim."""
        ),
        markdown_cell(
            """## 6. What the multimodal model adds

Aggregate metrics alone do not show whether combining image and text helps on
specific cases. The next comparison joins the predictions and surfaces cases
where the multimodal model is correct while one or both unimodal models fail."""
        ),
        code_cell(
            """def load_predictions(slug, prefix):
    frame = pd.read_csv(
        ROOT / f"reports/integrated_training/{slug}/validation_predictions.csv"
    )
    return frame[["sample_id", "predicted_label", "is_correct"]].rename(
        columns={
            "predicted_label": f"{prefix}_prediction",
            "is_correct": f"{prefix}_correct",
        }
    )

joined = (
    retained_predictions.rename(
        columns={
            "predicted_label": "multimodal_prediction",
            "is_correct": "multimodal_correct",
        }
    )
    .merge(load_predictions("keras_text", "text"), on="sample_id")
    .merge(load_predictions("keras_image", "image"), on="sample_id")
)

multimodal_wins = joined[
    joined["multimodal_correct"]
    & (~joined["text_correct"] | ~joined["image_correct"])
].copy()
columns = [
    "sample_id",
    "source",
    "description",
    "true_label",
    "multimodal_prediction",
    "text_prediction",
    "image_prediction",
]
display(multimodal_wins[columns].head(8))

counts = pd.DataFrame(
    {
        "model": ["Text-only", "Image-only", "Multimodal"],
        "correct": [
            int(joined["text_correct"].sum()),
            int(joined["image_correct"].sum()),
            int(joined["multimodal_correct"].sum()),
        ],
    }
)
figure, axis = plt.subplots(figsize=(7, 4.5))
axis.bar(counts["model"], counts["correct"])
axis.set_ylim(0, 60)
axis.set_ylabel("Correct validation samples")
axis.set_title("Correct predictions by neural input setup")
for index, value in enumerate(counts["correct"]):
    axis.text(index, value + 1, str(value), ha="center")
figure.tight_layout()
plt.show()"""
        ),
        markdown_cell(
            """The examples support a limited claim: the two-modality model
resolves cases that one-modality models miss. They do not prove that every
prediction uses both inputs correctly, so the conclusion remains comparative
rather than causal."""
        ),
        markdown_cell(
            f"""## 7. Where the model fails

The same frozen model makes {summary['error_count']} errors among 60 validation
samples. It predicts `PARTIAL_MATCH` {summary['predicted_partial_match_count']}
times and correctly recovers {summary['true_partial_match_recovered']} of 20
true partial matches. The weakest class recall is `MATCH` at
{match_recall:.2f}: ten true matches are classified as `MISMATCH`."""
        ),
        code_cell(
            """display(confusion)

figure, axis = plt.subplots(figsize=(6, 5))
image = axis.imshow(confusion.to_numpy(), aspect="auto")
axis.set_xticks(range(3), ["MATCH", "PARTIAL", "MISMATCH"])
axis.set_yticks(range(3), ["MATCH", "PARTIAL", "MISMATCH"])
axis.set_xlabel("Predicted")
axis.set_ylabel("Actual")
axis.set_title("Frozen keras_multimodal confusion matrix")
for row in range(3):
    for column in range(3):
        axis.text(
            column,
            row,
            int(confusion.iloc[row, column]),
            ha="center",
            va="center",
        )
figure.colorbar(image, ax=axis)
figure.tight_layout()
plt.show()

display(
    errors.groupby("error_pair")
    .size()
    .rename("errors")
    .sort_values(ascending=False)
    .reset_index()
)

display(pd.DataFrame(error_summary["class_performance"]).T)"""
        ),
        code_cell(
            """example_errors = (
    errors.sort_values(
        ["source", "true_label", "predicted_label", "sample_id"]
    )
    .groupby("source", sort=True, group_keys=False)
    .head(2)
    .head(4)
)

figure, axes = plt.subplots(2, 2, figsize=(12, 9))
for axis, (_, row) in zip(axes.flat, example_errors.iterrows()):
    image = Image.open(ROOT / row["image_path"]).convert("RGB")
    axis.imshow(image)
    axis.axis("off")
    axis.set_title(
        f"True: {row['true_label']} | Pred: {row['predicted_label']}\\n"
        f"{row['description']}",
        fontsize=9,
    )
figure.suptitle("Concrete errors from the frozen prediction set")
figure.tight_layout()
plt.show()

display(
    example_errors[
        [
            "sample_id",
            "source",
            "true_label",
            "predicted_label",
            "description",
        ]
    ]
)"""
        ),
        code_cell(
            """source_errors = (
    errors.groupby("source").size().rename("errors").reset_index()
)
source_totals = (
    validation.groupby("source").size().rename("samples").reset_index()
)
source_rates = source_errors.merge(source_totals, on="source")
source_rates["error_rate"] = (
    source_rates["errors"] / source_rates["samples"]
)
display(source_rates)

figure, axis = plt.subplots(figsize=(8, 4.5))
axis.bar(source_rates["source"], source_rates["error_rate"])
axis.set_ylim(0.0, 1.0)
axis.set_ylabel("Error rate")
axis.set_title("Frozen model: real open-license images are harder")
axis.tick_params(axis="x", rotation=15)
figure.tight_layout()
plt.show()"""
        ),
        markdown_cell(
            f"""## 8. Limits of the conclusion

The evidence supports the statement that the frozen multimodal model is the
strongest of the compared models on this grouped validation split. It does not
support claims of production readiness, general performance across all vehicle
parts, calibrated probabilities, human-level explanation, or final test
performance.

The main limitations are the small dataset, low `MATCH` recall
({match_recall:.2f}), and the higher error rate on real images
({real_rate:.1%} versus {generated_rate:.1%}). `PARTIAL_MATCH` remains
imperfect, but the frozen run does recover 12 of 20 cases."""
        ),
        markdown_cell(
            """## 9. Reproduction and evaluation boundary

From the repository root:

```powershell
python -m src.project_cli build-exam-first-submission
python -m src.project_cli verify-exam-first-submission
```

The exam-facing build reads committed train and validation evidence. It does
not train a model, open locked test CSV files, authorize final test evaluation,
or change the retained production model."""
        ),
        code_cell(
            """freeze_path = (
    ROOT / "reports/final_model_freeze/final_model_freeze_status.json"
)
with freeze_path.open(encoding="utf-8") as handle:
    freeze_status = json.load(handle)

boundary = pd.DataFrame(
    [
        {
            "model_training_performed": False,
            "locked_test_csv_files_opened": False,
            "test_split_used": False,
            "final_test_evaluation_authorized": False,
            "production_final_model_changed": False,
            "retained_model": freeze_status["final_model_slug"],
            "selection_decision": freeze_status["selection_decision"],
            "primary_prediction_artifact": error_summary[
                "prediction_artifact"
            ],
        }
    ]
)
display(boundary)
assert freeze_status["protocol_frozen"] is True
assert freeze_status["test_lock_preserved"] is True
assert error_summary["single_prediction_set_used"] is True
print("Evaluation and single-model evidence boundary: PASS")"""
        ),
        markdown_cell(
            f"""## 10. Conclusion

The multimodal network is the best model in the controlled comparison. Its
advantage over the unimodal neural baselines is visible both in macro F1 and in
specific examples. This is consistent with a benefit from combining image and
text, but it does not prove causal use of both modalities in every decision.

The frozen model still makes {summary['error_count']} errors, has low `MATCH`
recall, and performs worse on real images. The correct conclusion is measured
improvement, not completion. The next scientific step should improve real-image
robustness and `MATCH`/`MISMATCH` separation before any final test evaluation is
authorized."""
        ),
        markdown_cell(
            """## References

1. Faghri, F. et al. *VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.* BMVC, 2018.
2. Li, L. H. et al. *VisualBERT: A Simple and Performant Baseline for Vision and Language.* 2019.
3. Radford, A. et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021.
4. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS, 2017.
5. He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
6. scikit-learn documentation: model evaluation and grouped cross-validation."""
        ),
    ]

    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    notebook.metadata["project"] = {
        "step": STEP,
        "teacher_facing": True,
        "primary_research_question": PRIMARY_QUESTION,
        "primary_prediction_artifact": project_relative(
            RETAINED_PREDICTIONS_PATH
        ),
        "primary_prediction_artifact_sha256": normalized_sha256(
            RETAINED_PREDICTIONS_PATH
        ),
        "single_prediction_set_used": True,
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
        "supporting_evidence_is_secondary": True,
    }
    return notebook


def execute_notebook(notebook: nbformat.NotebookNode) -> None:
    EXAM_DIR.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    try:
        client.execute()
    except CellExecutionError as error:
        raise RuntimeError("Focused exam notebook execution failed.") from error
    nbformat.write(notebook, NOTEBOOK_PATH)


def notebook_metrics() -> dict[str, int]:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
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
    return {
        "notebook_cells": len(notebook.cells),
        "code_cells": len(code_cells),
        "executed_code_cells": sum(
            cell.get("execution_count") is not None for cell in code_cells
        ),
        "saved_outputs": len(outputs),
        "figures": len(figures),
        "error_outputs": sum(
            output.get("output_type") == "error" for output in outputs
        ),
    }


def update_legacy_manifest_readme_hash() -> None:
    manifest_paths = (
        PROJECT_ROOT / "reports/final_submission/final_submission_manifest.json",
        PROJECT_ROOT
        / "reports/exam_submission_readiness/"
        "exam_submission_readiness_manifest.json",
    )
    readme_hash = normalized_sha256(PROJECT_ROOT / "README.md")
    for legacy_manifest_path in manifest_paths:
        manifest = read_json(legacy_manifest_path)
        source_hashes = manifest.get("source_artifact_sha256", {})
        if "README.md" in source_hashes:
            source_hashes["README.md"] = readme_hash
        legacy_manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def build_status(evidence: dict[str, Any], metrics: dict[str, int]) -> dict[str, Any]:
    best = evidence["comparison"].sort_values("validation_rank").iloc[0]
    error_summary = evidence["error_summary"]
    return {
        "schema_version": "1.1",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "primary_research_question": PRIMARY_QUESTION,
        "base_checkpoint_commit": BASE_CHECKPOINT_COMMIT,
        "base_checkpoint_commit_count": BASE_CHECKPOINT_COMMIT_COUNT,
        "source_archive": SOURCE_ARCHIVE,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "primary_notebook": project_relative(NOTEBOOK_PATH),
        "validation_sample_count": int(len(evidence["validation"])),
        "validation_part_group_count": int(
            evidence["validation"]["part_group_id"].nunique()
        ),
        "retained_model_slug": str(best["model_slug"]),
        "primary_prediction_artifact": error_summary[
            "prediction_artifact"
        ],
        "primary_prediction_artifact_sha256": error_summary[
            "prediction_artifact_sha256"
        ],
        "single_prediction_set_used": True,
        "validation_accuracy": float(
            best["integrated_validation_accuracy"]
        ),
        "validation_macro_f1": float(
            best["integrated_validation_macro_f1"]
        ),
        "validation_correct_count": int(error_summary["correct_count"]),
        "validation_error_count": int(error_summary["error_count"]),
        "partial_match_error_count": int(
            error_summary["partial_match_error_count"]
        ),
        "partial_match_recovered_count": int(
            error_summary["true_partial_match_recovered"]
        ),
        "predicted_partial_match_count": int(
            error_summary["predicted_partial_match_count"]
        ),
        "generated_error_rate": float(
            error_summary["errors_by_source"]["generated_development"][
                "error_rate"
            ]
        ),
        "real_image_error_rate": float(
            error_summary["errors_by_source"][
                "wikimedia_commons_open_license"
            ]["error_rate"]
        ),
        "match_recall": float(
            error_summary["class_performance"]["MATCH"]["recall"]
        ),
        **metrics,
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
        "model_selection_changed": False,
        "physical_source_files_moved": False,
        "historical_controlled_retraining_is_primary": False,
    }


def build_summary(status: dict[str, Any]) -> str:
    return f"""# Step {STEP} — Single-Model Evidence Consistency Correction

Status: **{status['status']}**

Readiness: `{status['readiness']}`

## Purpose

The focused teacher-facing submission now derives its aggregate score,
confusion matrix, error table, source error rates, class analysis, and concrete
examples from one exact frozen `keras_multimodal` validation-prediction
artifact.

## Primary result

- retained model: `{status['retained_model_slug']}`;
- prediction artifact: `{status['primary_prediction_artifact']}`;
- grouped validation samples: {status['validation_sample_count']};
- independent physical-part groups: {status['validation_part_group_count']};
- validation accuracy: {status['validation_accuracy']:.4f};
- validation macro F1: {status['validation_macro_f1']:.4f};
- correct predictions: {status['validation_correct_count']};
- validation errors analyzed: {status['validation_error_count']};
- `PARTIAL_MATCH` recovered: {status['partial_match_recovered_count']}/20;
- `PARTIAL_MATCH` errors: {status['partial_match_error_count']};
- generated-image error rate: {status['generated_error_rate']:.1%};
- real-image error rate: {status['real_image_error_rate']:.1%};
- `MATCH` recall: {status['match_recall']:.2f}.

## Evidence separation

The historical Step 010.4 controlled retraining with 35 errors remains
preserved as supplementary stability evidence. It is not used for the primary
0.5333 accuracy / 0.5208 macro-F1 claim.

## Safety boundary

- model training performed: false;
- locked test CSV files opened: false;
- test split used: false;
- final test evaluation authorized: false;
- production final model changed: false;
- model selection changed: false.
"""


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
        "schema_version": "1.1",
        "step": STEP,
        "status": "PASS",
        "readiness": READINESS,
        "hash_normalization": "utf-8-lf for text; raw bytes for binary",
        "base_checkpoint_commit": BASE_CHECKPOINT_COMMIT,
        "base_checkpoint_commit_count": BASE_CHECKPOINT_COMMIT_COUNT,
        "source_archive": SOURCE_ARCHIVE,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_artifact_sha256": source_hashes,
        "generated_artifact_sha256": generated_hashes,
        "source_artifact_count": len(source_hashes),
        "generated_artifact_count": len(generated_hashes),
        "single_prediction_set_used": True,
        "primary_prediction_artifact": project_relative(
            RETAINED_PREDICTIONS_PATH
        ),
        "primary_prediction_artifact_sha256": normalized_sha256(
            RETAINED_PREDICTIONS_PATH
        ),
        "model_training_performed": False,
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
        "production_final_model_changed": False,
        "model_selection_changed": False,
    }


def main() -> None:
    evidence = load_evidence()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXAM_DIR.mkdir(parents=True, exist_ok=True)

    write_focused_evidence(evidence)
    build_figures(evidence)
    clarify_historical_readme()

    historical_readme = HISTORICAL_README_PATH.read_text(encoding="utf-8-sig")
    root_readme = (
        build_root_readme().rstrip()
        + "\n\n<details>\n<summary>Complete engineering and historical "
        "documentation</summary>\n\n"
        + historical_readme.rstrip()
        + "\n\n</details>"
    )
    write_text(ROOT_README_PATH, root_readme)

    historical_catalogue = HISTORICAL_NOTEBOOK_CATALOGUE_PATH.read_text(
        encoding="utf-8-sig"
    )
    catalogue = (
        build_notebook_catalogue().rstrip()
        + "\n\n<details>\n<summary>Historical notebook catalogue "
        "and verification commands</summary>\n\n"
        + historical_catalogue.rstrip()
        + "\n\n</details>"
    )
    write_text(NOTEBOOK_CATALOGUE_PATH, catalogue)

    write_text(EXAM_README_PATH, build_exam_readme(evidence))
    write_text(REPRODUCTION_PATH, build_reproduction())
    write_text(DEFENSE_NOTES_PATH, build_defense_notes())
    write_text(COURSE_ALIGNMENT_PATH, build_course_alignment())
    write_text(SUPPLEMENTARY_INDEX_PATH, build_supplementary_index())

    notebook = build_notebook(evidence)
    execute_notebook(notebook)
    metrics = notebook_metrics()
    status = build_status(evidence, metrics)
    STATUS_PATH.write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_text(SUMMARY_PATH, build_summary(status))

    update_legacy_manifest_readme_hash()
    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Step {STEP} single-model evidence correction built")
    print(f"- notebook: {project_relative(NOTEBOOK_PATH)}")
    print(f"- notebook cells: {metrics['notebook_cells']}")
    print(f"- executed code cells: {metrics['executed_code_cells']}")
    print(f"- saved outputs: {metrics['saved_outputs']}")
    print(f"- figures: {metrics['figures']}")
    print(f"- validation correct: {status['validation_correct_count']}")
    print(f"- validation errors: {status['validation_error_count']}")
    print("- single prediction set used: true")
    print("- model training performed: false")
    print("- locked test CSV files opened: false")
    print("- production final model changed: false")


if __name__ == "__main__":
    main()
