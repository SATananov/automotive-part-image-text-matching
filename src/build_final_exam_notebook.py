from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from src.final_exam_notebook_config import (
    BASE_CHECKPOINT_COMMIT,
    BASE_VERIFIED_TEST_COUNT,
    BASE_VERIFIED_WARNING_COUNT,
    CONTROLLED_EXPERIMENT_PATH,
    ERROR_ANALYSIS_PATH,
    FINAL_EVALUATION_PROTOCOL_PATH,
    FINAL_EXAM_NOTEBOOK_MANIFEST_PATH,
    FINAL_EXAM_NOTEBOOK_PATH,
    FINAL_EXAM_NOTEBOOK_STATUS_PATH,
    FINAL_EXAM_NOTEBOOK_SUMMARY_PATH,
    FINAL_MODEL_FREEZE_STATUS_PATH,
    FINAL_MODEL_SPECIFICATION_PATH,
    FINAL_TEST_AUTHORIZATION_PATH,
    FORBIDDEN_NOTEBOOK_CODE_TOKENS,
    INTEGRATED_COMPARISON_PATH,
    INTEGRATED_STATUS_PATH,
    INTEGRATED_TRAIN_PATH,
    INTEGRATED_VALIDATION_PATH,
    LOCKED_TEST_CONTRACT_PATH,
    MODEL_SELECTION_DECISION_PATH,
    MULTIMODAL_CONFUSION_MATRIX_PATH,
    MULTIMODAL_METRICS_PATH,
    MULTIMODAL_PREDICTIONS_PATH,
    NOTEBOOK_READINESS,
    REQUIRED_NOTEBOOK_HEADINGS,
)
from src.real_dataset_config import PROJECT_ROOT
from src.validate_external_training_readiness import project_relative_path


class FinalExamNotebookError(RuntimeError):
    """Raised when the final exam notebook cannot be built safely."""


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FinalExamNotebookError(
            f"Required notebook input is missing: {project_relative_path(path)}."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalExamNotebookError(
            f"Cannot read notebook input {project_relative_path(path)}: {error}."
        ) from error
    if not isinstance(payload, dict):
        raise FinalExamNotebookError(
            f"Notebook input is not a JSON object: {project_relative_path(path)}."
        )
    return payload


def sha256_file(path: Path) -> str:
    """Hash text evidence after UTF-8/LF normalization for portability."""
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_prerequisites() -> dict[str, dict[str, Any]]:
    integrated_status = read_json(INTEGRATED_STATUS_PATH)
    selection = read_json(MODEL_SELECTION_DECISION_PATH)
    model_specification = read_json(FINAL_MODEL_SPECIFICATION_PATH)
    evaluation_protocol = read_json(FINAL_EVALUATION_PROTOCOL_PATH)
    freeze_status = read_json(FINAL_MODEL_FREEZE_STATUS_PATH)
    authorization = read_json(FINAL_TEST_AUTHORIZATION_PATH)
    locked_contract = read_json(LOCKED_TEST_CONTRACT_PATH)

    if integrated_status.get("status") != "PASS":
        raise FinalExamNotebookError("Integrated training status is not PASS.")
    if integrated_status.get("test_split_used") is not False:
        raise FinalExamNotebookError("Integrated training reports test use.")
    if selection.get("decision") != "REFERENCE_RETAINED":
        raise FinalExamNotebookError("The frozen model decision is not retained.")
    if selection.get("final_test_evaluation_authorized") is not False:
        raise FinalExamNotebookError("Model selection already authorizes test use.")
    if model_specification.get("freeze_state") != "FROZEN":
        raise FinalExamNotebookError("Final model specification is not frozen.")
    if evaluation_protocol.get("protocol_state") != "FROZEN_NOT_AUTHORIZED":
        raise FinalExamNotebookError("Final evaluation protocol is not closed.")
    if freeze_status.get("readiness") != (
        "FINAL_MODEL_AND_EVALUATION_PROTOCOL_FROZEN_TEST_LOCKED"
    ):
        raise FinalExamNotebookError("Step 010.5 readiness differs.")
    if authorization.get("authorized") is not False:
        raise FinalExamNotebookError("Final test authorization is open.")
    if locked_contract.get("test_split_used") is not False:
        raise FinalExamNotebookError("Locked-test contract reports test use.")

    for path in (
        INTEGRATED_TRAIN_PATH,
        INTEGRATED_VALIDATION_PATH,
        INTEGRATED_COMPARISON_PATH,
        MULTIMODAL_METRICS_PATH,
        MULTIMODAL_CONFUSION_MATRIX_PATH,
        MULTIMODAL_PREDICTIONS_PATH,
        CONTROLLED_EXPERIMENT_PATH,
        ERROR_ANALYSIS_PATH,
    ):
        if not path.is_file():
            raise FinalExamNotebookError(
                f"Required committed evidence is missing: {project_relative_path(path)}."
            )

    return {
        "integrated_status": integrated_status,
        "selection": selection,
        "model_specification": model_specification,
        "evaluation_protocol": evaluation_protocol,
        "freeze_status": freeze_status,
        "authorization": authorization,
        "locked_contract": locked_contract,
    }


def markdown(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> nbformat.NotebookNode:
    source = text.strip() + "\n"
    lowered = source.lower()
    for token in FORBIDDEN_NOTEBOOK_CODE_TOKENS:
        if token.lower() in lowered:
            raise FinalExamNotebookError(
                f"Forbidden notebook code token detected: {token}."
            )
    return nbformat.v4.new_code_cell(source)


def build_notebook() -> nbformat.NotebookNode:
    cells: list[nbformat.NotebookNode] = []

    cells.append(markdown(r"""
# Automotive Part Image-Text Matching

## Final exam research notebook

**Task:** classify the relationship between an automotive-part image and a short description as `MATCH`, `PARTIAL_MATCH`, or `MISMATCH`.

**Evaluation boundary:** this notebook presents committed development and integrated **validation** evidence only. The final test split remains locked, unused, and unauthorized.
"""))

    cells.append(markdown(r"""
## 1. Abstract

This project studies a three-class image-text relationship problem for automotive parts. The work begins with a deterministic generated development dataset, extends the pipeline with reviewed open-license photographs, and compares classical, unimodal neural, and multimodal neural models under group-isolated train and validation splits. The integrated multimodal model achieved the strongest validation result (`accuracy = 0.5333`, `macro F1 = 0.5208`). A controlled improvement study did not satisfy the predefined incumbent guard, so the original multimodal recipe was retained. The model recipe and a one-shot future evaluation protocol are frozen, while the locked test split remains unused.
"""))

    cells.append(markdown(r"""
## 2. Problem statement and motivation

Automotive catalogues and warehouses often combine a photograph with a short product description. A wrong pairing can mislead catalogue users, create picking errors, or require additional manual review. This research asks whether a supervised model can classify the **relationship** between the visual and textual information rather than only predict a part category.

The three labels represent progressively weaker semantic agreement:

- **MATCH:** image and description refer to the same category;
- **PARTIAL_MATCH:** categories differ but belong to the same automotive system;
- **MISMATCH:** categories belong to different systems.

This formulation is useful because a near miss, such as a brake disc paired with a brake-pad description, is not equivalent to an unrelated pairing such as a brake disc and a starter motor.
"""))

    cells.append(markdown(r"""
## 3. Research question and hypothesis

**Research question:** Does combining image and text information improve validation performance over text-only and image-only models for three-class automotive part relationship classification?

**Hypothesis:** A multimodal model will outperform unimodal alternatives because the label is defined by the interaction between two inputs. Image-only models cannot know which description was supplied, while text-only models cannot verify the visible part.

**Primary selection metric:** macro F1 on the integrated validation split. Macro F1 gives equal importance to all three labels and is more informative than accuracy when a model neglects one class.
"""))

    cells.append(markdown(r"""
## 4. Related work

Image-text research commonly learns representations that connect visual and linguistic information. VSE++ demonstrated the value of hard negatives for visual-semantic embedding and cross-modal retrieval. VisualBERT introduced a single Transformer stack that aligns text tokens and image-region representations. CLIP scaled image-text alignment through contrastive pretraining on a very large corpus. These systems are much larger than the compact supervised models used here, but they motivate the central idea that visual and textual evidence should be modeled jointly.

The current project is deliberately narrower: it uses a small, auditable dataset, a three-class relation label, simple late fusion, and explicit train/validation group isolation. Keras Functional API is appropriate for this architecture because it supports multiple input branches in one directed model graph.

| Source | Relevance to this project |
|---|---|
| Faghri et al., **VSE++** (BMVC 2018) | Image-text alignment and informative negative pairs |
| Li et al., **VisualBERT** (2019/ACL 2020) | Joint modeling of visual and textual representations |
| Radford et al., **CLIP** (ICML 2021) | Large-scale contrastive image-text pairing |
| TensorFlow, **Keras Functional API** | Multi-input neural-network construction |
"""))

    cells.append(code(r"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import display

ROOT = Path.cwd()
if not (ROOT / "src").is_dir():
    ROOT = ROOT.parent

assert (ROOT / "src").is_dir(), "Run the notebook from the repository root or notebooks directory."

TRAIN_CSV = ROOT / "data" / "processed" / "integrated_train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "integrated_validation.csv"
COMPARISON_CSV = ROOT / "reports" / "integrated_training" / "validation_comparison.csv"
MULTIMODAL_METRICS_JSON = ROOT / "reports" / "integrated_training" / "keras_multimodal" / "validation_metrics.json"
CONFUSION_MATRIX_CSV = ROOT / "reports" / "integrated_training" / "keras_multimodal" / "validation_confusion_matrix.csv"
PREDICTIONS_CSV = ROOT / "reports" / "integrated_training" / "keras_multimodal" / "validation_predictions.csv"
CONTROLLED_COMPARISON_CSV = ROOT / "reports" / "validation_model_improvement" / "controlled_experiment_comparison.csv"
ERROR_ANALYSIS_CSV = ROOT / "reports" / "validation_model_improvement" / "validation_error_analysis.csv"
SELECTION_JSON = ROOT / "reports" / "validation_model_improvement" / "model_selection_decision.json"
MODEL_SPEC_JSON = ROOT / "reports" / "final_model_freeze" / "final_model_specification.json"
PROTOCOL_JSON = ROOT / "reports" / "final_model_freeze" / "final_evaluation_protocol.json"
FREEZE_STATUS_JSON = ROOT / "reports" / "final_model_freeze" / "final_model_freeze_status.json"
AUTHORIZATION_JSON = ROOT / "reports" / "final_model_freeze" / "final_test_authorization.json"

train_df = pd.read_csv(TRAIN_CSV)
validation_df = pd.read_csv(VALIDATION_CSV)
comparison_df = pd.read_csv(COMPARISON_CSV)
metrics = json.loads(MULTIMODAL_METRICS_JSON.read_text(encoding="utf-8"))
selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
model_spec = json.loads(MODEL_SPEC_JSON.read_text(encoding="utf-8"))
protocol = json.loads(PROTOCOL_JSON.read_text(encoding="utf-8"))
freeze_status = json.loads(FREEZE_STATUS_JSON.read_text(encoding="utf-8"))
authorization = json.loads(AUTHORIZATION_JSON.read_text(encoding="utf-8"))

print("Committed evidence loaded successfully.")
print(f"Training rows: {len(train_df)} | Validation rows: {len(validation_df)}")
print(f"Final-test authorization: {authorization['authorized']}")
"""))

    cells.append(markdown(r"""
## 5. Dataset construction, licensing and ethics

The project uses two data sources:

1. **Generated development images** for deterministic pipeline development.
2. **Reviewed open-license photographs** collected from Wikimedia Commons with source page, author, licence, licence URL, image hash, dimensions, and approval decision recorded in the repository.

Each physical part group generates three relationship samples by pairing one image with descriptions representing the three labels. This keeps the label semantics controlled, but it also means rows are not statistically independent. The split therefore operates on `part_group_id`, not on individual rows.

The open-license workflow accepts only reviewed images with documented attribution. Local original capture folders and staging material are excluded from Git. No personal or sensitive data is required by the experiment.
"""))

    cells.append(code(r"""
def profile_split(frame: pd.DataFrame, split_name: str) -> dict[str, object]:
    return {
        "split": split_name,
        "samples": len(frame),
        "images": frame["image_id"].nunique(),
        "groups": frame["part_group_id"].nunique(),
        "categories": frame["part_category"].nunique(),
        "generated samples": int((frame["source"] == "generated_development").sum()),
        "open-license samples": int((frame["source"] == "wikimedia_commons_open_license").sum()),
    }

split_profile = pd.DataFrame([
    profile_split(train_df, "train"),
    profile_split(validation_df, "validation"),
])

display(split_profile)
"""))

    cells.append(code(r"""
composition = (
    pd.concat([
        train_df.assign(split="train"),
        validation_df.assign(split="validation"),
    ])
    .groupby(["split", "source"])
    .size()
    .unstack(fill_value=0)
)

ax = composition.plot(kind="bar", figsize=(9, 4))
ax.set_title("Integrated dataset composition by committed split")
ax.set_xlabel("Split")
ax.set_ylabel("Samples")
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Source")
plt.tight_layout()
plt.show()
"""))

    cells.append(code(r"""
examples = pd.concat([
    train_df[train_df["source"] == "generated_development"].head(1),
    train_df[train_df["source"] == "wikimedia_commons_open_license"].head(1),
], ignore_index=True)

for row in examples.itertuples(index=False):
    image = Image.open(ROOT / row.image_path).convert("RGB")
    plt.figure(figsize=(6, 4))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{row.source}: {row.part_category}\n{row.description} → {row.label}")
    plt.tight_layout()
    plt.show()
"""))

    cells.append(markdown(r"""
## 6. Data cleaning and grouped split

The cleaning and formatting pipeline validates identifiers, category-family mappings, safe relative paths, readable images, hashes, duplicate content, approval state, and label balance. The integrated training and validation inputs contain ten automotive categories and equal numbers of all three relationship labels.

The critical statistical safeguard is **group isolation**. All three rows derived from the same physical part stay in one split. This prevents the model from seeing one pairing from a part during training and another pairing from the same part during validation.
"""))

    cells.append(code(r"""
train_groups = set(train_df["part_group_id"])
validation_groups = set(validation_df["part_group_id"])

group_audit = pd.DataFrame([
    {
        "train groups": len(train_groups),
        "validation groups": len(validation_groups),
        "train-validation overlap": len(train_groups & validation_groups),
        "train label counts": train_df["label"].value_counts().sort_index().to_dict(),
        "validation label counts": validation_df["label"].value_counts().sort_index().to_dict(),
    }
])

display(group_audit)
assert not (train_groups & validation_groups)
"""))

    cells.append(markdown(r"""
## 7. Models and experimental design

Six model families were compared under the same integrated train/validation boundary:

- majority baseline;
- TF-IDF with Logistic Regression;
- resized image pixels with Logistic Regression;
- Keras text neural network;
- Keras image neural network;
- Keras multimodal neural network.

The final multimodal architecture has separate text and image branches. The text branch uses token embeddings and global average pooling. The image branch uses rescaling, flattening, and dense layers. Their representations are concatenated and passed through a fusion classifier. The design is intentionally compact so that the contribution of multimodal information can be compared without a large pretrained backbone.
"""))

    cells.append(code(r"""
model_table = comparison_df[[
    "validation_rank",
    "model",
    "input_modality",
    "integrated_validation_accuracy",
    "integrated_validation_macro_f1",
    "development_validation_accuracy",
    "development_validation_macro_f1",
]].copy()

model_table.columns = [
    "rank",
    "model",
    "modality",
    "integrated accuracy",
    "integrated macro F1",
    "development accuracy",
    "development macro F1",
]

display(model_table.round(4))
"""))

    cells.append(markdown(r"""
## 8. Development and integrated validation results

The multimodal model ranked first on the integrated validation split. Its performance declined relative to the easier generated development split, which is an important result rather than a failure: the open-license photographs introduce realistic variation in background, viewpoint, scale, and object presentation.

The gap between development and integrated validation also warns against reporting only generated-data results. The integrated comparison is the more credible estimate for the current project stage.
"""))

    cells.append(code(r"""
plot_df = comparison_df.sort_values("integrated_validation_macro_f1")
ax = plot_df.plot(
    x="model",
    y=["integrated_validation_accuracy", "integrated_validation_macro_f1"],
    kind="barh",
    figsize=(10, 6),
)
ax.set_title("Integrated validation comparison")
ax.set_xlabel("Score")
ax.set_ylabel("")
ax.set_xlim(0, 0.60)
ax.legend(["Accuracy", "Macro F1"])
plt.tight_layout()
plt.show()
"""))

    cells.append(code(r"""
best = comparison_df.iloc[0]
performance_summary = pd.DataFrame([
    {
        "selected model": best["model"],
        "integrated validation accuracy": best["integrated_validation_accuracy"],
        "integrated validation macro F1": best["integrated_validation_macro_f1"],
        "development validation accuracy": best["development_validation_accuracy"],
        "development validation macro F1": best["development_validation_macro_f1"],
        "macro F1 change": best["macro_f1_change"],
    }
])
display(performance_summary.round(4))
"""))

    cells.append(code(r"""
confusion = pd.read_csv(CONFUSION_MATRIX_CSV, index_col=0)

fig = plt.figure(figsize=(6, 5))
plt.imshow(confusion.to_numpy(), aspect="auto")
plt.xticks(range(len(confusion.columns)), [c.replace("predicted_", "") for c in confusion.columns], rotation=25)
plt.yticks(range(len(confusion.index)), [i.replace("actual_", "") for i in confusion.index])
plt.title("Keras multimodal validation confusion matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
for row_index in range(confusion.shape[0]):
    for column_index in range(confusion.shape[1]):
        plt.text(column_index, row_index, int(confusion.iloc[row_index, column_index]), ha="center", va="center")
plt.colorbar(label="Samples")
plt.tight_layout()
plt.show()
"""))

    cells.append(code(r"""
per_class = pd.DataFrame(metrics["per_class"]).T
per_class.index.name = "label"
display(per_class[["precision", "recall", "f1", "support"]].round(4))
"""))

    cells.append(markdown(r"""
## 9. Validation error analysis

The integrated multimodal model made 35 errors among 60 validation samples. The most difficult distinction is the middle relationship class: `PARTIAL_MATCH` requires the model to identify both a category difference and a shared automotive family. Errors can therefore arise even when one modality is interpreted correctly.

The analysis below reports predefined validation errors only. It does not introduce a new split or inspect the locked test data.
"""))

    cells.append(code(r"""
errors = pd.read_csv(ERROR_ANALYSIS_CSV)
error_pairs = errors["error_pair"].value_counts().rename_axis("error pair").reset_index(name="count")
display(error_pairs)

ax = error_pairs.sort_values("count").plot(
    x="error pair",
    y="count",
    kind="barh",
    legend=False,
    figsize=(9, 5),
)
ax.set_title("Integrated validation error types")
ax.set_xlabel("Errors")
ax.set_ylabel("")
plt.tight_layout()
plt.show()
"""))

    cells.append(code(r"""
predictions = pd.read_csv(PREDICTIONS_CSV)
source_accuracy = (
    predictions.groupby("source")["is_correct"]
    .mean()
    .rename("validation accuracy")
    .reset_index()
)
category_accuracy = (
    predictions.groupby("part_category")["is_correct"]
    .mean()
    .sort_values()
    .rename("validation accuracy")
    .reset_index()
)

display(source_accuracy.round(4))
display(category_accuracy.round(4))
"""))

    cells.append(markdown(r"""
## 10. Controlled model improvement and selection

Three predefined multimodal candidates were trained across fixed seeds after the error analysis. The relation-aware candidate improved the repeated-seed mean and aggregate validation result relative to the newly retrained reference candidate. However, the project used an **incumbent guard** against the stronger Step 010.3 multimodal result. The candidate did not satisfy all predefined conditions, so the decision was `REFERENCE_RETAINED`.

This conservative decision avoids promoting a model merely because it wins within a weaker rerun. It also documents a negative experimental result rather than hiding it.
"""))

    cells.append(code(r"""
controlled = pd.read_csv(CONTROLLED_COMPARISON_CSV)
display(controlled[[
    "validation_rank",
    "candidate",
    "aggregate_validation_accuracy",
    "aggregate_validation_macro_f1",
    "mean_seed_macro_f1",
    "std_seed_macro_f1",
    "worst_class_f1",
    "parameter_count",
]].round(4))

selection_summary = pd.DataFrame([
    {
        "decision": selection["decision"],
        "selected family": selection["selected_candidate_slug"],
        "Step 010.3 incumbent macro F1": selection["incumbent_validation_macro_f1"],
        "final-test authorization": selection["final_test_evaluation_authorized"],
    }
])
display(selection_summary.round(4))
"""))

    cells.append(markdown(r"""
## 11. Final model and locked-test protocol freeze

Step 010.5 freezes the **model recipe**, preprocessing contract, label order, training settings, checkpoint-selection rule, environment fingerprint, and future one-shot evaluation metrics. Serialized trained weights are not falsely claimed as committed; the future protocol specifies exact reconstruction from the frozen recipe.

Protocol freeze is not test authorization. The current authorization flag remains `false`, and this notebook does not open, parse, predict on, or evaluate the final test split.
"""))

    cells.append(code(r"""
architecture = model_spec["architecture_contract"]
training_contract = model_spec["training_contract"]

freeze_table = pd.DataFrame([
    {"item": "Final model", "frozen value": model_spec["final_model"]},
    {"item": "Model family", "frozen value": model_spec["final_model_family"]},
    {"item": "Validation macro F1", "frozen value": model_spec["validation_evidence"]["macro_f1"]},
    {"item": "Trainable parameters", "frozen value": architecture["trainable_parameter_count"]},
    {"item": "Random state", "frozen value": training_contract["random_state"]},
    {"item": "Batch size", "frozen value": training_contract["batch_size"]},
    {"item": "Checkpoint rule", "frozen value": training_contract["checkpoint_selection_rule"]},
    {"item": "Protocol state", "frozen value": protocol["protocol_state"]},
    {"item": "Serialized weights committed", "frozen value": model_spec["serialized_weights_committed"]},
    {"item": "Final-test authorized", "frozen value": authorization["authorized"]},
])

display(freeze_table)
"""))

    cells.append(markdown(r"""
## 12. Testing and reproducibility

The repository separates dataset creation, validation, model workflows, reporting, and verification into reusable modules exposed through a central CLI. Tests cover grouped splitting, model outputs, licensing and attribution, real-data intake, transactional operations, repository hygiene, validation-only model selection, final model freeze, and locked-test safeguards.

At the Step 010.5 checkpoint, the full project environment reported **275 passing tests** with **154 known dependency warnings**. The dependency environment is pinned in `requirements-lock.txt`. Random seeds and deterministic TensorFlow operations are enabled where supported, while the report acknowledges that small numerical differences may occur across hardware and TensorFlow builds.
"""))

    cells.append(code(r"""
reproducibility = pd.DataFrame([
    {"control": "Base checkpoint", "value": "d517668"},
    {"control": "Verified tests at checkpoint", "value": 275},
    {"control": "Known dependency warnings", "value": 154},
    {"control": "Dependency lock", "value": "requirements-lock.txt"},
    {"control": "Group overlap", "value": len(train_groups & validation_groups)},
    {"control": "Test split used", "value": freeze_status["test_split_used"]},
    {"control": "Final-test authorized", "value": authorization["authorized"]},
])
display(reproducibility)
"""))

    cells.append(markdown(r"""
## 13. Limitations and threats to validity

- The integrated dataset is still small: 180 training and 60 validation rows derived from 80 physical part groups.
- Three rows are generated per image, so group-level splitting is essential and the effective independent sample size is the number of groups rather than rows.
- The generated source is cleaner than the open-license photographs and can overestimate performance.
- The image branch is a compact dense model over resized pixels rather than a pretrained convolutional or vision-transformer encoder.
- Short templated descriptions limit linguistic diversity.
- Validation results have uncertainty because the validation split contains only 20 physical groups.
- The controlled improvement study did not produce a candidate that passed the incumbent guard.
- No final test metric is reported because the test split remains locked and unauthorized at this stage.

These limitations make the project an honest baseline and a reproducible research foundation, not a production-ready automotive catalogue system.
"""))

    cells.append(markdown(r"""
## 14. Conclusion and future work

The results support the central hypothesis: the multimodal model is the strongest of the evaluated integrated validation models because the task depends on the relationship between image and text. Its macro F1 of `0.5208` exceeds both text-only and image-only alternatives, although the integrated result is substantially lower than the generated development result.

The project contributes a reproducible end-to-end workflow: controlled label construction, open-license data provenance, grouped splitting, classical and neural baselines, validation error analysis, guarded model selection, extensive testing, and a frozen future evaluation protocol.

Future work should expand the number of independently photographed part groups, increase description diversity, evaluate pretrained visual encoders, and investigate better relation-aware fusion. Any final test evaluation must remain a separate, explicitly authorized, one-shot procedure after the notebook and submission package are fully reviewed.
"""))

    cells.append(markdown(r"""
## 15. References

1. Faghri, F., Fleet, D. J., Kiros, J. R., & Fidler, S. (2018). **VSE++: Improving Visual-Semantic Embeddings with Hard Negatives.** British Machine Vision Conference. https://arxiv.org/abs/1707.05612
2. Li, L. H., Yatskar, M., Yin, D., Hsieh, C.-J., & Chang, K.-W. (2019). **VisualBERT: A Simple and Performant Baseline for Vision and Language.** https://arxiv.org/abs/1908.03557
3. Radford, A., et al. (2021). **Learning Transferable Visual Models From Natural Language Supervision.** Proceedings of ICML. https://arxiv.org/abs/2103.00020
4. TensorFlow. **The Functional API.** https://www.tensorflow.org/guide/keras/functional_api
5. scikit-learn. **TfidfVectorizer documentation.** https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
6. scikit-learn. **LogisticRegression documentation.** https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

### Reproduction commands

```powershell
python -m src.project_cli build-final-exam-notebook
python -m src.project_cli verify-final-exam-notebook
python -m pytest -q
python -m src.project_cli verify-project
```
"""))

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "project": {
                "step": "010.6",
                "base_checkpoint": BASE_CHECKPOINT_COMMIT,
                "test_split_used": False,
                "final_test_evaluation_authorized": False,
            },
        },
    )
    return notebook


def execute_notebook(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    try:
        return client.execute()
    except CellExecutionError as error:
        raise FinalExamNotebookError(
            f"Final exam notebook execution failed: {error}."
        ) from error


def notebook_statistics(notebook: nbformat.NotebookNode) -> dict[str, int]:
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    output_count = sum(len(cell.get("outputs", [])) for cell in code_cells)
    display_output_count = sum(
        1
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
    )
    return {
        "cell_count": len(notebook.cells),
        "markdown_cell_count": sum(
            cell.cell_type == "markdown" for cell in notebook.cells
        ),
        "code_cell_count": len(code_cells),
        "executed_code_cell_count": sum(
            cell.get("execution_count") is not None for cell in code_cells
        ),
        "output_count": output_count,
        "display_output_count": display_output_count,
    }


def validate_executed_notebook(notebook: nbformat.NotebookNode) -> dict[str, int]:
    statistics = notebook_statistics(notebook)
    if statistics["code_cell_count"] == 0:
        raise FinalExamNotebookError("Final exam notebook has no code cells.")
    if statistics["executed_code_cell_count"] != statistics["code_cell_count"]:
        raise FinalExamNotebookError("Not all final exam code cells were executed.")
    if statistics["output_count"] < 10:
        raise FinalExamNotebookError("Final exam notebook has too few saved outputs.")

    markdown_text = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    for heading in REQUIRED_NOTEBOOK_HEADINGS:
        if heading not in markdown_text:
            raise FinalExamNotebookError(f"Missing notebook heading: {heading}.")

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        lowered = cell.source.lower()
        for token in FORBIDDEN_NOTEBOOK_CODE_TOKENS:
            if token.lower() in lowered:
                raise FinalExamNotebookError(
                    f"Forbidden code token in executed notebook: {token}."
                )
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                raise FinalExamNotebookError("Executed notebook contains an error output.")

    return statistics


def build_reports(
    prerequisites: dict[str, dict[str, Any]],
    statistics: dict[str, int],
) -> None:
    source_paths = (
        INTEGRATED_TRAIN_PATH,
        INTEGRATED_VALIDATION_PATH,
        INTEGRATED_COMPARISON_PATH,
        INTEGRATED_STATUS_PATH,
        MULTIMODAL_METRICS_PATH,
        MULTIMODAL_CONFUSION_MATRIX_PATH,
        MULTIMODAL_PREDICTIONS_PATH,
        CONTROLLED_EXPERIMENT_PATH,
        ERROR_ANALYSIS_PATH,
        MODEL_SELECTION_DECISION_PATH,
        FINAL_MODEL_SPECIFICATION_PATH,
        FINAL_EVALUATION_PROTOCOL_PATH,
        FINAL_MODEL_FREEZE_STATUS_PATH,
        FINAL_TEST_AUTHORIZATION_PATH,
        LOCKED_TEST_CONTRACT_PATH,
    )
    source_fingerprints = {
        project_relative_path(path): sha256_file(path) for path in source_paths
    }
    notebook_hash = sha256_file(FINAL_EXAM_NOTEBOOK_PATH)

    status = {
        "status": "PASS",
        "readiness": NOTEBOOK_READINESS,
        "base_checkpoint_commit": BASE_CHECKPOINT_COMMIT,
        "notebook": project_relative_path(FINAL_EXAM_NOTEBOOK_PATH),
        **statistics,
        "related_work_source_count": 6,
        "base_verified_test_count": BASE_VERIFIED_TEST_COUNT,
        "base_verified_warning_count": BASE_VERIFIED_WARNING_COUNT,
        "final_model_slug": prerequisites["model_specification"][
            "final_model_slug"
        ],
        "selection_decision": prerequisites["selection"]["decision"],
        "protocol_state": prerequisites["evaluation_protocol"][
            "protocol_state"
        ],
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "model_retraining_performed": False,
        "model_selection_changed": False,
        "final_test_evaluation_authorized": False,
    }
    manifest = {
        "status": "PASS",
        "base_checkpoint_commit": BASE_CHECKPOINT_COMMIT,
        "hash_normalization": "utf-8-lf",
        "notebook_sha256": notebook_hash,
        "source_artifact_sha256": source_fingerprints,
        "generated_artifact_sha256": {
            project_relative_path(FINAL_EXAM_NOTEBOOK_PATH): notebook_hash,
            project_relative_path(FINAL_EXAM_NOTEBOOK_STATUS_PATH): "GENERATED_AFTER_MANIFEST",
            project_relative_path(FINAL_EXAM_NOTEBOOK_SUMMARY_PATH): "GENERATED_AFTER_MANIFEST",
        },
        "locked_test_csv_files_opened": False,
        "test_split_used": False,
        "final_test_evaluation_authorized": False,
    }

    atomic_write_json(FINAL_EXAM_NOTEBOOK_STATUS_PATH, status)
    atomic_write_json(FINAL_EXAM_NOTEBOOK_MANIFEST_PATH, manifest)

    summary = f"""# Final Exam Notebook and Research Narrative Integration

Status: `PASS`

Readiness: `{NOTEBOOK_READINESS}`

The executed notebook `{project_relative_path(FINAL_EXAM_NOTEBOOK_PATH)}` integrates the complete validation-only research narrative through Step 010.5.

## Notebook evidence

- cells: {statistics['cell_count']};
- markdown cells: {statistics['markdown_cell_count']};
- code cells: {statistics['code_cell_count']};
- executed code cells: {statistics['executed_code_cell_count']};
- saved outputs: {statistics['output_count']};
- related-work sources: 6;
- base checkpoint tests: {BASE_VERIFIED_TEST_COUNT} passed with {BASE_VERIFIED_WARNING_COUNT} known warnings.

## Scientific scope

The notebook covers the problem statement, hypothesis, previous research, open-license data acquisition, grouped splitting, six model families, development versus integrated validation results, confusion matrix, per-class metrics, validation error analysis, controlled improvement, conservative model selection, final model freeze, reproducibility, limitations, conclusion, and references.

## Locked-test boundary

- locked test CSV files opened: `false`;
- test split used: `false`;
- model retraining performed: `false`;
- model selection changed: `false`;
- final test evaluation authorized: `false`.
"""
    atomic_write_text(FINAL_EXAM_NOTEBOOK_SUMMARY_PATH, summary)

    manifest["generated_artifact_sha256"] = {
        project_relative_path(FINAL_EXAM_NOTEBOOK_PATH): notebook_hash,
        project_relative_path(FINAL_EXAM_NOTEBOOK_STATUS_PATH): sha256_file(
            FINAL_EXAM_NOTEBOOK_STATUS_PATH
        ),
        project_relative_path(FINAL_EXAM_NOTEBOOK_SUMMARY_PATH): sha256_file(
            FINAL_EXAM_NOTEBOOK_SUMMARY_PATH
        ),
    }
    atomic_write_json(FINAL_EXAM_NOTEBOOK_MANIFEST_PATH, manifest)


def main() -> None:
    prerequisites = validate_prerequisites()
    notebook = build_notebook()
    executed = execute_notebook(notebook)
    statistics = validate_executed_notebook(executed)

    FINAL_EXAM_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = FINAL_EXAM_NOTEBOOK_PATH.with_name(
        f".{FINAL_EXAM_NOTEBOOK_PATH.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        nbformat.write(executed, temporary)
        temporary.replace(FINAL_EXAM_NOTEBOOK_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()

    build_reports(prerequisites, statistics)

    print("Final exam notebook and research narrative integration")
    print(f"- notebook: {project_relative_path(FINAL_EXAM_NOTEBOOK_PATH)}")
    print(f"- cells: {statistics['cell_count']}")
    print(f"- executed code cells: {statistics['executed_code_cell_count']}")
    print(f"- saved outputs: {statistics['output_count']}")
    print("- locked test CSV files opened: false")
    print("- test split used: false")
    print("- final test evaluation authorized: false")
    print(f"Readiness: {NOTEBOOK_READINESS}")
    print("Status: PASS")


if __name__ == "__main__":
    main()
