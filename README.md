# Automotive Part Image-Text Matching

This is my final Deep Learning exam project.

The project compares an automotive-part photograph with a short text description and predicts one of three relation labels:

- `MATCH` — the image and text describe the same part;
- `PARTIAL_MATCH` — they describe different parts from the same broad vehicle subsystem;
- `MISMATCH` — they describe parts from different subsystems.

The main executed submission report is [project_final.ipynb](project_final.ipynb).

The frozen development notebook used for model selection is preserved byte for byte in [project_v3.ipynb](project_v3.ipynb).

## Research question

Can a compact neural model that combines image and text information outperform image-only, text-only, and simple multimodal baselines on independently grouped automotive-part images?

A reliable image-text relation check can help online catalogues and automotive warehouses reduce incorrect listings and picking mistakes.

## Dataset V3

Dataset V3 uses a manually reviewed subset of the public **Car Parts 40 Classes** Kaggle dataset (`gpiosenka/car-parts-40-classes`). The source metadata records the dataset as Apache-2.0.

The eight categories are:

- alternator;
- brake disc;
- brake pad;
- coil spring;
- headlight;
- oil filter;
- starter;
- taillight.

The original source split labels were ignored. The project uses its own deterministic split within every category.

| Split | Independent images | Images per category | Relation rows | Purpose |
|---|---:|---:|---:|---|
| Train | 480 | 60 | 2,880 | model training |
| Validation | 80 | 10 | 480 | model comparison and selection |
| Locked test | 80 | 10 | 480 | one final evaluation |
| **Total** | **640** | **80** | **3,840** | |

Every image creates six balanced image-text rows: two `MATCH`, two `PARTIAL_MATCH`, and two `MISMATCH`. Rows from the same image are related. The image is the independent evaluation unit, represented by the complete six-row image group. Dataset V3 uses the largest balanced non-duplicate quota that can be maintained equally across all eight categories.

## Data quality and leakage control

Before integration, the curated image pool passed checks for:

- readable and valid image files;
- exact SHA-256 duplicates;
- perceptual near-duplicate review pairs;
- high-confidence cross-category conflicts;
- overlap with the earlier project image pool;
- train, validation, and test group separation;
- relation-label balance;
- description overlap across splits;
- source and license provenance.

The locked test images are physically separated under `data/locked_test/dataset_v3/`. They were not used for training, early stopping, model comparison, checkpoint selection, or notebook development.

## Models

Seven development models were compared:

- majority baseline;
- text Logistic Regression;
- image Logistic Regression;
- image + text Logistic Regression;
- neural text MLP;
- image-only CNN;
- multimodal CNN + text MLP.

The selected model is `torch_multimodal_dataset_v3`. It combines image features and TF-IDF text features before relation classification. Model selection was based only on the development validation split.

## Validation evidence

The selected multimodal model achieved:

- validation accuracy: `0.7854`;
- validation macro F1: `0.7873`;
- correct validation predictions: `377/480`;
- independent validation images: `80`.

The validation result is model-selection evidence and is not presented as an unbiased final estimate.

## Frozen selection and one-time final test

The selected checkpoint and development notebook were frozen before test access. A separate authorization permitted exactly one evaluation of the selected checkpoint.

The final evaluator was then executed once on the 80 locked-test images and 480 relation rows. The authorization is consumed. Further tuning, checkpoint replacement, and post-test model selection are prohibited.

The original final-test artifacts are preserved byte for byte. The final reporting notebook reads only those saved artifacts; it does not run training or inference.

## Final test result

The frozen model achieved:

- **354/480 correct predictions**;
- **accuracy: `0.7375`**;
- **macro F1: `0.7382299830`**;
- grouped 95% accuracy interval: **`[0.6604, 0.8146]`**;
- grouped 95% macro-F1 interval: **`[0.6536, 0.8102]`**;
- independent test images: **80**.

Detailed final results are available in:

- [project_final.ipynb](project_final.ipynb);
- [docs/dataset_v3/final_test_results.md](docs/dataset_v3/final_test_results.md);
- `results/dataset_v3_final_test/`.

## Statistical evaluation

Because six relation rows share each image, uncertainty and comparisons operate on complete image groups rather than treating all 480 rows as independent.

The evaluation includes:

- accuracy and macro F1;
- grouped bootstrap confidence intervals;
- grouped paired randomization comparisons during development;
- confusion matrix;
- classification report;
- per-category metrics;
- final error analysis from saved predictions.

## How to run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the Dataset V3 development checks:

```powershell
python -m src.build_dataset_v3
python -m src.audit_dataset_v3
python -m src.verify_dataset_v3_training
python -m src.verify_notebook_v3
```

Build and verify the final reporting notebook:

```powershell
python -m src.build_final_report
python -m src.verify_final_report
```

Verify the preserved one-time final-test result:

```powershell
python -m src.verify_dataset_v3_final_test_results
python -m pytest -q
```

The final-test evaluator must not be rerun. The existing result directory and one-time execution guard intentionally prevent a second evaluation.

## Project structure

- `project_final.ipynb` — main executed final exam report;
- `project_v3.ipynb` — frozen pre-test development and model-selection notebook;
- `src/build_final_report.py` — builds and executes the final reporting notebook without inference;
- `src/verify_final_report.py` — verifies the final report and frozen evidence;
- `src/build_dataset_v3.py` — creates the development relation tables;
- `src/audit_dataset_v3.py` — checks data quality, balance, leakage, and test isolation;
- `src/train_dataset_v3.py` — trains and compares the development models;
- `src/evaluate_dataset_v3_final_test.py` — protected one-time final evaluator;
- `src/verify_dataset_v3_final_test_results.py` — recomputes and verifies the saved final evidence;
- `results/dataset_v3/` — development training and validation artifacts;
- `results/dataset_v3_final_test/` — preserved final-test artifacts;
- `docs/dataset_v3/` — source, split, relation, training, lock, authorization, and result protocols;
- `tests/` — automated consistency and regression tests.

## Limitations

- The dataset contains 640 curated images from one public source collection.
- The test set contains 80 independent images, so the confidence intervals remain meaningful but not narrow.
- Text descriptions explicitly contain part names; this is relation classification, not open-vocabulary visual-language understanding.
- Public source data may retain label noise or source-specific visual patterns despite curation.
- The selected model is compact and trained from scratch rather than based on a large pretrained vision-language model.
- Validation was used for model selection, while the locked test was used once for final reporting.

## Previous Dataset V2 experiment

The repository also preserves the earlier Dataset V2 experiment in `project.ipynb` and the original `results/` artifacts. It is retained for development history and comparison, but Dataset V3 and `project_final.ipynb` are the primary submission evidence.

## References and provenance

The notebooks discuss ResNet, VSE++, VisualBERT, and CLIP as related work. Dataset source and license records are documented under `docs/dataset_v3/` and the Dataset V3 manifests. The project reports its own compact supervised experiment and does not claim to reproduce those larger architectures.
