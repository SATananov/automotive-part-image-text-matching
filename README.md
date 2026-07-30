# Automotive Part Image-Text Matching

This is my final Deep Learning exam project.

The project compares a photograph of an automotive part with a short text description and predicts one of three labels:

- `MATCH` — the image and text describe the same part;
- `PARTIAL_MATCH` — the parts are different but belong to the same automotive system;
- `MISMATCH` — the parts belong to different systems.

The full executed report is in [project.ipynb](project.ipynb).

## Research question

Can a small model that uses both image and text perform better than image-only, text-only, and non-neural baselines?

This is a real problem for online catalogues and automotive warehouses. A wrong image-text pair can lead to a wrong product listing or a picking mistake.

## Dataset V2

The project uses ten automotive-part categories:

- air filter;
- alternator;
- brake disc;
- brake pad;
- coil spring;
- headlight;
- oil filter;
- shock absorber;
- starter;
- taillight.

The data comes from three sources:

1. the original Wikimedia Commons photographs;
2. additional real images from the Apache-2.0 **50 Types of Car Parts** Kaggle dataset and Wikimedia Commons;
3. 50 simple synthetic drawings used only for training.

The importer checks that the files are readable, converts selected images to RGB, resizes them to 224×224 pixels, calculates hashes, rejects duplicates, and records source and license information.

It uses the **largest balanced non-duplicate quota** that all ten categories can support. In this saved version, five imported training images and one imported validation image are used for every category.

| Split | Real images | Synthetic images | Independent images | Paired rows | Purpose |
|---|---:|---:|---:|---:|---|
| Train | 100 | 50 | 150 | 900 | training |
| Validation | 20 | 0 | 20 | 120 | model comparison |
| Test | 10 | 0 | 10 | 60 | locked and not evaluated |

Every image creates six image-text rows: two `MATCH`, two `PARTIAL_MATCH`, and two `MISMATCH`. These rows are related, so the **image is the independent evaluation unit**.

## Leakage checks

All rows from one image stay in the same split. The project checks for overlap in:

- image and sample IDs;
- part and object groups;
- file paths;
- exact SHA-256 hashes;
- exact captions;
- visually similar real images.

Train and validation have separate caption templates. The original test images remain locked and are not loaded by the training, audit, notebook, or normal verification code.

## Models

The comparison contains eight models:

- majority baseline;
- text Logistic Regression;
- image Logistic Regression;
- image + text Logistic Regression;
- neural text MLP;
- image-only CNN;
- multimodal CNN + text MLP trained on real images;
- the same multimodal model trained on real and synthetic images.

The main multimodal model combines image and text features before predicting the relation label.

## Evaluation

Because several rows share one image, the statistical analysis works with complete image groups:

- grouped bootstrap confidence intervals;
- paired image-group randomization tests;
- accuracy and macro F1;
- confusion matrix and per-category results.

Validation is used for early stopping and model comparison. It is development evidence, not a final unbiased test result.

## Saved validation result

The selected real-only multimodal model achieved:

- accuracy: `0.5083`;
- macro F1: `0.5034`;
- correct predictions: `61/120`;
- grouped 95% accuracy interval: `[0.3417, 0.6667]`.

The image + text Logistic Regression baseline achieved `0.3333` accuracy and `0.1667` macro F1. The grouped paired comparison gave `p = 0.072876` over 20 independent images.

The real + synthetic multimodal model achieved `0.4583` accuracy and `0.4251` macro F1, so the synthetic drawings did not improve the result in this experiment.

The exact generated summary is in [results/result_summary.md](results/result_summary.md).

## How to run

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The saved results use Python `3.13.5` and PyTorch `2.10.0+cpu`. Training and verification check the PyTorch version before replacing saved artifacts.

Run the main checks:

```powershell
python -m src.build_dataset
python -m src.audit
python -m pytest -q
python -m src.verify
```

Both test commands are supported:

```powershell
pytest -q
python -m pytest -q
```

To rebuild the complete Dataset V2 from the public sources, also install the acquisition dependency and run the importer:

```powershell
python -m pip install -r requirements-acquisition.txt
python -m src.import_dataset_v2 --force
```

A local Kaggle directory can be supplied with `--source-root`. An offline reviewed Wikimedia fallback can be supplied with `--commons-source-root`.

## Project structure

- `project.ipynb` — executed report;
- `src/import_dataset_v2.py` — image acquisition, cleaning, duplicate checks, and provenance;
- `src/build_dataset.py` — split and image-text pair generation;
- `src/audit.py` — leakage, license, image, and test-lock checks;
- `src/evaluation.py` — grouped statistical comparison;
- `src/models.py` — neural network definitions;
- `src/train.py` — baselines, neural training, and saved results;
- `src/build_notebook_v2.py` — rebuilds the notebook structure;
- `src/record_environment.py` — records the package versions;
- `src/verify.py` — checks consistency between saved artifacts;
- `tests/` — automated tests;
- `docs/` — dataset and evaluation notes.

## Limitations

- The validation set has only 20 independent images.
- Validation is reused for model selection.
- Part names are visible in the descriptions, so this is relation classification rather than open-vocabulary language understanding.
- Public datasets can contain imperfect labels or source-specific visual patterns.
- Synthetic drawings are simple and are used only as a training ablation.
- The final test has only ten images and remains unopened during development.

## Previous research and sources

The notebook discusses VSE++, VisualBERT, CLIP, and ResNet. The external real-image dataset and Wikimedia Commons are also cited. Original Wikimedia attribution is stored in `data/licenses.csv`, and Dataset V2 provenance is stored in `data/dataset_v2_manifest.csv`.

## Dataset V3 development branch

The `dataset-v3` branch contains a separately curated eight-category image pool with 480 train images, 80 validation images, and 80 physically separated locked-test images. The original Kaggle split labels are retained only as provenance; the project uses its own deterministic `60/10/10` split per category.

Dataset V3 development relation tables are stored under `data/manifests/dataset_v3/`. Seven development models have now been trained and independently verified on the 80-image validation split. The saved Dataset V3 artifacts are isolated under `results/dataset_v3/`. The locked 80-image test split remains unopened and unevaluated.

Build and audit the Dataset V3 development relations with:

```bash
python -m src.build_dataset_v3
python -m src.audit_dataset_v3
python -m pytest -q tests/test_dataset_v3_integrity.py tests/test_dataset_v3_relations.py
```

The Dataset V3 relation protocol is documented in `docs/dataset_v3/relation_protocol.md`.

### Dataset V3 training workflow

The Dataset V3 development training implementation is isolated from the saved Dataset V2 report. It writes only to `results/dataset_v3/` and compares seven baselines and neural models on 480 train images and 80 independent validation images.

The independent evaluation unit is the curated image group. Confidence intervals and paired comparisons operate on complete image groups, not on the six relation rows as if they were independent.

The training implementation can be checked without running a long experiment:

```bash
python -m pytest -q tests/test_dataset_v3_integrity.py tests/test_dataset_v3_relations.py tests/test_dataset_v3_training.py
```

After the implementation checkpoint is reviewed, the controlled training command is:

```bash
python -m src.train_dataset_v3
```

The command does not expose or evaluate the locked test split. The exact protocol is documented in `docs/dataset_v3/training_protocol.md`.

### Dataset V3 executed notebook

`project_v3.ipynb` is the executed Dataset V3 development report. It reads only train, validation, and saved development artifacts. It presents the data protocol, leakage checks, model comparison, grouped uncertainty, error analysis, training histories, and limitations without opening the locked test split.

Rebuild and execute it with:

```bash
python -m src.build_notebook_v3 --execute
python -m src.verify_notebook_v3
```

The existing `project.ipynb` remains unchanged while the Dataset V3 notebook is reviewed independently.
