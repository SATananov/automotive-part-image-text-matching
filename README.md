# Automotive Part Image-Text Matching

Final Deep Learning exam project: a leakage-aware multimodal classifier that determines whether an automotive-part photograph and a short description are a `MATCH`, `PARTIAL_MATCH`, or `MISMATCH`.

The executed research report is **[project.ipynb](project.ipynb)**.

## Research question

Can a compact neural model that combines image and text evidence classify their relationship better than image-only, text-only, and non-neural baselines?

The task is genuinely multimodal: every image is paired equally often with all three labels, so the relation cannot be solved from the image alone. Text category is also exactly balanced across labels, preventing a text-category shortcut.

## Dataset V2

Dataset V2 strengthens the original proof of concept without changing the locked final-test identities. Public-source availability is treated as an explicit constraint rather than hidden by repeated rows.

The importer selects the **largest balanced non-duplicate quota supported by every category** after exact-hash and perceptual-similarity filtering:

- maximum imported quota: 20 train + 5 validation images per category;
- minimum accepted quota: 3 train + 1 validation image per category;
- the actual quota is written to `data/dataset_v2_import_summary.json` and verified against `data/dataset_v2_manifest.csv`;
- all ten categories receive the same imported train and validation count, preventing category-frequency shortcuts.

The final split sizes are data-derived:

| Split | Original Wikimedia | Imported Dataset V2 | Synthetic | Independent images | Paired rows |
|---|---:|---:|---:|---:|---:|
| Train | 50 | `10 × imported_train_quota` | 50 | `100 + imported_train_images` | `6 × train_images` |
| Validation | 10 | `10 × imported_validation_quota` | 0 | `10 + imported_validation_images` | `6 × validation_images` |
| Test | 10 | 0 | 0 | 10 | 60 |

Each image contributes six pairs: two `MATCH`, two `PARTIAL_MATCH`, and two `MISMATCH`. The paired rows are dependent; the **image is the independent evaluation unit**.

### Kaggle-first real-image acquisition

The importer first discovers all ten project categories in the Apache-2.0 **50 Types of Car Parts** Kaggle dataset by normalized directory aliases. For `air_filter` and `shock_absorber`, Wikimedia Commons remains an open-license fallback when the downloaded Kaggle snapshot does not contain enough matching images.

The importer:

- uses the largest balanced category quota that survives duplicate screening instead of failing on an arbitrary fixed count;
- stores per-image provider, source page, author/credit, license, original hash, standardized hash, and transformation record in `data/dataset_v2_manifest.csv`;
- applies EXIF orientation, white-background alpha compositing, RGB conversion, centre crop, and 224×224 standardisation;
- rejects blank or near-constant standardized images with variance, luminance-entropy, and dominant-pixel checks;
- rejects exact duplicates and close same-category candidates using dHash and normalized grayscale similarity;
- compares candidates with the original train, validation, and sealed-test images;
- stores raw downloads only under ignored `.cache/` storage.

The original ten-image Wikimedia test holdout remains sealed and is not parsed by training, audit, notebook, or normal verification code.

## Text and relation design

Dataset V2 expands the captions from one repeated sentence per category to split-specific template banks:

- 40 unique training descriptions;
- 20 unique validation descriptions;
- 20 unique test descriptions;
- zero exact caption overlap across splits.

Mismatch targets rotate through unrelated automotive systems while remaining exactly balanced by text category and label. This broadens relation coverage without inflating the number of independent images.

## Models

The comparison contains four transparent non-neural baselines and four PyTorch models:

- majority baseline;
- TF-IDF + Logistic Regression;
- image-pixel Logistic Regression;
- image + text Logistic Regression;
- neural text MLP;
- image-only CNN;
- real-only multimodal CNN + text MLP;
- the same multimodal architecture trained with real + synthetic images.

The selected multimodal model uses auxiliary image-category and text-category supervision during training. Only the relation head is scored.

## Strict evaluation

- grouped bootstrap confidence intervals resample complete images;
- paired comparisons use complete image groups;
- exact sign-flip enumeration is used for at most 20 images;
- paired randomization uses exact sign-flip enumeration for at most 20 validation images and a deterministic 100,000-draw Monte Carlo procedure above that threshold;
- validation is labelled development evidence because it is used for early stopping and model comparison;
- no general-superiority claim is made from a single development split;
- test evaluation remains forbidden until the whole protocol is frozen.

See [docs/strict_evaluation_protocol.md](docs/strict_evaluation_protocol.md).

## Reproducible Dataset V2 pipeline

The complete refresh sequence:

1. acquires or accepts the documented hybrid public image sources;
2. applies automotive-domain, license, integrity, duplicate, and low-information image checks;
3. rebuilds all CSVs and manifests deterministically;
4. audits leakage, similarity, shortcuts, provenance, licensing, and the test lock;
5. records the exact environment;
6. retrains all eight models under canonical PyTorch `2.10.0`;
7. rebuilds and executes the notebook;
8. runs the complete tests and cross-artifact verifier.

Manual commands are:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-acquisition.txt
python -m src.import_dataset_v2 --force
python -m src.build_dataset
python -m src.audit
python -m src.record_environment
python -m src.train
python -m src.build_notebook_v2
python -m jupyter nbconvert --to notebook --execute project.ipynb --inplace --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3
python -m pytest -q
python -m src.verify
```

A previously downloaded Kaggle directory can be supplied with `--source-root`. An offline, pre-reviewed fallback containing `air_filter/` and `shock_absorber/` directories can be supplied with `--commons-source-root` when those categories are unavailable or insufficient in the local Kaggle snapshot:

```powershell
python -m src.import_dataset_v2 `
  --source-root "D:\Datasets\50-Types-of-Car-Parts" `
  --commons-source-root "D:\Datasets\Commons-V2" `
  --force
```

Without `--commons-source-root`, the importer uses the Wikimedia Commons API only for a fallback category that is unavailable or insufficient in Kaggle. Thumbnail and original-file URLs are both attempted, and selected files retain per-file attribution and license metadata. Commons candidates must also carry automotive context, transparent sources are composited on white before RGB conversion, and every standardized image must pass variance, luminance-entropy, and dominant-pixel checks.

## Reproducibility and artifact consistency

- `requirements.txt` pins canonical PyTorch `2.10.0`;
- `results/environment_lock.txt` and `.json` record the exact final environment;
- `src.train` refuses to overwrite canonical artifacts with another PyTorch version;
- `results/result_summary.md` is generated from predictions, not typed manually;
- `src.verify` recomputes metrics and paired comparisons and verifies the executed notebook;
- `data/test_lock.json` stores the sealed test SHA-256;
- `src/create_clean_checkpoint.py` creates a Git-derived ZIP with the exact commit in its ZIP comment and verifies archive hygiene.

## Repository map

- `src/import_dataset_v2.py` — hybrid acquisition, automotive-domain screening, transparency-safe standardisation, content checks, deterministic selection, deduplication, and attribution;
- `src/captions_v2.py` — split-specific caption banks;
- `src/build_dataset.py` — balanced six-pair relation construction and manifests;
- `src/audit.py` — identity, hash, perceptual similarity, shortcut, provenance, licensing, and lock audit;
- `src/evaluation.py` — exact and Monte Carlo image-group paired randomization;
- `src/train.py` — eight-model comparison and grouped uncertainty;
- `src/build_notebook_v2.py` — reproducible English exam report;
- `src/record_environment.py` — exact final environment lock;
- `src/verify.py` — cross-artifact verification;
- `src/create_clean_checkpoint.py` — clean Git checkpoint creation and validation;
- `docs/dataset_v2_protocol.md` — acquisition and sampling protocol;
- `docs/final_exam_rubric_mapping.md` — evidence for every grading criterion;
- `tests/` — data, importer, audit, statistics, training utility, and artifact tests.

## Limitations

Dataset V2 is much stronger than the original ten-image validation design, but it remains a controlled course experiment. Validation is reused for early stopping and model ranking. Captions still name the part category, so this is relation classification rather than open-vocabulary language understanding. Public datasets may contain source-specific visual conventions or imperfect labels; automated checks reduce but do not eliminate that risk. Synthetic drawings remain simple training-only templates. The final test contains only ten independent images and is intentionally unopened while development continues.

## Research references and data attribution

The notebook discusses VSE++, VisualBERT, CLIP, ResNet, the external automotive-parts dataset, and Wikimedia Commons as an open-media source. Original Wikimedia attribution is stored in `data/licenses.csv`; all imported Dataset V2 provenance and license records are stored in `data/dataset_v2_manifest.csv`.
