# Automotive Part Image-Text Matching

This repository is my final Deep Learning exam project.

## Research question

Can a neural model that combines an automotive-part image and a short text description classify their relationship better than image-only, text-only, and non-neural baselines?

The relation labels are:

- `MATCH` — image and text refer to the same part category;
- `PARTIAL_MATCH` — the categories differ but belong to the same automotive system;
- `MISMATCH` — the categories belong to different systems.

The executed report is **[project.ipynb](project.ipynb)**.

## Why the experimental design changed

An earlier version placed very similar synthetic drawings in both training and validation. That could reward template recognition instead of generalization. The current design removes that risk:

- all 50 synthetic drawings are training-only;
- validation contains 10 newly sourced real Wikimedia photographs, one per category;
- test contains 10 different newly sourced real photographs and remains locked;
- 20 additional open-license real photographs were added, bringing the real-image inventory to 70;
- alternate views from known photographic series are grouped by `object_group_id`;
- train and validation use different sentences, with zero exact caption overlap;
- source, image category, and text category shortcut diagnostics remain at chance accuracy (`1/3`).

## Data summary

Each image is paired with three descriptions, one for each relation label.

| Split | Real images | Synthetic images | Paired rows | Purpose |
|---|---:|---:|---:|---|
| Train | 50 | 50 | 300 | model fitting and synthetic-data ablation |
| Validation | 10 | 0 | 30 | real-image model comparison |
| Test | 10 | 0 | 30 | locked; not evaluated |

Because the three rows from one image are dependent, the project reports bootstrap confidence intervals by resampling complete image groups rather than individual rows.

## Deep learning approach

The project uses PyTorch and contains:

- a neural text-only MLP;
- an image-only CNN;
- a multimodal CNN + text MLP trained on real images only;
- the same multimodal architecture trained on real + synthetic images;
- four non-neural baselines.

The multimodal model also receives auxiliary supervision for the image category and text category during training. These auxiliary heads help the encoders learn the two inputs; only the relation prediction is used for the final score.

## Saved validation result

The exact current scores, confidence interval, paired comparison, and environment are generated directly from the saved prediction artifacts:

**[Generated validation result summary](results/result_summary.md)**

The selected development result is the real-only multimodal CNN. The conclusion remains deliberately cautious: the validation set contains only ten independent images, so an observed improvement cannot establish general superiority. The synthetic-data ablation is retained as a transparent negative experiment because the current template-like drawings did not improve transfer to real photographs.

The locked test split was not used.

## Canonical environment and reproducibility lock

The committed result artifacts were generated with **PyTorch 2.13.0**. `requirements.txt` pins that exact version, and `python -m src.train` refuses to overwrite the results under a different PyTorch version. This prevents a broad dependency range from silently producing a different set of predictions while leaving stale prose behind.

Create an environment and install the packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Rebuild the CSV splits and image manifest:

```powershell
python -m src.build_dataset
```

Run the leakage and license audit:

```powershell
python -m src.audit
```

Train all baselines and neural models. This regenerates predictions, metrics, histories, architectures, `run_info.json`, and the generated Markdown result summary. It intentionally marks the final verification as pending until the notebook is executed again:

```powershell
python -m src.train
```

Execute the notebook from top to bottom:

```powershell
python -m jupyter nbconvert --to notebook --execute project.ipynb --inplace --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3
```

Run the tests and create the final artifact-consistency verification:

```powershell
python -m pytest -q
python -m src.verify
```

The final `results/verification_summary.json` receives `status: PASS` only when saved predictions, metric tables, paired comparisons, generated summary, canonical environment, data audit, and executed notebook agree.

Normal training and notebook execution can load only `train` and `validation`. The test CSV is protected by a stored SHA-256 lock and is never parsed by `src.train` or `src.audit`.

## Repository map

- `project.ipynb` — executed exam report;
- `src/build_dataset.py` — deterministic split and manifest construction;
- `src/data.py` — locked data loading and image preparation;
- `src/models.py` — PyTorch neural architectures;
- `src/train.py` — baselines, neural training, grouped bootstrap, and saved results;
- `src/audit.py` — identity, hash, similarity, shortcut, license, and test-lock checks;
- `src/verify.py` — environment and cross-artifact consistency verification;
- `data/image_manifest.csv` — one row per image with split and SHA-256;
- `data/licenses.csv` — Wikimedia authorship, source page, license, and hash records;
- `results/result_summary.md` — generated human-readable scores from saved artifacts;
- `results/verification_summary.json` — final machine-readable consistency status;
- `results/` — predictions, metrics, histories, architectures, and audit reports;
- `tests/` — automated integrity and consistency tests.

## Limitations

The validation set contains only 10 independent images. The three paired rows per image are not independent. The vocabulary is deliberately small, and the part name appears in the description. Validation is used for model selection, while the locked test set is intentionally unevaluated. The project is a controlled course experiment, not a production automotive-search system.

## Research references

The notebook discusses and cites VSE++, VisualBERT, CLIP, and ResNet. Full Wikimedia attribution is stored in `data/licenses.csv`.
