# Automotive Part Image-Text Matching

This is my Deep Learning exam project.

## Question

Can a small neural network that uses both an automotive-part image and a short text description classify their relationship better than image-only and text-only models?

The three labels are:

- `MATCH` - the image and text describe the same part;
- `PARTIAL_MATCH` - they describe different parts from the same system;
- `MISMATCH` - they describe parts from different systems.

## Start here

Open the executed notebook:

**[project.ipynb](project.ipynb)**

It contains the problem, the data checks, the models, the results, examples of errors, limitations, and references.

## Main result

The multimodal model was the best model on the validation data.

| Validation data | Accuracy | Macro F1 |
|---|---:|---:|
| All validation images | 0.5333 | 0.5208 |
| Real Wikimedia images only | 0.4667 | 0.4626 |

I report the real-image subset separately because the generated drawings contain visually similar train and validation examples. This makes the real-image result more useful, although it is still based on only 30 image-text pairs.

The test split was not used.

## Project files

- `project.ipynb` - main exam notebook;
- `src/data.py` - data loading and split checks;
- `src/models.py` - the three Keras models;
- `src/train.py` - training and validation script;
- `src/audit.py` - leakage and shortcut checks;
- `data/` - CSV files, images, and image licenses;
- `results/` - saved validation results;
- `tests/` - small automated test suite.

## Run the project

Create and activate a virtual environment, then install the packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the data audit and the tests:

```powershell
python -m src.audit
python -m pytest -q
```

Train the models again:

```powershell
python -m src.train
```

Open the notebook:

```powershell
python -m jupyter notebook project.ipynb
```

Training uses only `data/train.csv`. Model comparison uses `data/validation.csv`. The normal data loader refuses to open `data/test.csv`.

## Limitations

The dataset is small. The generated drawings are simple and some are visually similar across the train and validation splits. The real-image validation subset has only one image per category. The result is useful as a course experiment, but it is not enough for a production system.

## Sources

The notebook cites VSE++, VisualBERT, CLIP, and ResNet. The open-license image authors and license links are in `data/licenses.csv`.
