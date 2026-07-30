# Dataset V3 notebook protocol

`project_v3.ipynb` is the executed development report for Dataset V3.

## Scope

The notebook reads only:

- Dataset V3 train and validation image manifests;
- Dataset V3 train and validation relation tables;
- the 19 saved development training artifacts under `results/dataset_v3/`.

It does not read the locked test manifest or locked test images.

## Statistical interpretation

Each image creates six relation rows, so the independent validation unit is the curated image group. The report distinguishes 480 relation rows from 80 independent validation images. Confidence intervals and paired comparisons use complete image groups.

Validation was used for early stopping and model selection. The reported `377/480` relation accuracy and `0.7872843` macro F1 are development-validation results, not final test estimates.

## Notebook content

The executed notebook includes:

- the research question and motivation;
- source, curation, and split description;
- relation-label construction;
- leakage and shortcut checks;
- seven-model comparison;
- grouped uncertainty intervals;
- confusion matrix and class report;
- per-category performance and error examples;
- neural training histories;
- grouped paired comparisons;
- previous research, limitations, and conclusion;
- final assertions confirming that the locked test was not read.

## Rebuild and verification

```bash
python -m src.build_notebook_v3 --execute
python -m src.verify_notebook_v3
python -m pytest -q tests/test_dataset_v3_notebook.py
```

The original `project.ipynb` remains unchanged until the Dataset V3 report is independently reviewed and approved for promotion.
