# Dataset V3 development training protocol

## Scope

Dataset V3 training uses only:

- `data/manifests/dataset_v3/dataset_v3_train_relations.csv`;
- `data/manifests/dataset_v3/dataset_v3_validation_relations.csv`.

The 80-image locked test split is not exposed by the training loader. The training workflow does not read the test manifest, test images, test labels, or test predictions.

## Models

The development comparison contains seven models:

1. majority baseline;
2. TF-IDF with Logistic Regression;
3. low-resolution image pixels with Logistic Regression;
4. concatenated image and text features with Logistic Regression;
5. PyTorch neural text model;
6. PyTorch CNN image model;
7. PyTorch multimodal CNN with auxiliary image-category and text-category supervision.

The Dataset V3 multimodal network uses eight auxiliary categories. The relation output remains the three-class task: `MATCH`, `PARTIAL_MATCH`, and `MISMATCH`.

## Evaluation unit

Each validation image contributes six dependent relation rows. The independent evaluation unit is the curated image group. Therefore:

- confidence intervals bootstrap complete image groups;
- paired comparisons flip complete image-group score differences;
- the 80-image validation set uses deterministic Monte Carlo grouped randomization rather than row-level tests.

Validation is development evidence used for early stopping and model comparison. It is not a final unbiased test result.

## Artifact isolation

Dataset V2 artifacts remain under `results/`.

Dataset V3 artifacts are written transactionally to `results/dataset_v3/`. A temporary directory is removed after a failed run, and an existing completed Dataset V3 result directory is never silently overwritten.

The training run stores:

- model comparison and validation predictions;
- grouped paired comparisons;
- per-category metrics;
- neural histories and architectures;
- three neural state dictionaries;
- run and environment metadata;
- an independently recomputed verification summary.

## Reproduction

After the training-workflow checkpoint has been verified and pushed, execute:

```bash
python -m src.train_dataset_v3
python -m src.verify_dataset_v3_training
```

The project pins PyTorch `2.10.0`. No locked-test evaluation is authorized by these commands.
