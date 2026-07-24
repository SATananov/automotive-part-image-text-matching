# Deep Learning Fundamentals Experimental Suite

Step 011.1 executes Problems 1–10 from the supplied **Deep Learning Fundamentals Exercise** on the committed integrated automotive-part image–text dataset.

## Scope

The suite includes:

1. dataset dimensions, class balance, image values, text lengths, and representative examples;
2. multimodal batch shapes, dtypes, alignment, and train/validation shuffle policy;
3. a one-hidden-layer neural baseline with finite-gradient and weight-update diagnostics;
4. a deliberate one-batch overfit test with learning curves, using two complete part groups (six balanced relation samples), direct `train_on_batch` updates, and exact post-update metrics for portable CPU execution;
5. a correct Keras train/validation loop and a no-weight-update validation audit;
6. SGD, RMSprop, Adam, and AdamW comparisons across predefined learning rates, with early stopping, an exponential schedule, and three-seed champion stability;
7. small, medium, and large capacity comparisons with parameter counts and per-epoch probability tracking;
8. L2, dropout, batch normalization, learning-rate scheduling, residual fusion, a CNN image branch, and a frozen MobileNetV2 feature probe;
9. image resolution, text sequence length, and grayscale preprocessing comparisons;
10. safe controlled failures for unscaled images, extreme learning rates, excessive dropout, misaligned train-only labels, sigmoid timing/learning behavior, a deep-sigmoid gradient probe, a missing-optimizer-step probe, and a validation-training block.

## Evaluation boundary

Only these inputs are authorized:

- `data/processed/integrated_train.csv`
- `data/processed/integrated_validation.csv`

The experiment code contains no authorized test path. It does not load, inspect, score, tune against, or authorize the locked test split.

The validation champion recorded inside this suite is educational evidence. It does not replace the frozen Step 010.8 exam model and does not authorize a final held-out evaluation.

## Reproducibility

Run the complete suite from the project virtual environment:

```powershell
python -m src.project_cli run-fundamentals-suite
python -m src.project_cli verify-fundamentals-suite
```

The runner writes a machine-readable execution registry, full comparison tables, histories, validation predictions, confusion matrices, eight figures, a fully executed evidence notebook, a status document, and a normalized SHA-256 manifest.

The pretrained MobileNetV2 probe uses the official Keras ImageNet weights when they are already cached or can be resolved by Keras. If the external resource is unavailable, the run is recorded transparently as `SKIPPED_RESOURCE_UNAVAILABLE`; the remaining architecture ablations continue and no result is invented.

## Interpretation

Macro F1 is the primary validation metric because the three relation classes should contribute equally. Accuracy, per-class metrics, train/validation loss, parameter count, run time, generalization gap, and seed stability are secondary evidence.

Controlled-failure runs are not model candidates. Their purpose is to reproduce recognizable failure signatures and document prevention checks without modifying committed data.
