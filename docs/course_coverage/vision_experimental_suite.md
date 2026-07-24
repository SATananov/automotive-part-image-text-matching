# Step 011.3A — Vision Core Experimental Suite, Explainability & Controlled Gates

Step 011.3A executes the locally available core of the supplied **Vision Models Exercise** on the committed integrated automotive-part dataset. It completes VIS-001, VIS-003, VIS-004, VIS-005, VIS-008, and VIS-009 while keeping pretrained and human-annotation work behind explicit gates.

## Completed scope

- image dimensions, brightness, contrast, source/category balance, exact hashes, difficult-example flags, and representative image evidence;
- deterministic fixed-convolutional representations at 32, 48, and 64 pixels across seeds 42, 43, and 44;
- learned scalar image-text compatibility scores using ordinal regression and class-probability expected scores;
- model-independent 3×3 occlusion, center masking, border masking, and crop-and-retest diagnostics;
- failure-driven augmentation ablations for exposure, center crop, JPEG compression, and a combined policy;
- pairwise ranking accuracy, three-way ordering, equal-pair accuracy, flip consistency, scalar-score transitivity, parameter count, and inference timing.

The suite records 48 controlled train/validation-only training runs:

- 27 representation/resolution runs;
- 15 augmentation runs;
- 6 compatibility-scoring runs.

## Controlled gates

- **VIS-002** remains deferred until pretrained convolutional and vision-transformer downloads, exact revisions, and licenses receive explicit approval.
- **VIS-006** remains deferred until VIS-002 selects a frozen champion and Tier 4 fine-tuning receives explicit approval.
- **VIS-007** remains deferred until at least two genuine independent annotators provide labels and confidence before adjudication. Synthetic annotators are forbidden.

## Commands

```powershell
python -m src.run_vision_experimental_suite
python -m src.build_vision_experiment_notebooks
python -m src.verification.vision_experimental_suite
```

## Evaluation boundary

Only `data/processed/integrated_train.csv` and `data/processed/integrated_validation.csv` may be loaded. The test split remains unopened and unauthorized. Step 011.3A does not replace, retrain, or modify the frozen production final model.
