
# Step 011.3A — Vision Core Experimental Suite

Status: **PASS**

Readiness: `VISION_EXPERIMENTAL_SUITE_CORE_COMPLETE_PRETRAINED_AND_HUMAN_GATES_TEST_LOCKED`

## Completed course problems

- VIS-001 — image inventory, dimensions, brightness, contrast, source/category balance, exact hashes, and review flags.
- VIS-003 — two learned compatibility-score strategies with train-fitted equal-pair thresholds.
- VIS-004 — 27 fixed-convolutional representation/resolution training runs across seeds 42, 43, and 44.
- VIS-005 — model-independent 3×3 occlusion, center masking, border masking, and crop-and-retest evidence.
- VIS-008 — 15 failure-driven augmentation runs covering no augmentation, exposure, crop, compression, and a combined policy.
- VIS-009 — pairwise ranking, three-way ordering, flip consistency, scalar-score transitivity, equal-pair accuracy, parameters, and inference timing.

## Controlled gates

- VIS-002 is deferred until pretrained downloads and license/revision recording receive explicit approval.
- VIS-006 is deferred until VIS-002 selects a frozen champion and Tier 4 fine-tuning receives explicit approval.
- VIS-007 is deferred until at least two genuine independent human annotators provide pre-adjudication confidence labels. No synthetic human agreement is reported.

## Experimental totals

- Unique train/validation images profiled: **80**
- Controlled training runs: **48**
- Representation champion: `VIS004_global_pool_64_seed42`
- Augmentation champion: `VIS008_combined_seed44`
- Compatibility champion: `VIS003_class_probability_expected_score_seed42`
- Pairwise ranking accuracy: **0.8833**
- Occlusion examples: **8**

## Locked evaluation boundary

- Model training performed: `true` — experimental train/validation-only runs.
- Locked test CSV files opened: `false`
- Test split used: `false`
- Final test evaluation authorized: `false`
- Production final model changed: `false`
- Pretrained weights downloaded: `false`
- Synthetic human agreement reported: `false`
