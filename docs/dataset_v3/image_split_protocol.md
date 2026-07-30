# Dataset V3 image split protocol

## Scope

This protocol applies to the curated Dataset V3 image pool:

- 8 automotive-part categories
- 80 independently curated images per category
- 640 images total

## Split

Each category is divided into:

- 60 train images
- 10 validation images
- 10 locked test images

Totals:

- train: 480
- validation: 80
- test: 80

## Grouping

Each final curated image is one independent `image_group_id`.
Before splitting, the pool passed:

- exact duplicate audit: 0 groups
- perceptual near-duplicate review: 0 pairs
- cross-category high-confidence duplicate audit: 0 pairs
- overlap with the current Dataset V2 project images: 0 pairs

No image group may appear in more than one split.

## Reproducibility

The split is deterministic.

Seed:

`dataset-v3-split-v1`

Ordering is based on SHA-256 of:

`seed | category | candidate_id`

The original Kaggle train/validation/test folders are ignored.

## Locked test policy

The 80 test images are locked.

They must not be used for:

- training
- early stopping
- model selection
- threshold tuning
- prompt or text-template selection
- error analysis
- data-cleaning decisions

Test lock fingerprint:

`162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5`

Only an explicitly authorized final evaluation may read the test split.
