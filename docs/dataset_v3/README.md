# Dataset V3 curated image pool

Dataset V3 is an eight-class automotive-part image pool.

## Categories

- alternator
- brake_disc
- brake_pad
- coil_spring
- headlight
- oil_filter
- starter
- taillight

## Size and split

Each category contains 80 curated images:

- 60 train
- 10 validation
- 10 locked test

Totals:

- 480 train images
- 80 validation images
- 80 locked test images
- 640 images overall

## Curation

The source pool was restricted to the `car parts 50` root from the Kaggle
dataset `gpiosenka/car-parts-40-classes`.

Only the eight mapped classes listed above were retained. Model files and
other non-image artifacts were excluded.

Before integration, the final pool passed these audits:

- exact duplicate groups: 0
- perceptual near-duplicate review pairs: 0
- high-confidence cross-category pairs: 0
- overlap pairs with the existing project images: 0

The original Kaggle train, validation, and test folders were ignored.
Dataset V3 uses its own deterministic split.

## Test lock

The test split is physically separated under
`data/locked_test/dataset_v3/`.

It is not authorized for training, model selection, or validation analysis.
