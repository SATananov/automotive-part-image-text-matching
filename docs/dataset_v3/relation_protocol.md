# Dataset V3 relation-building protocol

## Development scope

Dataset V3 contains eight curated automotive-part categories. Only the 480 train images and 80 validation images are available to the relation-building code. The 80-image test split remains physically separated and locked.

The development relation tables are stored under `data/manifests/dataset_v3/` rather than replacing the saved Dataset V2 tables.

## Labels

Every image produces six rows: two rows for each relation label.

- `MATCH`: the image and text identify the same part category.
- `PARTIAL_MATCH`: the text identifies a different part from the same broad vehicle subsystem.
- `MISMATCH`: the text identifies a part from another broad subsystem.

The broad subsystem groups are:

- engine support: alternator, starter, oil filter;
- chassis: brake disc, brake pad, coil spring;
- lighting: headlight, taillight.

This grouping is deliberately broader than the Dataset V2 system pairs. It allows every retained Dataset V3 category to participate in a balanced three-label task without introducing weak or dependent source classes merely to complete a pair.

## Balance and leakage controls

Each image contributes two examples per label. Partial targets form a permutation within each subsystem. Mismatch targets use deterministic cross-subsystem permutations. Therefore every text category occurs equally often under every label.

Train and validation use separate caption templates. The builder rejects:

- duplicate sample identifiers;
- category or label imbalance;
- identity overlap between train and validation;
- exact description overlap;
- locked-test paths in development tables;
- partial relations that cross subsystem boundaries;
- mismatch relations that remain inside a subsystem.

## Independent unit

The curated image group is the independent evaluation unit. All uncertainty estimates and paired comparisons must operate on complete image groups rather than treating the six rows from one image as independent observations.

## Test lock

Development modules do not expose a test loader and do not create test relation rows. The test manifest, images, labels, and predictions remain unavailable until a separately authorized final evaluation.
