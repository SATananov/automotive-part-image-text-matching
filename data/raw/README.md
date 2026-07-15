# Raw dataset

The image files are stored in `data/raw/images`.

## Image naming

Each physical automotive part receives a group identifier.

Example: starter_001

Different images of the same physical part use the same group identifier:

starter_001_01.jpg
starter_001_02.jpg
starter_001_03.jpg

## Labels

Each image is paired with three descriptions:

1. MATCH - the description names the correct part category
2. PARTIAL_MATCH - the description names another category from the same part family
3. MISMATCH - the description names a category from a different part family

## Split rule

Images from the same `part_group_id` must not appear in more than one of the training, validation, or test subsets.

## Data scope

Vehicle compatibility is not used as a label criterion because it cannot always be verified from an image alone.
