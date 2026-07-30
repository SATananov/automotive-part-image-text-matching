# Dataset V2 Acquisition and Sampling

## Goal

The goal of Dataset V2 is to add more independent real photographs while keeping the original final test unchanged.

## Sources

The importer first searches the Apache-2.0 **50 Types of Car Parts** Kaggle dataset. The project uses the following categories: air filter, alternator, brake disc, brake pad, coil spring, headlight, oil filter, shock absorber, starter, and taillight.

Wikimedia Commons is used as an open-license fallback when a category does not have enough suitable Kaggle images. Every selected file keeps its source page, author or credit, license, and hashes.

## Image checks

Before an image is accepted, the importer:

1. checks that the file is readable and has enough visual information;
2. applies EXIF orientation;
3. composites transparent images on white;
4. converts to RGB;
5. centre-crops and resizes to 224×224;
6. calculates source and standardized SHA-256 hashes;
7. rejects exact duplicates;
8. rejects very close same-category images using dHash and grayscale cosine similarity.

## Balanced selection

The importer finds the largest equal quota supported by all ten categories after the checks. The maximum is 20 training and 5 validation images per category. At least 3 training and 1 validation image per category are required.

Selection is deterministic. With the same local source files, the same images and hashes are selected. The committed files and manifest are the fixed exam snapshot because online datasets can change later.

## Split design

Imported training images go only to training. Imported validation images join the original validation images. The original ten test images stay unchanged and locked.

Every image receives six image-text pairs: two for each relation label. All rows from one image stay together during splitting and statistical evaluation.

## Required checks

The pipeline must confirm:

- balanced source and category counts;
- readable files and matching hashes;
- balanced text categories and labels;
- no identity or group overlap between train and validation;
- no exact image-hash overlap;
- no exact caption overlap;
- no suspicious same-category near duplicates;
- complete provenance and license information;
- an unchanged and matching test lock.
