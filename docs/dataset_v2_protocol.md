# Dataset V2 Acquisition and Sampling Protocol

## Goal

Increase the number and diversity of independent real images without changing the sealed final-test identities or representing repeated image-text pairs as independent observations.

## Public source design

Dataset V2 uses a Kaggle-first acquisition policy. All ten ontology categories are discovered through normalized source-directory aliases. Wikimedia Commons is retained as a documented open-license fallback for `air_filter` and `shock_absorber` only when the local Kaggle snapshot does not provide enough candidates.

### Kaggle component

- dataset: **50 Types of Car Parts - Image Classification**;
- handle: `gpiosenka/car-parts-40-classes`;
- recorded dataset license: Apache-2.0;
- supported mappings: all ten project categories, including common aliases such as `AIR FILTER`, `SHOCK ABSORBER`, `STRUT`, `BRAKE ROTOR`, and `TAIL LIGHTS`;
- selected images: the balanced share supported by the final cross-category quota, up to 20 train and 5 validation photographs per category.

### Wikimedia Commons component

- fallback categories: air filter and shock absorber;
- only readable raster files of adequate dimensions are considered;
- accepted license records include CC BY, CC0, and public-domain markers;
- each selected file records title, source page, author/credit, license name, license URL, and hashes;
- selected images: the non-duplicate fallback photographs that survive licensing and similarity checks; the final importer then applies one equal quota to every category.

The unified importer preserves source-specific metadata in `data/dataset_v2_manifest.csv`.

## Category mapping

The established ontology contains air filter, alternator, brake disc, brake pad, coil spring, headlight, oil filter, shock absorber, starter, and taillight. Source directory aliases are normalized case-insensitively and punctuation is ignored.

## Deterministic sampling

The importer first validates and deduplicates every category, then computes one balanced quota from the smallest surviving category. This avoids inventing statistical strength when an open source cannot supply the originally planned fixed count.

1. validate readable image files, source metadata, open-license fields, and automotive-domain relevance;
2. remove exact duplicate source hashes;
3. compute dHash and normalized grayscale vectors;
4. reject candidates too close to original project images or previously accepted same-category candidates;
5. composite transparent source images on white before RGB conversion and reject standardized images with insufficient variance or luminance entropy;
6. count surviving non-duplicate candidates in every category;
7. choose the largest equal quota supported by all categories, capped at 20 train and 5 validation images per category;
8. require at least 3 imported train and 1 imported validation image per category;
9. prefer original validation/test source directories for the new development-validation subset where source splits exist;
10. standardize and hash the selected files;
11. write complete provenance, license metadata, actual quotas, and manifest hash.

The validation quota is approximately one quarter of the bottleneck category, capped at five; the remaining balanced quota is assigned to training. No random sampling is used. Re-running against the same local source content produces the same selected source hashes and standardized files. Public online collections can evolve; the committed selected files and their manifest are therefore the canonical exam data snapshot.

## Split design

The project does not reuse the external dataset's test directories as its final test. Selected external validation/test photographs may enter the larger **development validation** subset. The original project test identities remain sealed and unchanged.

This avoids silently redefining the final test after model development while still providing a more stable validation sample.

## Pair construction

Every independent image receives six relation rows. Rotating mismatch permutations are globally balanced, so label, image category, text category, and source cannot become trivial relation shortcuts.

## Audit gates

The pipeline stops unless all of the following hold:

- source/split counts derived exactly from the committed Dataset V2 manifest;
- readable files and matching hashes;
- exact text-category-by-label balance;
- zero train/validation identity and group overlap;
- zero exact train/validation image-hash overlap;
- zero exact train/validation caption overlap;
- zero flagged same-category near-image pairs at the stated thresholds;
- complete original-Wikimedia and unified Dataset V2 provenance/license manifests;
- no missing provider, source, attribution, license, or hash values;
- unchanged sealed-test policy and matching test lock.
