# Dataset V2 documentation

## Independent images and paired rows

Dataset V2 records the actual balanced import quota in `dataset_v2_import_summary.json`. Split sizes are derived from that manifest rather than asserted in advance.

| Split | Original real | Imported real | Synthetic | Images | Rows |
|---|---:|---:|---:|---:|---:|
| Train | 50 | `10 × imported_train_quota` | 50 | `100 + imported_train_images` | `6 × train_images` |
| Validation | 10 | `10 × imported_validation_quota` | 0 | `10 + imported_validation_images` | `6 × validation_images` |
| Test | 10 | 0 | 0 | 10 | 60 |

Every image contributes six rows: two per relation label. Rows from the same image are dependent and must remain together in resampling and paired tests.

## Kaggle-first acquisition

`python -m src.import_dataset_v2 --force` creates one unified imported image collection. The importer first looks for all ten categories in the downloaded Kaggle directory using normalized aliases. Wikimedia Commons is queried only if `air_filter` or `shock_absorber` does not have enough Kaggle candidates.

The raw Kaggle source and temporary Commons downloads are cached under `.cache/` and are not committed. After duplicate screening, the importer chooses the largest equal quota supported by all ten categories, capped at 20 train and 5 validation images per category and requiring at least 3 train and 1 validation image per category.

Use `--source-root PATH` for an already downloaded Kaggle copy. Use `--commons-source-root PATH` for an offline reviewed root with `air_filter/` and `shock_absorber/` directories when a fallback is needed.

## Selection and duplicate controls

Selection is deterministic and based on source SHA-256 and path order after image validation. The importer screens same-category candidates against original project assets and accepted candidates with:

- exact SHA-256;
- 64-bit difference hash, rejecting Hamming distance ≤ 2;
- normalized 48×48 grayscale cosine similarity, rejecting similarity ≥ 0.995.

Selected images are EXIF-corrected, converted to RGB, centre-cropped, resized to 224×224, and saved as JPEG quality 92. Both original and standardized hashes are retained.

## Relation construction

- `MATCH`: image and text category are identical;
- `PARTIAL_MATCH`: paired categories belong to the same automotive system;
- `MISMATCH`: paired categories belong to different systems.

Mismatch targets use balanced rotating permutations. For every split, every text category appears exactly equally often under all three labels. Train, validation, and test have separate caption templates and zero exact sentence overlap.

## Manifests and licenses

- `image_manifest.csv`: one row per committed image with source, split, group IDs, path, and SHA-256;
- `licenses.csv`: attribution and license information for the original 70 Wikimedia photographs;
- `dataset_v2_manifest.csv`: unified per-image provider, source path/page, author/credit, license, source hash, standardized hash, split, and transformation record for the balanced imported subset;
- `dataset_v2_import_summary.json`: deterministic import summary, provider counts, recorded licenses, and manifest hash.

## Test lock

The original ten Wikimedia test identities remain unchanged. `test.csv` is generated for integrity but normal loaders reject it. `test_lock.json` stores its current SHA-256 and explicitly forbids evaluation in the development checkpoint.
