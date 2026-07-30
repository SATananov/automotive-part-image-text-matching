# Dataset V2

## Data used in the project

The project uses ten automotive-part categories. The saved Dataset V2 contains:

| Split | Original real | Imported real | Synthetic | Images | Rows |
|---|---:|---:|---:|---:|---:|
| Train | 50 | 50 | 50 | 150 | 900 |
| Validation | 10 | 10 | 0 | 20 | 120 |
| Test | 10 | 0 | 0 | 10 | 60 |

Each image creates six rows: two `MATCH`, two `PARTIAL_MATCH`, and two `MISMATCH`. Rows from the same image are related and stay in the same split.

## Real-image acquisition

The additional real images are imported mainly from the Apache-2.0 **50 Types of Car Parts** Kaggle dataset. Wikimedia Commons is used as an open-license fallback for categories that do not have enough suitable Kaggle images.

Run:

```powershell
python -m src.import_dataset_v2 --force
```

Use `--source-root PATH` for a dataset already downloaded on the computer. Use `--commons-source-root PATH` for an offline reviewed Wikimedia fallback.

The importer:

- checks that each file is a readable image;
- applies EXIF orientation;
- converts to RGB and resizes to 224×224;
- rejects blank or very low-information files;
- calculates SHA-256 and perceptual hashes;
- rejects exact and near duplicates;
- keeps complete source, author, license, and transformation information.

The largest equal number of valid images that all ten categories can support is selected. The saved version uses five imported training images and one imported validation image per category.

## Relation labels

- `MATCH`: image category and text category are the same;
- `PARTIAL_MATCH`: the categories are different but belong to the same automotive system;
- `MISMATCH`: the categories belong to different systems.

Training, validation, and test use different caption templates. There is no exact caption overlap between splits.

## Main files

- `image_manifest.csv` — one row per committed image, including split, group IDs, path, source, and SHA-256;
- `licenses.csv` — attribution and licenses for the original Wikimedia images;
- `dataset_v2_manifest.csv` — source, license, hashes, split, and transformation details for imported Dataset V2 images;
- `dataset_v2_import_summary.json` — saved import counts and settings;
- `train.csv` and `validation.csv` — development data;
- `test.csv` — locked test data;
- `test_lock.json` — test hash and evaluation permission.

## Test lock

The ten original test images are unchanged. The normal data loader refuses to load `test.csv`. The test file is kept only so its SHA-256 can be checked until the final one-time evaluation is approved.
