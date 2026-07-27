# Dataset documentation

## Task construction

Every image belongs to one of ten automotive-part categories and one of five broader systems. For each image, the dataset creates three paired rows:

- `MATCH`: the text names the same category;
- `PARTIAL_MATCH`: the text names another category from the same system;
- `MISMATCH`: the text names a category from another system.

The explicit `text_category` column makes the relation construction auditable. Text category is exactly balanced across all three labels, so it cannot predict the relation by itself.

## Image inventory

| Split | Wikimedia photographs | Synthetic drawings | Images | Paired rows |
|---|---:|---:|---:|---:|
| Train | 50 | 50 | 100 | 300 |
| Validation | 10 | 0 | 10 | 30 |
| Test | 10 | 0 | 10 | 30 |

The current revision added 20 new open-license Wikimedia photographs: one validation and one locked-test image for each category. There are now 70 real photographs and 50 synthetic drawings.

Synthetic drawings are used only for training and for a controlled ablation. No reported validation score contains synthetic images.

## Independence controls

- `part_group_id` keeps all three paired rows for one image together.
- `object_group_id` groups alternate views from the same physical object or photographic series.
- Known same-series Wikimedia photographs remain inside training.
- Generated drawings are grouped into ten template families and remain inside training.
- Train, validation, and test have zero overlap in image ID, group ID, object group, path, and exact file hash.
- Train, validation, and test use separate sentence templates, with zero exact description overlap.
- The nearest-image audit found no suspicious same-category train/validation pair at the defined perceptual thresholds.

`image_manifest.csv` records every image, source, split, category, grouping IDs, local path, and SHA-256.

## Licensing

`licenses.csv` contains one row for each of the 70 Wikimedia photographs:

- Commons title and source page;
- author and credit;
- license name and license URL;
- local path and SHA-256;
- a note describing the local preview-size download.

The dataset builder rejects duplicate Commons titles, source pages, local paths, or hashes and verifies every recorded file hash.

## Test lock

`test.csv` is present for final evaluation but remains sealed. `test_lock.json` stores its SHA-256 and states that evaluation is not permitted in the current checkpoint. Normal project code cannot load the test split.
