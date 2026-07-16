# Real Automotive Part Dataset Collection Protocol

## Purpose

The real dataset contains photographs of automotive parts and short text descriptions for the final image-text matching experiment.

Each physical part receives one stable `part_group_id`. Every photograph of that same physical object must use the same group identifier, even when the view, crop, lighting, or background changes.

## Initial target

The minimum target is:

- 10 automotive part categories;
- 10 physical parts per category;
- 2 approved photographs per physical part;
- 100 physical part groups;
- 200 approved real images;
- 600 image-text samples after each image is paired with three descriptions.

The target labels are:

- `MATCH`;
- `PARTIAL_MATCH`;
- `MISMATCH`.

## Categories

The first real-data version uses these categories:

- `starter`
- `alternator`
- `brake_disc`
- `brake_pad`
- `shock_absorber`
- `coil_spring`
- `headlight`
- `taillight`
- `oil_filter`
- `air_filter`

The category-to-family mapping is defined in `src/dataset_config.py` and must not be changed only in an annotation file.

## File naming

Use lowercase ASCII names with underscores. The filename contains the category, physical part number, and view.

Examples:

```text
starter_001_front.jpg
starter_001_detail.jpg
brake_disc_004_front.jpg
brake_disc_004_rear.jpg
```

The corresponding identifiers are:

```text
part_group_id: starter_001
image_id: starter_001_front
```

A new photograph receives a new `image_id`, but photographs of the same physical part keep the same `part_group_id`.

## Directory layout

```text
data/real/
├── originals/              # untouched source photographs; excluded from Git
├── staging/                # temporary review and conversion work; excluded from Git
├── processed/images/       # approved and normalized images used by the project
└── annotations/
    ├── part_groups.csv
    └── images.csv
```

Never overwrite the only copy of an original photograph. Work on a copy under `staging/`, then place only the approved normalized result under `processed/images/`.

## Image capture requirements

Each approved image must satisfy all of the following:

- show one clearly identifiable physical part as the main subject;
- come from a part that the collector owns, manages, or has permission to photograph;
- contain no faces, license plates, addresses, customer documents, serial-number paperwork, or other personal information;
- be in focus and sufficiently illuminated;
- preserve the complete part whenever possible;
- avoid heavy filters, artificial backgrounds, watermarks, and text overlays;
- use a distinct view rather than a duplicate export of the same photograph;
- have the correct category confirmed by a knowledgeable reviewer;
- use one of the allowed views: `front`, `rear`, `left`, `right`, `top`, `detail`, or `other`.

JPEG and PNG source photographs are acceptable. The later processing pipeline must normalize format, orientation, color mode, and model input size consistently without modifying originals.

## Annotation files

### `part_groups.csv`

This file contains one row per physical part and uses exactly these columns:

| Column | Meaning |
|---|---|
| `part_group_id` | Stable identifier for one physical part |
| `part_family` | Automotive system defined in the project configuration |
| `part_category` | One of the ten approved categories |
| `match_description` | Description for the same category |
| `partial_description` | Description for another category in the same family |
| `mismatch_description` | Description for a category in another family |
| `source` | Short non-personal source note, such as `warehouse_photo` |
| `approved` | `yes` only after review; otherwise `no` |
| `notes` | Optional quality or identification note |

Example:

```csv
part_group_id,part_family,part_category,match_description,partial_description,mismatch_description,source,approved,notes
starter_001,electrical,starter,Automotive starter motor.,Automotive alternator.,Automotive brake disc.,warehouse_photo,no,
```

### `images.csv`

This file contains one row per photograph and uses exactly these columns:

| Column | Meaning |
|---|---|
| `image_id` | Unique identifier for one photograph |
| `part_group_id` | Existing physical-part identifier from `part_groups.csv` |
| `image_path` | Repository-relative path under `data/real/processed/images/` |
| `view` | One of the configured allowed views |
| `approved` | `yes` only after image review; otherwise `no` |

Example:

```csv
image_id,part_group_id,image_path,view,approved
starter_001_front,starter_001,data/real/processed/images/starter_001_front.jpg,front,no
```

Identifiers must be unique. Every `images.csv` group reference must exist in `part_groups.csv`, and every approved `image_path` must point to an existing readable file.

## Description rules

Descriptions must be short, neutral, and based on the visible part category rather than on background clues or filename text.

- `match_description` names the same category as the image;
- `partial_description` names the paired category from the same automotive family;
- `mismatch_description` names a category from a different automotive family.

Do not include supplier names, warehouse locations, filenames, part-group numbers, approval status, or wording that directly reveals the label. Use consistent English terminology across groups.

The three descriptions are stored once at group level and are later expanded into three image-text samples for each approved image.

## Approval workflow

1. Copy the untouched photograph to `data/real/originals/`.
2. Create a working copy under `data/real/staging/`.
3. Assign the physical part a unique `part_group_id`.
4. Add or review its row in `part_groups.csv` with `approved` set to `no`.
5. Normalize the image and place it under `data/real/processed/images/`.
6. Add its row to `images.csv` with `approved` set to `no`.
7. Check category, family, description labels, filename, path, view, image quality, and privacy.
8. Change `approved` to `yes` only after all checks pass.

A rejected image stays outside the training dataset. Corrections must not silently change the identity of an already approved physical group.

## Leakage prevention

The final train, validation, and test split must be created by `part_group_id`, never by image row or generated image-text sample.

All views of one physical part and all three descriptions paired with those views must remain in the same split. Near-duplicate photographs from the same capture sequence must also remain in that group.

Do not use the test split for model selection, threshold adjustment, vocabulary decisions, manual prompt changes, or repeated progress reporting. Freeze the final model and evaluation procedure before opening test results.

## Validation checklist

Before a real-data version is accepted, verify that:

- all CSV files use the exact configured columns;
- identifiers are present, normalized, and unique;
- categories and families match the project configuration;
- every image group exists in `part_groups.csv`;
- every approved image path exists and is readable;
- approved groups contain valid `MATCH`, `PARTIAL_MATCH`, and `MISMATCH` descriptions;
- no photograph or description contains personal or customer information;
- each physical group has the required number of distinct approved views;
- image hashes and manual review reveal no cross-group duplicates;
- split groups are disjoint and every category is represented as planned;
- original and staging files remain excluded from Git;
- the test split has not been used during development.

## Change control

Changes to categories, families, allowed views, CSV columns, approval values, or target counts must first be made in the project configuration and tests. Update this protocol in the same commit so that code and collection instructions remain synchronized.
