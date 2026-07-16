# Real Automotive Part Dataset Collection Protocol

## Purpose

The real dataset contains photographs of automotive parts and short text descriptions for the final image-text matching experiment.

Each physical part receives one stable `part_group_id`. Every photograph of that same physical object must use the same group identifier, even when the view, crop, lighting, or background changes.

The generated development dataset and the real dataset are separate data domains. Real identifiers use a dedicated `real_` prefix and real files stay under `data/real/` so they cannot be confused with generated development samples.

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

Use lowercase ASCII names with underscores. Real-data identifiers must follow the dedicated namespace:

```text
part_group_id: real_<category>_<number>
image_id: real_<category>_<number>_<view>
```

Examples:

```text
real_starter_001_front.jpg
real_starter_001_detail.jpg
real_brake_disc_004_front.jpg
real_brake_disc_004_rear.jpg
```

The corresponding identifiers are:

```text
part_group_id: real_starter_001
image_id: real_starter_001_front
```

A new photograph receives a new `image_id`, but photographs of the same physical part keep the same `part_group_id`. The filename stem must equal `image_id` exactly.

The `real_` prefix is mandatory. It prevents identifier overlap with the generated development dataset, which already uses identifiers such as `starter_001` and `starter_001_01`.

## Directory layout

```text
data/real/
├── originals/                         # untouched photographs; ignored by Git
├── staging/                           # temporary review work; ignored by Git
├── processed/
│   ├── images/                        # approved normalized images
│   └── real_image_manifest.csv        # generated approved-image manifest
└── annotations/
    ├── part_groups.csv
    └── images.csv
```

Never overwrite the only copy of an original photograph. Work on a copy under `staging/`, then place only an approved normalized result under `processed/images/`.

The annotation CSV files and the generated manifest are reproducible project data. Original and staging photographs remain local and must not be committed.

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

JPEG and PNG source photographs are acceptable. Approved files may use `.jpg`, `.jpeg`, or `.png`. A later model-preparation step may normalize image size and color mode consistently without modifying originals.

## Annotation files

### `part_groups.csv`

This file contains one row per physical part and uses exactly these columns:

| Column | Meaning |
|---|---|
| `part_group_id` | Stable real-data identifier for one physical part |
| `part_family` | Automotive system defined in project configuration |
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
real_starter_001,electrical,starter,Automotive starter motor.,Automotive alternator.,Automotive brake disc.,warehouse_photo,no,
```

### `images.csv`

This file contains one row per photograph and uses exactly these columns:

| Column | Meaning |
|---|---|
| `image_id` | Unique identifier for one photograph |
| `part_group_id` | Existing identifier from `part_groups.csv` |
| `image_path` | Repository-relative path under `data/real/processed/images/` |
| `view` | One of the configured allowed views |
| `approved` | `yes` only after image review; otherwise `no` |

Example:

```csv
image_id,part_group_id,image_path,view,approved
real_starter_001_front,real_starter_001,data/real/processed/images/real_starter_001_front.jpg,front,no
```

Identifiers and paths must be unique. Every image group reference must exist in `part_groups.csv`. An approved image may only belong to an approved group, and every approved `image_path` must point to an existing readable file.

### `real_image_manifest.csv`

This file is generated by the validator and must not be edited manually. It contains only approved, readable images and uses these columns:

| Column | Meaning |
|---|---|
| `image_id` | Approved image identifier |
| `part_group_id` | Approved physical-part group |
| `image_path` | Safe repository-relative processed path |
| `part_family` | Validated family |
| `part_category` | Validated category |
| `view` | Validated image view |
| `source` | Non-personal source label from the group annotation |
| `approved` | Always `yes` for manifest rows |
| `sha256` | SHA-256 content hash used for duplicate checks |
| `file_size_bytes` | File size in bytes |
| `width` | Image width |
| `height` | Image height |
| `mode` | Pillow image mode |
| `format` | Detected image format |

Generate or refresh it with:

```powershell
python -m src.project_cli validate-real-data
```

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
3. Assign the physical part a unique `real_<category>_<number>` identifier.
4. Add or review its row in `part_groups.csv` with `approved` set to `no`.
5. Normalize the image and place it under `data/real/processed/images/`.
6. Add its row to `images.csv` with `approved` set to `no`.
7. Check category, family, descriptions, filename, path, view, image quality, permission, and privacy.
8. Run `python -m src.project_cli validate-real-data`.
9. Change the group and image rows to `yes` only after all checks pass.
10. Run the validator again and review the generated manifest and report.

A rejected image stays outside the training dataset. Corrections must not silently change the identity of an already approved physical group.

## Duplicate and leakage prevention

The validator calculates SHA-256 hashes for all approved real images.

- Exact duplicate content within one group is rejected.
- Cross-group duplicate content is rejected because it can create leakage after splitting.
- An approved real image whose hash matches a generated development image is rejected.
- Duplicate `image_id` values and duplicate `image_path` values are rejected.
- Repeated approved views inside one physical group are rejected.
- Real `part_group_id` and `image_id` values are checked against development identifiers.
- Paths outside `data/real/processed/images/`, absolute paths, and paths containing `..` are rejected.

Near-duplicates that are not byte-identical still require manual review. Photographs from the same capture sequence must remain under the same physical `part_group_id`.

## Leakage prevention

The final train, validation, and test split must be created by `part_group_id`, never by image row or generated image-text sample.

All views of one physical part and all three descriptions paired with those views must remain in the same split. Near-duplicate photographs from the same capture sequence must also remain in that group.

Do not use the test split for model selection, threshold adjustment, vocabulary decisions, manual prompt changes, or repeated progress reporting. Freeze the final model and evaluation procedure before opening test results.

## Automated validation

Run the intake validator from the repository root:

```powershell
python -m src.project_cli validate-real-data
```

The command validates schemas, required values, identifiers, family/category mapping, approval relationships, paths, readable image files, exact hashes, duplicate views, real/development separation, and manifest generation.

Run the Step 009 foundation verifier with:

```powershell
python -m src.project_cli verify-step-009
```

An empty pair of annotation templates is a valid `EMPTY_FOUNDATION` state. The validator remains `PASS` until collection begins, while still generating an empty manifest with the correct schema.

## Validation checklist

Before a real-data version is accepted, verify that:

- all CSV files use the exact configured columns;
- identifiers are present, normalized, unique, and use the `real_` namespace;
- categories and families match the project configuration;
- every image group exists in `part_groups.csv`;
- every approved image belongs to an approved group;
- every approved image path exists, stays inside the processed directory, and is readable;
- approved groups contain valid `MATCH`, `PARTIAL_MATCH`, and `MISMATCH` descriptions;
- no photograph or description contains personal or customer information;
- each physical group has the required number of distinct approved views;
- image hashes and manual review reveal no within-group, Cross-group, or development overlap;
- split groups are disjoint and every category is represented as planned;
- original and staging files remain excluded from Git;
- the test split has not been used during development.

## Change control

Changes to categories, families, allowed views, CSV columns, approval values, target counts, identifier format, or manifest schema must first be made in project configuration and tests. Update this protocol in the same commit so that code and collection instructions remain synchronized.
