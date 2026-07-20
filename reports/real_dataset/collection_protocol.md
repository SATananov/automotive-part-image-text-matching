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
python -m src.project_cli verify-real-dataset-foundation
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

## Step 009.1 sample intake and approval workflow

The tracked intake queue is:

```text
data/real/annotations/sample_intake.csv
```

The tracked decision history is:

```text
data/real/processed/approval_log.csv
```

The queue is the only supported bridge between the ignored local staging directory and the approved reproducible dataset. Do not copy a photograph directly into `data/real/processed/images/` and do not edit the approval log manually.

### Intake identifiers and staged files

Use a monotonically increasing identifier with at least six digits:

```text
intake_000001
intake_000002
```

Copy one candidate image directly under `data/real/staging/` and rename the file so that its filename stem equals the intake identifier:

```text
data/real/staging/intake_000001.jpg
```

The staging path must be project-relative, must not contain `..`, and must not point outside the staging directory. Supported source extensions are JPEG and PNG.

### Queue decisions

The `decision` column accepts exactly:

- `pending` — validate and review the candidate without changing approved data;
- `approved` — apply the candidate after review;
- `rejected` — record the rejection without creating an approved image.

A rejected row requires a clear `rejection_reason`. Pending and approved rows require the complete part-group metadata, label descriptions, view, source, and staged image file.

### Review before approval

Run:

```powershell
python -m src.project_cli review-real-intake
```

The review is read-only. It checks:

- exact CSV schema and unique intake identifiers;
- safe staged paths and filename-to-intake-ID consistency;
- `real_` group namespace, category-family mapping, and label semantics;
- conflicts with existing group and image annotations;
- readable dimensions, image mode, file size, luminance, contrast, and aspect ratio;
- exact SHA-256 duplicates within the queue;
- overlap with approved real images and the development dataset;
- repeated derived image IDs such as two rows for the same group and view.

Minimum dimensions are enforced. Lower-than-recommended resolution, unusual luminance, low contrast, non-RGB input, and extreme aspect ratio are warnings that require manual judgment. A warning does not automatically reject a useful photograph.

Review the generated reports:

```text
reports/real_dataset/sample_intake_review.json
reports/real_dataset/sample_intake_review.md
```

### Transactional apply

After the review report is acceptable and the intended rows are marked `approved` or `rejected`, run:

```powershell
python -m src.project_cli apply-real-intake
```

The apply operation is a transaction:

1. Rebuild and validate the review report.
2. Normalize approved photographs with EXIF orientation applied.
3. Convert approved photographs to EXIF-free RGB PNG files.
4. Build prospective part-group, image, queue, and approval-log tables.
5. Recheck normalized hashes against approved real and development content.
6. Write all files atomically.
7. Regenerate the real image manifest and run the complete Step 009 validator.
8. Keep the changes only when final validation passes.

If any stage fails, the workflow restores the previous CSV files and reports and removes any newly created processed images.

Approved rows create or update the approved physical group, add one image annotation, append one approval-log record, and leave the original staged source unchanged. Rejected rows append an audit record but do not create an image. Applied approved and rejected rows are removed from `sample_intake.csv`; `pending` rows remain for later review.

### Processed image policy

The processed filename is derived from the physical group and view:

```text
real_starter_101_front.png
```

The approved file is stored directly under:

```text
data/real/processed/images/
```

Normalization strips camera metadata and converts the pixel data to RGB PNG. This reduces privacy risk and makes the approved representation reproducible. The ignored originals and staging files remain local and are not part of the Git checkpoint.

### Workflow verification

Run:

```powershell
python -m src.project_cli verify-sample-intake
```

The verifier checks the intake and approval-log schemas, required modules and reports, CLI documentation, transaction safeguards, and the current queue review state.

## Step 009.2 first balanced batch

The first controlled real-data batch is planned in
`data/real/annotations/first_batch_plan.csv`. It contains 20 planned images
from 10 physical parts: one part from every supported category and two views
per part (`front` and `detail`). The plan is collection metadata only; it is
not an approved dataset and does not prove that a physical part or photograph
exists.

Use the exact reserved intake IDs and staging filenames from the plan. A
missing file is an expected `AWAITING_CAPTURE` condition. Do not insert plan
rows directly into `sample_intake.csv` until the corresponding file exists and
has been checked.

Run preparation before editing the live intake queue:

```powershell
python -m src.project_cli prepare-first-real-batch
```

The preparation command validates category balance, identifiers, group/view
coverage, metadata consistency, staging-file presence, quality checks, and
conflicts with the live queue or approval log. It generates a queue preview
but does not approve or modify real data.

Run the controlled dry run after one or more planned files are captured:

```powershell
python -m src.project_cli dry-run-first-real-batch
```

The dry run simulates an approved decision only inside temporary storage. It
normalizes captured candidates to temporary RGB PNG files, builds prospective
annotations, applies duplicate and leakage checks, and verifies that the live
annotations, queue, approval log, manifest, and processed images are unchanged.
A successful dry run is evidence that the workflow can accept the candidates;
it is not an approval decision.

Actual intake still follows Step 009.1: copy reviewed rows into
`sample_intake.csv`, keep them `pending` during review, set explicit decisions,
and run `apply-real-intake` only after the review report is accepted.

## Step 009.3 capture, staging, and review readiness

Place first-batch photographs in `data/real/capture_inbox/batch_001/`
using the exact descriptive names from `first_batch_capture_file_map.csv`, such
as `real_starter_001_front.jpg`. JPEG and PNG are accepted. Run
`import-first-real-batch` to copy the unchanged bytes into
`data/real/originals/batch_001/`. Do not add unplanned files or use two
extensions for one planned capture.

After local import, run:

```powershell
python -m src.project_cli stage-first-real-batch-capture
```

The command is the supported bridge from local originals to temporary staging.
It applies EXIF orientation, converts pixels to RGB, writes deterministic JPEG
staging files, and preserves every original unchanged. It rejects duplicate
originals, duplicate normalized content, exact overlap with development or
approved real content, multiple source candidates, unexpected filenames, and
existing staging destinations with different bytes. All new staging writes
are rolled back if the subsequent Step 009.2 preparation review fails.

The generated capture inventory records source presence, staging status, image
metrics, review errors, warnings, and queue readiness for all 20 planned rows.
The generated review queue draft contains only rows whose staged image has a
`PASS` or `WARN` review status and whose intake ID is not already queued. Every
draft decision remains `pending`.

The draft must not be treated as a live approval queue. Compare it with the
capture inventory and report, perform the manual visual review, and only then
copy selected rows into `sample_intake.csv`. The Step 009.1 review and
transactional apply commands remain mandatory. Step 009.3 never edits the live
queue, approval log, approved manifest, or processed images.

Run the verifier with:

```powershell
python -m src.project_cli verify-capture-staging
```

## First-batch descriptive file naming and local import

The first controlled batch uses exact descriptive filenames rather than raw intake identifiers:

```text
real_<part_category>_001_<view>.jpg
```

The canonical mapping is `data/real/annotations/first_batch_capture_file_map.csv`, and the operator checklist is `reports/real_dataset/first_batch_capture_checklist.md`.

Local photographs must be placed in `data/real/capture_inbox/batch_001/` and imported with `python -m src.project_cli import-first-real-batch`. The importer copies original bytes without pixel conversion, preserves the inbox files, detects duplicates and destination conflicts, and proves that staging and live dataset state remain unchanged. Only after the import report reaches `READY_FOR_STAGING` should the Step 009.3 staging command be run.

## First batch operator session

The first capture session is organized by physical part rather than by technical intake identifier. The operator uses `reports/real_dataset/first_batch_operator_guide.md` and photographs each physical part as one `front` view and one meaningful `detail` view.

Run the session preparation command at any point during capture:

```powershell
python -m src.project_cli prepare-first-real-batch-session
```

The generated `data/real/processed/first_batch_capture_session.csv` contains one row per physical part, the two descriptive filenames, inbox and originals status, pair readiness, and the next required action. The command must not modify the capture files, staging, annotations, live queue, approval log, or manifest.

The session is complete for local import only when all 20 planned slots are available. Local import, staging, manual queue review, and approval remain separate operations.

## First batch dashboard and progress tracking

The first-batch operational state is summarized with:

```powershell
python -m src.project_cli build-first-real-batch-dashboard
```

The generated `first_batch_capture_progress.csv` has one row per planned
photograph and records capture, import, staging, review, queue, decision, and
approval state. The self-contained `first_batch_capture_dashboard.html` gives
the operator summary cards, category progress, exact filenames, pipeline
stages, percentages, and next actions.

The dashboard is read-only with respect to capture files and live dataset
state. It may update only its own CSV, JSON, Markdown, and HTML snapshot. It
must report `live_state_unchanged: PASS`. Approval still requires the existing
manual review and transactional apply workflow.

## First batch capture execution and live progress

During the physical capture session, use the descriptive filenames from the
capture map and place available files in the batch inbox. Run:

```powershell
python -m src.project_cli run-first-real-batch-capture-session
```

The supported execution cycle validates the inbox, imports unchanged bytes to
local originals, stages deterministic review JPEG files, and updates a live
runtime dashboard. The complete operator procedure is documented in
`reports/real_dataset/first_batch_capture_execution_and_live_progress.md`.

All live execution outputs belong under
`data/real/runtime/first_batch_capture/`, which is ignored by Git. Runtime
isolation prevents repeated progress updates from changing tracked reports.
The cycle must report `live_dataset_unchanged: PASS` and
`tracked_outputs_unchanged: PASS`. If a downstream phase fails, originals and
staging created by that cycle are rolled back.

A read-only update is available with:

```powershell
python -m src.project_cli refresh-first-real-batch-live-progress
```

Neither command edits the live intake queue or records approval decisions.
Manual review and the existing transactional approval workflow remain required.

## First batch review queue activation and manual decisions

A generated review queue draft is not a live approval queue. Activate validated pending rows only through:

```powershell
python -m src.project_cli activate-first-real-batch-review-queue
```

The activation command must prove that every candidate belongs to the canonical first-batch plan, matches its metadata, has not been processed before, and passes the existing staged-image review. Exact rows already in the live queue are treated idempotently. Conflicts block the complete transaction. Activation must not modify part groups, image annotations, the approval log, approved image manifest, or processed images.

Prepare manual visual decisions with:

```powershell
python -m src.project_cli prepare-first-real-batch-manual-decisions
```

The runtime workbook preserves the operator fields `operator_decision`, `rejection_reason`, and `operator_notes`. A rejected row requires a reason. Decision preparation never edits the live queue and never applies an image. Automatic approval and automatic rejection are prohibited.

## Step 009.9 — Manual decision validation and controlled application

After Step 009.8 reports that all manual decisions are ready, validate the
workbook against the current live queue, current staged-image review, and the
canonical first-batch plan:

```powershell
python -m src.project_cli validate-first-real-batch-manual-decisions
```

Do not apply when readiness is anything other than `READY_TO_APPLY`. The
validation plan is runtime-only and is bound to SHA-256 fingerprints of the
queue, workbook, and canonical first-batch plan.

Apply the validated decisions with:

```powershell
python -m src.project_cli apply-first-real-batch-manual-decisions
```

The apply command revalidates all inputs, rejects stale plans, updates only the
explicit first-batch decisions, and delegates dataset writes to the
transactional Step 009.1 intake application. A second transaction layer
restores the queue, annotations, approval log, manifest, reports, and processed
images if any downstream operation or post-application check fails.

Verify the safeguards with:

```powershell
python -m src.project_cli verify-manual-decisions
```

## Step 010 — First real dataset capture and approved sample ingestion

After adding one or more descriptively named photographs to the first-batch
capture inbox, run:

```powershell
python -m src.project_cli run-first-real-dataset-capture
```

The command imports, stages, refreshes progress, activates validated pending
review rows, and prepares the runtime manual-decision workbook. It proves that
the approved dataset remains unchanged and never creates an automatic decision.

Inspect and edit only the operator columns in:

```text
data/real/runtime/first_batch_review/manual_decision_workbook.csv
```

After readiness becomes `READY_TO_APPLY`, run:

```powershell
python -m src.project_cli finalize-first-real-dataset-ingestion
```

The command applies the exact Step 009.9 fingerprinted decisions, audits every
approved output, and restores the pre-application snapshot if the audit fails.
Rejected captures produce `RECAPTURE_REQUIRED`; the batch is complete only when
all planned categories have approved front/detail pairs.

Verify Step 010 with:

```powershell
python -m src.project_cli verify-real-dataset-ingestion
```

## Step 010.1 — Open-license external development images

Internet images are collected only into the separate external-data boundary:

```text
data/external/open_license/
```

They must never be relabeled as `warehouse_photo` or copied into the controlled
Step 010 real-capture queue.

Run:

```powershell
python -m src.project_cli collect-open-license-images
python -m src.project_cli build-open-license-review-gallery
python -m src.project_cli validate-open-license-images
python -m src.project_cli verify-open-license-dataset
```

The collector accepts only JPEG or PNG files with an allowlisted public-domain,
CC0, CC BY, or CC BY-SA license. It stores the Commons source page, creator or
credit, license and license URL, download URL, dimensions, and SHA-256.

All semantic category decisions remain manual. New rows are always `pending`;
rejected rows require a reason. The external collection is not ready until each
category has five manually approved images.
