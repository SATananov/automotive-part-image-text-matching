# Step 010.2 — External Dataset Integration, Grouped Split & Training Readiness

## Goal

Step 010.2 converts the manually approved Step 010.1 image collection into a
training-ready image-text matching dataset while preserving the locked-test
policy.

Only rows with `operator_decision=approved` are integrated. Rejected and
pending rows never enter the generated metadata or splits.

## Generated external dataset

Each of the 50 approved images becomes one independent `part_group_id`. Three
samples are generated for every image:

- `MATCH` — description from the same part category;
- `PARTIAL_MATCH` — description from the paired category in the same family;
- `MISMATCH` — description from a category in another family.

This creates 150 external samples from 50 image groups.

Build the approved catalog, metadata, external grouped split, and integrated
development + external split with:

```powershell
python -m src.project_cli integrate-external-dataset
```

Generated external artifacts:

```text
data/external/integrated/approved_external_images.csv
data/external/integrated/external_matching_metadata.csv
data/external/integrated/external_train.csv
data/external/integrated/external_validation.csv
data/external/integrated/external_test.csv
data/external/integrated/external_split_manifest.csv
```

Generated integrated artifacts:

```text
data/processed/integrated_train.csv
data/processed/integrated_validation.csv
data/processed/integrated_test.csv
data/processed/integrated_split_manifest.csv
data/processed/integrated_test_lock.json
```

## Grouped split

The split is deterministic and category-balanced:

- train: 3 approved groups per category;
- validation: 1 approved group per category;
- test: 1 approved group per category.

All samples from one `part_group_id` remain in one split. External identifiers
use a dedicated `external_` namespace and cannot overlap with the generated
development identifiers.

## Test-lock policy

`integrated_test.csv` and `external_test.csv` are created only so the split is
complete and reproducible. They are fingerprinted in
`integrated_test_lock.json`. All paths stored in the lock and integration
reports are project-relative POSIX paths, so the committed artifacts remain
portable across Windows, Linux, fresh clones, and different user profiles.

Training-ready inputs contain only:

```text
data/processed/integrated_train.csv
data/processed/integrated_validation.csv
```

Step 010.2 does not train a model and does not evaluate the test split.

## Validation

Run:

```powershell
python -m src.project_cli validate-external-training-readiness
```

The validator checks:

- Step 010.1 remains `READY_FOR_EXTERNAL_DATASET`;
- exactly 50 approved images and five per category;
- rejected audit candidates are excluded;
- 150 generated samples and all three labels per image;
- group isolation across train, validation, and test;
- expected external split counts;
- development/external identifier separation;
- integrated split composition;
- image SHA-256 integrity;
- test fingerprints and the train/validation-only input policy.

The expected readiness is:

```text
READY_FOR_TRAINING
```

## Verification

Run:

```powershell
python -m src.project_cli verify-step-010-2
```
