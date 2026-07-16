# Step 009 — Real Automotive Parts Dataset Intake and Validation Foundation

## Objective

Create a safe, reproducible boundary for collecting real automotive-part photographs without mixing them with the generated development dataset.

## Implemented foundation

- dedicated `data/real/` directories for originals, staging, approved processed images, annotations, and the generated manifest;
- mandatory `real_` namespace for physical-part and image identifiers;
- exact schemas for `part_groups.csv`, `images.csv`, and `real_image_manifest.csv`;
- automatic validation of required values, identifiers, categories, families, descriptions, approvals, paths, file readability, and image metadata;
- SHA-256 duplicate detection within groups, across groups, and against development images;
- explicit identifier-overlap checks between real and development data;
- safe-path enforcement under `data/real/processed/images/`;
- CLI commands for intake validation and Step 009 verification;
- automated tests for manifest generation, duplicate detection, leakage protection, and the empty-foundation state;
- Git rules that keep originals and staging local while allowing approved reproducible data to be tracked.

## Initial state

The annotation templates are intentionally empty. This is a valid `EMPTY_FOUNDATION` state: the structure and safeguards are ready, but no real photographs have been approved yet.

## Commands

```powershell
python -m src.project_cli validate-real-data
python -m src.project_cli verify-step-009
python -m pytest -q
```

## Boundary

Step 009 does not create a real train, validation, or test split and does not train a model. Those actions must wait until a sufficient number of approved physical groups has passed intake validation.
