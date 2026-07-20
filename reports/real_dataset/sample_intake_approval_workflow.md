# Step 009.1 — Real Dataset Sample Intake & Approval Workflow

## Objective

Add a controlled, auditable path from ignored staged photographs to the approved real automotive-parts dataset without weakening the Step 009 duplicate and leakage protections.

## Implemented

- Added `sample_intake.csv` as the tracked review queue.
- Added `approval_log.csv` as the tracked decision history.
- Added read-only queue review with metadata, image-quality, hash, duplicate, and development-overlap checks.
- Added explicit `pending`, `approved`, and `rejected` decisions.
- Added deterministic image IDs derived from physical group and view.
- Added EXIF orientation handling and EXIF-free RGB PNG normalization.
- Added transactional apply with snapshots, atomic CSV writes, final Step 009 validation, and rollback.
- Added CLI commands for review, apply, and Step 009.1 verification.
- Added automated tests for approval, rejection, duplicate protection, image quality, and rollback behavior.

## Empty checkpoint state

The current repository contains an empty intake queue and an empty approval log. This is the expected clean foundation before the first user-supplied real photograph is staged.

Expected commands:

```powershell
python -m src.project_cli review-real-intake
python -m src.project_cli apply-real-intake
python -m src.project_cli verify-sample-intake
```

Expected state:

- review: `PASS / EMPTY_QUEUE`;
- apply: `PASS / NO_DECISIONS`;
- Step 009 validation: `PASS / EMPTY_FOUNDATION`;
- no real image is added automatically.
