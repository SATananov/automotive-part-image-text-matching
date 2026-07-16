# Step 009.2 — First Real Sample Batch Preparation & Controlled Intake Dry Run

## Goal

Prepare the first balanced real-photograph collection batch and exercise the
complete intake planning path without mutating the approved dataset.

## Implemented

- committed `batch_001` plan with 20 image slots;
- one physical part from each of the 10 configured categories;
- `front` and `detail` views for every planned group;
- deterministic `intake_000001`–`intake_000020` identifiers;
- preparation report and queue-preview generation;
- staged-file, metadata, quality, duplicate, and queue-conflict checks;
- temporary approval/normalization simulation;
- live-state fingerprint comparison proving dry-run immutability;
- CLI registration, documentation, verifier, and regression tests.

## Initial checkpoint

No real image is bundled or automatically approved. With an empty staging
directory the correct state is:

```text
Preparation: PASS / AWAITING_CAPTURE
Dry run: PASS / AWAITING_CAPTURE
Live-state immutability: PASS
```

## Commands

```powershell
python -m src.project_cli prepare-first-real-batch
python -m src.project_cli dry-run-first-real-batch
python -m src.project_cli verify-step-009-2
```
