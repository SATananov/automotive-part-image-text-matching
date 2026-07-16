# Step 009.3 — First Real Batch Capture, Staging & Review Readiness

## Goal

Provide a safe operational bridge from locally captured `batch_001`
photographs to review-ready staging files without modifying the live intake
queue or approving any sample.

## Implemented

- dedicated local originals location for `batch_001`;
- reserved intake-ID filename matching and unexpected-file detection;
- EXIF orientation handling and deterministic RGB JPEG staging;
- transactional staging writes with conflict-safe rollback;
- exact duplicate checks across originals and normalized staging content;
- development and approved-real overlap checks;
- capture inventory for all 20 planned image slots;
- pending-only review queue draft separated from the live queue;
- live queue, approval log, and manifest fingerprint protection;
- CLI registration, documentation, verifier, and regression tests.

## Initial checkpoint

No real photograph is bundled or approved. With empty local originals and
staging directories, the expected state is:

```text
Status: PASS
Readiness: AWAITING_CAPTURE
Original captures: 0
Staged images: 0
Queue draft rows: 0
Live queue unchanged: PASS
```

## Commands

```powershell
python -m src.project_cli stage-first-real-batch-capture
python -m src.project_cli verify-step-009-3
```
