# First Batch Manual Decision Validation and Controlled Application

## Scope

Step 009.9 converts the runtime manual decision workbook into an explicit,
fingerprinted application plan and applies it only through the existing
transactional real-sample intake workflow.

The implementation does not invent decisions. It accepts only operator-entered
`approved` or `rejected` values from the Step 009.8 runtime workbook.

## Validation

Run:

```powershell
python -m src.project_cli validate-first-real-batch-manual-decisions
```

The validation command:

- requires every live queue row to remain `pending`;
- compares workbook identifiers and immutable metadata with the live queue and
  the canonical `batch_001` plan;
- rebuilds the current staged-image review and blocks stale quality results;
- requires a rejection reason for every rejected item;
- requires all first-batch queue rows to have explicit decisions before
  application;
- writes a runtime-only application plan, status JSON, and Markdown summary;
- creates a SHA-256 `plan_id` from the queue, workbook, canonical plan, and
  explicit operator decisions.

No annotations, approval log, processed image, manifest, tracked report, or
live queue row is changed by validation.

## Controlled application

After validation reports `READY_TO_APPLY`, run:

```powershell
python -m src.project_cli apply-first-real-batch-manual-decisions
```

The application command rebuilds the plan and compares it with the saved
validation fingerprints. Any workbook, queue, or canonical-plan change makes
the saved plan stale and blocks application.

Only then are the explicit decisions copied into the live queue. The command
delegates normalization, annotation updates, approval logging, manifest
generation, and final validation to `src.apply_real_sample_intake.apply_intake`.

An outer Step 009.9 transaction snapshots:

- the live queue;
- real part-group and image annotations;
- the approval log;
- the real-image manifest;
- intake review, apply, and validation reports;
- every existing approved processed image.

Any downstream exception, count mismatch, or unhandled intake ID restores the
complete snapshot and records `ROLLED_BACK` in the runtime application report.

## Runtime outputs

All Step 009.9 operator-control outputs remain under:

```text
data/real/runtime/first_batch_review/
```

This directory is excluded from Git.

## Verification

Run:

```powershell
python -m src.project_cli verify-manual-decisions
```
