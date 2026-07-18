# Step 010 — First Real Dataset Capture & Approved Sample Ingestion

## Purpose

Step 010 joins the existing capture, import, staging, review queue, manual
decision, controlled application, and real-dataset validation layers into one
explicit first real-data workflow.

It never invents photographs, labels, approvals, or rejection decisions.

## Phase 1 — Capture and manual review preparation

Place descriptively named photographs in:

```text
data/real/capture_inbox/batch_001/
```

Run:

```powershell
python -m src.project_cli run-first-real-dataset-capture
```

The command imports and stages valid local captures, refreshes progress,
activates validated pending review rows, prepares the runtime decision workbook,
and validates any existing operator entries. It fingerprints the approved
dataset before and after the cycle and never records automatic decisions.

Safe readiness states are:

- `AWAITING_CAPTURE`
- `CAPTURE_IN_PROGRESS`
- `MANUAL_DECISIONS_REQUIRED`
- `READY_TO_APPLY`

## Phase 2 — Approved sample ingestion

Edit only:

- `operator_decision`
- `rejection_reason`
- `operator_notes`

When readiness is `READY_TO_APPLY`, run:

```powershell
python -m src.project_cli finalize-first-real-dataset-ingestion
```

The command requires the Step 009.9 fingerprinted plan, delegates all writes to
the controlled transactional application layer, and audits the approval log,
annotations, manifest, processed files, views, categories, and remaining queue.

A failed post-application audit restores the complete pre-application snapshot.

If all 20 planned images are approved and all ten physical parts have approved
`front` and `detail` views, readiness becomes `FIRST_BATCH_INGESTED`.

If one or more captures are rejected, valid approved samples remain ingested
and readiness becomes `RECAPTURE_REQUIRED`.

## Verification

```powershell
python -m src.project_cli verify-step-010
```
