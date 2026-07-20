# First Batch Capture Execution and Live Progress

This workflow turns the prepared first-batch capture plan into a safe local
operator cycle. It is designed for repeated use while photographs are being
made. The command imports available descriptive capture files, stages valid
originals, refreshes the operator worksheet, and rebuilds a live dashboard.
It never queues, approves, rejects, or publishes a real-data sample.

## Capture cycle

Place one or more correctly named JPEG or PNG files in:

```text
data/real/capture_inbox/batch_001/
```

Run one safe execution cycle:

```powershell
python -m src.project_cli run-first-real-batch-capture-session
```

The cycle performs these operations in order:

1. Validate the planned filenames and local inbox.
2. Copy valid source bytes into immutable local originals storage.
3. Normalize imported originals into temporary staging JPEG files.
4. Rebuild the capture-session worksheet and review queue draft.
5. Update the runtime-only progress CSV, JSON, Markdown, HTML dashboard, and
   execution journal.
6. Prove that the live queue, approval log, annotations, manifest, and tracked
   operational snapshots were not changed.

If import, normalization, duplicate detection, staging, or downstream review
fails, all originals and staging changes made by that cycle are restored.
The inbox files remain untouched so the operator can correct the problem.

## Runtime isolation

Live execution outputs are stored under:

```text
data/real/runtime/first_batch_capture/
```

The runtime directory is ignored by Git. Repeated progress refreshes therefore
do not make the repository dirty. Important runtime files include:

```text
live_dashboard.html
live_progress.csv
execution_status.json
execution_summary.md
execution_journal.csv
capture_session.csv
capture_inventory.csv
review_queue_draft.csv
```

The journal appends one row per execution cycle with the result, readiness,
newly imported and staged counts, progress percentage, and immutability checks.

## Read-only refresh

To refresh capture and pipeline visibility without importing or staging files,
run:

```powershell
python -m src.project_cli refresh-first-real-batch-live-progress
```

Open the local dashboard in a browser:

```text
data/real/runtime/first_batch_capture/live_dashboard.html
```

The refresh command may update only files inside the ignored runtime directory.
It must report both `live_dataset_unchanged: PASS` and
`tracked_outputs_unchanged: PASS`.

## Operator results

The execution result is one of:

- `NO_CAPTURE_FILES` — no planned capture is currently available;
- `NO_NEW_CHANGES` — captures were already imported or staged;
- `PROGRESS_UPDATED` — at least one new original or staging file was created;
- `READY_FOR_MANUAL_REVIEW` — staged candidates are ready for operator review;
- `BATCH_APPROVED` — all planned photographs are already in the approved
  dataset;
- `ROLLED_BACK` — the cycle failed and local originals/staging were restored.

A successful capture cycle does not mean that a sample is approved. Manual
visual review, explicit queue decisions, Step 009.1 review, and transactional
apply remain mandatory.

## Verification

Run:

```powershell
python -m src.project_cli verify-capture-execution
```

The verifier checks runtime isolation, CLI registration, semantic filenames,
execution journal schema, dashboard availability, rollback markers, current
execution status, and repository-safe output behavior.
