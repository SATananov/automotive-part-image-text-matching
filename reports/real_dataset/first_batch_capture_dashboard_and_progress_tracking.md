# First Batch Capture Dashboard and Progress Tracking

The first real-data batch uses one reproducible dashboard snapshot to track all
20 planned photographs through the controlled pipeline. The dashboard is an
operator aid and an audit view. It does not import, stage, queue, approve,
reject, delete, or rewrite any photograph or live dataset record.

## Build the dashboard

From the repository root, with the project virtual environment active, run:

```powershell
python -m src.project_cli build-first-real-batch-dashboard
```

The command writes four semantic outputs:

```text
data/real/processed/first_batch_capture_progress.csv
reports/real_dataset/first_batch_capture_dashboard.html
reports/real_dataset/first_batch_capture_dashboard.json
reports/real_dataset/first_batch_capture_progress_summary.md
```

Open `first_batch_capture_dashboard.html` in a browser for the visual operator
view. It is a self-contained local HTML file with no external JavaScript,
tracking service, network request, or remote asset.

## Tracked pipeline

Every planned `front` and `detail` photograph has one current pipeline stage:

```text
AWAITING_CAPTURE
CAPTURED
IMPORTED
STAGED
REVIEW_READY
QUEUED_FOR_DECISION
DECISION_RECORDED
APPROVED_DATASET
```

The progress CSV records the descriptive filename, physical part, view,
current stage, percentage, and exact next action. The HTML dashboard shows
summary cards, category progress bars, readiness, and the complete photograph
pipeline table.

## Readiness states

The batch-level readiness is derived only from observed local files and
committed workflow tables:

```text
AWAITING_CAPTURE
CAPTURE_SESSION_IN_PROGRESS
READY_FOR_LOCAL_IMPORT
READY_FOR_STAGING
STAGING_IN_PROGRESS
READY_FOR_MANUAL_REVIEW
REVIEW_IN_PROGRESS
BATCH_APPROVED
CAPTURE_DASHBOARD_BLOCKED
```

A dashboard state never replaces the mandatory commands for local import,
staging, manual review, explicit decision, or transactional approval.

## Safety and reproducibility

Before and after scanning, the command fingerprints the capture file map,
local inbox, immutable originals, staging, capture inventory, queue draft,
live queue, approval log, approved manifest, and real annotations. Only the
four dashboard outputs may change. `live_state_unchanged` must remain `PASS`.

The HTML renderer escapes filenames, categories, stages, actions, and errors
before insertion. Dashboard output is deterministic for an unchanged project
state and contains no current timestamp.

Verify the implementation with:

```powershell
python -m src.project_cli verify-step-009-6
```
