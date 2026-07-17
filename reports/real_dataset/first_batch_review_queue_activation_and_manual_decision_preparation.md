# First Batch Review Queue Activation and Manual Decision Preparation

This workflow is the controlled bridge between the generated first-batch review queue draft and the existing real-sample approval workflow.

## Queue activation

Run:

```powershell
python -m src.project_cli activate-first-real-batch-review-queue
```

The command selects the latest runtime review queue draft when available and otherwise uses the committed first-batch draft snapshot. It accepts only `pending` rows that belong to the canonical `batch_001` plan, match the planned metadata, have unique intake and staging identifiers, and pass the existing real-sample review validation.

Activation is transactional and idempotent. Exact rows already present in `sample_intake.csv` are not duplicated. A conflicting live row, previously processed intake ID, invalid staged image, metadata mismatch, duplicate content, or approval-state change blocks the operation. No approval or rejection decision is created automatically.

The only live file that activation may change is:

```text
data/real/annotations/sample_intake.csv
```

Part groups, image annotations, the approval log, approved image manifest, and processed image directory must remain byte-for-byte unchanged.

## Manual decision workbook

Run:

```powershell
python -m src.project_cli prepare-first-real-batch-manual-decisions
```

The command creates a runtime-only workbook under:

```text
data/real/runtime/first_batch_review/manual_decision_workbook.csv
```

The operator may edit only:

- `operator_decision`: blank, `approved`, or `rejected`;
- `rejection_reason`: mandatory for a rejected item;
- `operator_notes`: optional visual-review notes.

Regenerating the workbook preserves valid operator entries by `intake_id`, refreshes image metrics and review warnings, and validates decision completeness. It never writes decisions to `sample_intake.csv` and never applies an image to the approved dataset.

## Runtime outputs

All activation summaries, decision workbooks, guides, and validation reports are local runtime state under:

```text
data/real/runtime/first_batch_review/
```

The directory is ignored by Git. The tracked repository remains stable while an operator reviews real photographs.

## Required separation

The supported sequence is:

```text
capture execution
→ review-ready draft
→ pending queue activation
→ manual decision preparation
→ explicit decision recording
→ transactional apply
```

Queue activation and decision preparation are deliberately separate from decision recording and dataset application.
