# Experiment Execution Policy

- Step: **011.0**
- Planned experiments: **29**
- Current execution status: all entries are `PLANNED`.

## Lifecycle

`PLANNED → READY → RUNNING → COMPLETED → RETAINED or REJECTED`

`FAILED_DIAGNOSTIC` is used when a deliberately small or broken configuration provides useful failure evidence but must never participate in model selection.

## Resource gates

- **TIER_0** — Structural, data-contract, and no-training checks. Operator approval: no.
- **TIER_1** — Short diagnostic or classical baseline on train and validation only. Operator approval: no.
- **TIER_2** — Controlled comparison with fixed splits and bounded configurations. Operator approval: no.
- **TIER_3** — Retained-model candidate or broader ablation after diagnostic gates pass. Operator approval: yes.
- **TIER_4** — Optional pretrained download, transfer learning, or expensive fine-tuning proof of concept. Operator approval: yes.

Tier 3 and Tier 4 runs may begin only after cheaper structural and diagnostic gates pass. Pretrained assets require a recorded source, license, architecture name, and exact revision.

## Evidence contract

Every executed experiment must save its resolved configuration, metrics, timing, parameter counts, test-lock assertion, and the report or notebook cells listed in the registry.
No result may be entered in the registry before its artifacts exist and pass verification.

## Human annotation safeguard

VIS-007 requires at least two genuine independent human annotators. Confidence must be recorded before adjudication. Synthetic annotators, inferred agreement, or duplicated single-author labels must not be presented as human agreement evidence.

## Failure experiments

Intentional faults use copied train-only data in isolated configurations. They must not overwrite canonical metadata, validation artifacts, model-selection tables, or any locked-test file.
