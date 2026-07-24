# VIS-006 — Controlled fine-tuning

- Step: `011.3A`
- Status: **DEFERRED_CONTROLLED_GATE**
- Exercise problem: 6 of 9
- Test split used: `false`
- Final test evaluation authorized: `false`
- Production final model changed: `false`

## Evidence

- [`reports/course_coverage/vision/fine_tuning_gate.json`](../../../../reports/course_coverage/vision/fine_tuning_gate.json)

## Requirement

Gradually compare frozen features, trainable head, final block, final third, and limited full fine-tuning with controlled optimization.

## Safety boundary

Only committed train and validation splits are in scope. The locked test split remains unopened.
