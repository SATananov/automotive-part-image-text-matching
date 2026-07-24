# VIS-002 — Frozen architecture backbones

- Step: `011.3A`
- Status: **DEFERRED_CONTROLLED_GATE**
- Exercise problem: 2 of 9
- Test split used: `false`
- Final test evaluation authorized: `false`
- Production final model changed: `false`

## Evidence

- [`reports/course_coverage/vision/pretrained_backbone_gate.json`](../../../../reports/course_coverage/vision/pretrained_backbone_gate.json)

## Requirement

Compare two or three frozen backbones, including convolutional and transformer families, with constant splits, resizing, metrics, and dense heads.

## Safety boundary

Only committed train and validation splits are in scope. The locked test split remains unopened.
