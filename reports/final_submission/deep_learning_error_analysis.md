# Deep Learning Error Analysis and Failure Diagnostics

- Validation samples: **60**
- Errors: **35**
- Error rate: **58.33%**
- High-confidence errors at threshold 0.60: **0**

## Dominant confusion structure

- `PARTIAL_MATCH->MATCH`: 10
- `PARTIAL_MATCH->MISMATCH`: 10
- `MATCH->MISMATCH`: 10
- `MISMATCH->MATCH`: 4
- `MISMATCH->PARTIAL_MATCH`: 1

The intermediate PARTIAL_MATCH class accounts for 20 errors. The model frequently collapses it toward one of the two extremes, which is consistent with the semantic difficulty of related but non-identical parts.

## Domain shift

- Generated-image error rate: **50.00%**
- Real open-license image error rate: **66.67%**

The higher real-image error rate is consistent with uncontrolled backgrounds, lighting, scale, perspective, and partial occlusion.

## Controlled failure experiments

The Fundamentals suite intentionally tested unscaled images, excessive and tiny learning rates, excessive dropout, misaligned labels, deep sigmoid gradients, and a missing optimizer update. The repository reports measured signatures and prevention rules rather than hiding failed configurations.

## Explainability boundary

The Vision suite reports an automated foreground-proxy alignment rate of **75.00%**, but no human plausible-region score is claimed. Occlusion is treated as model diagnostics, not proof of human-like reasoning.

## Safety and evaluation boundary

This Step 011.4 layer reads committed validation reports only. It performs no training, does not open the locked test CSV, does not authorize final evaluation, and does not change the production model.
