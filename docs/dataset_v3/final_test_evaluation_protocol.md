# Dataset V3 final locked-test evaluation protocol

The selected Dataset V3 model and notebook are frozen. A separate,
machine-readable authorization permits exactly one final evaluation of the
frozen checkpoint.

This protocol is committed and verified before locked-test image access.

## Frozen inputs

- Model: `torch_multimodal_dataset_v3`
- Checkpoint:
  `results/dataset_v3/models/torch_multimodal_dataset_v3_state.pt`
- Checkpoint SHA-256:
  `bce8a98fc96140294f3043fd845e1a7b6f491c67869a6c775877cd6ca1aa2a17`
- Text dimension: 342
- TF-IDF fingerprint:
  `aad4f36d127ea8eb06cc3e407c1873bf93ee24576e136129f837eb2f32eab5ac`
- Relation protocol fingerprint:
  `db89b8c2aa5b94e9eec2913d80d17d0b5ebfbbd90a0e5e763706bd03b5fb4cc7`

The vectorizer is reconstructed from the locked training relation text with
the original training parameters. It is never fitted on validation or test
text.

## Test relations

Each of the 80 locked images receives six relation rows:

- two `MATCH`
- two `PARTIAL_MATCH`
- two `MISMATCH`

The result is 480 rows, balanced at 160 rows per relation label. Mismatch
targets use a deterministic SHA-256 mapping fixed in the protocol code.

Two final-test text templates are used. Their 16 category captions have zero
exact overlap with train descriptions and zero exact overlap with validation
descriptions. Every caption has non-zero features under the frozen training
vectorizer.

## Preparation-snapshot correction

The external preparation snapshot excluded locked-test image bytes. It did,
however, include test metadata through the full image manifest and public
test-lock JSON. The protocol records this explicitly.

Metadata review did not consume the evaluation because no locked-test image
was opened and no test prediction or metric was produced.

## One-time execution guard

Final execution requires an explicit command and the exact authorization
SHA-256. Before any test access, the evaluator verifies the frozen selection,
authorization, protocol, checkpoint, notebook, code hashes, vectorizer, and
model interface.

At the point where repository test metadata access begins, the execution
state records the authorization as consumed. No automatic rerun is allowed
after that point.

The evaluator then verifies all 80 image hashes, performs one inference run,
and writes:

- test image manifest
- generated relation rows
- row-level predictions
- accuracy and macro F1 with image-group bootstrap intervals
- confusion matrix
- per-category metrics
- authorization-consumption record
- execution state
- environment record

Test results are for final reporting and error analysis only. They may not be
used for model, threshold, preprocessing, caption, relation, or checkpoint
changes.

## Status at this checkpoint

- Test evaluation authorized: yes
- Maximum evaluations: 1
- Completed evaluations: 0
- Authorization consumed: no
- Repository test manifest read by the evaluator: no
- Locked-test images read: no
- Test evaluation executed: no
