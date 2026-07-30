# Dataset V3 one-time final-test authorization

The Dataset V3 model selection was frozen at commit
`7f8f8c11c0e4a339eb8df3485bd6bbdcad3d3de1` before any locked-test
evaluation.

This checkpoint authorizes exactly one final evaluation of the already
selected checkpoint:

`results/dataset_v3/models/torch_multimodal_dataset_v3_state.pt`

The selected model, checkpoint, notebook, preprocessing, caption rules,
relation rules, and all other model-selection decisions remain frozen.

## Authorization scope

- One final locked-test evaluation only
- Frozen checkpoint only
- No additional training
- No hyperparameter or threshold changes
- No checkpoint replacement
- No post-test tuning
- Test metrics may be used only for final reporting and error analysis
- The authorization is consumed after the single evaluation

At this authorization checkpoint, the test manifest and images are still
unread and the final evaluation is not yet executed.

## Machine-readable evidence

The authorization is stored in:

`data/manifests/dataset_v3/dataset_v3_final_test_authorization.json`

Its SHA-256 checksum is stored beside it in:

`data/manifests/dataset_v3/dataset_v3_final_test_authorization.sha256.txt`
