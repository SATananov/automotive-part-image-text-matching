# Dataset V3 final test results

The frozen `torch_multimodal_dataset_v3` checkpoint was evaluated exactly
once on the locked Dataset V3 test split.

## Final metrics

- independent test images: 80;
- relation rows: 480;
- correct predictions: 354;
- accuracy: 0.7375;
- macro F1: 0.7382299830;
- grouped accuracy 95% interval: [0.6604166667, 0.8145833333];
- grouped macro-F1 95% interval: [0.6536213360, 0.8101866315];
- image-category auxiliary accuracy: 0.675;
- text-category auxiliary accuracy: 1.0.

The test split contains ten images from each of the eight categories. Every
image contributes two rows for each relation label. The saved relation table
contains 160 MATCH, 160 MISMATCH, and 160 PARTIAL_MATCH rows.

## One-time authorization

The evaluator was invoked once. The authorization is consumed and records
one completed evaluation. The test manifest and locked images were read.
Further tuning, checkpoint replacement, and post-test model selection are
not permitted. The results are used only for final reporting and error
analysis.

## Metadata reconciliation

The evaluator completed successfully and wrote all nine expected result
artifacts. Its relation summary in `test_metrics.json` omitted the
`relation_protocol_fingerprint` field. An external wrapper described the
missing field as a wrong fingerprint and stopped after evaluation.

The original result artifacts are preserved byte for byte. The missing
fingerprint is reconciled outside those files by:

1. computing the fingerprint from the frozen protocol source;
2. rebuilding all 480 relation rows from the saved 80-image manifest;
3. requiring exact equality with the saved relation table;
4. rerunning the relation, leakage, balance, and grouped-split checks;
5. recomputing metrics, grouped intervals, the confusion matrix, and all
   eight per-category rows from the saved predictions.

The derived fingerprint is:

`db89b8c2aa5b94e9eec2913d80d17d0b5ebfbbd90a0e5e763706bd03b5fb4cc7`

The original `test_metrics.json` SHA-256 remains:

`5c4c2ea793779a5ea894e5124afd5a8f48c2df8fa0d7266e22436f144f084818`

No result file was rewritten and inference was not run again.

## Verification

Run:

```bash
python -m src.verify_dataset_v3_final_test_results
python -m pytest -q tests/test_dataset_v3_final_test_results.py
```

The deterministic results manifest is stored at
`data/manifests/dataset_v3/dataset_v3_final_test_results.json`.
