# Dataset V3 final model-selection lock

The Dataset V3 development workflow is frozen before any locked-test
evaluation.

## Frozen selection

- Selected model: `torch_multimodal_dataset_v3`
- Selected checkpoint:
  `results/dataset_v3/models/torch_multimodal_dataset_v3_state.pt`
- Selected report: `project_v3.ipynb`
- Selection basis: development validation only
- Development validation: 377/480 correct
- Validation accuracy: 0.7854166667
- Validation macro F1: 0.7872843193
- Independent validation image groups: 80
- Independent training-artifact audit: 76/76 PASS

## Methodological rule

After this lock is committed, no model architecture, weight, preprocessing
rule, caption rule, relation rule, hyperparameter, threshold, checkpoint,
or model-selection decision may be changed on the basis of locked-test
performance.

The locked test is still not authorized for evaluation at this checkpoint.
Its manifest and images remain unread by the model-development workflow.

A later one-time authorization must identify this exact lock and may only
evaluate the already selected checkpoint. The resulting test metrics must
be reported as final evaluation metrics and must not be used for additional
tuning.

## Machine-readable evidence

The canonical lock is stored in:

`data/manifests/dataset_v3/dataset_v3_final_selection_lock.json`

Its SHA-256 checksum is stored beside it in:

`data/manifests/dataset_v3/dataset_v3_final_selection_lock.sha256.txt`
