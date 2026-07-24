# Step 011.1 — Deep Learning Fundamentals Experimental Suite

- Status: **PASS**
- Readiness: `FUNDAMENTALS_EXPERIMENTAL_SUITE_COMPLETE_TEST_LOCKED`
- Base commit: `58d6236`
- Exercise problems completed: **10/10**
- Model training performed: **true**
- Frozen exam model changed: **false**
- Test split used: **false**
- Final test evaluation authorized: **false**

## Main findings

- Optimizer champion: `Adam` at `0.0003`; stability Macro F1 mean `0.2963` and standard deviation `0.0147`.
- Best capacity variant: `medium` with validation Macro F1 `0.3058` and `88619` parameters.
- Best architecture variant: `pretrained_mobilenet_v2` with validation Macro F1 `0.3303`.
- Best preprocessing variant: `grayscale_24_seq12` with validation Macro F1 `0.3126`.
- Pretrained MobileNetV2 probe: `COMPLETED`. Frozen ImageNet MobileNetV2 feature extractor.

## Exercise coverage

| ID | Evidence | Status |
|---|---|---|
| FND-001 | Dataset dimensions, balance, pixels, text lengths, examples | COMPLETED |
| FND-002 | Batch shapes, dtypes, alignment, deterministic shuffle policy | COMPLETED |
| FND-003 | One-hidden-layer baseline, gradients, weights, probability contract | COMPLETED |
| FND-004 | Deliberate one-batch overfit with learning curves | COMPLETED |
| FND-005 | Correct train/validation loop and validation no-update audit | COMPLETED |
| FND-006 | SGD/RMSprop/Adam/AdamW, LR grid, schedule, early stopping | COMPLETED |
| FND-007 | Small/medium/large capacity and probability tracking | COMPLETED |
| FND-008 | L2, dropout, batch norm, schedule, skip connection, CNN, pretrained probe | COMPLETED |
| FND-009 | Resolution, sequence length, and grayscale preprocessing | COMPLETED |
| FND-010 | Nine safe controlled-failure diagnostics | COMPLETED |

## Controlled failures

- `unscaled_images` — OBSERVED: Validation Macro F1=0.1667 versus the correct-loop reference 0.2886; train loss=1.0987544059753418.
- `excessive_learning_rate` — OBSERVED: Validation Macro F1=0.1667 versus the correct-loop reference 0.2886; train loss=1.13113272190094.
- `tiny_learning_rate` — OBSERVED: Validation Macro F1=0.2558 versus the correct-loop reference 0.2886; train loss=1.1205341815948486.
- `excessive_dropout` — OBSERVED: Validation Macro F1=0.1667 versus the correct-loop reference 0.2886; train loss=1.0986475944519043.
- `misaligned_train_labels` — OBSERVED: Validation Macro F1=0.2668 versus the correct-loop reference 0.2886; train loss=1.101723551750183.
- `sigmoid_activation` — NO_STRONG_SIGNATURE: Validation Macro F1=0.2402 versus the correct-loop reference 0.2886; train loss=1.1439261436462402.
- `deep_sigmoid_gradient_probe` — OBSERVED: Deep sigmoid networks can produce strongly uneven gradient norms; the measured first/last kernel ratio is reported without forcing a result.
- `missing_optimizer_step_probe` — OBSERVED: Repeated forward passes without an optimizer update left all weights unchanged and did not create systematic learning.
- `validation_training_blocked` — BLOCKED_BY_DESIGN: The suite exposes fixed train and validation arrays; every fit call uses train inputs for updates and validation inputs only through validation_data.

## Interpretation policy

These experiments demonstrate course concepts and produce validation-only comparisons. They do not replace the Step 010.8 frozen final model, do not authorize test access, and must not be presented as held-out test performance.
