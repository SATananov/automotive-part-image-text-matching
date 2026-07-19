# Step 010.2 — External Training Readiness

- Status: **PASS**
- Readiness: **READY_FOR_TRAINING**
- Approved external images: **50**
- Generated external samples: **150**
- Test locked: **true**
- Test evaluation permitted: **false**

## External split

| Split | Samples | Groups |
|---|---:|---:|
| Train | 90 | 30 |
| Validation | 30 | 10 |
| Test | 30 | 10 |

## Integrated split

| Split | Samples | Groups |
|---|---:|---:|
| Train | 180 | 60 |
| Validation | 60 | 20 |
| Test | 60 | 20 |

## Training policy

- Model training may use only `integrated_train.csv`.
- Model selection may use only `integrated_validation.csv`.
- `integrated_test.csv` remains fingerprinted and locked.
- Step 010.2 performs structural validation only; it does not train or evaluate a model.
