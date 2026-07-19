# Step 010.2 — External Dataset Integration

- Status: **PASS**
- Readiness: **READY_FOR_TRAINING_VALIDATION**
- Approved external images: **50**
- External samples: **150**
- Grouping column: `part_group_id`
- Test evaluation permitted: **no**

## External grouped split

| Split | Samples | Images | Part groups |
|---|---:|---:|---:|
| Train | 90 | 30 | 30 |
| Validation | 30 | 10 | 10 |
| Test | 30 | 10 | 10 |

## Integrated development + external split

| Split | Samples | Images | Part groups |
|---|---:|---:|---:|
| Train | 180 | 60 | 60 |
| Validation | 60 | 20 | 20 |
| Test | 60 | 20 | 20 |

## Leakage and test-lock policy

- Train and validation group overlap: 0
- Train and test group overlap: 0
- Validation and test group overlap: 0
- External groups use a dedicated `external_group_` namespace.
- Training-ready inputs include only integrated train and validation CSV files.
- External and integrated test CSV files are fingerprinted and locked.
- No model is trained or evaluated by Step 010.2.

The 29 rejected open-license candidates remain in the Step 010.1 audit workbook but are excluded from every integrated dataset.
