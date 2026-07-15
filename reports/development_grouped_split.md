# Development Grouped Split

**Status:** PASS

- Split strategy: category_balanced_deterministic
- Grouping column: `part_group_id`

## Split summary

| Split | Samples | Images | Part groups |
|---|---:|---:|---:|
| Train | 90 | 30 | 30 |
| Validation | 30 | 10 | 10 |
| Test | 30 | 10 | 10 |

## Label distribution

| Split | MATCH | PARTIAL_MATCH | MISMATCH |
|---|---:|---:|---:|
| Train | 30 | 30 | 30 |
| Validation | 10 | 10 | 10 |
| Test | 10 | 10 | 10 |

## Leakage check

- Train and validation group overlap: 0
- Train and test group overlap: 0
- Validation and test group overlap: 0

All rows belonging to the same physical part remain in one subset.

This split is intended for development and pipeline testing.
