# Development Grouped Split

**Status:** PASS

- Random state: 42
- Grouping column: `part_group_id`

## Split summary

| Split | Samples | Images | Part groups |
|---|---:|---:|---:|
| Train | 36 | 12 | 12 |
| Validation | 12 | 4 | 4 |
| Test | 12 | 4 | 4 |

## Label distribution

| Split | MATCH | PARTIAL_MATCH | MISMATCH |
|---|---:|---:|---:|
| Train | 12 | 12 | 12 |
| Validation | 4 | 4 | 4 |
| Test | 4 | 4 | 4 |

## Leakage check

- Train and validation group overlap: 0
- Train and test group overlap: 0
- Validation and test group overlap: 0

All rows belonging to the same physical part remain in one subset.

This split is intended for development and pipeline testing.
