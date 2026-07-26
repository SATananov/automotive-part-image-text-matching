# Data

The task has three labels:

- `MATCH`: the image and text describe the same part category;
- `PARTIAL_MATCH`: the two categories are from the same automotive system;
- `MISMATCH`: the categories are from different systems.

Each physical part has a `part_group_id`. All three text pairs for that part stay in the same split.

| File | Samples | Part groups |
|---|---:|---:|
| `train.csv` | 180 | 60 |
| `validation.csv` | 60 | 20 |
| `test.csv` | 60 | 20 |

Half of the images are simple generated drawings. The other half are open-license Wikimedia Commons images. Their authors and licenses are listed in `licenses.csv`.

The test split is present but locked. It is not read by the notebook or by the normal training command.
