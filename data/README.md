# Data

The task has three labels:

- `MATCH`: the image and text describe the same part category;
- `PARTIAL_MATCH`: the two categories are from the same automotive system;
- `MISMATCH`: the categories are from different systems.

Each physical part has a `part_group_id`. The same image is paired with one text for each label, and all three rows stay in the same split.

| File | Samples | Part groups |
|---|---:|---:|
| `train.csv` | 180 | 60 |
| `validation.csv` | 60 | 20 |
| `test.csv` | 60 | 20 |

Half of the images are simple synthetic drawings used as development data. They are marked with `source=generated`, are not presented as real photographs, and are evaluated separately because some of them are visually similar across splits.

The other half are open-license photographs from Wikimedia Commons. Their authors, source pages, licenses, and file hashes are listed in `licenses.csv`.

The descriptions use a small fixed vocabulary. There are 14 unique descriptions, and the validation descriptions also occur in training. This is recorded as a limitation of the text-only experiment.

The test split is present but locked. It is not read by the notebook or by the normal training command. `test_lock.json` stores its exact SHA-256 so accidental changes can be detected.
