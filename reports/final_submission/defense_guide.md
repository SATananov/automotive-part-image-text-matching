# Final Defense Guide

## Thirty-second project explanation

The project evaluates whether an automotive-part photograph and a short description are a full match, a related same-system partial match, or a mismatch. It compares six primary model families, uses group-isolated validation, and keeps the final test split locked until explicit authorization.

## Questions the lecturer is likely to ask

### Why is PARTIAL_MATCH difficult?
It is a semantic boundary class: brake disc versus brake pad is related but not identical. The reference error analysis records 20 PARTIAL_MATCH errors, split evenly toward MATCH and MISMATCH.

### Why group by part_group_id?
Different descriptions of the same physical image must not cross train and validation boundaries. Grouping prevents identity leakage.

### Why macro F1?
Macro F1 gives equal weight to all three relationship labels and exposes a model that ignores the difficult intermediate class.

### Why is the test split still locked?
Model selection and error analysis must not adapt to final-test outcomes. The repository records a one-shot authorization gate.

### Why retain the reference model?
A controlled candidate improved over the multi-seed reference aggregate but did not pass the predefined gate against the frozen incumbent. The decision therefore remained REFERENCE_RETAINED.

### What do the Deep Learning failures teach?
Unscaled images, excessive learning rate, excessive dropout, and misaligned labels produce recognizable validation or gradient signatures. A missing optimizer step leaves weights unchanged. These experiments demonstrate pipeline understanding rather than only score chasing.

### Why can a classical text baseline outperform LSTM or Transformer runs?
The text alone often names only one part. It does not contain the image context needed to determine MATCH versus PARTIAL_MATCH versus MISMATCH, and the dataset is intentionally small.

### What is the strongest limitation?
The dataset is small and the real open-license images show a higher error rate than generated images, indicating domain shift. Pretrained and human-annotation experiments remain behind explicit gates.
