# Strict Evaluation Protocol

## Purpose

This protocol prevents the 30 validation rows from being misrepresented as 30 independent observations. Each of the ten validation images appears in three dependent image-text pairs, one per relation label. The independent experimental unit is therefore the **image**.

## Development data

- Training contains 50 real and 50 synthetic images.
- Validation contains 10 different real images and is used for early stopping and model comparison.
- Test contains 10 further real images and remains locked.
- Alternate views of a known physical object or photographic series share one `object_group_id` and may not cross splits.

The current validation result is a development-set result. Reusing the same small validation split for early stopping and model comparison can introduce selection bias. This is reported as a limitation rather than hidden.

## Uncertainty interval

Accuracy and macro F1 confidence intervals are produced with a grouped bootstrap. Complete image groups are sampled with replacement, so all three dependent rows from an image remain together.

## Paired model comparison

Two models are compared using an exact image-group sign-flip randomization test:

1. align both models on the same validation `sample_id` values;
2. compute the difference in correct predictions within each image group;
3. enumerate all `2^G` swaps of the model labels for the `G` independent images;
4. compare the absolute randomized total difference with the observed absolute difference;
5. report the exact two-sided proportion as the p-value.

For the current ten validation images, the calculation enumerates all 1,024 assignments. Individual rows are never treated as independent permutations.

The p-value describes the fixed development comparison only. It does not remove model-selection bias, compensate for the small number of images, or establish production-level generalization.

## Multiple comparisons

Eight models are shown for transparency, but the main pre-specified comparison is:

- real-only multimodal CNN;
- non-neural image-plus-text Logistic Regression baseline.

Other comparisons, including the synthetic-data ablation, are interpreted as secondary development analyses. No broad superiority claim is based on ranking eight models on ten images.

## Locked test split

The test remains locked: its CSV is protected by a stored SHA-256 lock and is not parsed by normal training, audit, notebook, or verification code. A final test evaluation is permissible only after all of the following are frozen:

- dataset and split identities;
- preprocessing;
- architecture;
- hyperparameters;
- random seeds;
- model-selection rule;
- reported metrics and statistical procedure.

After opening the test split once, no further tuning may use the test result. Until that point, all reported scores must be labelled as validation or development results.

## Interpretation rules

- Report paired rows and independent images separately.
- Do not describe 300 train rows as 300 independent images.
- Do not treat a p-value above 0.05 as proof of equality.
- Do not treat a p-value below 0.05 as proof of practical or general superiority.
- State that the dataset is small and domain-specific.
- Keep the test split locked while methodology is still changing.
