# Strict Evaluation Protocol — Dataset V2

## Independent unit

Every validation image appears in six dependent relation pairs. The actual row and image counts are read from the generated CSV and manifest. The **image**, not the row, is therefore the unit for uncertainty and model comparison.

## Development data

- train: 50 original real images + the balanced imported Dataset V2 train subset + 50 synthetic images;
- validation: 10 original real images + the balanced imported Dataset V2 validation subset;
- sealed test: 10 original Wikimedia images;
- alternate known views share an `object_group_id` and may not cross splits.

Validation is used for early stopping and model comparison. Every validation result is explicitly described as development evidence and may contain model-selection bias.

## Grouped bootstrap

Accuracy and macro-F1 intervals resample complete `image_id` groups with replacement. All six rows belonging to an image remain together.

## Paired randomization

For two models:

1. align predictions by `sample_id`;
2. calculate the difference in correct rows within each image;
3. randomly swap the model labels for complete images;
4. compare the absolute randomized total with the observed absolute total;
5. report a two-sided image-group p-value.

Exact enumeration is used for at most 20 groups. When the generated Dataset V2 validation set contains more than 20 images, the project uses 100,000 deterministic Monte Carlo sign-flip assignments with seed 42 and the plus-one correction `(extreme + 1)/(repeats + 1)`.

## Multiple comparisons

The pre-specified main comparison is the real-only multimodal CNN against the non-neural image-plus-text Logistic Regression baseline. The real-plus-synthetic comparison is a secondary ablation. Rankings of all eight models are descriptive.

## Locked test

The test remains locked. It may be opened once only after freezing dataset identities, preprocessing, architectures, hyperparameters, seeds, model-selection rule, metrics, and statistical procedure. No subsequent tuning may use the test result.

## Interpretation rules

- report paired rows and independent images separately;
- do not convert more text pairs into claims of more visual evidence;
- do not interpret `p > 0.05` as proof of equality;
- do not interpret `p < 0.05` as automatic practical superiority;
- report confidence intervals, effect size, class behaviour, and limitations together;
- call current results validation/development results;
- preserve the sealed test while methodology changes.
