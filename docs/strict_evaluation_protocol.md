# Evaluation Protocol — Dataset V2

## Independent unit

Each validation image creates six related image-text rows. The image, not the row, is the independent unit used for confidence intervals and model comparison.

## Development splits

- training: 100 real images and 50 synthetic images;
- validation: 20 real images;
- locked test: 10 original Wikimedia images.

Validation is used for early stopping and model comparison. For this reason, validation results are development results and may contain model-selection bias.

## Grouped bootstrap

Accuracy and macro-F1 intervals resample complete `image_id` groups. All six rows from one image stay together.

## Paired comparison

For two models, the project:

1. aligns predictions by `sample_id`;
2. counts the correctness difference inside each image;
3. swaps model labels for complete image groups;
4. compares randomized totals with the observed total;
5. reports a two-sided image-group p-value.

Exact enumeration is used for at most 20 images. Above 20 images, the project uses **100,000 deterministic Monte Carlo** sign-flip assignments with seed 42 and a plus-one correction.

The main planned comparison is the real-only multimodal CNN against image + text Logistic Regression. The real + synthetic comparison is a secondary ablation. The full ranking of eight models is descriptive.

## Locked test

The test remains locked while the dataset, preprocessing, models, hyperparameters, seeds, metrics, and statistical method are still being developed. It should be evaluated once only after all decisions are final.

## Interpretation

- report paired rows and independent images separately;
- do not treat extra text pairs as extra independent photographs;
- do not treat `p > 0.05` as proof that models are equal;
- do not treat `p < 0.05` as automatic practical superiority;
- report confidence intervals, class results, and limitations together;
- call the current results validation or development results.
