# Oral defense notes

## State the question first

Does a multimodal neural network that combines an automotive-part image and a short description classify their relationship better than image-only and text-only neural baselines?

## Explain the split before discussing accuracy

All rows that belong to one physical part share a `part_group_id` and remain in
one split. This prevents the model from seeing the same physical object during
training and validation under a different image-text pairing.

## Explain what the comparison shows

The multimodal model is compared with majority, classical text, classical
image, neural text-only, and neural image-only baselines on the same grouped
validation split. Its advantage is evidence consistent with a benefit from
combining modalities, not proof that every individual prediction causally uses
both inputs.

## Show concrete examples

The notebook contains examples where the multimodal model is correct while one
or both unimodal models fail. The error examples, confusion matrix, source
rates, and aggregate score all come from the same frozen
`keras_multimodal/validation_predictions.csv` artifact.

## Be direct about weaknesses

The frozen model makes 28 errors on 60 validation samples. It recovers 12 of 20
`PARTIAL_MATCH` cases, so that class is not a total failure. The weakest recall
is `MATCH` at 0.30: ten true matches are predicted as `MISMATCH`. Real images
produce a 53.3% error rate versus 40.0% for generated images. The results do
not establish deployment readiness or calibrated probabilities.

## Keep the historical experiment separate

The historical Step 010.4 controlled retraining produced 35 errors and a very
different confusion matrix. It remains useful as stability evidence, but it is
not the prediction set used for the primary 0.5333 / 0.5208 result.

## Evaluation boundary

The test split remains locked. The exam-facing build does not train models and
does not change the retained production recipe.
