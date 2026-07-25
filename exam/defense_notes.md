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
validation split. Its advantage is evidence that both modalities help, not
proof that every individual prediction uses both modalities correctly.

## Show concrete examples

The notebook contains examples where the multimodal model is correct while one
or both unimodal models fail. It also shows difficult errors, especially
`PARTIAL_MATCH`, where the description is related to the image but is not the
same part category.

## Be direct about weaknesses

The validation set is small, real images produce more errors than generated
images, the model predicts no `PARTIAL_MATCH` cases in the retained confusion
matrix, and the results do not establish deployment readiness or calibration.

## Evaluation boundary

The test split remains locked. The exam-facing build does not train models and
does not change the retained production recipe.
