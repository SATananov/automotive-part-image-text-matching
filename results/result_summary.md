# Generated Validation Result Summary

This file is generated from the saved Dataset V2 prediction and metric artifacts. Do not edit it manually.

- Environment: Python `3.13.5`, PyTorch `2.10.0+cpu`
- Evaluation: 120 paired rows from 20 independent real validation images
- Training: 100 real and 50 synthetic images
- Locked test split used: no

| Model | Accuracy | Macro F1 | Correct |
|---|---:|---:|---:|
| Real-only multimodal CNN | 0.5083 | 0.5034 | 61/120 |
| Image + text Logistic Regression | 0.3333 | 0.1667 | 40/120 |
| Real + synthetic multimodal CNN | 0.4583 | 0.4251 | 55/120 |

The exact_image_group_sign_flip image-group paired comparison for the real-only neural model versus the non-neural multimodal baseline gives p = `0.072876` over 20 independent images.
The real-only model's grouped-bootstrap 95% accuracy interval is `[0.3417, 0.6667]`.

The result is a development-set comparison, not proof of general superiority. The final Wikimedia test images remain sealed.
