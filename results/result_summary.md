# Generated Validation Result Summary

This file is generated from the saved prediction and metric artifacts. Do not edit it manually.

- Environment: Python `3.13.13`, PyTorch `2.13.0+cpu`
- Evaluation: 30 paired rows from 10 independent real validation images
- Locked test split used: no

| Model | Accuracy | Macro F1 | Correct |
|---|---:|---:|---:|
| Real-only multimodal CNN | 0.5333 | 0.4977 | 16/30 |
| Image + text Logistic Regression | 0.3333 | 0.1667 | 10/30 |
| Real + synthetic multimodal CNN | 0.3000 | 0.2290 | 9/30 |

The exact paired two-sided p-value for the real-only neural model versus the non-neural multimodal baseline is `0.109375`.
The real-only model's grouped-bootstrap 95% accuracy interval is `[0.4000, 0.6667]`.

The result is an observed development-set comparison, not proof of general superiority. The current synthetic templates did not improve transfer to real validation images.
