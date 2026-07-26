# Teacher entry point

## Open this notebook

The only canonical notebook for the current exam submission is:

- Repository path: `exam/01_focused_deep_learning_project.ipynb`
- GitHub URL: https://github.com/SATananov/automotive-part-image-text-matching/blob/main/exam/01_focused_deep_learning_project.ipynb

The notebook presents one research question, one frozen retained model result,
grouped leakage protection, concrete successes and errors, limitations, and the
reproduction boundary.

## Current evidence boundary

- Primary prediction artifact: `reports/integrated_training/keras_multimodal/validation_predictions.csv`
- Validation samples: 60
- Independent physical-part groups: 20
- Validation accuracy: 0.5333
- Validation macro F1: 0.5208
- Correct predictions: 32
- Errors: 28
- Test split used: no
- Final test evaluation authorized: no
- Production final model changed: no

## Verify locally

```powershell
python -m src.project_cli verify-final-delivery
python -m jupyter notebook exam/01_focused_deep_learning_project.ipynb
```

The larger notebooks under `notebooks/` remain historical or supplementary
evidence. They are not alternative current submission entry points.
