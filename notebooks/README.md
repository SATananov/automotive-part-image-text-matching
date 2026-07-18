# Jupyter notebooks

## Development experiment

Open the main presentation notebook from the repository root:

```powershell
python -m jupyter notebook notebooks/01_development_experiment.ipynb
```

Run the cells from top to bottom.

The notebook:

- reads the committed development metadata, train split, validation split and report artifacts;
- checks train/validation group isolation;
- shows one image paired with the three relationship labels;
- compares all baseline and neural validation results;
- displays the best-model confusion matrix, prediction examples and training curves;
- does not load or evaluate `development_test.csv`;
- does not retrain models unless `RUN_TRAINING = True` is set manually.

The real-dataset workflow remains separate from this development presentation.
