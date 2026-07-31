# Final submission presentation

## Primary public files

- `README.md` is the repository landing page.
- `project_final.ipynb` is the main executed final report.
- `project_v3.ipynb` is the byte-preserved pre-test development and model-selection notebook.
- `docs/dataset_v3/final_test_results.md` is the concise final-test evidence note.

## Why there are two Dataset V3 notebooks

`project_v3.ipynb` was frozen before the locked test was opened. Its SHA-256 is recorded in the final-selection lock, so it must not be edited.

`project_final.ipynb` is a reporting-only notebook created after the authorized one-time test evaluation. It reads the saved development and final-test artifacts. It does not train a model, load locked images, or run inference.

## GitHub presentation

After the final Dataset V3 commit is pushed and fast-forwarded to `main`, keep `main` as the default branch.

Suggested repository description:

> Compact multimodal PyTorch project for automotive-part image-text relation classification with grouped evaluation and a locked final test.

Suggested topics:

- `deep-learning`
- `pytorch`
- `multimodal-learning`
- `computer-vision`
- `natural-language-processing`
- `image-text-matching`
- `jupyter-notebook`
- `automotive`

## Final submission link

Submit the repository root URL after confirming that GitHub opens `main` and immediately shows the Dataset V3 README and `project_final.ipynb`.
