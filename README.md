# Automotive Part Image–Text Matching

This project explores whether a deep learning model can determine if an image of an automotive part matches a short text description.

The task is treated as a three-class classification problem:

- `MATCH` - the image and description refer to the same automotive part
- `PARTIAL_MATCH` - the general category is correct, but some detail is different
- `MISMATCH` - the image and description refer to different automotive parts

The project will compare text-only, image-only, and multimodal models.

## Project structure

- `data/` - dataset files and images
- `notebooks/` - experiments and model training
- `src/` - reusable Python code
- `models/` - saved model files
- `reports/` - evaluation results
- `app/` - demonstration application
- `tests/` - automated tests
