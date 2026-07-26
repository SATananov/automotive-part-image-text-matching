from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LABELS = ("MATCH", "MISMATCH", "PARTIAL_MATCH")
COLUMNS = (
    "sample_id",
    "image_id",
    "part_group_id",
    "image_path",
    "part_family",
    "part_category",
    "description",
    "label",
    "source",
)


def load_split(name: str) -> pd.DataFrame:
    """Load train or validation data. The test split stays locked."""
    if name not in {"train", "validation"}:
        raise ValueError("Only train and validation are available in normal runs.")
    path = DATA_DIR / f"{name}.csv"
    data = pd.read_csv(path)
    if tuple(data.columns) != COLUMNS:
        raise ValueError(f"Unexpected columns in {path.name}.")
    if set(data["label"]) != set(LABELS):
        raise ValueError(f"Unexpected labels in {path.name}.")
    return data


def check_split(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, int]:
    """Return overlap counts. All of them should be zero."""
    return {
        "group_overlap": len(set(train["part_group_id"]) & set(validation["part_group_id"])),
        "image_id_overlap": len(set(train["image_id"]) & set(validation["image_id"])),
        "image_path_overlap": len(set(train["image_path"]) & set(validation["image_path"])),
    }


def load_images(data: pd.DataFrame, size: tuple[int, int] = (24, 24)) -> np.ndarray:
    """Load RGB images as a NumPy array."""
    cache: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for relative_path in data["image_path"].astype(str):
        if relative_path not in cache:
            image_path = PROJECT_ROOT / relative_path
            with Image.open(image_path) as image:
                cache[relative_path] = np.asarray(
                    image.convert("RGB").resize(size, Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
        rows.append(cache[relative_path])
    return np.stack(rows)


def encode_labels(labels: pd.Series) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    return np.asarray([label_to_index[label] for label in labels], dtype=np.int32)
