from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data import COLUMNS, DATA_DIR, LABELS, PROJECT_ROOT

MANIFEST_DIR = DATA_DIR / "manifests" / "dataset_v3"
TRAIN_IMAGE_MANIFEST = MANIFEST_DIR / "dataset_v3_train_images.csv"
VALIDATION_IMAGE_MANIFEST = MANIFEST_DIR / "dataset_v3_validation_images.csv"
TRAIN_RELATIONS = MANIFEST_DIR / "dataset_v3_train_relations.csv"
VALIDATION_RELATIONS = MANIFEST_DIR / "dataset_v3_validation_relations.csv"
RELATION_SUMMARY = MANIFEST_DIR / "dataset_v3_relation_summary.json"
DEVELOPMENT_AUDIT = MANIFEST_DIR / "dataset_v3_development_audit.json"
SPLIT_SUMMARY = MANIFEST_DIR / "dataset_v3_split_summary.json"
CURATION_SUMMARY = MANIFEST_DIR / "dataset_v3_curation_summary.json"

CATEGORIES = (
    "alternator",
    "brake_disc",
    "brake_pad",
    "coil_spring",
    "headlight",
    "oil_filter",
    "starter",
    "taillight",
)

FAMILIES = {
    "alternator": "engine_support",
    "oil_filter": "engine_support",
    "starter": "engine_support",
    "brake_disc": "chassis",
    "brake_pad": "chassis",
    "coil_spring": "chassis",
    "headlight": "lighting",
    "taillight": "lighting",
}

PARTIAL_TARGET = {
    "alternator": "starter",
    "starter": "oil_filter",
    "oil_filter": "alternator",
    "brake_disc": "brake_pad",
    "brake_pad": "coil_spring",
    "coil_spring": "brake_disc",
    "headlight": "taillight",
    "taillight": "headlight",
}

IMAGE_MANIFEST_COLUMNS = (
    "candidate_id",
    "project_category",
    "image_group_id",
    "split",
    "split_rank_within_category",
    "repository_relative_path",
    "sha256",
    "split_seed",
    "split_score",
    "source_dataset_slug",
    "source_dataset_root",
    "source_split_ignored",
    "test_locked",
    "split_policy",
)

EXPECTED_IMAGES_PER_CATEGORY = {
    "train": 60,
    "validation": 10,
}


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _development_manifest_path(split: str) -> Path:
    if split == "train":
        return TRAIN_IMAGE_MANIFEST
    if split == "validation":
        return VALIDATION_IMAGE_MANIFEST
    raise ValueError("Dataset V3 development code exposes only train and validation.")


def load_v3_image_manifest(split: str, *, verify_files: bool = True) -> pd.DataFrame:
    """Load one Dataset V3 development image manifest.

    The locked test manifest is deliberately not addressable through this function.
    """
    path = _development_manifest_path(split)
    data = pd.read_csv(path)
    if tuple(data.columns) != IMAGE_MANIFEST_COLUMNS:
        raise ValueError(f"Unexpected columns in {path.name}: {tuple(data.columns)}")
    if set(data["split"].astype(str)) != {split}:
        raise ValueError(f"Unexpected split values in {path.name}.")
    if set(data["project_category"].astype(str)) != set(CATEGORIES):
        raise ValueError(f"Unexpected categories in {path.name}.")
    per_category = data["project_category"].value_counts().sort_index()
    expected = EXPECTED_IMAGES_PER_CATEGORY[split]
    if len(per_category) != len(CATEGORIES) or not (per_category == expected).all():
        raise ValueError(f"Unexpected category balance in {path.name}: {per_category.to_dict()}")
    locked = data["test_locked"].astype(str).str.strip().str.lower()
    if not locked.isin({"false", "0"}).all():
        raise ValueError(f"Development manifest contains a locked-test row: {path.name}")
    normalized_paths = data["repository_relative_path"].astype(str).str.replace("\\", "/", regex=False)
    if normalized_paths.str.startswith("data/locked_test/").any():
        raise ValueError(f"Development manifest points into the locked-test area: {path.name}")
    for column in ("candidate_id", "image_group_id", "repository_relative_path", "sha256"):
        if data[column].duplicated().any():
            raise ValueError(f"Duplicate {column} in {path.name}.")
    if verify_files:
        for row in data.itertuples(index=False):
            image_path = PROJECT_ROOT / str(row.repository_relative_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing Dataset V3 development image: {image_path}")
            if file_sha256(image_path) != str(row.sha256):
                raise ValueError(f"Dataset V3 image hash mismatch: {row.candidate_id}")
    return data.sort_values(
        ["project_category", "split_rank_within_category", "candidate_id"],
        kind="stable",
    ).reset_index(drop=True)


def _relation_path(split: str) -> Path:
    if split == "train":
        return TRAIN_RELATIONS
    if split == "validation":
        return VALIDATION_RELATIONS
    raise ValueError("Dataset V3 relation loader exposes only train and validation.")


def load_v3_split(split: str) -> pd.DataFrame:
    """Load a generated Dataset V3 relation table without exposing locked test data."""
    path = _relation_path(split)
    data = pd.read_csv(path)
    if tuple(data.columns) != COLUMNS:
        raise ValueError(f"Unexpected columns in {path.name}: {tuple(data.columns)}")
    if set(data["label"].astype(str)) != set(LABELS):
        raise ValueError(f"Unexpected labels in {path.name}.")
    if set(data["part_category"].astype(str)) != set(CATEGORIES):
        raise ValueError(f"Unexpected image categories in {path.name}.")
    if set(data["text_category"].astype(str)) != set(CATEGORIES):
        raise ValueError(f"Unexpected text categories in {path.name}.")
    if set(data["source"].astype(str)) != {"dataset_v3"}:
        raise ValueError(f"Unexpected sources in {path.name}.")
    if data["image_path"].astype(str).str.replace("\\", "/", regex=False).str.startswith(
        "data/locked_test/"
    ).any():
        raise ValueError(f"Locked-test path found in {path.name}.")
    return data


def check_v3_split_overlap(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, int]:
    return {
        "part_group_overlap": len(set(train["part_group_id"]) & set(validation["part_group_id"])),
        "object_group_overlap": len(set(train["object_group_id"]) & set(validation["object_group_id"])),
        "image_id_overlap": len(set(train["image_id"]) & set(validation["image_id"])),
        "image_path_overlap": len(set(train["image_path"]) & set(validation["image_path"])),
    }
