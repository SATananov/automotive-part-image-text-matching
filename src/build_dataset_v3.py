from __future__ import annotations

import hashlib
import json
from itertools import permutations

import pandas as pd

from src.captions_v3 import caption_count, render_caption
from src.data import COLUMNS
from src.data_v3 import (
    CATEGORIES,
    FAMILIES,
    MANIFEST_DIR,
    PARTIAL_TARGET,
    RELATION_SUMMARY,
    TRAIN_RELATIONS,
    VALIDATION_RELATIONS,
    check_v3_split_overlap,
    load_v3_image_manifest,
)

PAIRS_PER_LABEL = 2
ROWS_PER_IMAGE = PAIRS_PER_LABEL * 3

_VALID_MISMATCH_PERMUTATIONS = tuple(
    candidate
    for candidate in permutations(CATEGORIES)
    if all(FAMILIES[source] != FAMILIES[target] for source, target in zip(CATEGORIES, candidate))
)

if not _VALID_MISMATCH_PERMUTATIONS:
    raise AssertionError("No valid Dataset V3 mismatch permutations were found.")


def mismatch_mapping(split: str, image_rank: int, pair_index: int) -> dict[str, str]:
    if split not in {"train", "validation"}:
        raise ValueError("Dataset V3 mismatch mappings are development-only.")
    token = f"dataset-v3|{split}|{image_rank}|{pair_index}".encode("utf-8")
    index = int(hashlib.sha256(token).hexdigest(), 16) % len(_VALID_MISMATCH_PERMUTATIONS)
    targets = _VALID_MISMATCH_PERMUTATIONS[index]
    return dict(zip(CATEGORIES, targets))


def relation_rows(image_manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in {"train", "validation"}:
        raise ValueError("Dataset V3 relations are generated only for train and validation.")
    rows: list[dict[str, str]] = []
    per_category = image_manifest["project_category"].value_counts()
    if len(per_category) != len(CATEGORIES) or per_category.nunique() != 1:
        raise ValueError(f"Balanced Dataset V3 image counts are required in {split}.")

    for category in CATEGORIES:
        selected = image_manifest[image_manifest["project_category"].eq(category)].sort_values(
            ["split_rank_within_category", "candidate_id"], kind="stable"
        )
        for image_rank, image in enumerate(selected.itertuples(index=False)):
            for pair_index in range(PAIRS_PER_LABEL):
                mismatch_target = mismatch_mapping(split, image_rank, pair_index)[category]
                relations = (
                    ("MATCH", category),
                    ("PARTIAL_MATCH", PARTIAL_TARGET[category]),
                    ("MISMATCH", mismatch_target),
                )
                for label, text_category in relations:
                    variant = (image_rank + pair_index) % caption_count(split)
                    rows.append(
                        {
                            "sample_id": (
                                f"v3_{split}_{image.candidate_id}_{label.lower()}_{pair_index + 1}"
                            ),
                            "image_id": str(image.candidate_id),
                            "part_group_id": str(image.image_group_id),
                            "object_group_id": str(image.image_group_id),
                            "image_path": str(image.repository_relative_path),
                            "part_family": FAMILIES[category],
                            "part_category": category,
                            "text_category": text_category,
                            "description": render_caption(split, text_category, variant),
                            "label": label,
                            "source": "dataset_v3",
                        }
                    )
    return pd.DataFrame(rows, columns=COLUMNS)


def validate_relation_design(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_images: pd.DataFrame,
    validation_images: pd.DataFrame,
) -> None:
    expected_images = {"train": 480, "validation": 80}
    frames = {
        "train": (train, train_images),
        "validation": (validation, validation_images),
    }
    for split, (relations, images) in frames.items():
        expected = expected_images[split]
        if len(images) != expected or len(relations) != expected * ROWS_PER_IMAGE:
            raise ValueError(f"Unexpected Dataset V3 {split} size.")
        if relations["image_id"].nunique() != expected:
            raise ValueError(f"Unexpected Dataset V3 image count in {split}.")
        if relations["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate Dataset V3 sample_id in {split}.")
        per_image = relations.groupby(["image_id", "label"]).size().unstack(fill_value=0)
        if not (per_image == PAIRS_PER_LABEL).all().all():
            raise ValueError(f"Every Dataset V3 image must have two rows per label in {split}.")
        label_counts = relations["label"].value_counts()
        if label_counts.nunique() != 1:
            raise ValueError(f"Dataset V3 labels are not balanced in {split}.")
        table = pd.crosstab(relations["text_category"], relations["label"])
        if table.shape != (len(CATEGORIES), 3) or table.nunique().max() != 1:
            raise ValueError(f"Dataset V3 text-category shortcut imbalance in {split}.")
        if set(relations["source"]) != {"dataset_v3"}:
            raise ValueError(f"Unexpected Dataset V3 source in {split}.")
        if relations["image_path"].str.replace("\\", "/", regex=False).str.startswith(
            "data/locked_test/"
        ).any():
            raise ValueError(f"Locked-test path entered Dataset V3 {split} relations.")

        match = relations[relations["label"].eq("MATCH")]
        partial = relations[relations["label"].eq("PARTIAL_MATCH")]
        mismatch = relations[relations["label"].eq("MISMATCH")]
        if not match["part_category"].eq(match["text_category"]).all():
            raise ValueError(f"Invalid MATCH relation in Dataset V3 {split}.")
        if not partial.apply(
            lambda row: row["part_category"] != row["text_category"]
            and FAMILIES[row["part_category"]] == FAMILIES[row["text_category"]],
            axis=1,
        ).all():
            raise ValueError(f"Invalid PARTIAL_MATCH relation in Dataset V3 {split}.")
        if not mismatch.apply(
            lambda row: FAMILIES[row["part_category"]] != FAMILIES[row["text_category"]],
            axis=1,
        ).all():
            raise ValueError(f"Invalid MISMATCH relation in Dataset V3 {split}.")

    overlaps = check_v3_split_overlap(train, validation)
    if any(overlaps.values()):
        raise ValueError(f"Dataset V3 development split overlap: {overlaps}")
    if set(train["description"]) & set(validation["description"]):
        raise ValueError("Exact Dataset V3 descriptions overlap across development splits.")


def write_v3_development_relations() -> dict[str, object]:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    train_images = load_v3_image_manifest("train", verify_files=True)
    validation_images = load_v3_image_manifest("validation", verify_files=True)
    train = relation_rows(train_images, "train")
    validation = relation_rows(validation_images, "validation")
    validate_relation_design(train, validation, train_images, validation_images)
    train.to_csv(TRAIN_RELATIONS, index=False, lineterminator="\n")
    validation.to_csv(VALIDATION_RELATIONS, index=False, lineterminator="\n")

    summary = {
        "status": "PASS_DATASET_V3_DEVELOPMENT_RELATIONS_READY",
        "dataset_version": "3.0-development",
        "categories": list(CATEGORIES),
        "broad_families": FAMILIES,
        "partial_target": PARTIAL_TARGET,
        "train_images": int(train["image_id"].nunique()),
        "validation_images": int(validation["image_id"].nunique()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "rows_per_image": ROWS_PER_IMAGE,
        "pairs_per_label_per_image": PAIRS_PER_LABEL,
        "split_overlap": check_v3_split_overlap(train, validation),
        "exact_description_overlap": len(set(train["description"]) & set(validation["description"])),
        "test_manifest_read": False,
        "test_images_read": False,
        "test_evaluation_executed": False,
        "train_relations": TRAIN_RELATIONS.relative_to(MANIFEST_DIR.parents[2]).as_posix(),
        "validation_relations": VALIDATION_RELATIONS.relative_to(MANIFEST_DIR.parents[2]).as_posix(),
    }
    RELATION_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(write_v3_development_relations(), indent=2))


if __name__ == "__main__":
    main()
