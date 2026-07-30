from __future__ import annotations

import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_v3 import (
    CURATION_SUMMARY,
    DEVELOPMENT_AUDIT,
    FAMILIES,
    SPLIT_SUMMARY,
    check_v3_split_overlap,
    load_v3_image_manifest,
    load_v3_split,
)


def shortcut_score(train: pd.DataFrame, validation: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    model.fit(train[columns], train["label"])
    predicted = model.predict(validation[columns])
    return {
        "features": "+".join(columns),
        "accuracy": float(accuracy_score(validation["label"], predicted)),
        "macro_f1": float(
            f1_score(validation["label"], predicted, average="macro", zero_division=0)
        ),
    }


def image_only_ceiling(data: pd.DataFrame) -> float:
    best_per_image = data.groupby("image_id")["label"].value_counts().groupby(level=0).max()
    return float(best_per_image.sum() / len(data))


def run_v3_development_audit() -> dict[str, object]:
    train_images = load_v3_image_manifest("train", verify_files=True)
    validation_images = load_v3_image_manifest("validation", verify_files=True)
    train = load_v3_split("train")
    validation = load_v3_split("validation")
    overlaps = check_v3_split_overlap(train, validation)

    split_summary = json.loads(SPLIT_SUMMARY.read_text(encoding="utf-8-sig"))
    curation_summary = json.loads(CURATION_SUMMARY.read_text(encoding="utf-8-sig"))

    shortcut_rows = [
        shortcut_score(train, validation, ["source"]),
        shortcut_score(train, validation, ["part_category"]),
        shortcut_score(train, validation, ["text_category"]),
        shortcut_score(train, validation, ["source", "part_category"]),
    ]

    summary = {
        "status": "PASS_DATASET_V3_DEVELOPMENT_AUDIT",
        "dataset_version": "3.0-development",
        "train_images": int(train["image_id"].nunique()),
        "validation_images": int(validation["image_id"].nunique()),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_image_manifest_rows": len(train_images),
        "validation_image_manifest_rows": len(validation_images),
        "broad_families": FAMILIES,
        "overlap": overlaps,
        "exact_cross_split_image_hash_overlap": len(
            set(train_images["sha256"]) & set(validation_images["sha256"])
        ),
        "unique_descriptions_train": int(train["description"].nunique()),
        "unique_descriptions_validation": int(validation["description"].nunique()),
        "exact_description_overlap": len(set(train["description"]) & set(validation["description"])),
        "validation_rows_with_exact_seen_description": int(
            validation["description"].isin(set(train["description"])).sum()
        ),
        "label_counts_train": train["label"].value_counts().sort_index().to_dict(),
        "label_counts_validation": validation["label"].value_counts().sort_index().to_dict(),
        "shortcut_baselines": shortcut_rows,
        "image_only_relation_ceiling_accuracy": image_only_ceiling(validation),
        "curation_exact_duplicate_groups": curation_summary["exact_duplicate_groups"],
        "curation_near_duplicate_review_pairs": curation_summary[
            "near_duplicate_review_pairs"
        ],
        "curation_current_project_overlap_pairs": curation_summary[
            "current_project_review_pairs"
        ],
        "test_lock_status": split_summary["test_lock_status"],
        "test_lock_sha256": split_summary["test_lock_sha256"],
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "test_evaluation_executed": False,
        "warnings": [
            "Each image contributes six dependent rows; validation statistics must use complete image groups.",
            "PARTIAL_MATCH means a different part from the same broad vehicle subsystem: engine support, chassis, or lighting.",
            "Validation contains 80 independent curated images and is used for development model comparison.",
            "The locked 80-image test split remains unavailable to development loaders and audits.",
        ],
    }
    DEVELOPMENT_AUDIT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    print(json.dumps(run_v3_development_audit(), indent=2))


if __name__ == "__main__":
    main()
