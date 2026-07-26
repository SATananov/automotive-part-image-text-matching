from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import DATA_DIR, PROJECT_ROOT, check_split, load_split

RESULTS_DIR = PROJECT_ROOT / "results"
SIMILARITY_LIMIT = 0.99


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def small_gray_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).reshape(-1) / 255.0


def image_table(data: pd.DataFrame) -> pd.DataFrame:
    return data[["image_id", "image_path", "part_category", "source"]].drop_duplicates("image_id")


def similarity_pairs(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source",
        "train_image_id",
        "validation_image_id",
        "train_category",
        "validation_category",
        "cosine_similarity",
    ]
    rows: list[dict[str, object]] = []
    for source in sorted(set(train["source"]) & set(validation["source"])):
        left = image_table(train[train["source"].eq(source)])
        right = image_table(validation[validation["source"].eq(source)])
        left_features = np.stack([small_gray_image(PROJECT_ROOT / p) for p in left["image_path"]])
        right_features = np.stack([small_gray_image(PROJECT_ROOT / p) for p in right["image_path"]])
        scores = cosine_similarity(left_features, right_features)
        for i, left_row in enumerate(left.itertuples(index=False)):
            for j, right_row in enumerate(right.itertuples(index=False)):
                score = float(scores[i, j])
                if score >= SIMILARITY_LIMIT:
                    rows.append({
                        "source": source,
                        "train_image_id": left_row.image_id,
                        "validation_image_id": right_row.image_id,
                        "train_category": left_row.part_category,
                        "validation_category": right_row.part_category,
                        "cosine_similarity": score,
                    })
    return pd.DataFrame(rows, columns=columns)


def shortcut_score(train: pd.DataFrame, validation: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(train[columns], train["label"])
    predicted = model.predict(validation[columns])
    return {
        "features": "+".join(columns),
        "accuracy": accuracy_score(validation["label"], predicted),
        "macro_f1": f1_score(validation["label"], predicted, average="macro", zero_division=0),
    }


def description_category(text: str) -> str:
    value = str(text).lower().strip().rstrip(".")
    return value.removeprefix("automotive ").replace(" ", "_")


def image_only_ceiling(data: pd.DataFrame) -> float:
    best_per_image = data.groupby("image_id")["label"].value_counts().groupby(level=0).max()
    return float(best_per_image.sum() / len(data))


def test_lock_status() -> dict[str, object]:
    lock = json.loads((DATA_DIR / "test_lock.json").read_text(encoding="utf-8"))
    actual_hash = file_sha256(DATA_DIR / "test.csv")
    return {
        "test_locked": bool(lock["test_locked"]),
        "test_evaluation_permitted": bool(lock["test_evaluation_permitted"]),
        "expected_sha256": str(lock["test_sha256"]),
        "actual_sha256": actual_hash,
        "sha256_matches": str(lock["test_sha256"]) == actual_hash,
    }


def run_audit() -> dict[str, object]:
    RESULTS_DIR.mkdir(exist_ok=True)
    train = load_split("train")
    validation = load_split("validation")
    overlaps = check_split(train, validation)

    train_images = image_table(train)
    validation_images = image_table(validation)
    train_hashes = {file_sha256(PROJECT_ROOT / p) for p in train_images["image_path"]}
    validation_hashes = {file_sha256(PROJECT_ROOT / p) for p in validation_images["image_path"]}

    near_pairs = similarity_pairs(train, validation)
    near_pairs.to_csv(
        RESULTS_DIR / "similar_image_pairs.csv",
        index=False,
        lineterminator="\n",
    )

    train = train.copy()
    validation = validation.copy()
    train["description_category"] = train["description"].map(description_category)
    validation["description_category"] = validation["description"].map(description_category)

    shortcut_rows = [
        shortcut_score(train, validation, ["source"]),
        shortcut_score(train, validation, ["part_category"]),
        shortcut_score(train, validation, ["source", "part_category"]),
        shortcut_score(train, validation, ["description_category"]),
    ]
    pd.DataFrame(shortcut_rows).to_csv(
        RESULTS_DIR / "shortcut_baselines.csv",
        index=False,
        lineterminator="\n",
    )

    word_lengths = train.assign(words=train["description"].str.split().str.len()).groupby("label")["words"].mean()
    generated_pairs = near_pairs[near_pairs["source"].eq("generated")]
    real_pairs = near_pairs[near_pairs["source"].eq("wikimedia")]
    train_descriptions = set(train["description"])
    validation_descriptions = set(validation["description"])

    summary = {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_groups": train["part_group_id"].nunique(),
        "validation_groups": validation["part_group_id"].nunique(),
        "overlap": overlaps,
        "exact_cross_split_image_hash_overlap": len(train_hashes & validation_hashes),
        "label_counts_train": train["label"].value_counts().sort_index().to_dict(),
        "label_counts_validation": validation["label"].value_counts().sort_index().to_dict(),
        "generated_similar_pairs_at_0_99": len(generated_pairs),
        "wikimedia_similar_pairs_at_0_99": len(real_pairs),
        "unique_descriptions_train": len(train_descriptions),
        "unique_descriptions_validation": len(validation_descriptions),
        "validation_unique_descriptions_seen_in_train": len(
            validation_descriptions & train_descriptions
        ),
        "validation_rows_with_seen_description": int(
            validation["description"].isin(train_descriptions).sum()
        ),
        "image_only_relation_ceiling_accuracy": image_only_ceiling(validation),
        "mean_text_words_by_label": {key: float(value) for key, value in word_lengths.items()},
        "shortcut_baselines": shortcut_rows,
        "test_lock": test_lock_status(),
        "warnings": [
            (
                "The generated drawings contain visually similar train/validation pairs. "
                "The real-image validation subset is reported separately for this reason."
            ),
            (
                "All validation descriptions are also present in training. Text-only results "
                "therefore measure this fixed vocabulary, not general language understanding."
            ),
            (
                "Each image is paired once with every relation label. An image-only model cannot "
                "identify the relation without the text and is limited to one correct row per image."
            ),
        ],
        "test_split_used": False,
    }
    with (RESULTS_DIR / "data_audit.json").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    summary = run_audit()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
