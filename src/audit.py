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

from src.data import PROJECT_ROOT, check_split, load_split

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
    return pd.DataFrame(rows)


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


def run_audit() -> dict[str, object]:
    train = load_split("train")
    validation = load_split("validation")
    overlaps = check_split(train, validation)

    train_images = image_table(train)
    validation_images = image_table(validation)
    train_hashes = {file_sha256(PROJECT_ROOT / p) for p in train_images["image_path"]}
    validation_hashes = {file_sha256(PROJECT_ROOT / p) for p in validation_images["image_path"]}

    near_pairs = similarity_pairs(train, validation)
    near_pairs.to_csv(RESULTS_DIR / "similar_image_pairs.csv", index=False)

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
    pd.DataFrame(shortcut_rows).to_csv(RESULTS_DIR / "shortcut_baselines.csv", index=False)

    word_lengths = train.assign(words=train["description"].str.split().str.len()).groupby("label")["words"].mean()
    generated_pairs = near_pairs[near_pairs["source"].eq("generated")] if not near_pairs.empty else near_pairs
    real_pairs = near_pairs[near_pairs["source"].eq("wikimedia")] if not near_pairs.empty else near_pairs

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
        "mean_text_words_by_label": {key: float(value) for key, value in word_lengths.items()},
        "shortcut_baselines": shortcut_rows,
        "warning": (
            "The generated drawings contain visually similar train/validation pairs. "
            "The real-image validation subset is reported separately for this reason."
        ),
        "test_split_used": False,
    }
    (RESULTS_DIR / "data_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    summary = run_audit()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
