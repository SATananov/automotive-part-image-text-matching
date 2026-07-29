from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, check_split, load_split, unique_images

SIMILARITY_LIMIT = 0.995
DHASH_LIMIT = 2


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalized_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        vector = np.asarray(
            image.convert("L").resize((48, 48), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).reshape(-1)
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def difference_hash(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((9, 8), Image.Resampling.BILINEAR),
            dtype=np.int16,
        )
    return (pixels[:, 1:] > pixels[:, :-1]).reshape(-1)


def nearest_same_category_pairs(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Find the closest real training image for each real validation image."""
    train_images = unique_images(train[train["source"].ne("generated")])
    validation_images = unique_images(validation)
    rows: list[dict[str, object]] = []
    cache_gray: dict[str, np.ndarray] = {}
    cache_hash: dict[str, np.ndarray] = {}

    def features(relative_path: str) -> tuple[np.ndarray, np.ndarray]:
        if relative_path not in cache_gray:
            path = PROJECT_ROOT / relative_path
            cache_gray[relative_path] = normalized_gray(path)
            cache_hash[relative_path] = difference_hash(path)
        return cache_gray[relative_path], cache_hash[relative_path]

    for val in validation_images.itertuples(index=False):
        candidates = train_images[train_images["part_category"].eq(val.part_category)]
        val_vector, val_hash = features(val.image_path)
        best: dict[str, object] | None = None
        for candidate in candidates.itertuples(index=False):
            candidate_vector, candidate_hash = features(candidate.image_path)
            cosine = round(float(np.dot(candidate_vector, val_vector)), 8)
            hamming = int(np.sum(candidate_hash != val_hash))
            row = {
                "part_category": val.part_category,
                "validation_image_id": val.image_id,
                "validation_source": val.source,
                "validation_image_path": val.image_path,
                "nearest_train_image_id": candidate.image_id,
                "nearest_train_source": candidate.source,
                "nearest_train_image_path": candidate.image_path,
                "normalized_cosine_similarity": cosine,
                "dhash_hamming_distance": hamming,
            }
            if best is None or cosine > float(best["normalized_cosine_similarity"]):
                best = row
        if best is None:
            raise ValueError(f"No training images for {val.part_category}")
        rows.append(best)
    return pd.DataFrame(rows).sort_values(
        ["part_category", "validation_image_id"], kind="stable"
    ).reset_index(drop=True)


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


def test_lock_status() -> dict[str, object]:
    lock = json.loads((DATA_DIR / "test_lock.json").read_text(encoding="utf-8"))
    actual_hash = file_sha256(DATA_DIR / "test.csv")
    return {
        "test_locked": bool(lock["test_locked"]),
        "test_evaluation_permitted": bool(lock["test_evaluation_permitted"]),
        "test_rows": int(lock["test_rows"]),
        "test_images": int(lock["test_images"]),
        "expected_sha256": str(lock["test_sha256"]),
        "actual_sha256": actual_hash,
        "sha256_matches": str(lock["test_sha256"]) == actual_hash,
        "rows_parsed_by_audit": False,
    }


def license_status() -> dict[str, object]:
    wikimedia_licenses = pd.read_csv(DATA_DIR / "licenses.csv")
    dataset_v2_licenses = pd.read_csv(DATA_DIR / "dataset_v2_manifest.csv")
    manifest = pd.read_csv(DATA_DIR / "image_manifest.csv")
    wikimedia = manifest[manifest["source"].eq("wikimedia")]
    dataset_v2 = manifest[manifest["source"].eq("dataset_v2")]

    wikimedia_hash_matches = []
    for row in wikimedia_licenses.itertuples(index=False):
        path = PROJECT_ROOT / row.local_path
        wikimedia_hash_matches.append(path.is_file() and file_sha256(path) == row.sha256)

    dataset_v2_hash_matches = []
    for row in dataset_v2_licenses.itertuples(index=False):
        path = PROJECT_ROOT / row.local_path
        dataset_v2_hash_matches.append(path.is_file() and file_sha256(path) == row.sha256)

    wikimedia_status = {
        "license_rows": len(wikimedia_licenses),
        "images": len(wikimedia),
        "all_images_covered": set(wikimedia_licenses["local_path"]) == set(wikimedia["image_path"]),
        "unique_description_urls": not wikimedia_licenses["description_url"].duplicated().any(),
        "unique_file_hashes": not wikimedia_licenses["sha256"].duplicated().any(),
        "all_hashes_match": bool(all(wikimedia_hash_matches)),
    }
    dataset_v2_status = {
        "license_rows": len(dataset_v2_licenses),
        "images": len(dataset_v2),
        "providers": dataset_v2_licenses["provider"].value_counts().sort_index().to_dict(),
        "recorded_licenses": sorted(dataset_v2_licenses["license_short_name"].unique().tolist()),
        "all_provenance_fields_present": bool(
            dataset_v2_licenses[
                [
                    "provider",
                    "source_dataset",
                    "dataset_url",
                    "source_title",
                    "description_url",
                    "author",
                    "credit",
                    "license_short_name",
                    "license_url",
                ]
            ]
            .fillna("")
            .astype(str)
            .apply(lambda column: column.str.strip().ne(""))
            .all()
            .all()
        ),
        "all_images_covered": set(dataset_v2_licenses["local_path"]) == set(dataset_v2["image_path"]),
        "unique_file_hashes": not dataset_v2_licenses["sha256"].duplicated().any(),
        "all_hashes_match": bool(all(dataset_v2_hash_matches)),
    }
    return {
        "wikimedia": wikimedia_status,
        "dataset_v2": dataset_v2_status,
        "license_rows": len(wikimedia_licenses) + len(dataset_v2_licenses),
        "all_license_hashes_match": (
            wikimedia_status["all_hashes_match"]
            and dataset_v2_status["all_hashes_match"]
            and wikimedia_status["all_images_covered"]
            and dataset_v2_status["all_images_covered"]
            and dataset_v2_status["all_provenance_fields_present"]
        ),
    }


def run_audit() -> dict[str, object]:
    RESULTS_DIR.mkdir(exist_ok=True)
    train = load_split("train")
    validation = load_split("validation")
    overlaps = check_split(train, validation)

    train_images = unique_images(train)
    validation_images = unique_images(validation)
    train_hashes = {file_sha256(PROJECT_ROOT / path) for path in train_images["image_path"]}
    validation_hashes = {
        file_sha256(PROJECT_ROOT / path) for path in validation_images["image_path"]
    }

    nearest = nearest_same_category_pairs(train, validation)
    nearest.to_csv(
        RESULTS_DIR / "nearest_real_image_pairs.csv", index=False, lineterminator="\n"
    )
    suspicious = nearest[
        (nearest["normalized_cosine_similarity"] >= SIMILARITY_LIMIT)
        | (nearest["dhash_hamming_distance"] <= DHASH_LIMIT)
    ]
    suspicious.to_csv(
        RESULTS_DIR / "similar_image_pairs.csv", index=False, lineterminator="\n"
    )

    shortcut_rows = [
        shortcut_score(train, validation, ["source"]),
        shortcut_score(train, validation, ["part_category"]),
        shortcut_score(train, validation, ["text_category"]),
        shortcut_score(train, validation, ["source", "part_category"]),
    ]
    pd.DataFrame(shortcut_rows).to_csv(
        RESULTS_DIR / "shortcut_baselines.csv", index=False, lineterminator="\n"
    )

    train_descriptions = set(train["description"])
    validation_descriptions = set(validation["description"])
    manifest = pd.read_csv(DATA_DIR / "image_manifest.csv")
    source_split_counts = (
        manifest.groupby(["split", "source"]).size().rename("images").reset_index()
    )

    summary = {
        "dataset_version": "2.0",
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_images": train["image_id"].nunique(),
        "validation_images": validation["image_id"].nunique(),
        "train_real_images": int(
            train_images[train_images["source"].ne("generated")]["image_id"].nunique()
        ),
        "train_synthetic_images": int(
            train_images[train_images["source"].eq("generated")]["image_id"].nunique()
        ),
        "validation_real_images": int(validation_images["image_id"].nunique()),
        "validation_synthetic_images": 0,
        "source_split_image_counts": source_split_counts.to_dict(orient="records"),
        "overlap": overlaps,
        "exact_cross_split_image_hash_overlap": len(train_hashes & validation_hashes),
        "suspicious_real_image_pairs": len(suspicious),
        "similarity_limit": SIMILARITY_LIMIT,
        "dhash_limit": DHASH_LIMIT,
        "maximum_same_category_similarity": round(
            float(nearest["normalized_cosine_similarity"].max()), 8
        ),
        "minimum_nearest_dhash_distance": int(nearest["dhash_hamming_distance"].min()),
        "label_counts_train": train["label"].value_counts().sort_index().to_dict(),
        "label_counts_validation": validation["label"].value_counts().sort_index().to_dict(),
        "unique_descriptions_train": len(train_descriptions),
        "unique_descriptions_validation": len(validation_descriptions),
        "exact_description_overlap": len(train_descriptions & validation_descriptions),
        "validation_rows_with_exact_seen_description": int(
            validation["description"].isin(train_descriptions).sum()
        ),
        "text_category_label_table_train": pd.crosstab(
            train["text_category"], train["label"]
        ).to_dict(),
        "text_category_label_table_validation": pd.crosstab(
            validation["text_category"], validation["label"]
        ).to_dict(),
        "image_only_relation_ceiling_accuracy": image_only_ceiling(validation),
        "shortcut_baselines": shortcut_rows,
        "licenses": license_status(),
        "test_lock": test_lock_status(),
        "warnings": [
            "Each image contributes six dependent rows; uncertainty and paired tests use complete image groups.",
            f"Validation contains {int(validation_images['image_id'].nunique())} independent real images and is used for model comparison, so the locked test remains the only untouched final holdout.",
            "The hybrid imported subset is deterministically selected after exact-hash, dHash, and normalized-cosine duplicate screening, with per-image provenance and license records.",
            "Synthetic drawings remain training-only and are retained solely for a transparent ablation.",
        ],
        "test_split_used": False,
    }
    (RESULTS_DIR / "data_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(run_audit(), indent=2))


if __name__ == "__main__":
    main()
