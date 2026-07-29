from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.captions_v2 import caption_count, render_caption
from src.data import COLUMNS, DATA_DIR, PROJECT_ROOT

CATEGORIES = (
    "air_filter",
    "alternator",
    "brake_disc",
    "brake_pad",
    "coil_spring",
    "headlight",
    "oil_filter",
    "shock_absorber",
    "starter",
    "taillight",
)

FAMILIES = {
    "air_filter": "filtration",
    "oil_filter": "filtration",
    "alternator": "electrical",
    "starter": "electrical",
    "brake_disc": "braking",
    "brake_pad": "braking",
    "coil_spring": "suspension",
    "shock_absorber": "suspension",
    "headlight": "lighting",
    "taillight": "lighting",
}

PARTIAL_TARGET = {
    "air_filter": "oil_filter",
    "oil_filter": "air_filter",
    "alternator": "starter",
    "starter": "alternator",
    "brake_disc": "brake_pad",
    "brake_pad": "brake_disc",
    "coil_spring": "shock_absorber",
    "shock_absorber": "coil_spring",
    "headlight": "taillight",
    "taillight": "headlight",
}

PAIRS_PER_LABEL = 2
ROWS_PER_IMAGE = PAIRS_PER_LABEL * 3
BASE_SOURCE_SPLIT_IMAGES = {
    ("train", "generated"): 50,
    ("train", "wikimedia"): 50,
    ("validation", "wikimedia"): 10,
    ("test", "wikimedia"): 10,
}

# Manually reviewed same-object or same-photographic-series groups. All of these
# remain inside the same split, preventing alternate views from crossing holdouts.
OBJECT_SERIES = {
    "commons_air_filter_475880": "wikimedia_air_filter_opel_astra_series",
    "commons_air_filter_475881": "wikimedia_air_filter_opel_astra_series",
    "commons_alternator_10186168": "wikimedia_alternator_101861_series",
    "commons_alternator_10186172": "wikimedia_alternator_101861_series",
    "commons_oil_filter_1158213": "wikimedia_oil_filter_115821_series",
    "commons_oil_filter_1158214": "wikimedia_oil_filter_115821_series",
    "commons_shock_absorber_1196179": "wikimedia_shock_absorber_11961_series",
    "commons_shock_absorber_1196183": "wikimedia_shock_absorber_11961_series",
    "commons_shock_absorber_1196186": "wikimedia_shock_absorber_11961_series",
    "commons_starter_1650041": "wikimedia_starter_165004_series",
    "commons_starter_1650046": "wikimedia_starter_165004_series",
    "commons_taillight_18026985": "wikimedia_taillight_1802_series",
    "commons_taillight_18027695": "wikimedia_taillight_1802_series",
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_dataset_v2_manifest() -> pd.DataFrame:
    path = DATA_DIR / "dataset_v2_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Dataset V2 import is missing. Run python -m src.import_dataset_v2 --force first."
        )
    manifest = pd.read_csv(path)
    required = {
        "asset_id",
        "part_category",
        "assigned_split",
        "object_group_id",
        "local_path",
        "provider",
        "source_dataset",
        "dataset_url",
        "source_relative_path",
        "source_original_split",
        "source_title",
        "description_url",
        "author",
        "credit",
        "license_short_name",
        "license_url",
        "source_sha256",
        "sha256",
        "dhash",
        "modifications",
    }
    if set(manifest.columns) != required:
        raise ValueError(f"Unexpected dataset_v2_manifest.csv columns: {tuple(manifest.columns)}")
    return manifest


def expected_source_split_images() -> dict[tuple[str, str], int]:
    manifest = _load_dataset_v2_manifest()
    counts = manifest.groupby("assigned_split").size().to_dict()
    expected = dict(BASE_SOURCE_SPLIT_IMAGES)
    expected[("train", "dataset_v2")] = int(counts.get("train", 0))
    expected[("validation", "dataset_v2")] = int(counts.get("validation", 0))
    return expected


def expected_images_by_split() -> dict[str, int]:
    source_counts = expected_source_split_images()
    return {
        split: sum(
            count
            for (candidate_split, _source), count in source_counts.items()
            if candidate_split == split
        )
        for split in ("train", "validation", "test")
    }


def _dataset_v2_lookup() -> dict[str, dict[str, str]]:
    manifest = _load_dataset_v2_manifest()
    return {
        str(row.local_path): {
            "asset_id": str(row.asset_id),
            "part_category": str(row.part_category),
            "split": str(row.assigned_split),
            "object_group_id": str(row.object_group_id),
            "sha256": str(row.sha256),
        }
        for row in manifest.itertuples(index=False)
    }


def source_split_and_category(
    path: Path,
    dataset_v2_lookup: dict[str, dict[str, str]],
) -> tuple[str, str, str]:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    if "generated" in path.parts:
        category = path.stem.rsplit("_", 2)[0]
        return "generated", "train", category
    if "wikimedia" in path.parts:
        stem = path.stem
        if stem.endswith("_new_validation"):
            split = "validation"
        elif stem.endswith("_new_test"):
            split = "test"
        else:
            split = "train"
        return "wikimedia", split, path.parent.name
    if "dataset_v2" in path.parts:
        metadata = dataset_v2_lookup.get(relative)
        if metadata is None:
            raise ValueError(f"Imported V2 image is missing from manifest: {relative}")
        return "dataset_v2", metadata["split"], metadata["part_category"]
    raise ValueError(f"Unknown image source: {path}")


def identity(
    path: Path,
    source: str,
    category: str,
    dataset_v2_lookup: dict[str, dict[str, str]],
) -> tuple[str, str, str]:
    stem = path.stem
    if source == "generated":
        image_id = stem
        part_group_id = stem.rsplit("_", 1)[0]
        object_group_id = f"synthetic_family_{category}"
        return image_id, part_group_id, object_group_id

    if source == "dataset_v2":
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        metadata = dataset_v2_lookup[relative]
        image_id = metadata["asset_id"]
        return image_id, f"external_group_{image_id}", metadata["object_group_id"]

    suffix = stem.removeprefix("commons_")
    image_id = f"external_image_{suffix}"
    part_group_id = f"external_group_{suffix}"
    object_group_id = OBJECT_SERIES.get(stem, f"wikimedia_object_{suffix}")
    return image_id, part_group_id, object_group_id


def image_inventory() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    dataset_v2_lookup = _dataset_v2_lookup()
    source_dirs = (
        DATA_DIR / "images" / "generated",
        DATA_DIR / "images" / "wikimedia",
        DATA_DIR / "images" / "dataset_v2",
    )
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {source_dir}")
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            source, split, category = source_split_and_category(path, dataset_v2_lookup)
            if category not in FAMILIES:
                raise ValueError(f"Unknown part category for {path}")
            image_id, part_group_id, object_group_id = identity(
                path, source, category, dataset_v2_lookup
            )
            digest = sha256(path)
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if source == "dataset_v2" and digest != dataset_v2_lookup[relative]["sha256"]:
                raise ValueError(f"Dataset V2 manifest hash mismatch: {relative}")
            rows.append(
                {
                    "image_id": image_id,
                    "part_group_id": part_group_id,
                    "object_group_id": object_group_id,
                    "image_path": relative,
                    "part_family": FAMILIES[category],
                    "part_category": category,
                    "source": source,
                    "split": split,
                    "sha256": digest,
                }
            )
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise ValueError("No images found")
    for column in ("image_id", "image_path", "sha256"):
        if inventory[column].duplicated().any():
            duplicate = inventory.loc[inventory[column].duplicated(), column].iloc[0]
            raise ValueError(f"Duplicate {column}: {duplicate}")
    return inventory.sort_values(
        ["split", "part_category", "source", "image_id"], kind="stable"
    ).reset_index(drop=True)


# Adjacent entries belong to the same semantic family. Offsets 2..8 are
# therefore guaranteed to point to a different family for every source entry.
# Applying one fixed offset to all ten categories is a permutation, which lets
# every image rank use two different mismatch offsets while preserving exact
# global text-category balance.
MISMATCH_RING = (
    "air_filter",
    "oil_filter",
    "alternator",
    "starter",
    "brake_disc",
    "brake_pad",
    "coil_spring",
    "shock_absorber",
    "headlight",
    "taillight",
)
MISMATCH_OFFSETS = (2, 3, 4, 5, 6, 7, 8)


def mismatch_target(category: str, image_rank: int, pair_index: int) -> str:
    source_index = MISMATCH_RING.index(category)
    # Two distinct offsets per image; the offsets rotate between image ranks so
    # every source category is paired with a broad set of unrelated categories.
    offset_position = (2 * image_rank + 3 * pair_index) % len(MISMATCH_OFFSETS)
    offset = MISMATCH_OFFSETS[offset_position]
    target = MISMATCH_RING[(source_index + offset) % len(MISMATCH_RING)]
    if FAMILIES[target] == FAMILIES[category]:
        raise AssertionError(f"Invalid mismatch family assignment: {category} -> {target}")
    return target


def relation_rows(inventory: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    selected = inventory[inventory["split"].eq(split)]
    image_counts = selected.groupby("part_category").size()
    if len(image_counts) != len(CATEGORIES) or image_counts.nunique() != 1:
        raise ValueError(f"Balanced category image counts are required before pairing in {split}")

    for category, category_images in selected.groupby("part_category", sort=True):
        ordered = category_images.sort_values(["source", "image_id"], kind="stable").reset_index(drop=True)
        for image_rank, image in enumerate(ordered.itertuples(index=False)):
            for pair_index in range(PAIRS_PER_LABEL):
                relations = (
                    ("MATCH", category),
                    ("PARTIAL_MATCH", PARTIAL_TARGET[category]),
                    ("MISMATCH", mismatch_target(category, image_rank, pair_index)),
                )
                for label, text_category in relations:
                    variant = (image_rank + pair_index) % caption_count(split)
                    rows.append(
                        {
                            "sample_id": (
                                f"{split}_{image.image_id}_{label.lower()}_{pair_index + 1}"
                            ),
                            "image_id": image.image_id,
                            "part_group_id": image.part_group_id,
                            "object_group_id": image.object_group_id,
                            "image_path": image.image_path,
                            "part_family": image.part_family,
                            "part_category": image.part_category,
                            "text_category": text_category,
                            "description": render_caption(split, text_category, variant),
                            "label": label,
                            "source": image.source,
                        }
                    )
    return pd.DataFrame(rows, columns=COLUMNS)


def validate_licenses(inventory: pd.DataFrame) -> None:
    licenses = pd.read_csv(DATA_DIR / "licenses.csv")
    expected_columns = {
        "asset_id",
        "part_category",
        "commons_title",
        "description_url",
        "author",
        "credit",
        "license_short_name",
        "license_url",
        "local_path",
        "sha256",
        "modifications",
    }
    if set(licenses.columns) != expected_columns:
        raise ValueError("Unexpected licenses.csv columns")
    wikimedia = inventory[inventory["source"].eq("wikimedia")].copy()
    if len(licenses) != len(wikimedia):
        raise ValueError(
            f"License rows ({len(licenses)}) do not match Wikimedia images ({len(wikimedia)})"
        )
    for column in ("asset_id", "commons_title", "description_url", "local_path", "sha256"):
        if licenses[column].duplicated().any():
            raise ValueError(f"Duplicate Wikimedia license field: {column}")
    if set(licenses["local_path"]) != set(wikimedia["image_path"]):
        raise ValueError("licenses.csv does not cover exactly the Wikimedia image inventory")
    for row in licenses.itertuples(index=False):
        path = PROJECT_ROOT / row.local_path
        if not path.is_file() or sha256(path) != row.sha256:
            raise ValueError(f"Wikimedia license hash mismatch: {row.asset_id}")
        if path.stem != row.asset_id:
            raise ValueError(f"asset_id does not match filename: {row.asset_id}")

    dataset_v2 = _load_dataset_v2_manifest()
    dataset_v2_inventory = inventory[inventory["source"].eq("dataset_v2")]
    if len(dataset_v2) != len(dataset_v2_inventory) or dataset_v2.empty:
        raise ValueError("Dataset V2 provenance manifest count is invalid")
    category_counts = dataset_v2.groupby(["assigned_split", "part_category"]).size().unstack(fill_value=0)
    if set(category_counts.index) != {"train", "validation"}:
        raise ValueError("Dataset V2 must contain train and validation images")
    if category_counts.loc["train"].nunique() != 1 or category_counts.loc["validation"].nunique() != 1:
        raise ValueError("Dataset V2 category counts must remain balanced")
    if set(dataset_v2["local_path"]) != set(dataset_v2_inventory["image_path"]):
        raise ValueError("dataset_v2_manifest.csv does not cover the imported V2 images")
    required_text_columns = (
        "provider",
        "source_dataset",
        "dataset_url",
        "source_title",
        "description_url",
        "author",
        "credit",
        "license_short_name",
        "license_url",
    )
    for column in required_text_columns:
        if dataset_v2[column].fillna("").astype(str).str.strip().eq("").any():
            raise ValueError(f"Dataset V2 contains a missing provenance value in {column}")
    if dataset_v2["asset_id"].duplicated().any() or dataset_v2["local_path"].duplicated().any():
        raise ValueError("Dataset V2 contains duplicate asset identifiers or local paths")
    for row in dataset_v2.itertuples(index=False):
        path = PROJECT_ROOT / row.local_path
        if not path.is_file() or sha256(path) != row.sha256:
            raise ValueError(f"Dataset V2 hash mismatch: {row.asset_id}")



def validate_design(splits: dict[str, pd.DataFrame], inventory: pd.DataFrame) -> None:
    source_split_counts = inventory.groupby(["split", "source"]).size().to_dict()
    expected_source_counts = expected_source_split_images()
    if source_split_counts != expected_source_counts:
        raise ValueError(
            f"Unexpected source/split image counts: {source_split_counts}; "
            f"expected {expected_source_counts}"
        )

    expected_images = expected_images_by_split()
    for split, data in splits.items():
        expected = expected_images[split]
        if len(data) != expected * ROWS_PER_IMAGE or data["image_id"].nunique() != expected:
            raise ValueError(f"Unexpected size for {split}: {len(data)} rows")
        label_counts_per_image = data.groupby(["image_id", "label"]).size().unstack(fill_value=0)
        if not (label_counts_per_image == PAIRS_PER_LABEL).all().all():
            raise ValueError(f"Every image must have {PAIRS_PER_LABEL} rows per label in {split}")
        if data["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample_id in {split}")
        per_category = data.drop_duplicates("image_id")["part_category"].value_counts()
        if len(per_category) != len(CATEGORIES) or per_category.nunique() != 1:
            raise ValueError(f"Image categories are not balanced in {split}")
        label_counts = data["label"].value_counts()
        if label_counts.nunique() != 1:
            raise ValueError(f"Relation labels are not balanced in {split}")
        if set(data["text_category"]) != set(CATEGORIES):
            raise ValueError(f"Text categories are incomplete in {split}")
        relation_text_counts = pd.crosstab(data["text_category"], data["label"])
        if relation_text_counts.nunique().max() != 1:
            raise ValueError(
                f"Text-category shortcut imbalance detected in {split}: "
                f"{relation_text_counts.to_dict()}"
            )

    if "generated" in set(splits["validation"]["source"]):
        raise ValueError("Validation must contain real images only")
    if set(splits["test"]["source"]) != {"wikimedia"}:
        raise ValueError("Test must remain the original locked Wikimedia holdout")
    if set(splits["train"]["source"]) != {"generated", "wikimedia", "dataset_v2"}:
        raise ValueError("Training must contain generated, original Wikimedia, and imported Dataset V2 images")

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for column in ("image_id", "part_group_id", "object_group_id", "image_path"):
            overlap = set(splits[left][column]) & set(splits[right][column])
            if overlap:
                raise ValueError(
                    f"{column} overlap between {left} and {right}: {sorted(overlap)[:3]}"
                )
        if set(splits[left]["description"]) & set(splits[right]["description"]):
            raise ValueError(f"Exact description overlap between {left} and {right}")


def write_splits() -> dict[str, pd.DataFrame]:
    inventory = image_inventory()
    validate_licenses(inventory)
    splits = {name: relation_rows(inventory, name) for name in ("train", "validation", "test")}
    validate_design(splits, inventory)

    manifest_columns = [
        "image_id",
        "part_group_id",
        "object_group_id",
        "image_path",
        "part_family",
        "part_category",
        "source",
        "split",
        "sha256",
    ]
    inventory[manifest_columns].to_csv(
        DATA_DIR / "image_manifest.csv", index=False, lineterminator="\n"
    )
    for name, data in splits.items():
        data.to_csv(DATA_DIR / f"{name}.csv", index=False, lineterminator="\n")

    lock = {
        "test_locked": True,
        "test_evaluation_permitted": False,
        "test_rows": int(len(splits["test"])),
        "test_images": int(splits["test"]["image_id"].nunique()),
        "test_sha256": sha256(DATA_DIR / "test.csv"),
        "note": (
            "The test rows are sealed and are not parsed by training, audit, or notebook execution. "
            "Dataset V2 increases development data while preserving the original independent test images."
        ),
    }
    (DATA_DIR / "test_lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    return splits


def main() -> None:
    splits = write_splits()
    for name, data in splits.items():
        sources = data.drop_duplicates("image_id")["source"].value_counts().to_dict()
        print(
            f"{name}: {len(data)} rows, {data['image_id'].nunique()} images, "
            f"sources={sources}, unique_descriptions={data['description'].nunique()}"
        )
    print(f"test_sha256={sha256(DATA_DIR / 'test.csv')}")


if __name__ == "__main__":
    main()
