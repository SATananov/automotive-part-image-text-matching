from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data import COLUMNS, DATA_DIR, PROJECT_ROOT

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

MISMATCH_TARGET = {
    "air_filter": "alternator",
    "oil_filter": "starter",
    "alternator": "brake_disc",
    "starter": "brake_pad",
    "brake_disc": "coil_spring",
    "brake_pad": "shock_absorber",
    "coil_spring": "headlight",
    "shock_absorber": "taillight",
    "headlight": "air_filter",
    "taillight": "oil_filter",
}

# Every split uses different sentences, while the actual part names remain present.
# This removes exact-text leakage without turning validation into an out-of-vocabulary test.
CAPTIONS = {
    "train": {
        "air_filter": "An automotive air filter for the engine intake system.",
        "alternator": "A vehicle alternator used by the electrical charging system.",
        "brake_disc": "A metal brake disc used in a vehicle braking system.",
        "brake_pad": "An automotive brake pad that presses against a disc.",
        "coil_spring": "A coil spring from a passenger-vehicle suspension.",
        "headlight": "A front headlight assembly used on an automobile.",
        "oil_filter": "An engine oil filter designed for a road vehicle.",
        "shock_absorber": "A vehicle shock absorber used as a suspension damper.",
        "starter": "An electric starter motor for an internal-combustion engine.",
        "taillight": "A rear taillight assembly from an automobile.",
    },
    "validation": {
        "air_filter": "A pleated engine air filter that removes particles from intake air.",
        "alternator": "An automotive alternator that converts rotation into electrical power.",
        "brake_disc": "A circular brake disc or rotor mounted near a vehicle wheel.",
        "brake_pad": "A friction brake pad used to slow an automobile disc.",
        "coil_spring": "A helical coil spring supporting a car suspension.",
        "headlight": "A forward-facing vehicle headlight that illuminates the road.",
        "oil_filter": "A replaceable automotive oil filter that cleans engine lubricant.",
        "shock_absorber": "A suspension shock absorber that controls vehicle movement.",
        "starter": "A high-torque automotive starter motor that cranks an engine.",
        "taillight": "A rear vehicle taillight used for visibility and signalling.",
    },
    "test": {
        "air_filter": "A replaceable air filter for a motor vehicle's intake path.",
        "alternator": "A belt-driven car alternator supplying current to the electrical system.",
        "brake_disc": "An automobile brake disc providing a rotor surface for braking.",
        "brake_pad": "A replaceable vehicle brake pad made from friction material.",
        "coil_spring": "A steel coil spring carrying part of an automobile's weight.",
        "headlight": "An automobile headlight unit intended for forward illumination.",
        "oil_filter": "A vehicle oil filter that traps contaminants in engine lubricant.",
        "shock_absorber": "A hydraulic shock absorber damping motion in a car suspension.",
        "starter": "An engine starter motor powered by a vehicle battery.",
        "taillight": "An automobile taillight assembly for rear position and warning signals.",
    },
}

# Manually reviewed same-object or same-photographic-series groups. All of these
# remain inside training, preventing alternate views from crossing into holdouts.
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

EXPECTED_IMAGES = {"train": 100, "validation": 10, "test": 10}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_and_split(path: Path) -> tuple[str, str]:
    source = "generated" if "generated" in path.parts else "wikimedia"
    stem = path.stem
    if stem.endswith("_new_validation"):
        return source, "validation"
    if stem.endswith("_new_test"):
        return source, "test"
    return source, "train"


def identity(path: Path, source: str, category: str) -> tuple[str, str, str]:
    stem = path.stem
    if source == "generated":
        image_id = stem
        part_group_id = stem.rsplit("_", 1)[0]
        # The five drawings in a category are variations of one simple template family.
        object_group_id = f"synthetic_family_{category}"
        return image_id, part_group_id, object_group_id

    suffix = stem.removeprefix("commons_")
    image_id = f"external_image_{suffix}"
    part_group_id = f"external_group_{suffix}"
    object_group_id = OBJECT_SERIES.get(stem, f"wikimedia_object_{suffix}")
    return image_id, part_group_id, object_group_id


def image_inventory() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    source_dirs = (DATA_DIR / "images" / "generated", DATA_DIR / "images" / "wikimedia")
    for source_dir in source_dirs:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            source, split = source_and_split(path)
            category = path.stem.rsplit("_", 2)[0] if source == "generated" else path.parent.name
            if category not in FAMILIES:
                raise ValueError(f"Unknown part category for {path}")
            image_id, part_group_id, object_group_id = identity(path, source, category)
            rows.append(
                {
                    "image_id": image_id,
                    "part_group_id": part_group_id,
                    "object_group_id": object_group_id,
                    "image_path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "part_family": FAMILIES[category],
                    "part_category": category,
                    "source": source,
                    "split": split,
                    "sha256": sha256(path),
                }
            )
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise ValueError("No images found")
    for column in ("image_id", "image_path", "sha256"):
        if inventory[column].duplicated().any():
            duplicate = inventory.loc[inventory[column].duplicated(), column].iloc[0]
            raise ValueError(f"Duplicate {column}: {duplicate}")
    return inventory.sort_values(["split", "part_category", "source", "image_id"], kind="stable")


def relation_rows(inventory: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for image in inventory[inventory["split"].eq(split)].itertuples(index=False):
        relations = (
            ("MATCH", image.part_category),
            ("PARTIAL_MATCH", PARTIAL_TARGET[image.part_category]),
            ("MISMATCH", MISMATCH_TARGET[image.part_category]),
        )
        for label, text_category in relations:
            rows.append(
                {
                    "sample_id": f"{split}_{image.image_id}_{label.lower()}",
                    "image_id": image.image_id,
                    "part_group_id": image.part_group_id,
                    "object_group_id": image.object_group_id,
                    "image_path": image.image_path,
                    "part_family": image.part_family,
                    "part_category": image.part_category,
                    "text_category": text_category,
                    "description": CAPTIONS[split][text_category],
                    "label": label,
                    "source": image.source,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def validate_licenses(inventory: pd.DataFrame) -> None:
    licenses = pd.read_csv(DATA_DIR / "licenses.csv")
    expected_columns = {
        "asset_id", "part_category", "commons_title", "description_url", "author", "credit",
        "license_short_name", "license_url", "local_path", "sha256", "modifications",
    }
    if set(licenses.columns) != expected_columns:
        raise ValueError("Unexpected licenses.csv columns")
    wikimedia = inventory[inventory["source"].eq("wikimedia")].copy()
    if len(licenses) != len(wikimedia):
        raise ValueError(f"License rows ({len(licenses)}) do not match Wikimedia images ({len(wikimedia)})")
    for column in ("asset_id", "commons_title", "description_url", "local_path", "sha256"):
        if licenses[column].duplicated().any():
            raise ValueError(f"Duplicate license field: {column}")
    expected_paths = set(wikimedia["image_path"])
    if set(licenses["local_path"]) != expected_paths:
        raise ValueError("licenses.csv does not cover exactly the Wikimedia image inventory")
    for row in licenses.itertuples(index=False):
        path = PROJECT_ROOT / row.local_path
        if not path.is_file():
            raise ValueError(f"Missing licensed image: {row.local_path}")
        if sha256(path) != row.sha256:
            raise ValueError(f"License hash mismatch: {row.asset_id}")
        if path.stem != row.asset_id:
            raise ValueError(f"asset_id does not match filename: {row.asset_id}")


def validate_design(splits: dict[str, pd.DataFrame]) -> None:
    for split, data in splits.items():
        expected = EXPECTED_IMAGES[split]
        if len(data) != expected * 3 or data["image_id"].nunique() != expected:
            raise ValueError(f"Unexpected size for {split}: {len(data)} rows")
        if not (data.groupby("image_id")["label"].nunique() == 3).all():
            raise ValueError(f"Every image must have all three labels in {split}")
        if data["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample_id in {split}")
        per_category = data.drop_duplicates("image_id")["part_category"].value_counts()
        if len(per_category) != 10 or per_category.nunique() != 1:
            raise ValueError(f"Image categories are not balanced in {split}")
        relation_counts = pd.crosstab(data["text_category"], data["label"])
        if relation_counts.nunique().nunique() != 1 or relation_counts.shape != (10, 3):
            raise ValueError(f"Text categories are not label-balanced in {split}")

    if set(splits["validation"]["source"]) != {"wikimedia"}:
        raise ValueError("Validation must contain real Wikimedia images only")
    if set(splits["test"]["source"]) != {"wikimedia"}:
        raise ValueError("Test must contain real Wikimedia images only")
    if set(splits["train"]["source"]) != {"generated", "wikimedia"}:
        raise ValueError("Training must contain real and synthetic images")

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for column in ("image_id", "part_group_id", "object_group_id", "image_path"):
            overlap = set(splits[left][column]) & set(splits[right][column])
            if overlap:
                raise ValueError(f"{column} overlap between {left} and {right}: {sorted(overlap)[:3]}")
        if set(splits[left]["description"]) & set(splits[right]["description"]):
            raise ValueError(f"Exact description overlap between {left} and {right}")


def write_splits() -> dict[str, pd.DataFrame]:
    inventory = image_inventory()
    validate_licenses(inventory)
    splits = {name: relation_rows(inventory, name) for name in ("train", "validation", "test")}
    validate_design(splits)

    manifest_columns = [
        "image_id", "part_group_id", "object_group_id", "image_path", "part_family",
        "part_category", "source", "split", "sha256",
    ]
    inventory[manifest_columns].to_csv(DATA_DIR / "image_manifest.csv", index=False, lineterminator="\n")
    for name, data in splits.items():
        data.to_csv(DATA_DIR / f"{name}.csv", index=False, lineterminator="\n")

    lock = {
        "test_locked": True,
        "test_evaluation_permitted": False,
        "test_sha256": sha256(DATA_DIR / "test.csv"),
        "note": "The test rows are sealed and are not parsed by training, audit, or notebook execution.",
    }
    (DATA_DIR / "test_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return splits


def main() -> None:
    splits = write_splits()
    for name, data in splits.items():
        sources = data.drop_duplicates("image_id")["source"].value_counts().to_dict()
        print(f"{name}: {len(data)} rows, {data['image_id'].nunique()} images, sources={sources}")
    print(f"test_sha256={sha256(DATA_DIR / 'test.csv')}")


if __name__ == "__main__":
    main()
