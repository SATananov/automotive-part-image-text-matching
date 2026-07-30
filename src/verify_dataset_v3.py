from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "data" / "manifests" / "dataset_v3"
    / "dataset_v3_image_manifest.csv"
)
LOCK_ROOT = ROOT / "data" / "locked_test" / "dataset_v3"
LOCK_JSON = LOCK_ROOT / "dataset_v3_test_lock.json"
LOCK_CHECKSUM = LOCK_ROOT / "dataset_v3_test_lock.sha256.txt"

EXPECTED_LOCK = (
    "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
)
CATEGORIES = {
    "alternator",
    "brake_disc",
    "brake_pad",
    "coil_spring",
    "headlight",
    "oil_filter",
    "starter",
    "taillight",
}


def load_rows() -> list[dict[str, str]]:
    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def verify() -> dict[str, object]:
    rows = load_rows()

    if len(rows) != 640:
        raise AssertionError(
            f"Expected 640 manifest rows, found {len(rows)}."
        )

    split_counts = Counter(row["split"] for row in rows)

    if split_counts != Counter(
        {"train": 480, "validation": 80, "test": 80}
    ):
        raise AssertionError(
            f"Unexpected split counts: {dict(split_counts)}"
        )

    category_counts = Counter(
        (row["project_category"], row["split"])
        for row in rows
    )

    for category in CATEGORIES:
        for split, expected in (
            ("train", 60),
            ("validation", 10),
            ("test", 10),
        ):
            actual = category_counts[(category, split)]

            if actual != expected:
                raise AssertionError(
                    f"{category}/{split}: expected {expected}, "
                    f"found {actual}."
                )

    ids: set[str] = set()
    groups: set[str] = set()
    hashes: set[str] = set()
    groups_by_split: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        path = ROOT / row["repository_relative_path"]

        if not path.is_file():
            raise AssertionError(f"Missing Dataset V3 image: {path}")

        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        if digest != row["sha256"]:
            raise AssertionError(f"SHA-256 mismatch: {path}")

        ids.add(row["candidate_id"])
        groups.add(row["image_group_id"])
        hashes.add(digest)
        groups_by_split[row["split"]].add(row["image_group_id"])

        expected_locked = row["split"] == "test"

        if str(row["test_locked"]).lower() != str(
            expected_locked
        ).lower():
            raise AssertionError(
                f"Incorrect lock flag: {row['candidate_id']}"
            )

        if expected_locked:
            normalized = row[
                "repository_relative_path"
            ].replace("\\", "/")

            if not normalized.startswith(
                "data/locked_test/dataset_v3/"
            ):
                raise AssertionError(
                    f"Test image outside locked area: {normalized}"
                )

    if len(ids) != 640:
        raise AssertionError("Candidate IDs are not unique.")

    if len(groups) != 640:
        raise AssertionError("Image groups are not unique.")

    if len(hashes) != 640:
        raise AssertionError("Image SHA-256 values are not unique.")

    if groups_by_split["train"] & groups_by_split["validation"]:
        raise AssertionError("Train/validation group leakage.")

    if groups_by_split["train"] & groups_by_split["test"]:
        raise AssertionError("Train/test group leakage.")

    if groups_by_split["validation"] & groups_by_split["test"]:
        raise AssertionError("Validation/test group leakage.")

    lock = json.loads(LOCK_JSON.read_text(encoding="utf-8-sig"))
    checksum = LOCK_CHECKSUM.read_text(
        encoding="utf-8-sig"
    ).split()[0]

    if lock["status"] != (
        "LOCKED_NOT_AUTHORIZED_FOR_TRAINING_OR_SELECTION"
    ):
        raise AssertionError("Unexpected test-lock status.")

    if lock["test_lock_sha256"] != EXPECTED_LOCK:
        raise AssertionError("Unexpected test-lock JSON fingerprint.")

    if checksum != EXPECTED_LOCK:
        raise AssertionError("Unexpected test-lock checksum.")

    return {
        "status": "PASS",
        "images": len(rows),
        "split_counts": dict(split_counts),
        "unique_candidate_ids": len(ids),
        "unique_image_groups": len(groups),
        "unique_sha256": len(hashes),
        "group_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "test_lock_status": lock["status"],
        "test_lock_sha256": EXPECTED_LOCK,
    }


def main() -> None:
    print(
        json.dumps(
            verify(),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
