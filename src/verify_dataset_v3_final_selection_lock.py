from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_v3"
    / "dataset_v3_final_selection_lock.json"
)
CHECKSUM_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_v3"
    / "dataset_v3_final_selection_lock.sha256.txt"
)

EXPECTED_SOURCE_COMMIT = (
    "8be2a8d440c4809f5941008edd7cc45221db163a"
)
EXPECTED_MODEL = "torch_multimodal_dataset_v3"
EXPECTED_NOTEBOOK_SHA256 = (
    "4f87a9818cf91d1011b4b09794da086b6cc8fc9469e10d612a7c6ae6efb1900a"
)
EXPECTED_TEST_LOCK = (
    "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def verify_final_selection_lock() -> dict:
    if not LOCK_PATH.is_file():
        raise RuntimeError("Final selection lock JSON is missing.")

    if not CHECKSUM_PATH.is_file():
        raise RuntimeError("Final selection lock checksum is missing.")

    lock = json.loads(
        LOCK_PATH.read_text(encoding="utf-8-sig")
    )
    checksum = CHECKSUM_PATH.read_text(
        encoding="utf-8-sig"
    ).split()[0]
    actual_lock_sha256 = sha256(LOCK_PATH)

    if checksum != actual_lock_sha256:
        raise RuntimeError(
            "Final selection lock checksum does not match."
        )

    if lock["status"] != (
        "FINAL_SELECTION_LOCKED_TEST_NOT_AUTHORIZED"
    ):
        raise RuntimeError("Unexpected final-selection status.")

    if lock["source_commit"] != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError("Unexpected final-selection source commit.")

    current_head = git("rev-parse", "HEAD")
    current_parent = None

    try:
        current_parent = git("rev-parse", "HEAD^")
    except subprocess.CalledProcessError:
        current_parent = None

    if EXPECTED_SOURCE_COMMIT not in {
        current_head,
        current_parent,
    }:
        raise RuntimeError(
            "The final-selection lock is not based on the expected commit."
        )

    if lock["selected_model_slug"] != EXPECTED_MODEL:
        raise RuntimeError("Unexpected selected model.")

    if lock["selected_notebook_sha256"] != (
        EXPECTED_NOTEBOOK_SHA256
    ):
        raise RuntimeError("Unexpected selected-notebook fingerprint.")

    if sha256(
        PROJECT_ROOT / lock["selected_notebook"]
    ) != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError("Selected notebook changed after locking.")

    checkpoint = PROJECT_ROOT / lock["selected_checkpoint"]

    if sha256(checkpoint) != lock["selected_checkpoint_sha256"]:
        raise RuntimeError("Selected checkpoint changed after locking.")

    for relative, expected_hash in lock[
        "component_hashes"
    ].items():
        path = PROJECT_ROOT / relative

        if not path.is_file():
            raise RuntimeError(
                f"Locked component is missing: {relative}"
            )

        if sha256(path) != expected_hash:
            raise RuntimeError(
                f"Locked component changed: {relative}"
            )

    if git("rev-parse", "HEAD:results/dataset_v3") != (
        lock["training_results_tree"]
    ):
        raise RuntimeError(
            "Training-results Git tree differs from the lock."
        )

    if git(
        "rev-parse",
        "HEAD:data/locked_test/dataset_v3",
    ) != lock["locked_test_tree"]:
        raise RuntimeError(
            "Locked-test Git tree differs from the lock."
        )

    if lock["selection_basis"] != (
        "development_validation_only"
    ):
        raise RuntimeError("Unexpected model-selection basis.")

    if lock["selection_frozen"] is not True:
        raise RuntimeError("Selection is not marked frozen.")

    if lock["further_tuning_permitted"] is not False:
        raise RuntimeError("Further tuning is marked as permitted.")

    if lock["test_evaluation_authorized"] is not False:
        raise RuntimeError("Test evaluation is already authorized.")

    for flag in (
        "test_evaluation_executed",
        "test_manifest_read",
        "test_images_read",
    ):
        if lock[flag] is not False:
            raise RuntimeError(
                f"Unexpected final-selection flag: {flag}."
            )

    if lock["test_lock_sha256"] != EXPECTED_TEST_LOCK:
        raise RuntimeError("Unexpected test-lock fingerprint.")

    if lock["main_correct_predictions"] != 377:
        raise RuntimeError("Unexpected selected-model correct count.")

    if lock["main_total_predictions"] != 480:
        raise RuntimeError("Unexpected selected-model prediction count.")

    if lock["validation_images"] != 80:
        raise RuntimeError("Unexpected validation image-group count.")

    result = {
        "status": "PASS_DATASET_V3_FINAL_SELECTION_LOCK_VERIFIED",
        "source_commit": lock["source_commit"],
        "current_commit": current_head,
        "selected_model_slug": lock["selected_model_slug"],
        "selected_checkpoint_sha256": (
            lock["selected_checkpoint_sha256"]
        ),
        "selected_notebook_sha256": (
            lock["selected_notebook_sha256"]
        ),
        "final_selection_lock_sha256": actual_lock_sha256,
        "selection_frozen": True,
        "further_tuning_permitted": False,
        "test_evaluation_authorized": False,
        "test_evaluation_executed": False,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_lock_sha256": EXPECTED_TEST_LOCK,
        "component_hashes_verified": len(
            lock["component_hashes"]
        ),
    }

    print()
    print("=== DATASET V3 FINAL SELECTION LOCK VERIFIED ===")
    print(f"Selected model:      {lock['selected_model_slug']}")
    print("Selection frozen:    True")
    print("Further tuning:      False")
    print("Test authorized:     False")
    print("Test evaluated:      False")
    print(
        "Components verified: "
        f"{len(lock['component_hashes'])}"
    )
    print(
        "Status: "
        "PASS_DATASET_V3_FINAL_SELECTION_LOCK_VERIFIED"
    )

    return result


def main() -> None:
    verify_final_selection_lock()


if __name__ == "__main__":
    main()
