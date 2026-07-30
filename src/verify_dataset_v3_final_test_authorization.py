from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests" / "dataset_v3"

SELECTION_LOCK_PATH = (
    MANIFEST_DIR / "dataset_v3_final_selection_lock.json"
)
AUTHORIZATION_PATH = (
    MANIFEST_DIR / "dataset_v3_final_test_authorization.json"
)
AUTHORIZATION_CHECKSUM_PATH = (
    MANIFEST_DIR
    / "dataset_v3_final_test_authorization.sha256.txt"
)

EXPECTED_SOURCE_COMMIT = (
    "7f8f8c11c0e4a339eb8df3485bd6bbdcad3d3de1"
)
EXPECTED_SELECTION_LOCK_SHA256 = (
    "54be116ab887680dcca1db08154936e3ef01b0ca8c4f73807e9324f64a9b865f"
)
EXPECTED_MODEL = "torch_multimodal_dataset_v3"
EXPECTED_CHECKPOINT_SHA256 = (
    "bce8a98fc96140294f3043fd845e1a7b6f491c67869a6c775877cd6ca1aa2a17"
)
EXPECTED_NOTEBOOK_SHA256 = (
    "4f87a9818cf91d1011b4b09794da086b6cc8fc9469e10d612a7c6ae6efb1900a"
)
EXPECTED_TEST_LOCK_SHA256 = (
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
    return completed.stdout.rstrip("\r\n")


def verify_final_test_authorization() -> dict:
    if not SELECTION_LOCK_PATH.is_file():
        raise RuntimeError("Final-selection lock is missing.")

    if not AUTHORIZATION_PATH.is_file():
        raise RuntimeError("Final-test authorization is missing.")

    if not AUTHORIZATION_CHECKSUM_PATH.is_file():
        raise RuntimeError(
            "Final-test authorization checksum is missing."
        )

    selection_lock = json.loads(
        SELECTION_LOCK_PATH.read_text(encoding="utf-8-sig")
    )
    authorization = json.loads(
        AUTHORIZATION_PATH.read_text(encoding="utf-8-sig")
    )
    checksum = AUTHORIZATION_CHECKSUM_PATH.read_text(
        encoding="utf-8-sig"
    ).split()[0]
    actual_authorization_sha256 = sha256(
        AUTHORIZATION_PATH
    )

    if checksum != actual_authorization_sha256:
        raise RuntimeError(
            "Authorization checksum does not match."
        )

    if sha256(SELECTION_LOCK_PATH) != (
        EXPECTED_SELECTION_LOCK_SHA256
    ):
        raise RuntimeError(
            "Final-selection lock changed after authorization."
        )

    if authorization["status"] != (
        "FINAL_TEST_EVALUATION_AUTHORIZED_NOT_EXECUTED"
    ):
        raise RuntimeError(
            "Unexpected final-test authorization status."
        )

    if authorization["authorization_source_commit"] != (
        EXPECTED_SOURCE_COMMIT
    ):
        raise RuntimeError(
            "Unexpected authorization source commit."
        )

    if authorization["final_selection_lock_sha256"] != (
        EXPECTED_SELECTION_LOCK_SHA256
    ):
        raise RuntimeError(
            "Authorization references the wrong selection lock."
        )

    if authorization["selected_model_slug"] != EXPECTED_MODEL:
        raise RuntimeError("Unexpected authorized model.")

    if authorization["selected_checkpoint_sha256"] != (
        EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Unexpected authorized checkpoint fingerprint."
        )

    if authorization["selected_notebook_sha256"] != (
        EXPECTED_NOTEBOOK_SHA256
    ):
        raise RuntimeError(
            "Unexpected authorized notebook fingerprint."
        )

    if authorization["test_lock_sha256"] != (
        EXPECTED_TEST_LOCK_SHA256
    ):
        raise RuntimeError(
            "Unexpected authorized test-lock fingerprint."
        )

    if authorization["selection_basis"] != (
        "development_validation_only"
    ):
        raise RuntimeError(
            "Unexpected model-selection basis."
        )

    if authorization["selection_frozen"] is not True:
        raise RuntimeError("Selection is not frozen.")

    if authorization["further_tuning_permitted"] is not False:
        raise RuntimeError("Further tuning is permitted.")

    if authorization["post_test_tuning_permitted"] is not False:
        raise RuntimeError("Post-test tuning is permitted.")

    if authorization["test_evaluation_authorized"] is not True:
        raise RuntimeError(
            "Test evaluation is not authorized."
        )

    if authorization["authorization_scope"] != (
        "one_time_final_evaluation_of_frozen_checkpoint_only"
    ):
        raise RuntimeError(
            "Unexpected authorization scope."
        )

    if authorization["maximum_authorized_evaluations"] != 1:
        raise RuntimeError(
            "Authorization is not limited to one evaluation."
        )

    if authorization["authorized_evaluations_completed"] != 0:
        raise RuntimeError(
            "Authorization reports a completed evaluation."
        )

    if authorization["authorization_consumed"] is not False:
        raise RuntimeError(
            "Authorization is already marked consumed."
        )

    for flag in (
        "test_evaluation_executed",
        "test_manifest_read",
        "test_images_read",
        "test_results_may_be_used_for_tuning",
    ):
        if authorization[flag] is not False:
            raise RuntimeError(
                f"Unexpected authorization flag: {flag}."
            )

    if selection_lock["selection_frozen"] is not True:
        raise RuntimeError(
            "Underlying final selection is not frozen."
        )

    if selection_lock["further_tuning_permitted"] is not False:
        raise RuntimeError(
            "Underlying final selection permits tuning."
        )

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
            "Authorization is not based on the expected commit."
        )

    if git("rev-parse", "HEAD:results/dataset_v3") != (
        authorization["training_results_tree"]
    ):
        raise RuntimeError(
            "Training-results Git tree differs from authorization."
        )

    if git("rev-parse", "HEAD:project_v3.ipynb") != (
        authorization["selected_notebook_blob"]
    ):
        raise RuntimeError(
            "Selected-notebook blob differs from authorization."
        )

    if git(
        "rev-parse",
        "HEAD:data/locked_test/dataset_v3",
    ) != authorization["locked_test_tree"]:
        raise RuntimeError(
            "Locked-test Git tree differs from authorization."
        )

    result = {
        "status": (
            "PASS_DATASET_V3_FINAL_TEST_AUTHORIZATION_VERIFIED"
        ),
        "authorization_source_commit": (
            authorization["authorization_source_commit"]
        ),
        "current_commit": current_head,
        "selected_model_slug": (
            authorization["selected_model_slug"]
        ),
        "selected_checkpoint_sha256": (
            authorization["selected_checkpoint_sha256"]
        ),
        "selected_notebook_sha256": (
            authorization["selected_notebook_sha256"]
        ),
        "final_selection_lock_sha256": (
            authorization["final_selection_lock_sha256"]
        ),
        "authorization_sha256": (
            actual_authorization_sha256
        ),
        "selection_frozen": True,
        "further_tuning_permitted": False,
        "post_test_tuning_permitted": False,
        "test_evaluation_authorized": True,
        "authorization_scope": (
            authorization["authorization_scope"]
        ),
        "maximum_authorized_evaluations": 1,
        "authorized_evaluations_completed": 0,
        "authorization_consumed": False,
        "test_evaluation_executed": False,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_results_may_be_used_for_tuning": False,
        "test_lock_sha256": EXPECTED_TEST_LOCK_SHA256,
    }

    print()
    print("=== DATASET V3 FINAL TEST AUTHORIZATION VERIFIED ===")
    print(
        f"Selected model:      "
        f"{authorization['selected_model_slug']}"
    )
    print("Selection frozen:    True")
    print("Further tuning:      False")
    print("Test authorized:     True")
    print("Authorized runs:     1")
    print("Completed runs:      0")
    print("Test evaluated:      False")
    print("Test manifest read:  False")
    print("Test images read:    False")
    print(
        "Status: "
        "PASS_DATASET_V3_FINAL_TEST_AUTHORIZATION_VERIFIED"
    )

    return result


def main() -> None:
    verify_final_test_authorization()


if __name__ == "__main__":
    main()
