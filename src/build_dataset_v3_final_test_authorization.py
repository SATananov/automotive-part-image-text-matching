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
SELECTION_CHECKSUM_PATH = (
    MANIFEST_DIR / "dataset_v3_final_selection_lock.sha256.txt"
)
AUTHORIZATION_PATH = (
    MANIFEST_DIR / "dataset_v3_final_test_authorization.json"
)
AUTHORIZATION_CHECKSUM_PATH = (
    MANIFEST_DIR
    / "dataset_v3_final_test_authorization.sha256.txt"
)

SOURCE_COMMIT = "7f8f8c11c0e4a339eb8df3485bd6bbdcad3d3de1"
EXPECTED_BRANCH = "dataset-v3"
EXPECTED_SELECTION_LOCK_SHA256 = (
    "54be116ab887680dcca1db08154936e3ef01b0ca8c4f73807e9324f64a9b865f"
)
EXPECTED_SELECTED_MODEL = "torch_multimodal_dataset_v3"
EXPECTED_SELECTED_CHECKPOINT = (
    "results/dataset_v3/models/"
    "torch_multimodal_dataset_v3_state.pt"
)
EXPECTED_SELECTED_CHECKPOINT_SHA256 = (
    "bce8a98fc96140294f3043fd845e1a7b6f491c67869a6c775877cd6ca1aa2a17"
)
EXPECTED_SELECTED_NOTEBOOK = "project_v3.ipynb"
EXPECTED_SELECTED_NOTEBOOK_SHA256 = (
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


def build_authorization() -> dict:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    if branch != EXPECTED_BRANCH:
        raise RuntimeError(
            f"Expected branch {EXPECTED_BRANCH}, found {branch}."
        )

    if head != SOURCE_COMMIT:
        raise RuntimeError(
            f"Expected source commit {SOURCE_COMMIT}, found {head}."
        )

    if not SELECTION_LOCK_PATH.is_file():
        raise RuntimeError("Final-selection lock JSON is missing.")

    if not SELECTION_CHECKSUM_PATH.is_file():
        raise RuntimeError("Final-selection lock checksum is missing.")

    if AUTHORIZATION_PATH.exists():
        raise RuntimeError(
            "Final-test authorization JSON already exists."
        )

    if AUTHORIZATION_CHECKSUM_PATH.exists():
        raise RuntimeError(
            "Final-test authorization checksum already exists."
        )

    selection_lock = json.loads(
        SELECTION_LOCK_PATH.read_text(encoding="utf-8-sig")
    )
    selection_checksum = SELECTION_CHECKSUM_PATH.read_text(
        encoding="utf-8-sig"
    ).split()[0]
    actual_selection_sha256 = sha256(SELECTION_LOCK_PATH)

    if selection_checksum != actual_selection_sha256:
        raise RuntimeError(
            "Final-selection lock checksum does not match."
        )

    if actual_selection_sha256 != (
        EXPECTED_SELECTION_LOCK_SHA256
    ):
        raise RuntimeError(
            "Final-selection lock differs from the verified lock."
        )

    if selection_lock["status"] != (
        "FINAL_SELECTION_LOCKED_TEST_NOT_AUTHORIZED"
    ):
        raise RuntimeError(
            "Unexpected final-selection lock status."
        )

    if selection_lock["selected_model_slug"] != (
        EXPECTED_SELECTED_MODEL
    ):
        raise RuntimeError("Unexpected selected model.")

    if selection_lock["selected_checkpoint"] != (
        EXPECTED_SELECTED_CHECKPOINT
    ):
        raise RuntimeError("Unexpected selected checkpoint.")

    if selection_lock["selected_checkpoint_sha256"] != (
        EXPECTED_SELECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Unexpected selected-checkpoint fingerprint."
        )

    if selection_lock["selected_notebook"] != (
        EXPECTED_SELECTED_NOTEBOOK
    ):
        raise RuntimeError("Unexpected selected notebook.")

    if selection_lock["selected_notebook_sha256"] != (
        EXPECTED_SELECTED_NOTEBOOK_SHA256
    ):
        raise RuntimeError(
            "Unexpected selected-notebook fingerprint."
        )

    if selection_lock["test_lock_sha256"] != (
        EXPECTED_TEST_LOCK_SHA256
    ):
        raise RuntimeError(
            "Unexpected locked-test fingerprint."
        )

    if selection_lock["selection_frozen"] is not True:
        raise RuntimeError("Model selection is not frozen.")

    if selection_lock["further_tuning_permitted"] is not False:
        raise RuntimeError("Further tuning is marked permitted.")

    if selection_lock["test_evaluation_authorized"] is not False:
        raise RuntimeError(
            "The final-selection lock already authorizes test evaluation."
        )

    if selection_lock["test_evaluation_executed"] is not False:
        raise RuntimeError(
            "The final-selection lock reports a test evaluation."
        )

    results_tree = git("rev-parse", "HEAD:results/dataset_v3")
    notebook_blob = git(
        "rev-parse",
        "HEAD:project_v3.ipynb",
    )
    locked_test_tree = git(
        "rev-parse",
        "HEAD:data/locked_test/dataset_v3",
    )
    selection_lock_blob = git(
        "rev-parse",
        (
            "HEAD:data/manifests/dataset_v3/"
            "dataset_v3_final_selection_lock.json"
        ),
    )

    authorization = {
        "status": (
            "FINAL_TEST_EVALUATION_AUTHORIZED_NOT_EXECUTED"
        ),
        "dataset_version": "3.0-development",
        "branch": EXPECTED_BRANCH,
        "authorization_source_commit": SOURCE_COMMIT,
        "final_selection_lock_sha256": (
            EXPECTED_SELECTION_LOCK_SHA256
        ),
        "final_selection_lock_blob": selection_lock_blob,
        "selected_model_slug": EXPECTED_SELECTED_MODEL,
        "selected_checkpoint": EXPECTED_SELECTED_CHECKPOINT,
        "selected_checkpoint_sha256": (
            EXPECTED_SELECTED_CHECKPOINT_SHA256
        ),
        "selected_notebook": EXPECTED_SELECTED_NOTEBOOK,
        "selected_notebook_sha256": (
            EXPECTED_SELECTED_NOTEBOOK_SHA256
        ),
        "training_results_tree": results_tree,
        "selected_notebook_blob": notebook_blob,
        "locked_test_tree": locked_test_tree,
        "test_lock_sha256": EXPECTED_TEST_LOCK_SHA256,
        "selection_basis": "development_validation_only",
        "selection_frozen": True,
        "further_tuning_permitted": False,
        "post_test_tuning_permitted": False,
        "test_evaluation_authorized": True,
        "authorization_scope": (
            "one_time_final_evaluation_of_frozen_checkpoint_only"
        ),
        "maximum_authorized_evaluations": 1,
        "authorized_evaluations_completed": 0,
        "authorization_consumed": False,
        "test_evaluation_executed": False,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_results_may_be_used_for_tuning": False,
        "test_results_use": (
            "final_reporting_and_error_analysis_only"
        ),
        "main_validation_accuracy": selection_lock[
            "main_validation_accuracy"
        ],
        "main_validation_macro_f1": selection_lock[
            "main_validation_macro_f1"
        ],
        "main_correct_predictions": selection_lock[
            "main_correct_predictions"
        ],
        "main_total_predictions": selection_lock[
            "main_total_predictions"
        ],
        "validation_images": selection_lock[
            "validation_images"
        ],
        "independent_audit_status": selection_lock[
            "independent_audit_status"
        ],
        "independent_audit_checks": selection_lock[
            "independent_audit_checks"
        ],
    }

    AUTHORIZATION_PATH.write_text(
        json.dumps(
            authorization,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    authorization_sha256 = sha256(AUTHORIZATION_PATH)

    AUTHORIZATION_CHECKSUM_PATH.write_text(
        (
            f"{authorization_sha256}  "
            f"{AUTHORIZATION_PATH.name}\n"
        ),
        encoding="utf-8",
    )

    result = {
        **authorization,
        "authorization_sha256": authorization_sha256,
        "authorization_path": str(
            AUTHORIZATION_PATH.relative_to(PROJECT_ROOT)
        ),
        "authorization_checksum_path": str(
            AUTHORIZATION_CHECKSUM_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
    }

    print()
    print("=== DATASET V3 FINAL TEST AUTHORIZATION CREATED ===")
    print(f"Source commit:       {SOURCE_COMMIT}")
    print(f"Selected model:      {EXPECTED_SELECTED_MODEL}")
    print("Evaluation scope:    ONE TIME")
    print("Further tuning:      NOT PERMITTED")
    print("Test authorized:     True")
    print("Test executed:       False")
    print("Test manifest read:  False")
    print("Test images read:    False")
    print(f"Authorization SHA:   {authorization_sha256}")
    print(
        "Status: "
        "FINAL_TEST_EVALUATION_AUTHORIZED_NOT_EXECUTED"
    )

    return result


def main() -> None:
    build_authorization()


if __name__ == "__main__":
    main()
