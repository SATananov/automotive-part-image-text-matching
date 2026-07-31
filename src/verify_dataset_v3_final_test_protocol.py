from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from src.data_v3 import load_v3_split
from src.final_test_protocol_v3 import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_RELATION_PROTOCOL_FINGERPRINT,
    EXPECTED_TEXT_DIMENSION,
    EXPECTED_VECTORIZER_FINGERPRINT,
    all_final_test_captions,
    file_sha256,
    protocol_model_dry_run,
    relation_protocol_fingerprint,
)
from src.verify_dataset_v3_final_selection_lock import (
    verify_final_selection_lock,
)
from src.verify_dataset_v3_final_test_authorization import (
    verify_final_test_authorization,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_v3"
)
PROTOCOL_PATH = (
    MANIFEST_DIR
    / "dataset_v3_final_test_protocol.json"
)
PROTOCOL_CHECKSUM_PATH = (
    MANIFEST_DIR
    / "dataset_v3_final_test_protocol.sha256.txt"
)

EXPECTED_SOURCE_COMMIT = (
    "591ccda6294b03608a0f3b7c3fa0952af56c849d"
)
EXPECTED_SELECTION_LOCK_SHA256 = (
    "54be116ab887680dcca1db08154936e3ef01b0ca8c4f73807e9324f64a9b865f"
)
EXPECTED_AUTHORIZATION_SHA256 = (
    "a58005028f389c9c27ec737d079cc4a750b9c555fa4dfba350017ebdb8754a24"
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
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
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


def verify_final_test_protocol() -> dict[str, object]:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError(
            "Final-test protocol JSON is missing."
        )

    if not PROTOCOL_CHECKSUM_PATH.is_file():
        raise RuntimeError(
            "Final-test protocol checksum is missing."
        )

    protocol = json.loads(
        PROTOCOL_PATH.read_text(
            encoding="utf-8-sig"
        )
    )
    checksum = PROTOCOL_CHECKSUM_PATH.read_text(
        encoding="utf-8-sig"
    ).split()[0]
    actual_protocol_sha256 = sha256(
        PROTOCOL_PATH
    )

    if checksum != actual_protocol_sha256:
        raise RuntimeError(
            "Final-test protocol checksum mismatch."
        )

    if protocol["status"] != (
        "FINAL_TEST_EVALUATOR_PROTOCOL_READY_NOT_EXECUTED"
    ):
        raise RuntimeError(
            "Unexpected final-test protocol status."
        )

    if protocol["protocol_source_commit"] != (
        EXPECTED_SOURCE_COMMIT
    ):
        raise RuntimeError(
            "Unexpected final-test protocol source commit."
        )

    current_head = git("rev-parse", "HEAD")
    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            EXPECTED_SOURCE_COMMIT,
            current_head,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if ancestry.returncode not in {0, 1}:
        raise RuntimeError(
            "Could not verify protocol source ancestry: "
            f"{ancestry.stderr.strip()}"
        )

    if ancestry.returncode != 0:
        raise RuntimeError(
            "Final-test protocol source commit is not "
            "an ancestor of current HEAD."
        )

    selection = verify_final_selection_lock()
    authorization = verify_final_test_authorization()

    selection_path = (
        MANIFEST_DIR
        / "dataset_v3_final_selection_lock.json"
    )
    authorization_path = (
        MANIFEST_DIR
        / "dataset_v3_final_test_authorization.json"
    )
    checkpoint_path = (
        PROJECT_ROOT
        / protocol["selected_checkpoint"]
    )
    notebook_path = (
        PROJECT_ROOT
        / protocol["selected_notebook"]
    )

    if sha256(selection_path) != (
        EXPECTED_SELECTION_LOCK_SHA256
    ):
        raise RuntimeError(
            "Final-selection lock SHA-256 changed."
        )

    if sha256(authorization_path) != (
        EXPECTED_AUTHORIZATION_SHA256
    ):
        raise RuntimeError(
            "Final-test authorization SHA-256 changed."
        )

    if file_sha256(checkpoint_path) != (
        EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Selected checkpoint SHA-256 changed."
        )

    if sha256(notebook_path) != (
        EXPECTED_NOTEBOOK_SHA256
    ):
        raise RuntimeError(
            "Selected notebook SHA-256 changed."
        )

    if protocol["final_selection_lock_sha256"] != (
        EXPECTED_SELECTION_LOCK_SHA256
    ):
        raise RuntimeError(
            "Protocol references the wrong selection lock."
        )

    if protocol["final_test_authorization_sha256"] != (
        EXPECTED_AUTHORIZATION_SHA256
    ):
        raise RuntimeError(
            "Protocol references the wrong authorization."
        )

    if protocol["test_lock_sha256"] != (
        EXPECTED_TEST_LOCK_SHA256
    ):
        raise RuntimeError(
            "Protocol references the wrong test lock."
        )

    if protocol["text_dimension"] != (
        EXPECTED_TEXT_DIMENSION
    ):
        raise RuntimeError(
            "Unexpected final-test text dimension."
        )

    if protocol["vectorizer_fingerprint"] != (
        EXPECTED_VECTORIZER_FINGERPRINT
    ):
        raise RuntimeError(
            "Unexpected final-test vectorizer fingerprint."
        )

    if protocol["relation_protocol_fingerprint"] != (
        EXPECTED_RELATION_PROTOCOL_FINGERPRINT
    ):
        raise RuntimeError(
            "Unexpected relation protocol fingerprint."
        )

    if relation_protocol_fingerprint() != (
        EXPECTED_RELATION_PROTOCOL_FINGERPRINT
    ):
        raise RuntimeError(
            "Current relation protocol code changed."
        )

    train = load_v3_split("train")
    validation = load_v3_split("validation")
    captions = all_final_test_captions()

    if captions & set(
        train["description"].astype(str)
    ):
        raise RuntimeError(
            "Final-test captions overlap train descriptions."
        )

    if captions & set(
        validation["description"].astype(str)
    ):
        raise RuntimeError(
            "Final-test captions overlap validation descriptions."
        )

    dry_run = protocol_model_dry_run(
        train,
        checkpoint_path,
    )

    for relative, expected_hash in (
        protocol["code_hashes"].items()
    ):
        path = PROJECT_ROOT / relative

        if not path.is_file():
            raise RuntimeError(
                f"Protocol code file is missing: {relative}"
            )

        if sha256(path) != expected_hash:
            raise RuntimeError(
                f"Protocol code file changed: {relative}"
            )

    required_false = (
        "further_tuning_permitted",
        "post_test_tuning_permitted",
        "authorization_consumed",
        "test_evaluation_executed",
        "repository_test_manifest_read",
        "test_images_read",
    )

    for field in required_false:
        if protocol[field] is not False:
            raise RuntimeError(
                f"Unexpected protocol flag: {field}"
            )

    if protocol["selection_frozen"] is not True:
        raise RuntimeError(
            "Protocol selection is not frozen."
        )

    if protocol["test_evaluation_authorized"] is not True:
        raise RuntimeError(
            "Protocol does not authorize the final test."
        )

    if protocol["maximum_authorized_evaluations"] != 1:
        raise RuntimeError(
            "Protocol is not limited to one final evaluation."
        )

    if protocol["authorized_evaluations_completed"] != 0:
        raise RuntimeError(
            "Protocol reports a completed final evaluation."
        )

    if protocol[
        "external_snapshot_test_metadata_reviewed"
    ] is not True:
        raise RuntimeError(
            "Protocol does not record the external metadata review."
        )

    if protocol[
        "external_snapshot_locked_test_images_present"
    ] is not False:
        raise RuntimeError(
            "Protocol incorrectly reports locked-test images "
            "in the preparation snapshot."
        )

    result = {
        "status": (
            "PASS_DATASET_V3_FINAL_TEST_PROTOCOL_VERIFIED"
        ),
        "current_commit": current_head,
        "protocol_source_commit": (
            EXPECTED_SOURCE_COMMIT
        ),
        "protocol_source_is_ancestor": True,
        "protocol_sha256": (
            actual_protocol_sha256
        ),
        "selection_lock_sha256": (
            EXPECTED_SELECTION_LOCK_SHA256
        ),
        "authorization_sha256": (
            EXPECTED_AUTHORIZATION_SHA256
        ),
        "selected_checkpoint_sha256": (
            EXPECTED_CHECKPOINT_SHA256
        ),
        "selected_notebook_sha256": (
            EXPECTED_NOTEBOOK_SHA256
        ),
        "relation_protocol_fingerprint": (
            EXPECTED_RELATION_PROTOCOL_FINGERPRINT
        ),
        "vectorizer_fingerprint": (
            EXPECTED_VECTORIZER_FINGERPRINT
        ),
        "code_hashes_verified": len(
            protocol["code_hashes"]
        ),
        "test_captions": len(captions),
        "caption_overlap_with_train": 0,
        "caption_overlap_with_validation": 0,
        "model_dry_run": dry_run["status"],
        "selection_frozen": True,
        "further_tuning_permitted": False,
        "post_test_tuning_permitted": False,
        "test_evaluation_authorized": True,
        "maximum_authorized_evaluations": 1,
        "authorized_evaluations_completed": 0,
        "authorization_consumed": False,
        "test_evaluation_executed": False,
        "repository_test_manifest_read": False,
        "test_images_read": False,
        "selection_lock_verifier": selection["status"],
        "authorization_verifier": (
            authorization["status"]
        ),
    }

    print()
    print("=== DATASET V3 FINAL-TEST PROTOCOL VERIFIED ===")
    print(
        f"Protocol SHA-256:    "
        f"{actual_protocol_sha256}"
    )
    print("Code hashes:         PASS")
    print("Model dry run:       PASS")
    print("Test metadata read:  False (repository)")
    print("Test images read:    False")
    print("Test evaluated:      False")
    print("Authorization used:  False")
    print(
        "Status: PASS_DATASET_V3_FINAL_TEST_PROTOCOL_VERIFIED"
    )

    return result


def main() -> None:
    verify_final_test_protocol()


if __name__ == "__main__":
    main()
