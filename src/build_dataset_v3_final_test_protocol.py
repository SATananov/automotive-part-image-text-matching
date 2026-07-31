from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from src.data_v3 import (
    VALIDATION_IMAGE_MANIFEST,
    load_v3_split,
)
from src.final_test_protocol_v3 import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_RELATION_PROTOCOL_FINGERPRINT,
    EXPECTED_TEST_IMAGES,
    EXPECTED_TEST_ROWS,
    EXPECTED_TEXT_DIMENSION,
    EXPECTED_VECTORIZER_FINGERPRINT,
    FINAL_TEST_TEMPLATES,
    all_final_test_captions,
    build_final_test_relations,
    file_sha256,
    protocol_model_dry_run,
    relation_protocol_fingerprint,
    validate_final_test_relations,
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

SOURCE_COMMIT = "591ccda6294b03608a0f3b7c3fa0952af56c849d"
EXPECTED_BRANCH = "dataset-v3"
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
EXTERNAL_PREPARATION_SNAPSHOT_SHA256 = (
    "b6a66cdcdbc63f2fc6e22680ea8aa4f793b6629529d355a6a97427ef4878d54e"
)

CHECKPOINT_RELATIVE = (
    "results/dataset_v3/models/"
    "torch_multimodal_dataset_v3_state.pt"
)
NOTEBOOK_RELATIVE = "project_v3.ipynb"
SELECTION_LOCK_RELATIVE = (
    "data/manifests/dataset_v3/"
    "dataset_v3_final_selection_lock.json"
)
AUTHORIZATION_RELATIVE = (
    "data/manifests/dataset_v3/"
    "dataset_v3_final_test_authorization.json"
)

CODE_PATHS = (
    "src/final_test_protocol_v3.py",
    "src/evaluate_dataset_v3_final_test.py",
    "src/build_dataset_v3_final_test_protocol.py",
    "src/verify_dataset_v3_final_test_protocol.py",
    "tests/test_dataset_v3_final_test_protocol.py",
    "docs/dataset_v3/final_test_evaluation_protocol.md",
)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def build_final_test_protocol() -> dict[str, object]:
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

    if PROTOCOL_PATH.exists():
        raise RuntimeError(
            "Final-test protocol JSON already exists."
        )

    if PROTOCOL_CHECKSUM_PATH.exists():
        raise RuntimeError(
            "Final-test protocol checksum already exists."
        )

    selection = verify_final_selection_lock()
    authorization = verify_final_test_authorization()

    selection_path = PROJECT_ROOT / SELECTION_LOCK_RELATIVE
    authorization_path = PROJECT_ROOT / AUTHORIZATION_RELATIVE
    checkpoint_path = PROJECT_ROOT / CHECKPOINT_RELATIVE
    notebook_path = PROJECT_ROOT / NOTEBOOK_RELATIVE

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

    if sha256(checkpoint_path) != (
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

    train = load_v3_split("train")
    validation = load_v3_split("validation")
    model_dry_run = protocol_model_dry_run(
        train,
        checkpoint_path,
    )

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

    validation_images = pd.read_csv(
        VALIDATION_IMAGE_MANIFEST
    )
    dry_run_images = validation_images.copy()
    dry_run_images["split"] = "test"
    dry_run_images["test_locked"] = True

    relations = build_final_test_relations(
        dry_run_images
    )
    relation_dry_run = validate_final_test_relations(
        relations,
        dry_run_images,
        train,
        validation,
        require_locked_paths=False,
        require_group_disjoint=False,
    )

    fingerprint = relation_protocol_fingerprint()

    if fingerprint != (
        EXPECTED_RELATION_PROTOCOL_FINGERPRINT
    ):
        raise RuntimeError(
            "Final-test relation protocol fingerprint changed."
        )

    code_hashes = {}

    for relative in CODE_PATHS:
        path = PROJECT_ROOT / relative

        if not path.is_file():
            raise RuntimeError(
                f"Final-test protocol file missing: {relative}"
            )

        code_hashes[relative] = sha256(path)

    protocol = {
        "status": (
            "FINAL_TEST_EVALUATOR_PROTOCOL_READY_NOT_EXECUTED"
        ),
        "protocol_version": "dataset-v3-final-test-v1",
        "branch": EXPECTED_BRANCH,
        "protocol_source_commit": SOURCE_COMMIT,
        "final_selection_lock_sha256": (
            EXPECTED_SELECTION_LOCK_SHA256
        ),
        "final_test_authorization_sha256": (
            EXPECTED_AUTHORIZATION_SHA256
        ),
        "selected_model_slug": (
            authorization["selected_model_slug"]
        ),
        "selected_checkpoint": CHECKPOINT_RELATIVE,
        "selected_checkpoint_sha256": (
            EXPECTED_CHECKPOINT_SHA256
        ),
        "selected_notebook": NOTEBOOK_RELATIVE,
        "selected_notebook_sha256": (
            EXPECTED_NOTEBOOK_SHA256
        ),
        "test_lock_sha256": (
            EXPECTED_TEST_LOCK_SHA256
        ),
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
        "external_preparation_snapshot_sha256": (
            EXTERNAL_PREPARATION_SNAPSHOT_SHA256
        ),
        "external_snapshot_test_metadata_reviewed": True,
        "external_snapshot_locked_test_images_present": False,
        "snapshot_documentation_correction": (
            "The preparation snapshot excluded locked-test image "
            "bytes but contained 80 test metadata rows through the "
            "full image manifest and public test-lock JSON."
        ),
        "test_metadata_review_did_not_consume_evaluation": True,
        "relation_protocol_fingerprint": fingerprint,
        "final_test_templates": list(
            FINAL_TEST_TEMPLATES
        ),
        "final_test_caption_count": len(captions),
        "exact_caption_overlap_with_train": 0,
        "exact_caption_overlap_with_validation": 0,
        "pairs_per_label_per_image": 2,
        "expected_test_images": EXPECTED_TEST_IMAGES,
        "expected_test_rows": EXPECTED_TEST_ROWS,
        "expected_rows_per_label": 160,
        "text_dimension": EXPECTED_TEXT_DIMENSION,
        "vectorizer_fingerprint": (
            EXPECTED_VECTORIZER_FINGERPRINT
        ),
        "protocol_dry_run": {
            "uses_development_validation_metadata_only": True,
            "reads_locked_test_manifest": False,
            "reads_locked_test_images": False,
            "relation_design": relation_dry_run,
            "model_interface": model_dry_run,
        },
        "execution_command": (
            "python -m src.evaluate_dataset_v3_final_test "
            "--execute-authorized-final-test "
            "--confirm-authorization-sha "
            f"{EXPECTED_AUTHORIZATION_SHA256}"
        ),
        "execution_guard": {
            "explicit_command_required": True,
            "final_result_directory_must_not_exist": True,
            "temporary_execution_directory_must_not_exist": True,
            "authorization_is_consumed_before_first_test_image_read": True,
            "automatic_rerun_after_test_access": False,
        },
        "expected_result_directory": (
            "results/dataset_v3_final_test"
        ),
        "expected_result_files": [
            "execution_state.json",
            "test_image_manifest.csv",
            "test_relations.csv",
            "test_predictions.csv",
            "test_metrics.json",
            "test_per_category.csv",
            "test_confusion_matrix.csv",
            "authorization_consumption.json",
            "environment.json",
        ],
        "code_hashes": code_hashes,
        "selection_lock_verifier": selection["status"],
        "authorization_verifier": authorization["status"],
    }

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    PROTOCOL_PATH.write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    protocol_sha256 = sha256(PROTOCOL_PATH)
    PROTOCOL_CHECKSUM_PATH.write_text(
        f"{protocol_sha256}  {PROTOCOL_PATH.name}\n",
        encoding="utf-8",
    )

    result = {
        **protocol,
        "protocol_sha256": protocol_sha256,
    }

    print()
    print("=== DATASET V3 FINAL-TEST PROTOCOL CREATED ===")
    print(f"Source commit:       {SOURCE_COMMIT}")
    print(
        f"Selected model:      "
        f"{authorization['selected_model_slug']}"
    )
    print(
        f"Text dimension:      "
        f"{EXPECTED_TEXT_DIMENSION}"
    )
    print("Test captions:       16")
    print("Test metadata read:  False (repository)")
    print("Test images read:    False")
    print("Test evaluated:      False")
    print("Authorization used:  False")
    print(f"Protocol SHA-256:    {protocol_sha256}")
    print(
        "Status: FINAL_TEST_EVALUATOR_PROTOCOL_READY_NOT_EXECUTED"
    )

    return result


def main() -> None:
    build_final_test_protocol()


if __name__ == "__main__":
    main()
