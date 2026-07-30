from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests" / "dataset_v3"
LOCK_PATH = MANIFEST_DIR / "dataset_v3_final_selection_lock.json"
CHECKSUM_PATH = (
    MANIFEST_DIR / "dataset_v3_final_selection_lock.sha256.txt"
)

SOURCE_COMMIT = "8be2a8d440c4809f5941008edd7cc45221db163a"
SELECTED_MODEL_SLUG = "torch_multimodal_dataset_v3"
SELECTED_CHECKPOINT = (
    "results/dataset_v3/models/"
    "torch_multimodal_dataset_v3_state.pt"
)
NOTEBOOK_PATH = "project_v3.ipynb"
EXPECTED_NOTEBOOK_SHA256 = (
    "4f87a9818cf91d1011b4b09794da086b6cc8fc9469e10d612a7c6ae6efb1900a"
)
EXPECTED_TEST_LOCK_SHA256 = (
    "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
)

COMPONENT_PATHS = (
    "data/manifests/dataset_v3/dataset_v3_image_manifest.csv",
    "data/manifests/dataset_v3/dataset_v3_train_images.csv",
    "data/manifests/dataset_v3/dataset_v3_validation_images.csv",
    "data/manifests/dataset_v3/dataset_v3_train_relations.csv",
    "data/manifests/dataset_v3/dataset_v3_validation_relations.csv",
    "data/manifests/dataset_v3/dataset_v3_relation_summary.json",
    "data/manifests/dataset_v3/dataset_v3_development_audit.json",
    "results/dataset_v3/environment_lock.json",
    "results/dataset_v3/model_comparison.csv",
    "results/dataset_v3/multimodal_per_category.csv",
    "results/dataset_v3/multimodal_validation_predictions.csv",
    "results/dataset_v3/paired_comparisons.csv",
    "results/dataset_v3/run_info.json",
    "results/dataset_v3/training_summary.json",
    "results/dataset_v3/validation_predictions.csv",
    "results/dataset_v3/verification_summary.json",
    SELECTED_CHECKPOINT,
    NOTEBOOK_PATH,
    "src/models.py",
    "src/data_v3.py",
    "src/captions_v3.py",
    "src/train_dataset_v3.py",
    "src/verify_dataset_v3_training.py",
    "src/build_notebook_v3.py",
    "src/verify_notebook_v3.py",
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
    # Preserve leading spaces used by `git status --porcelain`.
    # Removing them shifts the two status columns and can turn
    # `README.md` into `EADME.md`.
    return completed.stdout.rstrip("\r\n")


def load_json(relative: str) -> dict:
    return json.loads(
        (PROJECT_ROOT / relative).read_text(encoding="utf-8-sig")
    )


def build_lock() -> dict:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")

    if branch != "dataset-v3":
        raise RuntimeError(
            f"Expected branch dataset-v3, found {branch}."
        )

    if head != SOURCE_COMMIT:
        raise RuntimeError(
            f"Expected source commit {SOURCE_COMMIT}, found {head}."
        )

    if git("status", "--porcelain", "--untracked-files=all"):
        allowed = {
            "README.md",
            "docs/dataset_v3/final_selection_lock.md",
            "src/build_dataset_v3_final_selection_lock.py",
            "src/verify_dataset_v3_final_selection_lock.py",
            "tests/test_dataset_v3_final_selection_lock.py",
        }
        status_lines = git(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).splitlines()

        invalid_status_lines = [
            line
            for line in status_lines
            if len(line) < 4 or line[2] != " "
        ]

        if invalid_status_lines:
            raise RuntimeError(
                "Unexpected Git porcelain status format: "
                f"{invalid_status_lines}"
            )

        actual = {
            line[3:].replace("\\", "/")
            for line in status_lines
        }
        if not actual.issubset(allowed):
            raise RuntimeError(
                "Unexpected working-tree paths before lock creation: "
                f"{sorted(actual - allowed)}"
            )

    for relative in COMPONENT_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise RuntimeError(
                f"Required lock component missing: {relative}"
            )

    training_summary = load_json(
        "results/dataset_v3/training_summary.json"
    )
    verification_summary = load_json(
        "results/dataset_v3/verification_summary.json"
    )
    run_info = load_json("results/dataset_v3/run_info.json")
    test_lock = load_json(
        "data/locked_test/dataset_v3/"
        "dataset_v3_test_lock.json"
    )

    if training_summary["status"] != (
        "PASS_DATASET_V3_DEVELOPMENT_TRAINING_ARTIFACTS_READY"
    ):
        raise RuntimeError("Unexpected training-summary status.")

    if verification_summary["status"] != (
        "PASS_DATASET_V3_DEVELOPMENT_TRAINING_VERIFIED"
    ):
        raise RuntimeError("Unexpected verification-summary status.")

    if training_summary["main_model_slug"] != SELECTED_MODEL_SLUG:
        raise RuntimeError("Unexpected selected model.")

    if test_lock["status"] != (
        "LOCKED_NOT_AUTHORIZED_FOR_TRAINING_OR_SELECTION"
    ):
        raise RuntimeError("Unexpected test-lock status.")

    if test_lock["test_lock_sha256"] != EXPECTED_TEST_LOCK_SHA256:
        raise RuntimeError("Unexpected test-lock fingerprint.")

    for name, payload in (
        ("training_summary", training_summary),
        ("verification_summary", verification_summary),
        ("run_info", run_info),
    ):
        if payload["test_lock_sha256"] != EXPECTED_TEST_LOCK_SHA256:
            raise RuntimeError(
                f"Unexpected test-lock fingerprint in {name}."
            )

        for flag in (
            "test_manifest_read",
            "test_images_read",
            "test_split_used",
            "test_evaluation_executed",
        ):
            if payload[flag] is not False:
                raise RuntimeError(
                    f"Unauthorized test flag in {name}: {flag}."
                )

    if run_info["test_evaluation_permitted"] is not False:
        raise RuntimeError(
            "Test evaluation is already marked as permitted."
        )

    notebook_sha256 = sha256(PROJECT_ROOT / NOTEBOOK_PATH)

    if notebook_sha256 != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError(
            "The selected notebook differs from the verified notebook."
        )

    component_hashes = {
        relative: sha256(PROJECT_ROOT / relative)
        for relative in COMPONENT_PATHS
    }

    results_tree = git("rev-parse", "HEAD:results/dataset_v3")
    locked_test_tree = git(
        "rev-parse",
        "HEAD:data/locked_test/dataset_v3",
    )

    payload = {
        "status": "FINAL_SELECTION_LOCKED_TEST_NOT_AUTHORIZED",
        "dataset_version": "3.0-development",
        "branch": branch,
        "source_commit": SOURCE_COMMIT,
        "selected_model_slug": SELECTED_MODEL_SLUG,
        "selected_checkpoint": SELECTED_CHECKPOINT,
        "selected_checkpoint_sha256": component_hashes[
            SELECTED_CHECKPOINT
        ],
        "selected_notebook": NOTEBOOK_PATH,
        "selected_notebook_sha256": notebook_sha256,
        "training_results_tree": results_tree,
        "locked_test_tree": locked_test_tree,
        "selection_basis": "development_validation_only",
        "selection_frozen": True,
        "further_tuning_permitted": False,
        "test_evaluation_authorized": False,
        "test_evaluation_executed": False,
        "test_manifest_read": False,
        "test_images_read": False,
        "main_validation_accuracy": training_summary[
            "main_validation_accuracy"
        ],
        "main_validation_macro_f1": training_summary[
            "main_validation_macro_f1"
        ],
        "main_correct_predictions": training_summary[
            "main_correct_predictions"
        ],
        "main_total_predictions": training_summary[
            "main_total_predictions"
        ],
        "validation_images": 80,
        "validation_rows": 480,
        "independent_audit_status": (
            "PASS_INDEPENDENT_DATASET_V3_TRAINING_AUDIT"
        ),
        "independent_audit_checks": "76/76",
        "audit_snapshot_sha256": (
            "fa81cb572db44b7b098b73137b59625456a0f57ef9596b515ce7edbef087db93"
        ),
        "environment_lock_sha256": training_summary[
            "environment_lock_sha256"
        ],
        "test_lock_status": test_lock["status"],
        "test_lock_sha256": EXPECTED_TEST_LOCK_SHA256,
        "component_hashes": component_hashes,
    }

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lock_sha256 = sha256(LOCK_PATH)
    CHECKSUM_PATH.write_text(
        f"{lock_sha256}  {LOCK_PATH.name}\n",
        encoding="utf-8",
    )

    result = {
        **payload,
        "final_selection_lock_sha256": lock_sha256,
        "final_selection_lock_path": str(
            LOCK_PATH.relative_to(PROJECT_ROOT)
        ),
        "final_selection_lock_checksum_path": str(
            CHECKSUM_PATH.relative_to(PROJECT_ROOT)
        ),
    }

    print()
    print("=== DATASET V3 FINAL SELECTION LOCK CREATED ===")
    print(f"Source commit:       {SOURCE_COMMIT}")
    print(f"Selected model:      {SELECTED_MODEL_SLUG}")
    print(
        "Validation result:  "
        f"{training_summary['main_correct_predictions']}/"
        f"{training_summary['main_total_predictions']}"
    )
    print("Further tuning:      NOT PERMITTED")
    print("Test authorization:  False")
    print("Test evaluation:     NOT EXECUTED")
    print(f"Lock SHA-256:        {lock_sha256}")
    print(
        "Status: "
        "FINAL_SELECTION_LOCKED_TEST_NOT_AUTHORIZED"
    )

    return result


def main() -> None:
    build_lock()


if __name__ == "__main__":
    main()
