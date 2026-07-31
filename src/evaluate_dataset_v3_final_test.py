from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data import LABELS, PROJECT_ROOT
from src.data_v3 import (
    CATEGORIES,
    IMAGE_MANIFEST_COLUMNS,
    load_v3_split,
)
from src.final_test_protocol_v3 import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TEST_IMAGES,
    EXPECTED_TEST_IMAGES_PER_CATEGORY,
    EXPECTED_TEST_ROWS,
    build_final_test_relations,
    file_sha256,
    fit_frozen_text_vectorizer,
    load_frozen_multimodal_model,
    validate_final_test_relations,
)
from src.train import (
    BATCH_SIZE,
    RANDOM_STATE,
    grouped_bootstrap_interval,
    require_canonical_torch_version,
    set_seed,
)
from src.verify_dataset_v3_final_test_protocol import (
    verify_final_test_protocol,
)

MANIFEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_v3"
)
FULL_IMAGE_MANIFEST = (
    MANIFEST_DIR
    / "dataset_v3_image_manifest.csv"
)
PUBLIC_TEST_LOCK = (
    PROJECT_ROOT
    / "data"
    / "locked_test"
    / "dataset_v3"
    / "dataset_v3_test_lock.json"
)
PUBLIC_TEST_LOCK_CHECKSUM = (
    PROJECT_ROOT
    / "data"
    / "locked_test"
    / "dataset_v3"
    / "dataset_v3_test_lock.sha256.txt"
)
AUTHORIZATION_PATH = (
    MANIFEST_DIR
    / "dataset_v3_final_test_authorization.json"
)
PROTOCOL_PATH = (
    MANIFEST_DIR
    / "dataset_v3_final_test_protocol.json"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "dataset_v3"
    / "models"
    / "torch_multimodal_dataset_v3_state.pt"
)
FINAL_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "dataset_v3_final_test"
)
TEMP_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / ".dataset_v3_final_test_execution"
)

EXPECTED_AUTHORIZATION_SHA256 = (
    "a58005028f389c9c27ec737d079cc4a750b9c555fa4dfba350017ebdb8754a24"
)
EXPECTED_TEST_LOCK_SHA256 = (
    "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
)
EXECUTION_CONFIRMATION = EXPECTED_AUTHORIZATION_SHA256


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def update_execution_state(
    state: dict[str, Any],
    **changes: Any,
) -> None:
    state.update(changes)
    write_json(
        TEMP_RESULTS_DIR / "execution_state.json",
        state,
    )


def verify_authorization_state() -> dict[str, Any]:
    authorization = load_json(
        AUTHORIZATION_PATH
    )

    if file_sha256(AUTHORIZATION_PATH) != (
        EXPECTED_AUTHORIZATION_SHA256
    ):
        raise RuntimeError(
            "Final-test authorization SHA-256 changed."
        )

    if authorization["status"] != (
        "FINAL_TEST_EVALUATION_AUTHORIZED_NOT_EXECUTED"
    ):
        raise RuntimeError(
            "Unexpected final-test authorization status."
        )

    if authorization[
        "test_evaluation_authorized"
    ] is not True:
        raise RuntimeError(
            "Final test is not authorized."
        )

    if authorization[
        "maximum_authorized_evaluations"
    ] != 1:
        raise RuntimeError(
            "Authorization is not limited to one evaluation."
        )

    if authorization[
        "authorized_evaluations_completed"
    ] != 0:
        raise RuntimeError(
            "Authorization reports a completed evaluation."
        )

    if authorization[
        "authorization_consumed"
    ] is not False:
        raise RuntimeError(
            "Authorization is already consumed."
        )

    for field in (
        "test_evaluation_executed",
        "test_manifest_read",
        "test_images_read",
        "test_results_may_be_used_for_tuning",
        "post_test_tuning_permitted",
        "further_tuning_permitted",
    ):
        if authorization[field] is not False:
            raise RuntimeError(
                f"Unexpected authorization flag: {field}"
            )

    if authorization[
        "test_lock_sha256"
    ] != EXPECTED_TEST_LOCK_SHA256:
        raise RuntimeError(
            "Authorization test-lock fingerprint changed."
        )

    return authorization


def preflight_without_test_access() -> dict[str, Any]:
    require_canonical_torch_version()
    protocol = verify_final_test_protocol()
    authorization = verify_authorization_state()

    if FINAL_RESULTS_DIR.exists():
        raise FileExistsError(
            "Final-test result directory already exists. "
            "The one-time evaluation cannot be rerun."
        )

    if TEMP_RESULTS_DIR.exists():
        raise FileExistsError(
            "A final-test execution state already exists. "
            "Do not rerun before independent review."
        )

    train = load_v3_split("train")
    vectorizer, vectorizer_fingerprint = (
        fit_frozen_text_vectorizer(
            train
        )
    )
    model, checkpoint = (
        load_frozen_multimodal_model(
            CHECKPOINT_PATH
        )
    )

    sample_text = vectorizer.transform(
        [
            "Automotive component: automotive alternator, "
            "part of the engine-support and electrical group."
        ]
    ).toarray().astype(np.float32)

    with torch.no_grad():
        outputs = model(
            torch.zeros(
                (1, 3, 48, 48),
                dtype=torch.float32,
            ),
            torch.from_numpy(sample_text),
        )

    if [list(output.shape) for output in outputs] != [
        [1, 3],
        [1, len(CATEGORIES)],
        [1, len(CATEGORIES)],
    ]:
        raise RuntimeError(
            "Selected model preflight output shapes changed."
        )

    return {
        "status": "PASS_FINAL_TEST_PREFLIGHT_NO_TEST_ACCESS",
        "protocol_sha256": (
            protocol["protocol_sha256"]
        ),
        "authorization_sha256": (
            EXPECTED_AUTHORIZATION_SHA256
        ),
        "checkpoint_sha256": (
            EXPECTED_CHECKPOINT_SHA256
        ),
        "checkpoint_model_slug": (
            checkpoint["model_slug"]
        ),
        "vectorizer_fingerprint": (
            vectorizer_fingerprint
        ),
        "test_evaluation_authorized": True,
        "maximum_authorized_evaluations": 1,
        "authorized_evaluations_completed": 0,
        "authorization_consumed": False,
        "repository_test_manifest_read": False,
        "test_images_read": False,
        "test_evaluation_executed": False,
    }


def load_authorized_test_images() -> pd.DataFrame:
    manifest = pd.read_csv(
        FULL_IMAGE_MANIFEST
    )

    if tuple(manifest.columns) != (
        IMAGE_MANIFEST_COLUMNS
    ):
        raise RuntimeError(
            "Unexpected Dataset V3 full image manifest columns."
        )

    test = manifest[
        manifest["split"].astype(str).eq("test")
    ].copy()

    if len(test) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_IMAGES} test images, "
            f"found {len(test)}."
        )

    per_category = (
        test["project_category"]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    if set(per_category.index) != set(CATEGORIES):
        raise RuntimeError(
            "Unexpected final-test categories."
        )

    if not (
        per_category
        == EXPECTED_TEST_IMAGES_PER_CATEGORY
    ).all():
        raise RuntimeError(
            "Final-test image categories are not balanced."
        )

    locked = (
        test["test_locked"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not locked.isin({"true", "1"}).all():
        raise RuntimeError(
            "A final-test manifest row is not marked locked."
        )

    paths = (
        test["repository_relative_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    if not paths.str.startswith(
        "data/locked_test/dataset_v3/images/"
    ).all():
        raise RuntimeError(
            "A final-test image path is outside "
            "the locked directory."
        )

    for column in (
        "candidate_id",
        "image_group_id",
        "repository_relative_path",
        "sha256",
    ):
        if test[column].duplicated().any():
            raise RuntimeError(
                f"Duplicate final-test {column}."
            )

    lock = load_json(
        PUBLIC_TEST_LOCK
    )
    checksum = (
        PUBLIC_TEST_LOCK_CHECKSUM.read_text(
            encoding="utf-8-sig"
        ).split()[0]
    )

    if lock["test_lock_sha256"] != (
        EXPECTED_TEST_LOCK_SHA256
    ):
        raise RuntimeError(
            "Unexpected public test-lock fingerprint."
        )

    if checksum != EXPECTED_TEST_LOCK_SHA256:
        raise RuntimeError(
            "Unexpected public test-lock checksum."
        )

    if lock["test_image_count"] != (
        EXPECTED_TEST_IMAGES
    ):
        raise RuntimeError(
            "Unexpected public test-lock image count."
        )

    locked_records = {
        (
            str(row["candidate_id"]),
            str(row["project_category"]),
            str(row["image_group_id"]),
            str(row["sha256"]),
        )
        for row in lock["images"]
    }
    manifest_records = {
        (
            str(row.candidate_id),
            str(row.project_category),
            str(row.image_group_id),
            str(row.sha256),
        )
        for row in test.itertuples(index=False)
    }

    if locked_records != manifest_records:
        raise RuntimeError(
            "Full test manifest does not match "
            "the public test-lock records."
        )

    return test.sort_values(
        [
            "project_category",
            "split_rank_within_category",
            "candidate_id",
        ],
        kind="stable",
    ).reset_index(drop=True)


def load_and_verify_test_images(
    relations: pd.DataFrame,
    test_images: pd.DataFrame,
) -> np.ndarray:
    expected_hashes = {
        str(row.repository_relative_path): str(
            row.sha256
        )
        for row in test_images.itertuples(index=False)
    }
    cache: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []

    for relative_path in (
        relations["image_path"].astype(str)
    ):
        if relative_path not in cache:
            path = PROJECT_ROOT / relative_path

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing locked-test image: {path}"
                )

            actual_hash = file_sha256(path)
            expected_hash = expected_hashes[
                relative_path
            ]

            if actual_hash != expected_hash:
                raise RuntimeError(
                    "Locked-test image SHA-256 mismatch: "
                    f"{relative_path}"
                )

            with Image.open(path) as image:
                cache[relative_path] = np.asarray(
                    image.convert("RGB").resize(
                        (48, 48),
                        Image.Resampling.BILINEAR,
                    ),
                    dtype=np.float32,
                ) / 255.0

        rows.append(cache[relative_path])

    if len(cache) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            "Unexpected unique locked-test image count."
        )

    images = np.stack(rows)
    return np.transpose(
        images,
        (0, 3, 1, 2),
    ).astype(np.float32)


def predict_in_batches(
    model: torch.nn.Module,
    images: np.ndarray,
    text: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relation_predictions: list[np.ndarray] = []
    image_category_predictions: list[np.ndarray] = []
    text_category_predictions: list[np.ndarray] = []

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(images),
            BATCH_SIZE,
        ):
            stop = min(
                start + BATCH_SIZE,
                len(images),
            )
            outputs = model(
                torch.from_numpy(
                    images[start:stop]
                ),
                torch.from_numpy(
                    text[start:stop]
                ),
            )
            relation_predictions.append(
                outputs[0].argmax(
                    dim=1
                ).cpu().numpy()
            )
            image_category_predictions.append(
                outputs[1].argmax(
                    dim=1
                ).cpu().numpy()
            )
            text_category_predictions.append(
                outputs[2].argmax(
                    dim=1
                ).cpu().numpy()
            )

    return (
        np.concatenate(relation_predictions),
        np.concatenate(image_category_predictions),
        np.concatenate(text_category_predictions),
    )


def environment_snapshot() -> dict[str, Any]:
    distributions = (
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pillow",
    )
    versions = {}

    for distribution in distributions:
        try:
            versions[distribution] = metadata.version(
                distribution
            )
        except metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def evaluate_authorized_final_test() -> dict[str, Any]:
    preflight = preflight_without_test_access()
    authorization = verify_authorization_state()

    TEMP_RESULTS_DIR.mkdir(
        parents=False,
        exist_ok=False,
    )

    state: dict[str, Any] = {
        "status": "FINAL_TEST_EXECUTION_PREPARED",
        "authorization_sha256": (
            EXPECTED_AUTHORIZATION_SHA256
        ),
        "authorization_consumed": False,
        "authorized_evaluations_completed": 0,
        "repository_test_manifest_read": False,
        "test_images_read": False,
        "test_evaluation_executed": False,
        "preflight": preflight,
    }
    update_execution_state(state)

    try:
        update_execution_state(
            state,
            status=(
                "FINAL_TEST_METADATA_ACCESS_STARTED"
            ),
            authorization_consumed=True,
            authorized_evaluations_completed=1,
            repository_test_manifest_read=True,
        )

        test_images = load_authorized_test_images()
        train = load_v3_split("train")
        validation = load_v3_split("validation")
        relations = build_final_test_relations(
            test_images
        )
        relation_summary = (
            validate_final_test_relations(
                relations,
                test_images,
                train,
                validation,
                require_locked_paths=True,
            )
        )

        test_images.to_csv(
            TEMP_RESULTS_DIR
            / "test_image_manifest.csv",
            index=False,
            lineterminator="\n",
        )
        relations.to_csv(
            TEMP_RESULTS_DIR
            / "test_relations.csv",
            index=False,
            lineterminator="\n",
        )

        vectorizer, vectorizer_fingerprint = (
            fit_frozen_text_vectorizer(
                train
            )
        )
        text = vectorizer.transform(
            relations["description"].astype(str)
        ).toarray().astype(np.float32)

        model, checkpoint = (
            load_frozen_multimodal_model(
                CHECKPOINT_PATH
            )
        )

        update_execution_state(
            state,
            status=(
                "FINAL_TEST_IMAGE_ACCESS_STARTED"
            ),
            test_images_read=True,
        )

        images = load_and_verify_test_images(
            relations,
            test_images,
        )

        set_seed(RANDOM_STATE)
        torch.set_num_threads(
            max(
                1,
                min(
                    4,
                    torch.get_num_threads(),
                ),
            )
        )

        (
            relation_index,
            image_category_index,
            text_category_index,
        ) = predict_in_batches(
            model,
            images,
            text,
        )

        labels = np.asarray(
            checkpoint["labels"]
        )
        categories = np.asarray(
            checkpoint["categories"]
        )
        predicted = labels[
            relation_index
        ]
        image_category_predicted = categories[
            image_category_index
        ]
        text_category_predicted = categories[
            text_category_index
        ]

        true = relations["label"].to_numpy()
        group_ids = (
            relations["image_id"].to_numpy()
        )
        accuracy = float(
            accuracy_score(
                true,
                predicted,
            )
        )
        macro_f1 = float(
            f1_score(
                true,
                predicted,
                average="macro",
                zero_division=0,
            )
        )
        accuracy_ci = grouped_bootstrap_interval(
            true,
            predicted,
            group_ids,
            lambda y_true, y_pred: float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
        )
        macro_f1_ci = grouped_bootstrap_interval(
            true,
            predicted,
            group_ids,
            lambda y_true, y_pred: float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),
            seed=RANDOM_STATE + 1,
        )

        image_category_accuracy = float(
            np.mean(
                image_category_predicted
                == relations[
                    "part_category"
                ].to_numpy()
            )
        )
        text_category_accuracy = float(
            np.mean(
                text_category_predicted
                == relations[
                    "text_category"
                ].to_numpy()
            )
        )

        predictions = relations.copy()
        predictions = predictions.rename(
            columns={
                "label": "true_label",
            }
        )
        predictions["predicted_label"] = (
            predicted
        )
        predictions["is_correct"] = (
            predictions["true_label"]
            .eq(
                predictions[
                    "predicted_label"
                ]
            )
        )
        predictions[
            "predicted_image_category"
        ] = image_category_predicted
        predictions[
            "predicted_text_category"
        ] = text_category_predicted
        predictions.to_csv(
            TEMP_RESULTS_DIR
            / "test_predictions.csv",
            index=False,
            lineterminator="\n",
        )

        per_category_rows = []

        for category, group in (
            predictions.groupby(
                "part_category",
                sort=True,
            )
        ):
            per_category_rows.append(
                {
                    "part_category": category,
                    "samples": len(group),
                    "independent_images": int(
                        group[
                            "image_id"
                        ].nunique()
                    ),
                    "correct": int(
                        group[
                            "is_correct"
                        ].sum()
                    ),
                    "accuracy": float(
                        accuracy_score(
                            group[
                                "true_label"
                            ],
                            group[
                                "predicted_label"
                            ],
                        )
                    ),
                    "macro_f1": float(
                        f1_score(
                            group[
                                "true_label"
                            ],
                            group[
                                "predicted_label"
                            ],
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }
            )

        pd.DataFrame(
            per_category_rows
        ).to_csv(
            TEMP_RESULTS_DIR
            / "test_per_category.csv",
            index=False,
            lineterminator="\n",
        )

        matrix = confusion_matrix(
            true,
            predicted,
            labels=list(LABELS),
        )
        pd.DataFrame(
            matrix,
            index=[
                f"true_{label}"
                for label in LABELS
            ],
            columns=[
                f"predicted_{label}"
                for label in LABELS
            ],
        ).to_csv(
            TEMP_RESULTS_DIR
            / "test_confusion_matrix.csv",
            lineterminator="\n",
        )

        report = classification_report(
            true,
            predicted,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "status": (
                "PASS_DATASET_V3_FINAL_TEST_EVALUATION_COMPLETE"
            ),
            "model_slug": checkpoint["model_slug"],
            "checkpoint_sha256": (
                EXPECTED_CHECKPOINT_SHA256
            ),
            "authorization_sha256": (
                EXPECTED_AUTHORIZATION_SHA256
            ),
            "authorization_consumed": True,
            "authorized_evaluations_completed": 1,
            "test_evaluation_executed": True,
            "repository_test_manifest_read": True,
            "test_images_read": True,
            "test_images": EXPECTED_TEST_IMAGES,
            "test_rows": EXPECTED_TEST_ROWS,
            "correct_predictions": int(
                np.sum(
                    true == predicted
                )
            ),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "accuracy_ci": list(
                accuracy_ci
            ),
            "macro_f1_ci": list(
                macro_f1_ci
            ),
            "image_category_accuracy": (
                image_category_accuracy
            ),
            "text_category_accuracy": (
                text_category_accuracy
            ),
            "classification_report": report,
            "relation_protocol": relation_summary,
            "vectorizer_fingerprint": (
                vectorizer_fingerprint
            ),
            "selection_frozen": True,
            "further_tuning_permitted": False,
            "post_test_tuning_permitted": False,
            "test_results_may_be_used_for_tuning": False,
            "test_results_use": (
                "final_reporting_and_error_analysis_only"
            ),
        }
        write_json(
            TEMP_RESULTS_DIR
            / "test_metrics.json",
            metrics,
        )

        consumption = {
            "status": (
                "FINAL_TEST_AUTHORIZATION_CONSUMED"
            ),
            "authorization_sha256": (
                EXPECTED_AUTHORIZATION_SHA256
            ),
            "selected_model_slug": (
                checkpoint["model_slug"]
            ),
            "selected_checkpoint_sha256": (
                EXPECTED_CHECKPOINT_SHA256
            ),
            "maximum_authorized_evaluations": 1,
            "authorized_evaluations_completed": 1,
            "authorization_consumed": True,
            "repository_test_manifest_read": True,
            "test_images_read": True,
            "test_evaluation_executed": True,
            "further_tuning_permitted": False,
            "post_test_tuning_permitted": False,
            "test_results_may_be_used_for_tuning": False,
        }
        write_json(
            TEMP_RESULTS_DIR
            / "authorization_consumption.json",
            consumption,
        )
        write_json(
            TEMP_RESULTS_DIR
            / "environment.json",
            environment_snapshot(),
        )

        update_execution_state(
            state,
            status=(
                "FINAL_TEST_EVALUATION_COMPLETE"
            ),
            test_evaluation_executed=True,
            result_status=metrics["status"],
            accuracy=accuracy,
            macro_f1=macro_f1,
            correct_predictions=metrics[
                "correct_predictions"
            ],
            total_predictions=EXPECTED_TEST_ROWS,
        )

        TEMP_RESULTS_DIR.replace(
            FINAL_RESULTS_DIR
        )

        print()
        print("=== DATASET V3 FINAL TEST COMPLETE ===")
        print(
            f"Model:               "
            f"{checkpoint['model_slug']}"
        )
        print(
            f"Correct:             "
            f"{metrics['correct_predictions']}/"
            f"{EXPECTED_TEST_ROWS}"
        )
        print(
            f"Accuracy:            {accuracy:.10f}"
        )
        print(
            f"Macro F1:            {macro_f1:.10f}"
        )
        print("Authorization used:  True")
        print("Further tuning:      False")
        print(
            "Status: PASS_DATASET_V3_FINAL_TEST_"
            "EVALUATION_COMPLETE"
        )

        return metrics

    except Exception as error:
        if TEMP_RESULTS_DIR.exists():
            update_execution_state(
                state,
                status=(
                    "FINAL_TEST_EVALUATION_FAILED_AFTER_"
                    "EXECUTION_STATE_CREATED"
                ),
                error_type=type(error).__name__,
                error_message=str(error),
            )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset V3 one-time final locked-test evaluator."
        )
    )
    mode = parser.add_mutually_exclusive_group(
        required=True
    )
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Verify the frozen evaluator without reading "
            "the test manifest or images."
        ),
    )
    mode.add_argument(
        "--execute-authorized-final-test",
        action="store_true",
        help=(
            "Consume the one-time authorization and execute "
            "the final locked-test evaluation."
        ),
    )
    parser.add_argument(
        "--confirm-authorization-sha",
        default="",
        help=(
            "Exact authorization SHA-256 required for "
            "final execution."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.preflight_only:
        print(
            json.dumps(
                preflight_without_test_access(),
                indent=2,
            )
        )
        return

    if args.confirm_authorization_sha != (
        EXECUTION_CONFIRMATION
    ):
        raise RuntimeError(
            "Exact authorization SHA-256 confirmation "
            "is required for final execution."
        )

    evaluate_authorized_final_test()


if __name__ == "__main__":
    main()
