from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
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
    EXPECTED_RELATION_PROTOCOL_FINGERPRINT,
    EXPECTED_TEST_IMAGES,
    EXPECTED_TEST_ROWS,
    build_final_test_relations,
    relation_protocol_fingerprint,
    validate_final_test_relations,
)
from src.train import (
    RANDOM_STATE,
    grouped_bootstrap_interval,
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "dataset_v3_final_test"
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
PROTOCOL_PATH = (
    MANIFEST_DIR
    / "dataset_v3_final_test_protocol.json"
)
AUTHORIZATION_PATH = (
    MANIFEST_DIR
    / "dataset_v3_final_test_authorization.json"
)
SELECTION_LOCK_PATH = (
    MANIFEST_DIR
    / "dataset_v3_final_selection_lock.json"
)
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "dataset_v3"
    / "models"
    / "torch_multimodal_dataset_v3_state.pt"
)

EXPECTED_PROTOCOL_SOURCE_COMMIT = (
    "a18dc841dc3db066d391ca5aac8f0da2b6f43b0e"
)
EXPECTED_PROTOCOL_SHA256 = (
    "4d0936a02578a4759754223eaf45d4cc0d201bf415c71c94e115ccd51296e000"
)
EXPECTED_AUTHORIZATION_SHA256 = (
    "a58005028f389c9c27ec737d079cc4a750b9c555fa4dfba350017ebdb8754a24"
)
EXPECTED_SELECTION_LOCK_SHA256 = (
    "54be116ab887680dcca1db08154936e3ef01b0ca8c4f73807e9324f64a9b865f"
)
EXPECTED_TEST_LOCK_SHA256 = (
    "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
)
EXPECTED_VECTORIZER_FINGERPRINT = (
    "aad4f36d127ea8eb06cc3e407c1873bf93ee24576e136129f837eb2f32eab5ac"
)
EXPECTED_EXTERNAL_EVIDENCE_SHA256 = (
    "e2c84eae85cb2f33b6241a9e2ebc7939a3b56e686a2a7281a9355f9571d8a506"
)
EXPECTED_ORIGINAL_METRICS_SHA256 = (
    "5c4c2ea793779a5ea894e5124afd5a8f48c2df8fa0d7266e22436f144f084818"
)

EXPECTED_RESULT_HASHES = {
    "authorization_consumption.json": (
        "81622d702324b5c8f1783c53a3cea9b1265fc894904637a0947288d47b158282"
    ),
    "environment.json": (
        "1f42dbf82ad4eee51b1d623cf6af43596a9d31ecc651a26109287e1495e3cb92"
    ),
    "execution_state.json": (
        "ca30c5fdcb1473245374b4353411f3c6386157349be6fba6c48450a93c595865"
    ),
    "test_confusion_matrix.csv": (
        "46c82464de40474f3f65d47a82f761cf76a74561b2521c9d7c1b58016e0f157e"
    ),
    "test_image_manifest.csv": (
        "7be43af6e1a7e352a32a86b4812a38d20b1e3f225b073779e9de92df816301f3"
    ),
    "test_metrics.json": EXPECTED_ORIGINAL_METRICS_SHA256,
    "test_per_category.csv": (
        "d165659faae2b5c8fbc16c9d88b60959589124a4ba9c0aa6c4f1382b54e93e89"
    ),
    "test_predictions.csv": (
        "d22b38a3ca3d780f3d0e1f2cfdf3a4f901ad50d5b81a75a769c381310e931a6c"
    ),
    "test_relations.csv": (
        "4e228e8b7afccc8db7b62e56304e5f3fe200b4bc90468788649115de6e31e05c"
    ),
}

EXPECTED_RESULT_FILES = set(EXPECTED_RESULT_HASHES)
EXPECTED_ROWS_PER_LABEL = EXPECTED_TEST_ROWS // len(LABELS)
EXPECTED_IMAGES_PER_CATEGORY = EXPECTED_TEST_IMAGES // len(CATEGORIES)
EXPECTED_ROWS_PER_CATEGORY = EXPECTED_TEST_ROWS // len(CATEGORIES)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def close_float(
    left: Any,
    right: Any,
    *,
    tolerance: float = 1e-12,
) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def require_float(
    actual: Any,
    expected: Any,
    field: str,
) -> None:
    if not close_float(actual, expected):
        raise RuntimeError(
            f"Numeric mismatch for {field}: "
            f"expected {expected}, found {actual}"
        )


def require_nested_close(
    actual: Any,
    expected: Any,
    path: str,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise RuntimeError(
                f"Expected a mapping at {path}."
            )

        if set(actual) != set(expected):
            raise RuntimeError(
                f"Mapping keys differ at {path}: "
                f"expected {sorted(expected)}, found {sorted(actual)}"
            )

        for key in expected:
            require_nested_close(
                actual[key],
                expected[key],
                f"{path}.{key}",
            )
        return

    if isinstance(expected, float):
        require_float(actual, expected, path)
        return

    if actual != expected:
        raise RuntimeError(
            f"Value mismatch at {path}: "
            f"expected {expected!r}, found {actual!r}"
        )


def parse_is_correct(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    if not normalized.isin(
        {"true", "false", "1", "0"}
    ).all():
        raise RuntimeError(
            "Unexpected is_correct values."
        )

    return normalized.isin({"true", "1"}).to_numpy()


def verify_result_artifact_hashes() -> dict[str, str]:
    if not RESULTS_DIR.is_dir():
        raise RuntimeError(
            f"Final-test result directory missing: {RESULTS_DIR}"
        )

    actual_files = {
        path.name
        for path in RESULTS_DIR.iterdir()
        if path.is_file()
    }

    if actual_files != EXPECTED_RESULT_FILES:
        missing = sorted(
            EXPECTED_RESULT_FILES - actual_files
        )
        unexpected = sorted(
            actual_files - EXPECTED_RESULT_FILES
        )
        raise RuntimeError(
            "Unexpected final-test result inventory. "
            f"Missing={missing}; unexpected={unexpected}"
        )

    actual_hashes = {
        name: file_sha256(RESULTS_DIR / name)
        for name in sorted(EXPECTED_RESULT_FILES)
    }

    if actual_hashes != EXPECTED_RESULT_HASHES:
        differences = {
            name: {
                "expected": EXPECTED_RESULT_HASHES[name],
                "actual": actual_hashes[name],
            }
            for name in EXPECTED_RESULT_HASHES
            if actual_hashes[name]
            != EXPECTED_RESULT_HASHES[name]
        }
        raise RuntimeError(
            "A final-test result artifact changed: "
            f"{differences}"
        )

    return actual_hashes


def verify_frozen_inputs() -> dict[str, str]:
    expected = {
        "protocol": (
            PROTOCOL_PATH,
            EXPECTED_PROTOCOL_SHA256,
        ),
        "authorization": (
            AUTHORIZATION_PATH,
            EXPECTED_AUTHORIZATION_SHA256,
        ),
        "selection_lock": (
            SELECTION_LOCK_PATH,
            EXPECTED_SELECTION_LOCK_SHA256,
        ),
        "checkpoint": (
            CHECKPOINT_PATH,
            EXPECTED_CHECKPOINT_SHA256,
        ),
    }
    actual: dict[str, str] = {}

    for name, (path, expected_hash) in expected.items():
        if not path.is_file():
            raise RuntimeError(
                f"Frozen input missing: {path}"
            )

        actual_hash = file_sha256(path)

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Frozen input changed: {name}"
            )

        actual[name] = actual_hash

    return actual


def verify_image_manifest(
    saved: pd.DataFrame,
) -> dict[str, Any]:
    if tuple(saved.columns) != IMAGE_MANIFEST_COLUMNS:
        raise RuntimeError(
            "Unexpected saved test-image manifest columns."
        )

    full = pd.read_csv(FULL_IMAGE_MANIFEST)
    test = (
        full[
            full["split"].astype(str).eq("test")
        ]
        .sort_values(
            [
                "project_category",
                "split_rank_within_category",
                "candidate_id",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    assert_frame_equal(
        saved.reset_index(drop=True),
        test,
        check_dtype=False,
        check_exact=True,
    )

    if len(saved) != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            "Unexpected saved test-image count."
        )

    for column in (
        "candidate_id",
        "image_group_id",
        "repository_relative_path",
        "sha256",
    ):
        if saved[column].astype(str).nunique() != (
            EXPECTED_TEST_IMAGES
        ):
            raise RuntimeError(
                f"Unexpected uniqueness for {column}."
            )

    category_counts = (
        saved["project_category"]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    if set(category_counts.index) != set(CATEGORIES):
        raise RuntimeError(
            "Unexpected test-image categories."
        )

    if not (
        category_counts
        == EXPECTED_IMAGES_PER_CATEGORY
    ).all():
        raise RuntimeError(
            "Test images are not category-balanced."
        )

    paths = (
        saved["repository_relative_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    if not paths.str.startswith(
        "data/locked_test/dataset_v3/images/"
    ).all():
        raise RuntimeError(
            "A saved test-image path is outside the locked directory."
        )

    lock = load_json(PUBLIC_TEST_LOCK)

    if lock["test_lock_sha256"] != (
        EXPECTED_TEST_LOCK_SHA256
    ):
        raise RuntimeError(
            "Public test-lock fingerprint changed."
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
    saved_records = {
        (
            str(row.candidate_id),
            str(row.project_category),
            str(row.image_group_id),
            str(row.sha256),
        )
        for row in saved.itertuples(index=False)
    }

    if locked_records != saved_records:
        raise RuntimeError(
            "Saved test-image metadata differs from the public lock."
        )

    return {
        "images": len(saved),
        "categories": len(category_counts),
        "images_per_category": EXPECTED_IMAGES_PER_CATEGORY,
        "full_manifest_match": True,
        "public_test_lock_match": True,
    }


def verify_relations(
    saved_images: pd.DataFrame,
    saved_relations: pd.DataFrame,
) -> dict[str, Any]:
    train = load_v3_split("train")
    validation = load_v3_split("validation")
    expected_relations = build_final_test_relations(
        saved_images
    )

    assert_frame_equal(
        saved_relations.reset_index(drop=True),
        expected_relations.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )

    relation_summary = validate_final_test_relations(
        saved_relations,
        saved_images,
        train,
        validation,
        require_locked_paths=True,
        require_group_disjoint=True,
    )

    fingerprint = relation_protocol_fingerprint()

    if fingerprint != (
        EXPECTED_RELATION_PROTOCOL_FINGERPRINT
    ):
        raise RuntimeError(
            "Frozen relation protocol fingerprint changed."
        )

    return {
        "summary": relation_summary,
        "relation_protocol_fingerprint": fingerprint,
        "saved_relations_match_frozen_builder": True,
        "rows": len(saved_relations),
        "images": int(
            saved_relations["image_id"]
            .astype(str)
            .nunique()
        ),
    }


def verify_predictions_and_metrics(
    relations: pd.DataFrame,
    predictions: pd.DataFrame,
    saved_metrics: dict[str, Any],
    saved_per_category: pd.DataFrame,
    saved_matrix: pd.DataFrame,
    relation_audit: dict[str, Any],
) -> dict[str, Any]:
    expected_prediction_columns = (
        list(relations.columns[:-2])
        + [
            "true_label",
            "source",
            "predicted_label",
            "is_correct",
            "predicted_image_category",
            "predicted_text_category",
        ]
    )

    if list(predictions.columns) != expected_prediction_columns:
        raise RuntimeError(
            "Unexpected final-test prediction columns."
        )

    relation_copy = relations.rename(
        columns={"label": "true_label"}
    )

    shared_columns = [
        column
        for column in relation_copy.columns
        if column in predictions.columns
    ]

    assert_frame_equal(
        predictions[shared_columns].reset_index(drop=True),
        relation_copy[shared_columns].reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )

    if len(predictions) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Unexpected final-test prediction count."
        )

    if predictions["sample_id"].astype(str).nunique() != (
        EXPECTED_TEST_ROWS
    ):
        raise RuntimeError(
            "Duplicate final-test prediction sample IDs."
        )

    true = predictions[
        "true_label"
    ].astype(str).to_numpy()
    predicted = predictions[
        "predicted_label"
    ].astype(str).to_numpy()
    groups = predictions[
        "image_id"
    ].astype(str).to_numpy()

    expected_correct_mask = true == predicted
    saved_correct_mask = parse_is_correct(
        predictions["is_correct"]
    )

    if not np.array_equal(
        expected_correct_mask,
        saved_correct_mask,
    ):
        raise RuntimeError(
            "Saved is_correct values are inconsistent."
        )

    correct = int(expected_correct_mask.sum())
    accuracy = float(
        accuracy_score(true, predicted)
    )
    macro_f1 = float(
        f1_score(
            true,
            predicted,
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
    )
    accuracy_ci = grouped_bootstrap_interval(
        true,
        predicted,
        groups,
        lambda y_true, y_pred: float(
            accuracy_score(y_true, y_pred)
        ),
    )
    macro_f1_ci = grouped_bootstrap_interval(
        true,
        predicted,
        groups,
        lambda y_true, y_pred: float(
            f1_score(
                y_true,
                y_pred,
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        seed=RANDOM_STATE + 1,
    )
    image_category_accuracy = float(
        np.mean(
            predictions[
                "predicted_image_category"
            ].astype(str).to_numpy()
            == predictions[
                "part_category"
            ].astype(str).to_numpy()
        )
    )
    text_category_accuracy = float(
        np.mean(
            predictions[
                "predicted_text_category"
            ].astype(str).to_numpy()
            == predictions[
                "text_category"
            ].astype(str).to_numpy()
        )
    )
    report = classification_report(
        true,
        predicted,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )

    expected_scalars = {
        "correct_predictions": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "image_category_accuracy": image_category_accuracy,
        "text_category_accuracy": text_category_accuracy,
    }

    if saved_metrics["correct_predictions"] != correct:
        raise RuntimeError(
            "Saved correct-prediction count is inconsistent."
        )

    for field in (
        "accuracy",
        "macro_f1",
        "image_category_accuracy",
        "text_category_accuracy",
    ):
        require_float(
            saved_metrics[field],
            expected_scalars[field],
            field,
        )

    for actual, expected, field in (
        (
            saved_metrics["accuracy_ci"],
            list(accuracy_ci),
            "accuracy_ci",
        ),
        (
            saved_metrics["macro_f1_ci"],
            list(macro_f1_ci),
            "macro_f1_ci",
        ),
    ):
        if len(actual) != 2:
            raise RuntimeError(
                f"Unexpected interval length for {field}."
            )

        for index in range(2):
            require_float(
                actual[index],
                expected[index],
                f"{field}[{index}]",
            )

    require_nested_close(
        saved_metrics["classification_report"],
        report,
        "classification_report",
    )

    saved_relation_summary = saved_metrics[
        "relation_protocol"
    ]

    if "relation_protocol_fingerprint" in (
        saved_relation_summary
    ):
        raise RuntimeError(
            "The original metrics artifact no longer has "
            "the documented fingerprint omission."
        )

    if saved_relation_summary != (
        relation_audit["summary"]
    ):
        raise RuntimeError(
            "Saved relation summary differs from independent validation."
        )

    if saved_metrics["vectorizer_fingerprint"] != (
        EXPECTED_VECTORIZER_FINGERPRINT
    ):
        raise RuntimeError(
            "Saved vectorizer fingerprint changed."
        )

    expected_matrix = confusion_matrix(
        true,
        predicted,
        labels=list(LABELS),
    )
    expected_index = [
        f"true_{label}"
        for label in LABELS
    ]
    expected_columns = [
        f"predicted_{label}"
        for label in LABELS
    ]

    if saved_matrix.index.tolist() != expected_index:
        raise RuntimeError(
            "Unexpected confusion-matrix row labels."
        )

    if saved_matrix.columns.tolist() != (
        expected_columns
    ):
        raise RuntimeError(
            "Unexpected confusion-matrix column labels."
        )

    if not np.array_equal(
        saved_matrix.to_numpy(dtype=int),
        expected_matrix,
    ):
        raise RuntimeError(
            "Saved confusion matrix is inconsistent."
        )

    if len(saved_per_category) != len(CATEGORIES):
        raise RuntimeError(
            "Unexpected per-category result count."
        )

    per_category_checks = 0

    for category in CATEGORIES:
        group = predictions[
            predictions[
                "part_category"
            ].astype(str).eq(category)
        ]
        saved = saved_per_category[
            saved_per_category[
                "part_category"
            ].astype(str).eq(category)
        ]

        if len(group) != EXPECTED_ROWS_PER_CATEGORY:
            raise RuntimeError(
                f"Unexpected row count for category {category}."
            )

        if len(saved) != 1:
            raise RuntimeError(
                f"Missing per-category row for {category}."
            )

        row = saved.iloc[0]
        category_correct = int(
            (
                group["true_label"].astype(str)
                == group["predicted_label"].astype(str)
            ).sum()
        )
        category_accuracy = float(
            accuracy_score(
                group["true_label"].astype(str),
                group["predicted_label"].astype(str),
            )
        )
        category_macro_f1 = float(
            f1_score(
                group["true_label"].astype(str),
                group["predicted_label"].astype(str),
                labels=list(LABELS),
                average="macro",
                zero_division=0,
            )
        )

        if int(row["samples"]) != EXPECTED_ROWS_PER_CATEGORY:
            raise RuntimeError(
                f"Wrong saved sample count for {category}."
            )

        if int(row["independent_images"]) != (
            EXPECTED_IMAGES_PER_CATEGORY
        ):
            raise RuntimeError(
                f"Wrong saved image count for {category}."
            )

        if int(row["correct"]) != category_correct:
            raise RuntimeError(
                f"Wrong saved correct count for {category}."
            )

        require_float(
            row["accuracy"],
            category_accuracy,
            f"{category}.accuracy",
        )
        require_float(
            row["macro_f1"],
            category_macro_f1,
            f"{category}.macro_f1",
        )
        per_category_checks += 1

    return {
        "correct_predictions": correct,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "accuracy_ci": list(accuracy_ci),
        "macro_f1_ci": list(macro_f1_ci),
        "image_category_accuracy": image_category_accuracy,
        "text_category_accuracy": text_category_accuracy,
        "classification_report": report,
        "confusion_matrix": expected_matrix.tolist(),
        "per_category_checks": per_category_checks,
        "predictions_recomputed": True,
        "confidence_intervals_recomputed": True,
        "confusion_matrix_recomputed": True,
    }


def verify_execution_records(
    execution_state: dict[str, Any],
    consumption: dict[str, Any],
    saved_metrics: dict[str, Any],
) -> dict[str, Any]:
    if execution_state["status"] != (
        "FINAL_TEST_EVALUATION_COMPLETE"
    ):
        raise RuntimeError(
            "Unexpected final-test execution status."
        )

    if saved_metrics["status"] != (
        "PASS_DATASET_V3_FINAL_TEST_EVALUATION_COMPLETE"
    ):
        raise RuntimeError(
            "Unexpected final-test metrics status."
        )

    if consumption["status"] != (
        "FINAL_TEST_AUTHORIZATION_CONSUMED"
    ):
        raise RuntimeError(
            "Unexpected authorization-consumption status."
        )

    for payload_name, payload in (
        ("execution_state", execution_state),
        ("metrics", saved_metrics),
        ("consumption", consumption),
    ):
        for field in (
            "authorization_consumed",
            "repository_test_manifest_read",
            "test_images_read",
            "test_evaluation_executed",
        ):
            if payload[field] is not True:
                raise RuntimeError(
                    f"{payload_name} does not confirm {field}."
                )

        if payload[
            "authorized_evaluations_completed"
        ] != 1:
            raise RuntimeError(
                f"{payload_name} does not record one evaluation."
            )

    for payload_name, payload in (
        ("metrics", saved_metrics),
        ("consumption", consumption),
    ):
        for field in (
            "further_tuning_permitted",
            "post_test_tuning_permitted",
            "test_results_may_be_used_for_tuning",
        ):
            if payload[field] is not False:
                raise RuntimeError(
                    f"{payload_name} contains unsafe flag {field}."
                )

    return {
        "execution_state_status": execution_state["status"],
        "metrics_status": saved_metrics["status"],
        "consumption_status": consumption["status"],
        "authorization_consumed": True,
        "authorized_evaluations_completed": 1,
        "test_evaluation_executed": True,
        "repository_test_manifest_read": True,
        "test_images_read": True,
        "further_tuning_permitted": False,
        "post_test_tuning_permitted": False,
        "test_results_may_be_used_for_tuning": False,
    }


@lru_cache(maxsize=1)
def build_final_test_results_manifest() -> dict[str, Any]:
    artifact_hashes = verify_result_artifact_hashes()
    frozen_hashes = verify_frozen_inputs()

    saved_images = pd.read_csv(
        RESULTS_DIR / "test_image_manifest.csv"
    )
    saved_relations = pd.read_csv(
        RESULTS_DIR / "test_relations.csv"
    )
    predictions = pd.read_csv(
        RESULTS_DIR / "test_predictions.csv"
    )
    saved_per_category = pd.read_csv(
        RESULTS_DIR / "test_per_category.csv"
    )
    saved_matrix = pd.read_csv(
        RESULTS_DIR / "test_confusion_matrix.csv",
        index_col=0,
    )
    saved_metrics = load_json(
        RESULTS_DIR / "test_metrics.json"
    )
    execution_state = load_json(
        RESULTS_DIR / "execution_state.json"
    )
    consumption = load_json(
        RESULTS_DIR / "authorization_consumption.json"
    )

    image_audit = verify_image_manifest(
        saved_images
    )
    relation_audit = verify_relations(
        saved_images,
        saved_relations,
    )
    metric_audit = verify_predictions_and_metrics(
        saved_relations,
        predictions,
        saved_metrics,
        saved_per_category,
        saved_matrix,
        relation_audit,
    )
    execution_audit = verify_execution_records(
        execution_state,
        consumption,
        saved_metrics,
    )

    return {
        "status": (
            "PASS_DATASET_V3_FINAL_TEST_RESULTS_RECORDED"
        ),
        "dataset": "dataset_v3",
        "protocol_source_commit": (
            EXPECTED_PROTOCOL_SOURCE_COMMIT
        ),
        "selected_model_slug": (
            "torch_multimodal_dataset_v3"
        ),
        "protocol_sha256": (
            EXPECTED_PROTOCOL_SHA256
        ),
        "authorization_sha256": (
            EXPECTED_AUTHORIZATION_SHA256
        ),
        "selection_lock_sha256": (
            EXPECTED_SELECTION_LOCK_SHA256
        ),
        "selected_checkpoint_sha256": (
            EXPECTED_CHECKPOINT_SHA256
        ),
        "test_lock_sha256": (
            EXPECTED_TEST_LOCK_SHA256
        ),
        "external_evidence_snapshot_sha256": (
            EXPECTED_EXTERNAL_EVIDENCE_SHA256
        ),
        "evaluator_invocations": 1,
        "evaluator_completed": True,
        "evaluator_rerun": False,
        "result_artifacts_modified": False,
        "result_artifact_count": len(
            artifact_hashes
        ),
        "result_artifact_hashes": (
            artifact_hashes
        ),
        "frozen_input_hashes": frozen_hashes,
        "metadata_reconciliation": {
            "status": (
                "PASS_MISSING_RELATION_FINGERPRINT_"
                "RECONCILED_WITH_FROZEN_PROTOCOL"
            ),
            "wrapper_error": (
                "Metrics contain the wrong relation fingerprint."
            ),
            "actual_issue": (
                "The original metrics relation summary omitted "
                "the relation_protocol_fingerprint field."
            ),
            "original_test_metrics_sha256": (
                EXPECTED_ORIGINAL_METRICS_SHA256
            ),
            "saved_fingerprint_present": False,
            "derived_relation_protocol_fingerprint": (
                relation_audit[
                    "relation_protocol_fingerprint"
                ]
            ),
            "frozen_protocol_fingerprint": (
                EXPECTED_RELATION_PROTOCOL_FINGERPRINT
            ),
            "saved_relations_match_frozen_builder": True,
            "saved_relation_summary_matches_validation": True,
            "result_file_rewritten": False,
            "inference_rerun": False,
        },
        "image_manifest_audit": image_audit,
        "relation_audit": relation_audit,
        "metric_audit": metric_audit,
        "execution_audit": execution_audit,
        "final_reporting": {
            "test_images": EXPECTED_TEST_IMAGES,
            "test_rows": EXPECTED_TEST_ROWS,
            "correct_predictions": metric_audit[
                "correct_predictions"
            ],
            "accuracy": metric_audit["accuracy"],
            "macro_f1": metric_audit["macro_f1"],
            "accuracy_ci": metric_audit[
                "accuracy_ci"
            ],
            "macro_f1_ci": metric_audit[
                "macro_f1_ci"
            ],
            "image_category_accuracy": (
                metric_audit[
                    "image_category_accuracy"
                ]
            ),
            "text_category_accuracy": (
                metric_audit[
                    "text_category_accuracy"
                ]
            ),
            "use": (
                "final_reporting_and_error_analysis_only"
            ),
        },
        "selection_frozen": True,
        "further_tuning_permitted": False,
        "post_test_tuning_permitted": False,
        "test_results_may_be_used_for_tuning": False,
    }
