from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.dataset_config import LABELS
from src.integrated_training_config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    INTEGRATED_METRIC_PATHS,
    TEXT_EMBEDDING_DIMENSION,
    TEXT_SEQUENCE_LENGTH,
)
from src.run_integrated_training_validation import (
    architecture_text,
    atomic_write_json,
    atomic_write_text,
    build_common_metadata,
    build_image_branch,
    build_multimodal_model,
    build_text_vocabulary,
    configure_keras_runtime,
    decode_label_indices,
    encode_labels,
    encode_text_sequences,
    extract_image_arrays,
    load_integrated_datasets,
    locked_test_fingerprints,
    read_test_lock,
    resolve_image_path,
)
from src.validate_external_training_readiness import project_relative_path
from src.verification.integrated_training_validation import (
    build_verification_report as build_integrated_training_verification,
)
from src.validation_model_improvement_config import (
    CANDIDATE_ARCHITECTURE_PATHS,
    CANDIDATE_CONFUSION_MATRIX_PATHS,
    CANDIDATE_DESCRIPTIONS,
    CANDIDATE_DIRECTORIES,
    CANDIDATE_HISTORY_PATHS,
    CANDIDATE_METRIC_PATHS,
    CANDIDATE_PREDICTION_PATHS,
    CANDIDATE_TITLES,
    DATA_DIAGNOSTICS_JSON_PATH,
    DISAGREEMENT_CSV_PATH,
    DISAGREEMENT_JSON_PATH,
    ERROR_ANALYSIS_CSV_PATH,
    ERROR_ANALYSIS_JSON_PATH,
    EXPERIMENT_COMPARISON_CSV_PATH,
    EXPERIMENT_COMPARISON_JSON_PATH,
    EXPERIMENT_REGISTRY_PATH,
    EXPERIMENT_SEEDS,
    HIGH_CONFIDENCE_ERROR_THRESHOLD,
    MAXIMUM_ACCURACY_REGRESSION,
    MAXIMUM_WORST_CLASS_F1_REGRESSION,
    MAXIMUM_INCUMBENT_MACRO_F1_REGRESSION,
    MINIMUM_MACRO_F1_GAIN,
    SELECTION_DECISION_PATH,
    VALIDATION_IMPROVEMENT_BATCH_SIZE,
    VALIDATION_IMPROVEMENT_MAX_EPOCHS,
    VALIDATION_IMPROVEMENT_PATIENCE,
    VALIDATION_IMPROVEMENT_ROOT,
    VALIDATION_IMPROVEMENT_STATUS_PATH,
    VALIDATION_IMPROVEMENT_SUMMARY_PATH,
)

LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


class ValidationImprovementError(RuntimeError):
    """Raised when Step 010.4 safeguards or artifacts are invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_description(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def build_data_diagnostics(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    train_groups = set(train_dataframe["part_group_id"])
    validation_groups = set(validation_dataframe["part_group_id"])

    def image_hashes(dataframe: pd.DataFrame) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        unique_rows = dataframe[["image_id", "image_path"]].drop_duplicates()
        for row in unique_rows.itertuples(index=False):
            digest = sha256_file(resolve_image_path(str(row.image_path)))
            result.setdefault(digest, []).append(str(row.image_id))
        return result

    train_hashes = image_hashes(train_dataframe)
    validation_hashes = image_hashes(validation_dataframe)
    shared_hashes = sorted(set(train_hashes) & set(validation_hashes))

    train_descriptions = {
        normalize_description(value)
        for value in train_dataframe["description"]
    }
    validation_descriptions = {
        normalize_description(value)
        for value in validation_dataframe["description"]
    }
    shared_descriptions = sorted(train_descriptions & validation_descriptions)

    cross_label_texts = []
    combined = pd.concat(
        [
            train_dataframe.assign(split="train"),
            validation_dataframe.assign(split="validation"),
        ],
        ignore_index=True,
    )
    combined["normalized_description"] = combined["description"].map(
        normalize_description
    )
    for description, group in combined.groupby("normalized_description"):
        labels = sorted(set(group["label"]))
        if len(labels) > 1:
            cross_label_texts.append(
                {
                    "description": description,
                    "labels": labels,
                    "sample_count": int(len(group)),
                }
            )

    risk_flags: list[str] = []
    if shared_hashes:
        risk_flags.append("exact_image_hash_overlap_across_train_validation")
    if shared_descriptions:
        risk_flags.append("exact_text_overlap_across_train_validation")
    if cross_label_texts:
        risk_flags.append("same_description_used_with_multiple_labels")

    return {
        "status": "PASS",
        "training_samples": int(len(train_dataframe)),
        "validation_samples": int(len(validation_dataframe)),
        "train_validation_group_overlap": int(
            len(train_groups & validation_groups)
        ),
        "train_validation_exact_image_hash_overlap": int(len(shared_hashes)),
        "shared_image_hashes": [
            {
                "sha256": digest,
                "training_image_ids": train_hashes[digest],
                "validation_image_ids": validation_hashes[digest],
            }
            for digest in shared_hashes
        ],
        "train_validation_exact_description_overlap": int(
            len(shared_descriptions)
        ),
        "shared_descriptions": shared_descriptions,
        "cross_label_description_count": int(len(cross_label_texts)),
        "cross_label_descriptions": cross_label_texts,
        "risk_flags": risk_flags,
        "test_split_used": False,
        "test_evaluation_permitted": False,
    }


def build_text_features(
    keras: Any,
    text_input: Any,
    vocabulary_size: int,
    *,
    regularized: bool,
) -> Any:
    features = keras.layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=(20 if regularized else TEXT_EMBEDDING_DIMENSION),
        name="token_embedding",
    )(text_input)
    average = keras.layers.GlobalAveragePooling1D(name="text_average")(features)
    maximum = keras.layers.GlobalMaxPooling1D(name="text_maximum")(features)
    features = keras.layers.Concatenate(name="text_pooling")([average, maximum])
    features = keras.layers.Dense(
        32,
        activation="relu",
        name="text_features",
    )(features)
    if regularized:
        features = keras.layers.Dropout(0.15, name="text_dropout")(features)
    return features


def build_relation_model(
    keras: Any,
    vocabulary_size: int,
    *,
    regularized: bool,
) -> Any:
    text_input = keras.Input(
        shape=(TEXT_SEQUENCE_LENGTH,),
        dtype="int32",
        name="description_tokens",
    )
    text_features = build_text_features(
        keras,
        text_input,
        vocabulary_size,
        regularized=regularized,
    )

    image_input = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype="float32",
        name="image",
    )
    image_features = build_image_branch(keras, image_input)
    if regularized:
        image_features = keras.layers.Dropout(
            0.15,
            name="image_dropout",
        )(image_features)

    product = keras.layers.Multiply(name="feature_product")(
        [text_features, image_features]
    )
    difference = keras.layers.Subtract(name="feature_difference")(
        [text_features, image_features]
    )
    absolute_difference = keras.layers.Lambda(
        lambda values: keras.ops.abs(values),
        name="absolute_feature_difference",
    )(difference)
    fused = keras.layers.Concatenate(name="relation_fusion")(
        [text_features, image_features, product, absolute_difference]
    )
    fused = keras.layers.Dense(
        48 if regularized else 64,
        activation="relu",
        name="fusion_dense",
    )(fused)
    fused = keras.layers.Dropout(
        0.30 if regularized else 0.15,
        name="fusion_dropout",
    )(fused)
    outputs = keras.layers.Dense(
        len(LABELS),
        activation="softmax",
        name="class_probabilities",
    )(fused)
    model = keras.Model(
        inputs={"description_tokens": text_input, "image": image_input},
        outputs=outputs,
        name=(
            "regularized_relation_multimodal_classifier"
            if regularized
            else "relation_aware_multimodal_classifier"
        ),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.0007 if regularized else 0.001
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def candidate_builders() -> dict[str, Callable[[Any, int], Any]]:
    return {
        "reference_multimodal": (
            lambda keras, vocabulary_size: build_multimodal_model(
                keras,
                vocabulary_size,
            )
        ),
        "relation_aware_multimodal": (
            lambda keras, vocabulary_size: build_relation_model(
                keras,
                vocabulary_size,
                regularized=False,
            )
        ),
        "regularized_relation_multimodal": (
            lambda keras, vocabulary_size: build_relation_model(
                keras,
                vocabulary_size,
                regularized=True,
            )
        ),
    }


def prediction_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, Any]:
    precision, recall, class_f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=LABELS,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": list(LABELS),
    }


def candidate_prediction_table(
    validation_dataframe: pd.DataFrame,
    predicted_labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    table = validation_dataframe[
        [
            "sample_id",
            "part_group_id",
            "image_id",
            "part_category",
            "source",
            "description",
            "label",
        ]
    ].copy()
    table = table.rename(columns={"label": "true_label"})
    table["predicted_label"] = predicted_labels
    for index, label in enumerate(LABELS):
        table[f"probability_{label.lower()}"] = probabilities[:, index]
    sorted_probabilities = np.sort(probabilities, axis=1)
    table["confidence"] = sorted_probabilities[:, -1]
    table["confidence_margin"] = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )
    table["is_correct"] = table["true_label"] == table["predicted_label"]
    table["error_pair"] = np.where(
        table["is_correct"],
        "CORRECT",
        table["true_label"] + "->" + table["predicted_label"],
    )
    return table


def train_candidate(
    *,
    keras: Any,
    backend: str,
    candidate_slug: str,
    builder: Callable[[Any, int], Any],
    vocabulary_size: int,
    training_inputs: dict[str, np.ndarray],
    validation_inputs: dict[str, np.ndarray],
    training_labels: np.ndarray,
    validation_labels: np.ndarray,
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    common_metadata: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    seed_probabilities: list[np.ndarray] = []
    seed_results: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    model_architecture = ""
    parameter_count = 0

    true_labels = validation_dataframe["label"].to_numpy(dtype=object)
    for seed in EXPERIMENT_SEEDS:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        model = builder(keras, vocabulary_size)
        parameter_count = int(model.count_params())
        if not model_architecture:
            model_architecture = architecture_text(model)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=VALIDATION_IMPROVEMENT_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        )
        history_object = model.fit(
            training_inputs,
            training_labels,
            validation_data=(validation_inputs, validation_labels),
            epochs=VALIDATION_IMPROVEMENT_MAX_EPOCHS,
            batch_size=VALIDATION_IMPROVEMENT_BATCH_SIZE,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0,
        )
        probabilities = np.asarray(
            model.predict(
                validation_inputs,
                batch_size=VALIDATION_IMPROVEMENT_BATCH_SIZE,
                verbose=0,
            )
        )
        if probabilities.shape != (len(validation_dataframe), len(LABELS)):
            raise ValidationImprovementError(
                f"Unexpected probability shape for {candidate_slug}: "
                f"{probabilities.shape}."
            )
        seed_probabilities.append(probabilities)
        predicted = decode_label_indices(np.argmax(probabilities, axis=1))
        metrics = prediction_metrics(true_labels, predicted)
        history = {
            key: [float(value) for value in values]
            for key, values in history_object.history.items()
        }
        validation_losses = history.get("val_loss", [])
        seed_results.append(
            {
                "seed": int(seed),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "epochs_completed": int(
                    len(next(iter(history.values()), []))
                ),
                "best_epoch": (
                    int(np.argmin(validation_losses)) + 1
                    if validation_losses
                    else None
                ),
                "best_validation_loss": (
                    float(min(validation_losses))
                    if validation_losses
                    else None
                ),
            }
        )
        for epoch_index in range(len(next(iter(history.values()), []))):
            row: dict[str, Any] = {
                "seed": int(seed),
                "epoch": epoch_index + 1,
            }
            for metric_name, values in history.items():
                row[metric_name] = values[epoch_index]
            history_rows.append(row)
        del model
        gc.collect()

    aggregate_probabilities = np.mean(np.stack(seed_probabilities), axis=0)
    aggregate_predictions = decode_label_indices(
        np.argmax(aggregate_probabilities, axis=1)
    )
    aggregate_metrics = prediction_metrics(true_labels, aggregate_predictions)
    seed_macro_f1 = np.asarray(
        [result["macro_f1"] for result in seed_results],
        dtype=float,
    )
    seed_accuracy = np.asarray(
        [result["accuracy"] for result in seed_results],
        dtype=float,
    )
    metric_payload: dict[str, Any] = {
        "candidate_slug": candidate_slug,
        "candidate": CANDIDATE_TITLES[candidate_slug],
        "description": CANDIDATE_DESCRIPTIONS[candidate_slug],
        "evaluation_split": "integrated_validation",
        "training_sample_count": int(len(train_dataframe)),
        "sample_count": int(len(validation_dataframe)),
        "aggregate_probability_method": "mean_across_fixed_seeds",
        "seeds": list(EXPERIMENT_SEEDS),
        "seed_results": seed_results,
        "mean_seed_accuracy": float(seed_accuracy.mean()),
        "std_seed_accuracy": float(seed_accuracy.std(ddof=0)),
        "mean_seed_macro_f1": float(seed_macro_f1.mean()),
        "std_seed_macro_f1": float(seed_macro_f1.std(ddof=0)),
        "accuracy": aggregate_metrics["accuracy"],
        "macro_f1": aggregate_metrics["macro_f1"],
        "per_class": aggregate_metrics["per_class"],
        "confusion_matrix": aggregate_metrics["confusion_matrix"],
        "confusion_matrix_labels": list(LABELS),
        "worst_class_f1": float(
            min(
                details["f1"]
                for details in aggregate_metrics["per_class"].values()
            )
        ),
        "parameter_count": parameter_count,
        "keras_backend": backend,
        "data_contract": common_metadata,
        "test_split_used": False,
        "test_evaluation_permitted": False,
    }
    predictions = candidate_prediction_table(
        validation_dataframe,
        aggregate_predictions,
        aggregate_probabilities,
    )

    candidate_directory = CANDIDATE_DIRECTORIES[candidate_slug]
    candidate_directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CANDIDATE_METRIC_PATHS[candidate_slug], metric_payload)
    predictions.to_csv(
        CANDIDATE_PREDICTION_PATHS[candidate_slug],
        index=False,
        lineterminator="\n",
    )
    confusion = pd.DataFrame(
        aggregate_metrics["confusion_matrix"],
        index=[f"actual_{label}" for label in LABELS],
        columns=[f"predicted_{label}" for label in LABELS],
    )
    confusion.to_csv(
        CANDIDATE_CONFUSION_MATRIX_PATHS[candidate_slug],
        index=True,
        lineterminator="\n",
    )
    pd.DataFrame(history_rows).to_csv(
        CANDIDATE_HISTORY_PATHS[candidate_slug],
        index=False,
        lineterminator="\n",
    )
    atomic_write_text(
        CANDIDATE_ARCHITECTURE_PATHS[candidate_slug],
        model_architecture.rstrip() + "\n",
    )
    return metric_payload, predictions


def build_comparison(candidate_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for order, slug in enumerate(CANDIDATE_TITLES, start=1):
        metrics = candidate_metrics[slug]
        rows.append(
            {
                "candidate_slug": slug,
                "candidate": CANDIDATE_TITLES[slug],
                "aggregate_validation_accuracy": metrics["accuracy"],
                "aggregate_validation_macro_f1": metrics["macro_f1"],
                "mean_seed_accuracy": metrics["mean_seed_accuracy"],
                "std_seed_accuracy": metrics["std_seed_accuracy"],
                "mean_seed_macro_f1": metrics["mean_seed_macro_f1"],
                "std_seed_macro_f1": metrics["std_seed_macro_f1"],
                "worst_class_f1": metrics["worst_class_f1"],
                "parameter_count": metrics["parameter_count"],
                "test_split_used": False,
                "_order": order,
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        by=[
            "mean_seed_macro_f1",
            "aggregate_validation_macro_f1",
            "aggregate_validation_accuracy",
            "_order",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    comparison.insert(0, "validation_rank", range(1, len(comparison) + 1))
    return comparison.drop(columns="_order")


def build_selection_decision(
    comparison: pd.DataFrame,
    candidate_metrics: dict[str, dict[str, Any]],
    *,
    incumbent_validation_accuracy: float,
    incumbent_validation_macro_f1: float,
) -> dict[str, Any]:
    reference = candidate_metrics["reference_multimodal"]
    eligible: list[dict[str, Any]] = []
    evaluated: list[dict[str, Any]] = []
    for slug in CANDIDATE_TITLES:
        if slug == "reference_multimodal":
            continue
        metrics = candidate_metrics[slug]
        macro_gain = float(
            metrics["mean_seed_macro_f1"] - reference["mean_seed_macro_f1"]
        )
        accuracy_change = float(metrics["accuracy"] - reference["accuracy"])
        worst_class_change = float(
            metrics["worst_class_f1"] - reference["worst_class_f1"]
        )
        incumbent_macro_f1_change = float(
            metrics["macro_f1"] - incumbent_validation_macro_f1
        )
        incumbent_accuracy_change = float(
            metrics["accuracy"] - incumbent_validation_accuracy
        )
        passes = (
            macro_gain >= MINIMUM_MACRO_F1_GAIN
            and accuracy_change >= -MAXIMUM_ACCURACY_REGRESSION
            and worst_class_change >= -MAXIMUM_WORST_CLASS_F1_REGRESSION
            and incumbent_macro_f1_change
            >= -MAXIMUM_INCUMBENT_MACRO_F1_REGRESSION
        )
        record = {
            "candidate_slug": slug,
            "mean_seed_macro_f1_gain": macro_gain,
            "aggregate_accuracy_change": accuracy_change,
            "worst_class_f1_change": worst_class_change,
            "incumbent_validation_macro_f1_change": (
                incumbent_macro_f1_change
            ),
            "incumbent_validation_accuracy_change": (
                incumbent_accuracy_change
            ),
            "passes_acceptance_gate": passes,
        }
        evaluated.append(record)
        if passes:
            eligible.append(record)

    if eligible:
        eligible_slugs = {record["candidate_slug"] for record in eligible}
        selected_slug = str(
            comparison[
                comparison["candidate_slug"].isin(eligible_slugs)
            ].iloc[0]["candidate_slug"]
        )
        decision = "IMPROVEMENT_ACCEPTED"
    else:
        selected_slug = "reference_multimodal"
        decision = "REFERENCE_RETAINED"

    selected = candidate_metrics[selected_slug]
    return {
        "status": "PASS",
        "decision": decision,
        "reference_candidate_slug": "reference_multimodal",
        "selected_candidate_slug": selected_slug,
        "selected_candidate": CANDIDATE_TITLES[selected_slug],
        "selection_metric": "mean_seed_macro_f1_with_incumbent_guard",
        "incumbent_validation_accuracy": incumbent_validation_accuracy,
        "incumbent_validation_macro_f1": incumbent_validation_macro_f1,
        "selected_mean_seed_macro_f1": selected["mean_seed_macro_f1"],
        "selected_aggregate_validation_macro_f1": selected["macro_f1"],
        "selected_aggregate_validation_accuracy": selected["accuracy"],
        "acceptance_gate": {
            "minimum_mean_seed_macro_f1_gain": MINIMUM_MACRO_F1_GAIN,
            "maximum_incumbent_aggregate_macro_f1_regression": (
                MAXIMUM_INCUMBENT_MACRO_F1_REGRESSION
            ),
            "maximum_aggregate_accuracy_regression": (
                MAXIMUM_ACCURACY_REGRESSION
            ),
            "maximum_worst_class_f1_regression": (
                MAXIMUM_WORST_CLASS_F1_REGRESSION
            ),
        },
        "evaluated_improvements": evaluated,
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "final_test_evaluation_authorized": False,
    }


def build_error_analysis(
    selected_slug: str,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    errors = predictions[~predictions["is_correct"]].copy()
    errors = errors.sort_values(
        by=["confidence", "confidence_margin"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
    errors.insert(0, "error_rank", range(1, len(errors) + 1))

    confusion_pairs = {
        str(pair): int(count)
        for pair, count in errors["error_pair"].value_counts().items()
    }
    by_category = {
        str(category): {
            "errors": int(len(group)),
            "samples": int(
                len(predictions[predictions["part_category"] == category])
            ),
            "error_rate": float(
                len(group)
                / len(predictions[predictions["part_category"] == category])
            ),
        }
        for category, group in errors.groupby("part_category")
    }
    by_source = {
        str(source): {
            "errors": int(len(group)),
            "samples": int(len(predictions[predictions["source"] == source])),
            "error_rate": float(
                len(group) / len(predictions[predictions["source"] == source])
            ),
        }
        for source, group in errors.groupby("source")
    }
    high_confidence = errors[
        errors["confidence"] >= HIGH_CONFIDENCE_ERROR_THRESHOLD
    ]
    payload = {
        "status": "PASS",
        "selected_candidate_slug": selected_slug,
        "validation_sample_count": int(len(predictions)),
        "correct_count": int(predictions["is_correct"].sum()),
        "error_count": int(len(errors)),
        "error_rate": float(len(errors) / len(predictions)),
        "confusion_pairs": confusion_pairs,
        "errors_by_category": by_category,
        "errors_by_source": by_source,
        "high_confidence_error_threshold": HIGH_CONFIDENCE_ERROR_THRESHOLD,
        "high_confidence_error_count": int(len(high_confidence)),
        "highest_confidence_errors": high_confidence.head(20).to_dict(
            orient="records"
        ),
        "test_split_used": False,
        "test_evaluation_permitted": False,
    }
    return errors, payload


def build_disagreement_analysis(
    candidate_predictions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    first_slug = next(iter(CANDIDATE_TITLES))
    base_columns = [
        "sample_id",
        "part_group_id",
        "image_id",
        "part_category",
        "source",
        "description",
        "true_label",
    ]
    table = candidate_predictions[first_slug][base_columns].copy()
    prediction_columns = []
    for slug in CANDIDATE_TITLES:
        values = candidate_predictions[slug][
            ["sample_id", "predicted_label", "confidence"]
        ].rename(
            columns={
                "predicted_label": f"prediction_{slug}",
                "confidence": f"confidence_{slug}",
            }
        )
        table = table.merge(values, on="sample_id", how="left", validate="1:1")
        prediction_columns.append(f"prediction_{slug}")
    table["unique_prediction_count"] = table[prediction_columns].nunique(axis=1)
    table["has_candidate_disagreement"] = table["unique_prediction_count"] > 1
    table["all_candidates_correct"] = table[prediction_columns].eq(
        table["true_label"], axis=0
    ).all(axis=1)
    table["all_candidates_wrong"] = ~table[prediction_columns].eq(
        table["true_label"], axis=0
    ).any(axis=1)
    table = table.sort_values(
        by=["has_candidate_disagreement", "all_candidates_wrong"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
    payload = {
        "status": "PASS",
        "sample_count": int(len(table)),
        "candidate_count": len(CANDIDATE_TITLES),
        "disagreement_count": int(table["has_candidate_disagreement"].sum()),
        "unanimously_correct_count": int(table["all_candidates_correct"].sum()),
        "unanimously_wrong_count": int(table["all_candidates_wrong"].sum()),
        "test_split_used": False,
        "test_evaluation_permitted": False,
    }
    return table, payload


def render_summary(
    comparison: pd.DataFrame,
    decision: dict[str, Any],
    error_analysis: dict[str, Any],
    diagnostics: dict[str, Any],
    backend: str,
) -> str:
    lines = [
        "# Validation Error Analysis and Controlled Model Improvement",
        "",
        "- Status: **PASS**",
        "- Readiness: **MODEL_IMPROVEMENT_DECISION_COMPLETE**",
        f"- Decision: **{decision['decision']}**",
        f"- Selected candidate: **{decision['selected_candidate']}**",
        (
            "- Step 010.3 incumbent validation: "
            f"accuracy `{decision['incumbent_validation_accuracy']:.4f}`, "
            f"macro F1 `{decision['incumbent_validation_macro_f1']:.4f}`"
        ),
        (
            "- Selected controlled aggregate: "
            f"accuracy `{decision['selected_aggregate_validation_accuracy']:.4f}`, "
            f"macro F1 `{decision['selected_aggregate_validation_macro_f1']:.4f}`"
        ),
        f"- Keras backend: `{backend}`",
        "- Three fixed seeds were used for every candidate.",
        "- Only integrated train and integrated validation were loaded.",
        "- The locked test split was not loaded, evaluated, or used for selection.",
        "",
        "## Controlled experiment comparison",
        "",
        "| Rank | Candidate | Mean seed macro F1 | Aggregate macro F1 | Accuracy | Worst-class F1 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in comparison.to_dict(orient="records"):
        lines.append(
            f"| {int(row['validation_rank'])} | {row['candidate']} "
            f"| {float(row['mean_seed_macro_f1']):.4f} "
            f"| {float(row['aggregate_validation_macro_f1']):.4f} "
            f"| {float(row['aggregate_validation_accuracy']):.4f} "
            f"| {float(row['worst_class_f1']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation error analysis",
            "",
            f"- Validation errors: {error_analysis['error_count']} / "
            f"{error_analysis['validation_sample_count']}",
            f"- High-confidence errors: "
            f"{error_analysis['high_confidence_error_count']}",
            f"- Exact image hash overlap across train/validation: "
            f"{diagnostics['train_validation_exact_image_hash_overlap']}",
            f"- Exact description overlap across train/validation: "
            f"{diagnostics['train_validation_exact_description_overlap']}",
            "",
            "The model-selection gate requires a stable mean-seed gain, no "
            "material accuracy or worst-class regression against the controlled "
            "reference, and no more than 0.01 aggregate macro-F1 regression "
            "against the Step 010.3 incumbent. Completing this step does not "
            "authorize test use.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_validation_error_analysis_and_model_improvement() -> dict[str, Any]:
    integrated_verification = build_integrated_training_verification()
    if integrated_verification.get("status") != "PASS":
        raise ValidationImprovementError(
            "Step 010.3 integrated training verification is not PASS."
        )

    lock = read_test_lock()
    before_fingerprints = locked_test_fingerprints(lock)
    train_dataframe, validation_dataframe = load_integrated_datasets()
    common_metadata = build_common_metadata(train_dataframe, validation_dataframe)
    diagnostics = build_data_diagnostics(train_dataframe, validation_dataframe)
    if diagnostics["train_validation_group_overlap"] != 0:
        raise ValidationImprovementError(
            "Train and validation groups overlap."
        )

    keras, backend = configure_keras_runtime()
    vocabulary = build_text_vocabulary(train_dataframe["description"].astype(str))
    vocabulary_size = max(vocabulary.values(), default=1) + 1
    training_inputs = {
        "description_tokens": encode_text_sequences(
            train_dataframe["description"].astype(str), vocabulary
        ),
        "image": extract_image_arrays(
            train_dataframe,
            size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            flatten=False,
        ),
    }
    validation_inputs = {
        "description_tokens": encode_text_sequences(
            validation_dataframe["description"].astype(str), vocabulary
        ),
        "image": extract_image_arrays(
            validation_dataframe,
            size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            flatten=False,
        ),
    }
    training_labels = encode_labels(train_dataframe["label"])
    validation_labels = encode_labels(validation_dataframe["label"])

    VALIDATION_IMPROVEMENT_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_metrics: dict[str, dict[str, Any]] = {}
    candidate_predictions: dict[str, pd.DataFrame] = {}
    for slug, builder in candidate_builders().items():
        metrics, predictions = train_candidate(
            keras=keras,
            backend=backend,
            candidate_slug=slug,
            builder=builder,
            vocabulary_size=vocabulary_size,
            training_inputs=training_inputs,
            validation_inputs=validation_inputs,
            training_labels=training_labels,
            validation_labels=validation_labels,
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            common_metadata=common_metadata,
        )
        candidate_metrics[slug] = metrics
        candidate_predictions[slug] = predictions

    comparison = build_comparison(candidate_metrics)
    step0103_reference = json.loads(
        INTEGRATED_METRIC_PATHS["keras_multimodal"].read_text(encoding="utf-8")
    )
    decision = build_selection_decision(
        comparison,
        candidate_metrics,
        incumbent_validation_accuracy=float(step0103_reference["accuracy"]),
        incumbent_validation_macro_f1=float(step0103_reference["macro_f1"]),
    )
    selected_slug = str(decision["selected_candidate_slug"])
    error_rows, error_payload = build_error_analysis(
        selected_slug,
        candidate_predictions[selected_slug],
    )
    disagreement_rows, disagreement_payload = build_disagreement_analysis(
        candidate_predictions
    )

    after_fingerprints = locked_test_fingerprints(lock)
    if before_fingerprints != after_fingerprints:
        raise ValidationImprovementError(
            "A locked test artifact changed during Step 010.4."
        )

    comparison.to_csv(
        EXPERIMENT_COMPARISON_CSV_PATH,
        index=False,
        lineterminator="\n",
    )
    atomic_write_json(
        EXPERIMENT_COMPARISON_JSON_PATH,
        {
            "status": "PASS",
            "selection_metric": "mean_seed_macro_f1_with_incumbent_guard",
            "candidate_count": len(CANDIDATE_TITLES),
            "candidates": comparison.to_dict(orient="records"),
            "test_split_used": False,
            "test_evaluation_permitted": False,
        },
    )
    error_rows.to_csv(ERROR_ANALYSIS_CSV_PATH, index=False, lineterminator="\n")
    atomic_write_json(ERROR_ANALYSIS_JSON_PATH, error_payload)
    disagreement_rows.to_csv(
        DISAGREEMENT_CSV_PATH,
        index=False,
        lineterminator="\n",
    )
    atomic_write_json(DISAGREEMENT_JSON_PATH, disagreement_payload)
    atomic_write_json(DATA_DIAGNOSTICS_JSON_PATH, diagnostics)
    atomic_write_json(SELECTION_DECISION_PATH, decision)
    atomic_write_json(
        EXPERIMENT_REGISTRY_PATH,
        {
            "status": "PASS",
            "protocol": "fixed_candidates_fixed_seeds_validation_only",
            "candidate_order": list(CANDIDATE_TITLES),
            "candidate_descriptions": CANDIDATE_DESCRIPTIONS,
            "seeds": list(EXPERIMENT_SEEDS),
            "max_epochs": VALIDATION_IMPROVEMENT_MAX_EPOCHS,
            "early_stopping_patience": VALIDATION_IMPROVEMENT_PATIENCE,
            "batch_size": VALIDATION_IMPROVEMENT_BATCH_SIZE,
            "keras_backend": backend,
            "step0103_reference_validation_accuracy": step0103_reference[
                "accuracy"
            ],
            "step0103_reference_validation_macro_f1": step0103_reference[
                "macro_f1"
            ],
            "test_split_used": False,
            "test_evaluation_permitted": False,
        },
    )
    atomic_write_text(
        VALIDATION_IMPROVEMENT_SUMMARY_PATH,
        render_summary(comparison, decision, error_payload, diagnostics, backend),
    )
    status = {
        "status": "PASS",
        "readiness": "MODEL_IMPROVEMENT_DECISION_COMPLETE",
        "decision": decision["decision"],
        "selected_candidate_slug": selected_slug,
        "selected_candidate": decision["selected_candidate"],
        "selected_validation_accuracy": decision[
            "selected_aggregate_validation_accuracy"
        ],
        "selected_validation_macro_f1": decision[
            "selected_aggregate_validation_macro_f1"
        ],
        "selected_mean_seed_macro_f1": decision[
            "selected_mean_seed_macro_f1"
        ],
        "candidate_count": len(CANDIDATE_TITLES),
        "seeds_per_candidate": len(EXPERIMENT_SEEDS),
        "keras_backend": backend,
        "validation_error_count": error_payload["error_count"],
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "final_test_evaluation_authorized": False,
        "locked_test_fingerprints_unchanged": True,
        "locked_test_fingerprints": after_fingerprints,
        "comparison_csv": project_relative_path(EXPERIMENT_COMPARISON_CSV_PATH),
        "selection_decision": project_relative_path(SELECTION_DECISION_PATH),
        "summary": project_relative_path(VALIDATION_IMPROVEMENT_SUMMARY_PATH),
    }
    atomic_write_json(VALIDATION_IMPROVEMENT_STATUS_PATH, status)
    return status


def main() -> None:
    status = run_validation_error_analysis_and_model_improvement()
    print("Validation error analysis and controlled model improvement")
    print(f"- Status: {status['status']}")
    print(f"- Readiness: {status['readiness']}")
    print(f"- Decision: {status['decision']}")
    print(f"- Selected candidate: {status['selected_candidate']}")
    print(
        "- Validation: "
        f"accuracy={status['selected_validation_accuracy']:.4f}, "
        f"macro_f1={status['selected_validation_macro_f1']:.4f}"
    )
    print(f"- Mean seed macro F1: {status['selected_mean_seed_macro_f1']:.4f}")
    print("- Locked test split used: no")


if __name__ == "__main__":
    main()
