from __future__ import annotations

import copy
import io
import json
import platform
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    IMAGE_SIZE,
    LABELS,
    RESULTS_DIR,
    check_split,
    encode_labels,
    load_images,
    load_split,
)
from src.models import ImageRelationCNN, MultimodalRelationCNN, TextRelationMLP

RANDOM_STATE = 42
MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 32
BOOTSTRAP_REPEATS = 2000
AUXILIARY_LOSS_WEIGHT = 0.40
REAL_ONLY_SLUG = "torch_multimodal_real_only"
SYNTHETIC_SLUG = "torch_multimodal_real_plus_synthetic"
MAIN_MODEL_SLUG = REAL_ONLY_SLUG


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def simple_image_features(data: pd.DataFrame) -> np.ndarray:
    """Low-resolution RGB pixels for transparent non-neural baselines."""
    return load_images(data, size=(16, 16)).reshape(len(data), -1).astype(np.float32) / 255.0


def torch_images(data: pd.DataFrame) -> np.ndarray:
    images = load_images(data, IMAGE_SIZE).astype(np.float32) / 255.0
    return np.transpose(images, (0, 3, 1, 2))


def grouped_bootstrap_interval(
    true: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Bootstrap complete image groups because each image contributes three paired rows."""
    unique_groups = np.asarray(pd.unique(groups))
    group_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        row_indices = np.concatenate([group_indices[group] for group in sampled])
        values[index] = metric(true[row_indices], predicted[row_indices])
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def score_model(
    name: str,
    slug: str,
    modality: str,
    training_data: str,
    validation: pd.DataFrame,
    predicted: np.ndarray,
    image_category_accuracy: float | None = None,
    text_category_accuracy: float | None = None,
) -> dict[str, object]:
    true = validation["label"].to_numpy()
    groups = validation["image_id"].to_numpy()
    accuracy = float(accuracy_score(true, predicted))
    macro_f1 = float(f1_score(true, predicted, average="macro", zero_division=0))
    accuracy_ci = grouped_bootstrap_interval(
        true,
        predicted,
        groups,
        lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
    )
    f1_ci = grouped_bootstrap_interval(
        true,
        predicted,
        groups,
        lambda y_true, y_pred: float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        seed=RANDOM_STATE + 1,
    )
    return {
        "model": name,
        "model_slug": slug,
        "modality": modality,
        "training_data": training_data,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "accuracy_ci_low": accuracy_ci[0],
        "accuracy_ci_high": accuracy_ci[1],
        "macro_f1_ci_low": f1_ci[0],
        "macro_f1_ci_high": f1_ci[1],
        "correct_predictions": int(np.sum(true == predicted)),
        "total_predictions": int(len(true)),
        "validation_images": int(validation["image_id"].nunique()),
        "validation_image_category_accuracy": image_category_accuracy,
        "validation_text_category_accuracy": text_category_accuracy,
    }


def prediction_rows(
    model: str,
    model_slug: str,
    validation: pd.DataFrame,
    predicted: np.ndarray,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "image_id",
        "part_group_id",
        "object_group_id",
        "part_category",
        "text_category",
        "source",
        "description",
        "label",
        "image_path",
    ]
    out = validation[columns].copy().rename(columns={"label": "true_label"})
    out["predicted_label"] = predicted
    out["is_correct"] = out["true_label"].eq(out["predicted_label"])
    out["model"] = model
    out["model_slug"] = model_slug
    return out[
        [
            "sample_id",
            "image_id",
            "part_group_id",
            "object_group_id",
            "part_category",
            "text_category",
            "source",
            "description",
            "true_label",
            "predicted_label",
            "is_correct",
            "image_path",
            "model",
            "model_slug",
        ]
    ]


def augment_images(images: torch.Tensor) -> torch.Tensor:
    """Apply lightweight in-memory augmentation only to training batches."""
    result = images.clone()
    flip_mask = torch.rand(len(result), device=result.device) < 0.5
    result[flip_mask] = torch.flip(result[flip_mask], dims=[3])
    brightness = 0.9 + 0.2 * torch.rand((len(result), 1, 1, 1), device=result.device)
    return torch.clamp(result * brightness, 0.0, 1.0)


def evaluate_single_input(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    correct = 0
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * len(y_batch)
            predicted = logits.argmax(dim=1)
            correct += int((predicted == y_batch).sum().item())
            predictions.append(predicted.cpu().numpy())
    return total_loss / len(labels), correct / len(labels), np.concatenate(predictions)


def fit_single_input_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    augment_image_batches: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    x_train = torch.from_numpy(train_features).float()
    y_train = torch.from_numpy(train_labels).long()
    x_validation = torch.from_numpy(validation_features).float()
    y_validation = torch.from_numpy(validation_labels).long()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        for x_batch, y_batch in loader:
            if augment_image_batches:
                x_batch = augment_images(x_batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y_batch)
            correct += int((logits.argmax(dim=1) == y_batch).sum().item())
        val_loss, val_accuracy, _ = evaluate_single_input(
            model, x_validation, y_validation
        )
        rows.append(
            {
                "epoch": epoch,
                "loss": total_loss / len(y_train),
                "accuracy": correct / len(y_train),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    _, _, predicted = evaluate_single_input(model, x_validation, y_validation)
    return predicted, pd.DataFrame(rows)


def evaluate_multimodal(
    model: MultimodalRelationCNN,
    images: torch.Tensor,
    texts: torch.Tensor,
    relations: torch.Tensor,
    image_categories: torch.Tensor,
    text_categories: torch.Tensor,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(images, texts, relations, image_categories, text_categories),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    totals = {"loss": 0.0, "relation": 0, "image_category": 0, "text_category": 0}
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for image_batch, text_batch, relation_batch, image_cat_batch, text_cat_batch in loader:
            relation_logits, image_logits, text_logits = model(image_batch, text_batch)
            relation_loss = criterion(relation_logits, relation_batch)
            image_loss = criterion(image_logits, image_cat_batch)
            text_loss = criterion(text_logits, text_cat_batch)
            loss = relation_loss + AUXILIARY_LOSS_WEIGHT * (image_loss + text_loss)
            totals["loss"] += float(loss.item()) * len(relation_batch)
            relation_pred = relation_logits.argmax(dim=1)
            totals["relation"] += int((relation_pred == relation_batch).sum().item())
            totals["image_category"] += int((image_logits.argmax(dim=1) == image_cat_batch).sum().item())
            totals["text_category"] += int((text_logits.argmax(dim=1) == text_cat_batch).sum().item())
            predictions.append(relation_pred.cpu().numpy())
    count = len(relations)
    metrics = {
        "loss": totals["loss"] / count,
        "relation_accuracy": totals["relation"] / count,
        "image_category_accuracy": totals["image_category"] / count,
        "text_category_accuracy": totals["text_category"] / count,
    }
    return metrics, np.concatenate(predictions)


def fit_multimodal_model(
    model: MultimodalRelationCNN,
    train_images: np.ndarray,
    train_texts: np.ndarray,
    train_relations: np.ndarray,
    train_image_categories: np.ndarray,
    train_text_categories: np.ndarray,
    validation_images: np.ndarray,
    validation_texts: np.ndarray,
    validation_relations: np.ndarray,
    validation_image_categories: np.ndarray,
    validation_text_categories: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    tensors = [
        torch.from_numpy(train_images).float(),
        torch.from_numpy(train_texts).float(),
        torch.from_numpy(train_relations).long(),
        torch.from_numpy(train_image_categories).long(),
        torch.from_numpy(train_text_categories).long(),
    ]
    validation_tensors = [
        torch.from_numpy(validation_images).float(),
        torch.from_numpy(validation_texts).float(),
        torch.from_numpy(validation_relations).long(),
        torch.from_numpy(validation_image_categories).long(),
        torch.from_numpy(validation_text_categories).long(),
    ]
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        totals = {"loss": 0.0, "relation": 0, "image_category": 0, "text_category": 0}
        for image_batch, text_batch, relation_batch, image_cat_batch, text_cat_batch in loader:
            image_batch = augment_images(image_batch)
            optimizer.zero_grad(set_to_none=True)
            relation_logits, image_logits, text_logits = model(image_batch, text_batch)
            relation_loss = criterion(relation_logits, relation_batch)
            image_loss = criterion(image_logits, image_cat_batch)
            text_loss = criterion(text_logits, text_cat_batch)
            loss = relation_loss + AUXILIARY_LOSS_WEIGHT * (image_loss + text_loss)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.item()) * len(relation_batch)
            totals["relation"] += int((relation_logits.argmax(dim=1) == relation_batch).sum().item())
            totals["image_category"] += int((image_logits.argmax(dim=1) == image_cat_batch).sum().item())
            totals["text_category"] += int((text_logits.argmax(dim=1) == text_cat_batch).sum().item())

        validation_metrics, _ = evaluate_multimodal(model, *validation_tensors)
        count = len(tensors[2])
        rows.append(
            {
                "epoch": epoch,
                "loss": totals["loss"] / count,
                "accuracy": totals["relation"] / count,
                "image_category_accuracy": totals["image_category"] / count,
                "text_category_accuracy": totals["text_category"] / count,
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["relation_accuracy"],
                "val_image_category_accuracy": validation_metrics["image_category_accuracy"],
                "val_text_category_accuracy": validation_metrics["text_category_accuracy"],
            }
        )
        if validation_metrics["loss"] < best_loss - 1e-4:
            best_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    final_metrics, predicted = evaluate_multimodal(model, *validation_tensors)
    return predicted, pd.DataFrame(rows), final_metrics


def save_model_summary(model: nn.Module, path: Path) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    text = f"{model}\n\nTrainable parameters: {trainable:,}\nTotal parameters: {total:,}\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def save_main_model_tables(all_predictions: pd.DataFrame) -> None:
    multimodal = all_predictions[all_predictions["model_slug"].eq(MAIN_MODEL_SLUG)].copy()
    if multimodal.empty:
        raise ValueError("Main multimodal predictions are missing")
    multimodal.to_csv(
        RESULTS_DIR / "multimodal_validation_predictions.csv",
        index=False,
        lineterminator="\n",
    )
    rows: list[dict[str, object]] = []
    for category, group in multimodal.groupby("part_category", sort=True):
        rows.append(
            {
                "part_category": category,
                "samples": len(group),
                "correct": int(group["is_correct"].sum()),
                "accuracy": accuracy_score(group["true_label"], group["predicted_label"]),
                "macro_f1": f1_score(
                    group["true_label"], group["predicted_label"], average="macro", zero_division=0
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        RESULTS_DIR / "multimodal_per_category.csv", index=False, lineterminator="\n"
    )


def paired_exact_comparison(
    all_predictions: pd.DataFrame, left_slug: str, right_slug: str
) -> dict[str, object]:
    left = all_predictions[all_predictions["model_slug"].eq(left_slug)].set_index("sample_id")
    right = all_predictions[all_predictions["model_slug"].eq(right_slug)].set_index("sample_id")
    if set(left.index) != set(right.index):
        raise ValueError("Paired models do not cover the same validation rows")
    right = right.loc[left.index]
    left_correct = left["is_correct"].astype(bool).to_numpy()
    right_correct = right["is_correct"].astype(bool).to_numpy()
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    p_value = 1.0 if discordant == 0 else float(binomtest(left_only, discordant, 0.5).pvalue)
    return {
        "left_model_slug": left_slug,
        "right_model_slug": right_slug,
        "left_correct_right_wrong": left_only,
        "left_wrong_right_correct": right_only,
        "discordant_predictions": discordant,
        "exact_two_sided_p_value": p_value,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    set_seed(RANDOM_STATE)

    train = load_split("train")
    validation = load_split("validation")
    if any(check_split(train, validation).values()):
        raise SystemExit("Train and validation identities overlap")
    if set(validation["source"]) != {"wikimedia"}:
        raise SystemExit("Validation must be real-image only")

    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    def record(
        name: str,
        slug: str,
        modality: str,
        training_data: str,
        predicted: np.ndarray,
        image_category_accuracy: float | None = None,
        text_category_accuracy: float | None = None,
    ) -> None:
        predicted = np.asarray(predicted)
        results.append(
            score_model(
                name,
                slug,
                modality,
                training_data,
                validation,
                predicted,
                image_category_accuracy,
                text_category_accuracy,
            )
        )
        predictions.append(prediction_rows(name, slug, validation, predicted))

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(train), 1)), train["label"])
    record(
        "Majority baseline",
        "majority",
        "none",
        "all train rows",
        majority.predict(np.zeros((len(validation), 1))),
    )

    text_vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), sublinear_tf=True, norm="l2"
    )
    train_tfidf_sparse = text_vectorizer.fit_transform(train["description"])
    validation_tfidf_sparse = text_vectorizer.transform(validation["description"])
    text_baseline = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    text_baseline.fit(train_tfidf_sparse, train["label"])
    record(
        "TF-IDF + Logistic Regression",
        "tfidf_logistic_regression",
        "text",
        "all train rows",
        text_baseline.predict(validation_tfidf_sparse),
    )

    flat_train = simple_image_features(train)
    flat_validation = simple_image_features(validation)
    image_baseline = LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    image_baseline.fit(flat_train, train["label"])
    record(
        "Image pixels + Logistic Regression",
        "image_logistic_regression",
        "image",
        "all train rows",
        image_baseline.predict(flat_validation),
    )

    combined_train = sparse.hstack(
        [sparse.csr_matrix(flat_train), train_tfidf_sparse], format="csr"
    )
    combined_validation = sparse.hstack(
        [sparse.csr_matrix(flat_validation), validation_tfidf_sparse], format="csr"
    )
    combined_baseline = LogisticRegression(max_iter=4000, random_state=RANDOM_STATE)
    combined_baseline.fit(combined_train, train["label"])
    record(
        "Image + text Logistic Regression",
        "image_text_logistic_regression",
        "image + text",
        "all train rows",
        combined_baseline.predict(combined_validation),
    )

    train_text = train_tfidf_sparse.toarray().astype(np.float32)
    validation_text = validation_tfidf_sparse.toarray().astype(np.float32)
    train_images = torch_images(train)
    validation_images = torch_images(validation)
    y_train = encode_labels(train["label"])
    y_validation = encode_labels(validation["label"])
    index_to_label = np.asarray(LABELS)
    text_dimension = train_text.shape[1]

    category_names = sorted(set(train["part_category"]) | set(train["text_category"]))
    category_to_index = {category: index for index, category in enumerate(category_names)}
    train_image_categories = train["part_category"].map(category_to_index).to_numpy(np.int64)
    train_text_categories = train["text_category"].map(category_to_index).to_numpy(np.int64)
    validation_image_categories = validation["part_category"].map(category_to_index).to_numpy(np.int64)
    validation_text_categories = validation["text_category"].map(category_to_index).to_numpy(np.int64)

    set_seed(RANDOM_STATE)
    text_model = TextRelationMLP(text_dimension)
    save_model_summary(text_model, RESULTS_DIR / "torch_text_architecture.txt")
    pred_index, history = fit_single_input_model(
        text_model,
        train_text,
        y_train,
        validation_text,
        y_validation,
        augment_image_batches=False,
    )
    history.to_csv(
        RESULTS_DIR / "torch_text_training_history.csv", index=False, lineterminator="\n"
    )
    record(
        "PyTorch neural text model",
        "torch_text",
        "text",
        "all train rows",
        index_to_label[pred_index],
    )

    set_seed(RANDOM_STATE + 1)
    image_model = ImageRelationCNN()
    save_model_summary(image_model, RESULTS_DIR / "torch_cnn_image_architecture.txt")
    pred_index, history = fit_single_input_model(
        image_model,
        train_images,
        y_train,
        validation_images,
        y_validation,
        augment_image_batches=True,
    )
    history.to_csv(
        RESULTS_DIR / "torch_cnn_image_training_history.csv", index=False, lineterminator="\n"
    )
    record(
        "PyTorch CNN image model",
        "torch_cnn_image",
        "image",
        "all train rows",
        index_to_label[pred_index],
    )

    real_mask = train["source"].eq("wikimedia").to_numpy()
    multimodal_jobs = [
        (
            "PyTorch multimodal CNN (real-only training)",
            REAL_ONLY_SLUG,
            "50 real train images",
            real_mask,
            RANDOM_STATE + 2,
        ),
        (
            "PyTorch multimodal CNN (real + synthetic training)",
            SYNTHETIC_SLUG,
            "50 real + 50 synthetic train images",
            np.ones(len(train), dtype=bool),
            RANDOM_STATE + 3,
        ),
    ]
    for name, slug, training_data, mask, seed in multimodal_jobs:
        set_seed(seed)
        model = MultimodalRelationCNN(text_dimension)
        save_model_summary(model, RESULTS_DIR / f"{slug}_architecture.txt")
        pred_index, history, auxiliary_metrics = fit_multimodal_model(
            model,
            train_images[mask],
            train_text[mask],
            y_train[mask],
            train_image_categories[mask],
            train_text_categories[mask],
            validation_images,
            validation_text,
            y_validation,
            validation_image_categories,
            validation_text_categories,
        )
        history.to_csv(
            RESULTS_DIR / f"{slug}_training_history.csv", index=False, lineterminator="\n"
        )
        record(
            name,
            slug,
            "image + text",
            training_data,
            index_to_label[pred_index],
            auxiliary_metrics["image_category_accuracy"],
            auxiliary_metrics["text_category_accuracy"],
        )

    result_table = pd.DataFrame(results).sort_values(
        ["validation_macro_f1", "validation_accuracy", "model"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    all_predictions = pd.concat(predictions, ignore_index=True)
    result_table.to_csv(RESULTS_DIR / "model_comparison.csv", index=False, lineterminator="\n")
    all_predictions.to_csv(
        RESULTS_DIR / "validation_predictions.csv", index=False, lineterminator="\n"
    )
    save_main_model_tables(all_predictions)

    result_table[
        result_table["model_slug"].isin([REAL_ONLY_SLUG, SYNTHETIC_SLUG])
    ].to_csv(RESULTS_DIR / "synthetic_ablation.csv", index=False, lineterminator="\n")

    comparisons = [
        paired_exact_comparison(
            all_predictions, MAIN_MODEL_SLUG, "image_text_logistic_regression"
        ),
        paired_exact_comparison(
            all_predictions, MAIN_MODEL_SLUG, SYNTHETIC_SLUG
        ),
    ]
    pd.DataFrame(comparisons).to_csv(
        RESULTS_DIR / "paired_comparisons.csv", index=False, lineterminator="\n"
    )

    run_info = {
        "result_source": "Generated by python -m src.train from the current project files.",
        "random_state": RANDOM_STATE,
        "deep_learning_framework": "PyTorch",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "training_split": "data/train.csv",
        "model_selection_split": "data/validation.csv (real images only)",
        "validation_rows": len(validation),
        "validation_images": validation["image_id"].nunique(),
        "test_split_used": False,
        "test_evaluation_permitted": False,
        "saved_model_count": len(result_table),
        "main_model_slug": MAIN_MODEL_SLUG,
        "auxiliary_loss_weight": AUXILIARY_LOSS_WEIGHT,
    }
    (RESULTS_DIR / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    print(result_table.to_string(index=False))


if __name__ == "__main__":
    main()
