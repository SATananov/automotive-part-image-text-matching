from __future__ import annotations

import hashlib
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.captions_v3 import PART_LANGUAGE
from src.data import COLUMNS, LABELS
from src.data_v3 import CATEGORIES, FAMILIES, PARTIAL_TARGET
from src.models import MultimodalRelationCNN

PAIRS_PER_LABEL = 2
ROWS_PER_IMAGE = PAIRS_PER_LABEL * len(LABELS)
EXPECTED_TEST_IMAGES = 80
EXPECTED_TEST_IMAGES_PER_CATEGORY = 10
EXPECTED_TEST_ROWS = EXPECTED_TEST_IMAGES * ROWS_PER_IMAGE
EXPECTED_ROWS_PER_LABEL = EXPECTED_TEST_ROWS // len(LABELS)
EXPECTED_TEXT_DIMENSION = 342

EXPECTED_MODEL_SLUG = "torch_multimodal_dataset_v3"
EXPECTED_CHECKPOINT_SHA256 = (
    "bce8a98fc96140294f3043fd845e1a7b6f491c67869a6c775877cd6ca1aa2a17"
)
EXPECTED_VECTORIZER_FINGERPRINT = (
    "aad4f36d127ea8eb06cc3e407c1873bf93ee24576e136129f837eb2f32eab5ac"
)
EXPECTED_RELATION_PROTOCOL_FINGERPRINT = (
    "db89b8c2aa5b94e9eec2913d80d17d0b5ebfbbd90a0e5e763706bd03b5fb4cc7"
)

FINAL_TEST_TEMPLATES = (
    "A {noun} from the {subsystem}; its purpose is to {function}.",
    "This vehicle part is a {noun}, recognized by {form}.",
)

_VALID_MISMATCH_PERMUTATIONS = tuple(
    candidate
    for candidate in permutations(CATEGORIES)
    if all(
        FAMILIES[source] != FAMILIES[target]
        for source, target in zip(CATEGORIES, candidate)
    )
)

if not _VALID_MISMATCH_PERMUTATIONS:
    raise AssertionError(
        "No valid Dataset V3 final-test mismatch permutations were found."
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def render_final_test_caption(category: str, variant: int) -> str:
    if category not in PART_LANGUAGE:
        raise ValueError(
            f"Unknown Dataset V3 final-test category: {category}"
        )

    language = PART_LANGUAGE[category]
    template = FINAL_TEST_TEMPLATES[
        variant % len(FINAL_TEST_TEMPLATES)
    ]

    return template.format(
        noun=language.noun,
        subsystem=language.subsystem,
        function=language.function,
        form=language.form,
    )


def all_final_test_captions() -> set[str]:
    return {
        render_final_test_caption(category, variant)
        for category in CATEGORIES
        for variant in range(len(FINAL_TEST_TEMPLATES))
    }


def final_test_mismatch_mapping(
    image_rank: int,
    pair_index: int,
) -> dict[str, str]:
    token = (
        f"dataset-v3|test|{image_rank}|{pair_index}"
    ).encode("utf-8")
    index = int(
        hashlib.sha256(token).hexdigest(),
        16,
    ) % len(_VALID_MISMATCH_PERMUTATIONS)

    return dict(
        zip(
            CATEGORIES,
            _VALID_MISMATCH_PERMUTATIONS[index],
        )
    )


def relation_protocol_fingerprint() -> str:
    payload = {
        "categories": list(CATEGORIES),
        "families": FAMILIES,
        "partial_target": PARTIAL_TARGET,
        "pairs_per_label": PAIRS_PER_LABEL,
        "templates": list(FINAL_TEST_TEMPLATES),
        "mismatch_token": (
            "dataset-v3|test|{image_rank}|{pair_index}"
        ),
        "valid_mismatch_permutations": [
            list(candidate)
            for candidate in _VALID_MISMATCH_PERMUTATIONS
        ],
    }
    canonical = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def build_final_test_relations(
    test_images: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "candidate_id",
        "project_category",
        "image_group_id",
        "split_rank_within_category",
        "repository_relative_path",
    }
    missing = sorted(required - set(test_images.columns))

    if missing:
        raise ValueError(
            f"Final-test image manifest columns are missing: {missing}"
        )

    per_category = (
        test_images["project_category"]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    if set(per_category.index) != set(CATEGORIES):
        raise ValueError(
            "Final-test image manifest has unexpected categories."
        )

    if not (
        per_category == EXPECTED_TEST_IMAGES_PER_CATEGORY
    ).all():
        raise ValueError(
            "Final-test image manifest is not balanced: "
            f"{per_category.to_dict()}"
        )

    if len(test_images) != EXPECTED_TEST_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_IMAGES} final-test images, "
            f"found {len(test_images)}."
        )

    rows: list[dict[str, str]] = []

    for category in CATEGORIES:
        selected = test_images[
            test_images["project_category"].astype(str).eq(
                category
            )
        ].sort_values(
            [
                "split_rank_within_category",
                "candidate_id",
            ],
            kind="stable",
        )

        for image_rank, image in enumerate(
            selected.itertuples(index=False)
        ):
            for pair_index in range(PAIRS_PER_LABEL):
                mismatch_target = final_test_mismatch_mapping(
                    image_rank,
                    pair_index,
                )[category]
                relations = (
                    ("MATCH", category),
                    (
                        "PARTIAL_MATCH",
                        PARTIAL_TARGET[category],
                    ),
                    ("MISMATCH", mismatch_target),
                )

                for label, text_category in relations:
                    variant = (
                        image_rank + pair_index
                    ) % len(FINAL_TEST_TEMPLATES)
                    rows.append(
                        {
                            "sample_id": (
                                "v3_test_"
                                f"{image.candidate_id}_"
                                f"{label.lower()}_"
                                f"{pair_index + 1}"
                            ),
                            "image_id": str(
                                image.candidate_id
                            ),
                            "part_group_id": str(
                                image.image_group_id
                            ),
                            "object_group_id": str(
                                image.image_group_id
                            ),
                            "image_path": str(
                                image.repository_relative_path
                            ),
                            "part_family": FAMILIES[
                                category
                            ],
                            "part_category": category,
                            "text_category": text_category,
                            "description": (
                                render_final_test_caption(
                                    text_category,
                                    variant,
                                )
                            ),
                            "label": label,
                            "source": (
                                "dataset_v3_final_test"
                            ),
                        }
                    )

    return pd.DataFrame(rows, columns=COLUMNS)


def validate_final_test_relations(
    relations: pd.DataFrame,
    test_images: pd.DataFrame,
    train_relations: pd.DataFrame,
    validation_relations: pd.DataFrame,
    *,
    require_locked_paths: bool,
    require_group_disjoint: bool = True,
) -> dict[str, object]:
    if tuple(relations.columns) != COLUMNS:
        raise ValueError(
            "Unexpected final-test relation columns."
        )

    if len(relations) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_ROWS} final-test rows, "
            f"found {len(relations)}."
        )

    if relations["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate final-test sample_id values."
        )

    if relations["image_id"].nunique() != (
        EXPECTED_TEST_IMAGES
    ):
        raise ValueError(
            "Unexpected final-test image count."
        )

    if set(relations["label"].astype(str)) != set(LABELS):
        raise ValueError(
            "Unexpected final-test labels."
        )

    label_counts = (
        relations["label"].value_counts().sort_index()
    )

    if not (
        label_counts == EXPECTED_ROWS_PER_LABEL
    ).all():
        raise ValueError(
            "Final-test labels are not balanced: "
            f"{label_counts.to_dict()}"
        )

    per_image = (
        relations.groupby(["image_id", "label"])
        .size()
        .unstack(fill_value=0)
    )

    if not (per_image == PAIRS_PER_LABEL).all().all():
        raise ValueError(
            "Every final-test image must have two rows "
            "for each relation label."
        )

    text_table = pd.crosstab(
        relations["text_category"],
        relations["label"],
    )

    if (
        text_table.shape != (len(CATEGORIES), len(LABELS))
        or text_table.nunique().max() != 1
    ):
        raise ValueError(
            "Final-test text-category balance is invalid."
        )

    if set(relations["source"].astype(str)) != {
        "dataset_v3_final_test"
    }:
        raise ValueError(
            "Unexpected final-test relation source."
        )

    match = relations[
        relations["label"].eq("MATCH")
    ]
    partial = relations[
        relations["label"].eq("PARTIAL_MATCH")
    ]
    mismatch = relations[
        relations["label"].eq("MISMATCH")
    ]

    if not match["part_category"].eq(
        match["text_category"]
    ).all():
        raise ValueError(
            "Invalid MATCH relation in final test."
        )

    if not partial.apply(
        lambda row: (
            row["part_category"]
            != row["text_category"]
            and FAMILIES[row["part_category"]]
            == FAMILIES[row["text_category"]]
        ),
        axis=1,
    ).all():
        raise ValueError(
            "Invalid PARTIAL_MATCH relation in final test."
        )

    if not mismatch.apply(
        lambda row: (
            FAMILIES[row["part_category"]]
            != FAMILIES[row["text_category"]]
        ),
        axis=1,
    ).all():
        raise ValueError(
            "Invalid MISMATCH relation in final test."
        )

    normalized_paths = (
        relations["image_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    if require_locked_paths:
        if not normalized_paths.str.startswith(
            "data/locked_test/dataset_v3/images/"
        ).all():
            raise ValueError(
                "A final-test relation points outside "
                "the locked image directory."
            )
    elif normalized_paths.str.startswith(
        "data/locked_test/"
    ).any():
        raise ValueError(
            "Development dry-run relations entered "
            "the locked-test area."
        )

    test_groups = set(
        test_images["image_group_id"].astype(str)
    )
    train_groups = set(
        train_relations["object_group_id"].astype(str)
    )
    validation_groups = set(
        validation_relations[
            "object_group_id"
        ].astype(str)
    )

    if require_group_disjoint:
        if test_groups & train_groups:
            raise ValueError(
                "Final-test/train group overlap detected."
            )

        if test_groups & validation_groups:
            raise ValueError(
                "Final-test/validation group overlap detected."
            )

    test_descriptions = set(
        relations["description"].astype(str)
    )
    train_descriptions = set(
        train_relations["description"].astype(str)
    )
    validation_descriptions = set(
        validation_relations["description"].astype(str)
    )

    if test_descriptions & train_descriptions:
        raise ValueError(
            "Final-test descriptions overlap train descriptions."
        )

    if test_descriptions & validation_descriptions:
        raise ValueError(
            "Final-test descriptions overlap validation descriptions."
        )

    return {
        "status": "PASS_FINAL_TEST_RELATION_PROTOCOL",
        "images": EXPECTED_TEST_IMAGES,
        "rows": EXPECTED_TEST_ROWS,
        "rows_per_label": EXPECTED_ROWS_PER_LABEL,
        "pairs_per_label_per_image": PAIRS_PER_LABEL,
        "categories": len(CATEGORIES),
        "test_train_group_overlap": 0,
        "test_validation_group_overlap": 0,
        "description_overlap_with_train": 0,
        "description_overlap_with_validation": 0,
        "locked_paths_required": require_locked_paths,
        "group_disjointness_required": require_group_disjoint,
    }


def fit_frozen_text_vectorizer(
    train_relations: pd.DataFrame,
) -> tuple[TfidfVectorizer, str]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(
        train_relations["description"].astype(str)
    )

    payload = {
        "params": {
            "lowercase": True,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "norm": "l2",
        },
        "vocabulary": sorted(
            vectorizer.vocabulary_.items(),
            key=lambda item: item[0],
        ),
        "idf": [
            float(value)
            for value in vectorizer.idf_.tolist()
        ],
    }
    canonical = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    fingerprint = hashlib.sha256(
        canonical
    ).hexdigest()

    if len(vectorizer.vocabulary_) != (
        EXPECTED_TEXT_DIMENSION
    ):
        raise ValueError(
            "Unexpected frozen TF-IDF dimension: "
            f"{len(vectorizer.vocabulary_)}"
        )

    if fingerprint != EXPECTED_VECTORIZER_FINGERPRINT:
        raise ValueError(
            "Frozen TF-IDF fingerprint changed."
        )

    return vectorizer, fingerprint


def load_frozen_multimodal_model(
    checkpoint_path: Path,
) -> tuple[MultimodalRelationCNN, dict[str, object]]:
    if file_sha256(checkpoint_path) != (
        EXPECTED_CHECKPOINT_SHA256
    ):
        raise ValueError(
            "Selected multimodal checkpoint SHA-256 changed."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["model_slug"] != EXPECTED_MODEL_SLUG:
        raise ValueError(
            "Unexpected model slug in selected checkpoint."
        )

    if checkpoint["text_dimension"] != (
        EXPECTED_TEXT_DIMENSION
    ):
        raise ValueError(
            "Unexpected text dimension in selected checkpoint."
        )

    if tuple(checkpoint["labels"]) != tuple(LABELS):
        raise ValueError(
            "Unexpected labels in selected checkpoint."
        )

    expected_categories = tuple(sorted(CATEGORIES))

    if tuple(checkpoint["categories"]) != (
        expected_categories
    ):
        raise ValueError(
            "Unexpected categories in selected checkpoint."
        )

    model = MultimodalRelationCNN(
        EXPECTED_TEXT_DIMENSION,
        number_of_part_categories=len(
            expected_categories
        ),
    )
    model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )
    model.eval()

    return model, checkpoint


def protocol_model_dry_run(
    train_relations: pd.DataFrame,
    checkpoint_path: Path,
) -> dict[str, object]:
    vectorizer, fingerprint = (
        fit_frozen_text_vectorizer(
            train_relations
        )
    )

    captions: list[str] = []
    categories: list[str] = []

    for category in CATEGORIES:
        for variant in range(
            len(FINAL_TEST_TEMPLATES)
        ):
            captions.append(
                render_final_test_caption(
                    category,
                    variant,
                )
            )
            categories.append(category)

    text = vectorizer.transform(
        captions
    ).toarray().astype(np.float32)

    nonzero_per_row = np.count_nonzero(
        text,
        axis=1,
    )

    if np.any(nonzero_per_row == 0):
        raise ValueError(
            "A final-test caption has no frozen TF-IDF features."
        )

    model, checkpoint = (
        load_frozen_multimodal_model(
            checkpoint_path
        )
    )

    images = torch.zeros(
        (
            len(captions),
            3,
            48,
            48,
        ),
        dtype=torch.float32,
    )
    text_tensor = torch.from_numpy(text)

    with torch.no_grad():
        (
            relation_logits,
            image_category_logits,
            text_category_logits,
        ) = model(images, text_tensor)

    if tuple(relation_logits.shape) != (
        len(captions),
        len(LABELS),
    ):
        raise ValueError(
            "Unexpected relation output shape in protocol dry run."
        )

    if tuple(image_category_logits.shape) != (
        len(captions),
        len(CATEGORIES),
    ):
        raise ValueError(
            "Unexpected image-category output shape "
            "in protocol dry run."
        )

    if tuple(text_category_logits.shape) != (
        len(captions),
        len(CATEGORIES),
    ):
        raise ValueError(
            "Unexpected text-category output shape "
            "in protocol dry run."
        )

    predicted_categories = np.asarray(
        checkpoint["categories"]
    )[
        text_category_logits.argmax(
            dim=1
        ).cpu().numpy()
    ]
    expected = np.asarray(categories)
    text_category_accuracy = float(
        np.mean(
            predicted_categories == expected
        )
    )

    if text_category_accuracy != 1.0:
        raise ValueError(
            "Frozen model did not identify all final-test "
            "caption categories in the no-image dry run."
        )

    return {
        "status": "PASS_FINAL_TEST_PROTOCOL_MODEL_DRY_RUN",
        "captions": len(captions),
        "zero_vector_captions": int(
            np.sum(nonzero_per_row == 0)
        ),
        "text_dimension": text.shape[1],
        "vectorizer_fingerprint": fingerprint,
        "relation_output_shape": list(
            relation_logits.shape
        ),
        "image_category_output_shape": list(
            image_category_logits.shape
        ),
        "text_category_output_shape": list(
            text_category_logits.shape
        ),
        "text_category_accuracy": (
            text_category_accuracy
        ),
        "checkpoint_model_slug": (
            checkpoint["model_slug"]
        ),
    }
