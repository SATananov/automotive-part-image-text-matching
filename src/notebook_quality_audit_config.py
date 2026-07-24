from __future__ import annotations

from pathlib import Path

from src.real_dataset_config import PROJECT_ROOT

QUALITY_AUDIT_STEP = "010.7"
QUALITY_AUDIT_BASE_COMMIT = "2f41d84"
QUALITY_AUDIT_READINESS = (
    "NOTEBOOK_EXECUTION_VISUAL_QA_AND_CITATION_AUDIT_PASS"
)

QUALITY_AUDIT_REPORT_ROOT = (
    PROJECT_ROOT / "reports" / "notebook_quality_audit"
)
NOTEBOOK_EXECUTION_AUDIT_PATH = (
    QUALITY_AUDIT_REPORT_ROOT / "notebook_execution_audit.json"
)
VISUAL_OUTPUT_AUDIT_PATH = (
    QUALITY_AUDIT_REPORT_ROOT / "visual_output_audit.json"
)
NUMERIC_CONSISTENCY_AUDIT_PATH = (
    QUALITY_AUDIT_REPORT_ROOT / "numeric_consistency_audit.json"
)
CITATION_AUDIT_PATH = (
    QUALITY_AUDIT_REPORT_ROOT / "citation_audit.json"
)
QUALITY_AUDIT_STATUS_PATH = (
    QUALITY_AUDIT_REPORT_ROOT / "notebook_quality_audit_status.json"
)
QUALITY_AUDIT_MANIFEST_PATH = (
    QUALITY_AUDIT_REPORT_ROOT / "notebook_quality_audit_manifest.json"
)
QUALITY_AUDIT_SUMMARY_PATH = (
    QUALITY_AUDIT_REPORT_ROOT / "notebook_quality_audit_summary.md"
)

CITATION_VERIFICATION_DATE = "2026-07-23"

CITATION_SOURCES = (
    {
        "reference_id": 1,
        "short_name": "VSE++",
        "title": "VSE++: Improving Visual-Semantic Embeddings with Hard Negatives",
        "url": "https://bmvc2018.org/contents/papers/0344.pdf",
        "expected_domain": "bmvc2018.org",
        "source_type": "conference_paper",
        "claim": "hard negatives for visual-semantic embedding and cross-modal retrieval",
    },
    {
        "reference_id": 2,
        "short_name": "VisualBERT",
        "title": "VisualBERT: A Simple and Performant Baseline for Vision and Language",
        "url": "https://arxiv.org/abs/1908.03557",
        "expected_domain": "arxiv.org",
        "source_type": "research_paper",
        "claim": "joint Transformer modeling of text tokens and image-region representations",
    },
    {
        "reference_id": 3,
        "short_name": "CLIP",
        "title": "Learning Transferable Visual Models From Natural Language Supervision",
        "url": "https://proceedings.mlr.press/v139/radford21a.html",
        "expected_domain": "proceedings.mlr.press",
        "source_type": "conference_paper",
        "claim": "large-scale contrastive image-text pretraining",
    },
    {
        "reference_id": 4,
        "short_name": "Keras Functional API",
        "title": "The Functional API",
        "url": "https://www.tensorflow.org/guide/keras/functional_api",
        "expected_domain": "www.tensorflow.org",
        "source_type": "official_documentation",
        "claim": "multi-input neural-network construction",
    },
    {
        "reference_id": 5,
        "short_name": "TfidfVectorizer",
        "title": "TfidfVectorizer documentation",
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html",
        "expected_domain": "scikit-learn.org",
        "source_type": "official_documentation",
        "claim": "TF-IDF feature extraction",
    },
    {
        "reference_id": 6,
        "short_name": "LogisticRegression",
        "title": "LogisticRegression documentation",
        "url": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "expected_domain": "scikit-learn.org",
        "source_type": "official_documentation",
        "claim": "regularized logistic-regression classifier",
    },
)

EXPECTED_FIGURE_TITLES = (
    "Integrated dataset composition by committed split",
    "Generated development:",
    "Reviewed open-license:",
    "Integrated validation comparison",
    "Retained multimodal model — validation confusion matrix",
    "Retained multimodal model — validation error types",
)

MINIMUM_FIGURE_WIDTH = 300
MINIMUM_FIGURE_HEIGHT = 300
MINIMUM_FIGURE_BYTES = 5_000
MINIMUM_PIXEL_STANDARD_DEVIATION = 2.0
