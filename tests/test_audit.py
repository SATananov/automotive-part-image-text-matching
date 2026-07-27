from __future__ import annotations

import json

from src.audit import run_audit
from src.data import RESULTS_DIR


def test_audit_confirms_independent_real_validation() -> None:
    summary = run_audit()
    assert summary["validation_real_images"] == 10
    assert summary["validation_synthetic_images"] == 0
    assert summary["exact_cross_split_image_hash_overlap"] == 0
    assert summary["suspicious_real_image_pairs"] == 0
    assert all(value == 0 for value in summary["overlap"].values())


def test_audit_confirms_no_exact_text_leakage() -> None:
    summary = run_audit()
    assert summary["exact_description_overlap"] == 0
    assert summary["validation_rows_with_exact_seen_description"] == 0


def test_shortcut_diagnostics_remain_at_chance() -> None:
    summary = run_audit()
    for row in summary["shortcut_baselines"]:
        assert row["accuracy"] == 1 / 3
    assert summary["image_only_relation_ceiling_accuracy"] == 1 / 3


def test_license_audit_passes() -> None:
    status = run_audit()["licenses"]
    assert status["license_rows"] == 70
    assert status["wikimedia_images"] == 70
    assert all(
        status[key]
        for key in (
            "all_wikimedia_images_covered",
            "unique_commons_titles",
            "unique_description_urls",
            "unique_file_hashes",
            "all_license_hashes_match",
        )
    )


def test_saved_audit_matches_current_critical_fields() -> None:
    current = run_audit()
    saved = json.loads((RESULTS_DIR / "data_audit.json").read_text(encoding="utf-8"))
    assert current["overlap"] == saved["overlap"]
    assert current["licenses"] == saved["licenses"]
    assert current["test_lock"] == saved["test_lock"]
    assert current["test_split_used"] is False
