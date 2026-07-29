from __future__ import annotations

import json

import pandas as pd

from src.audit import run_audit
from src.data import DATA_DIR, RESULTS_DIR


def test_audit_confirms_dataset_v2_independent_real_validation() -> None:
    summary = run_audit()
    manifest = pd.read_csv(DATA_DIR / "dataset_v2_manifest.csv")
    imported = manifest["assigned_split"].value_counts().to_dict()
    assert summary["dataset_version"] == "2.0"
    assert summary["train_images"] == 100 + imported["train"]
    assert summary["validation_images"] == 10 + imported["validation"]
    assert summary["train_real_images"] == 50 + imported["train"]
    assert summary["train_synthetic_images"] == 50
    assert summary["validation_real_images"] == 10 + imported["validation"]
    assert summary["validation_synthetic_images"] == 0
    assert summary["exact_cross_split_image_hash_overlap"] == 0
    assert summary["suspicious_real_image_pairs"] == 0
    assert all(value == 0 for value in summary["overlap"].values())


def test_audit_confirms_no_exact_text_leakage() -> None:
    summary = run_audit()
    assert summary["unique_descriptions_train"] == 40
    assert summary["unique_descriptions_validation"] == 20
    assert summary["exact_description_overlap"] == 0
    assert summary["validation_rows_with_exact_seen_description"] == 0


def test_shortcut_diagnostics_remain_at_chance() -> None:
    summary = run_audit()
    for row in summary["shortcut_baselines"]:
        assert row["accuracy"] == 1 / 3
    assert summary["image_only_relation_ceiling_accuracy"] == 1 / 3


def test_both_license_audits_pass() -> None:
    status = run_audit()["licenses"]
    manifest = pd.read_csv(DATA_DIR / "dataset_v2_manifest.csv")
    assert status["license_rows"] == 70 + len(manifest)
    assert status["all_license_hashes_match"] is True
    assert status["wikimedia"]["license_rows"] == status["wikimedia"]["images"] == 70
    assert status["dataset_v2"]["license_rows"] == status["dataset_v2"]["images"] == len(manifest)
    assert sum(status["dataset_v2"]["providers"].values()) == len(manifest)
    assert status["dataset_v2"]["all_provenance_fields_present"] is True
    assert status["wikimedia"]["all_hashes_match"] is True
    assert status["dataset_v2"]["all_hashes_match"] is True


def test_saved_audit_matches_current_critical_fields() -> None:
    current = run_audit()
    saved = json.loads((RESULTS_DIR / "data_audit.json").read_text(encoding="utf-8"))
    assert current["overlap"] == saved["overlap"]
    assert current["licenses"] == saved["licenses"]
    assert current["test_lock"] == saved["test_lock"]
    assert current["test_split_used"] is False
