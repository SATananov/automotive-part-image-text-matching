import json

from src.audit import RESULTS_DIR, run_audit


def test_audit_has_no_exact_or_group_overlap() -> None:
    summary = run_audit()
    assert summary["overlap"] == {
        "group_overlap": 0,
        "image_id_overlap": 0,
        "image_path_overlap": 0,
    }
    assert summary["exact_cross_split_image_hash_overlap"] == 0


def test_audit_reports_generated_similarity_warning() -> None:
    summary = run_audit()
    assert summary["generated_similar_pairs_at_0_99"] > 0
    assert summary["wikimedia_similar_pairs_at_0_99"] == 0


def test_category_and_source_do_not_predict_the_label() -> None:
    summary = run_audit()
    scores = {row["features"]: row["accuracy"] for row in summary["shortcut_baselines"]}
    assert scores["source"] == 1 / 3
    assert scores["part_category"] == 1 / 3
    assert scores["source+part_category"] == 1 / 3


def test_audit_records_text_repetition_and_image_only_ceiling() -> None:
    summary = run_audit()
    assert summary["unique_descriptions_train"] == 14
    assert summary["unique_descriptions_validation"] == 14
    assert summary["validation_unique_descriptions_seen_in_train"] == 14
    assert summary["validation_rows_with_seen_description"] == 60
    assert summary["image_only_relation_ceiling_accuracy"] == 1 / 3


def test_saved_audit_matches_current_audit() -> None:
    current = run_audit()
    saved = json.loads((RESULTS_DIR / "data_audit.json").read_text(encoding="utf-8"))
    assert current["overlap"] == saved["overlap"]
    assert current["test_lock"] == saved["test_lock"]
    assert current["test_split_used"] is False
