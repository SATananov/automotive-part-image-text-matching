from __future__ import annotations

import os
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import src.import_dataset_v2 as import_module

from src.import_dataset_v2 import (
    ALL_CATEGORIES,
    COMMONS_FALLBACK_CATEGORIES,
    COMMONS_SEARCH_TERMS,
    KAGGLE_CATEGORY_ALIASES,
    MANIFEST_COLUMNS,
    TOTAL_PER_CATEGORY,
    allowed_commons_license,
    normalize_name,
    safe_extract_zip,
)


def test_hybrid_source_mapping_covers_exact_project_ontology() -> None:
    assert len(KAGGLE_CATEGORY_ALIASES) == 10
    assert set(COMMONS_SEARCH_TERMS) == {"air_filter", "shock_absorber"}
    assert COMMONS_FALLBACK_CATEGORIES == {"air_filter", "shock_absorber"}
    assert set(COMMONS_SEARCH_TERMS).issubset(KAGGLE_CATEGORY_ALIASES)
    assert set(ALL_CATEGORIES) == {
        "air_filter",
        "alternator",
        "brake_disc",
        "brake_pad",
        "coil_spring",
        "headlight",
        "oil_filter",
        "shock_absorber",
        "starter",
        "taillight",
    }


def test_source_alias_normalization_is_punctuation_and_case_insensitive() -> None:
    assert normalize_name("Brake-Rotor") == "BRAKE ROTOR"
    assert normalize_name("tail_lights") == "TAIL LIGHTS"
    assert normalize_name("  starter motor  ") == "STARTER MOTOR"


def test_kaggle_discovery_accepts_air_filter_and_shock_aliases(tmp_path: Path) -> None:
    for directory_name in ("AIR FILTER", "SHOCK ABSORBER"):
        directory = tmp_path / "train" / directory_name
        directory.mkdir(parents=True)
        (directory / "sample.jpg").write_bytes(b"placeholder")

    discovered = import_module.discover_kaggle_class_directories(tmp_path)

    assert [path.name for path in discovered["air_filter"]] == ["sample.jpg"]
    assert [path.name for path in discovered["shock_absorber"]] == ["sample.jpg"]


def test_commons_license_filter_accepts_only_open_license_markers() -> None:
    for value in ("CC BY 4.0", "CC BY-SA 3.0", "CC0 1.0", "Public domain", "PDM"):
        assert allowed_commons_license(value)
    for value in ("All rights reserved", "Fair use", "Unknown", ""):
        assert not allowed_commons_license(value)


def test_dataset_v2_manifest_schema_contains_provenance_and_hash_fields() -> None:
    required = {
        "provider",
        "source_dataset",
        "dataset_url",
        "source_title",
        "description_url",
        "author",
        "credit",
        "license_short_name",
        "license_url",
        "source_sha256",
        "sha256",
        "dhash",
        "modifications",
    }
    assert required.issubset(MANIFEST_COLUMNS)
    assert len(MANIFEST_COLUMNS) == len(set(MANIFEST_COLUMNS))


def test_safe_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "not allowed")
    with pytest.raises(ValueError, match="Unsafe archive member"):
        safe_extract_zip(archive, tmp_path / "target")


def test_kaggle_download_uses_managed_cache_without_direct_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded_root = tmp_path / "downloaded"
    downloaded_root.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_dataset_download(handle: str, **kwargs: object) -> str:
        calls.append((handle, kwargs))
        return str(downloaded_root)

    fake_module = types.SimpleNamespace(dataset_download=fake_dataset_download)
    monkeypatch.setitem(sys.modules, "kagglehub", fake_module)
    monkeypatch.setattr(import_module, "CACHE_DIR", tmp_path / "project-cache")
    monkeypatch.delenv("KAGGLEHUB_CACHE", raising=False)

    resolved = import_module.download_kaggle_source()

    assert resolved == downloaded_root
    assert calls == [(import_module.KAGGLE_HANDLE, {})]
    assert os.environ["KAGGLEHUB_CACHE"] == str(
        (tmp_path / "project-cache" / "kagglehub").resolve()
    )


def test_kaggle_download_retries_without_nonempty_output_directory_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloaded_root = tmp_path / "downloaded"
    downloaded_root.mkdir()
    attempts = 0

    def flaky_dataset_download(handle: str, **kwargs: object) -> str:
        nonlocal attempts
        attempts += 1
        assert handle == import_module.KAGGLE_HANDLE
        assert kwargs == {}
        if attempts == 1:
            raise ConnectionError("temporary disconnect")
        return str(downloaded_root)

    fake_module = types.SimpleNamespace(dataset_download=flaky_dataset_download)
    monkeypatch.setitem(sys.modules, "kagglehub", fake_module)
    monkeypatch.setattr(import_module, "CACHE_DIR", tmp_path / "project-cache")
    monkeypatch.setattr(import_module.time, "sleep", lambda _seconds: None)

    assert import_module.download_kaggle_source() == downloaded_root
    assert attempts == 2


def test_import_prefers_kaggle_when_fallback_categories_are_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    fake_candidates = {
        category: [object() for _ in range(TOTAL_PER_CATEGORY)]
        for category in ALL_CATEGORIES
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(import_module, "resolve_kaggle_root", lambda path: path)
    monkeypatch.setattr(import_module, "build_kaggle_candidates", lambda _root: fake_candidates)

    def fail_commons(_category: str) -> list[object]:
        raise AssertionError("Commons fallback should not be called")

    monkeypatch.setattr(import_module, "build_commons_candidates_from_api", fail_commons)
    monkeypatch.setattr(
        import_module,
        "select_candidates",
        lambda candidates: captured.setdefault("candidates", candidates),
    )
    monkeypatch.setattr(import_module, "write_selected", lambda selected, force: selected)
    monkeypatch.setattr(import_module, "validate_import", lambda manifest: {"status": "PASS"})

    assert import_module.import_dataset(source_root, force=True) == {"status": "PASS"}
    assert captured["candidates"] is fake_candidates


def test_import_supplements_missing_kaggle_fallback_category(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    fake_candidates = {
        category: [object() for _ in range(TOTAL_PER_CATEGORY)]
        for category in ALL_CATEGORIES
    }
    fake_candidates["air_filter"] = []
    fallback_rows = [object() for _ in range(TOTAL_PER_CATEGORY)]
    called: list[str] = []

    monkeypatch.setattr(import_module, "resolve_kaggle_root", lambda path: path)
    monkeypatch.setattr(import_module, "build_kaggle_candidates", lambda _root: fake_candidates)

    def fake_commons(category: str) -> list[object]:
        called.append(category)
        return fallback_rows if category == "air_filter" else []

    monkeypatch.setattr(import_module, "build_commons_candidates_from_api", fake_commons)
    monkeypatch.setattr(import_module, "select_candidates", lambda candidates: candidates)
    monkeypatch.setattr(import_module, "write_selected", lambda selected, force: selected)
    monkeypatch.setattr(import_module, "validate_import", lambda manifest: {"status": "PASS"})

    assert import_module.import_dataset(source_root, force=True) == {"status": "PASS"}
    assert called == ["air_filter"]
    assert fake_candidates["air_filter"] == fallback_rows


def test_balanced_selection_quota_uses_full_canonical_target_when_available() -> None:
    counts = {category: 25 for category in ALL_CATEGORIES}
    assert import_module.balanced_selection_quota(counts) == (20, 5)


def test_balanced_selection_quota_adapts_to_sparse_open_source_category() -> None:
    counts = {category: 25 for category in ALL_CATEGORIES}
    counts["air_filter"] = 8
    counts["shock_absorber"] = 11
    assert import_module.balanced_selection_quota(counts) == (6, 2)


def test_balanced_selection_quota_rejects_too_few_independent_images() -> None:
    counts = {category: 25 for category in ALL_CATEGORIES}
    counts["air_filter"] = 3
    with pytest.raises(ValueError, match="minimum balanced quota"):
        import_module.balanced_selection_quota(counts)

def test_commons_relevance_requires_automotive_context() -> None:
    assert import_module.commons_candidate_relevant(
        "shock_absorber",
        "File:Land Rover dampers.jpg",
        "Vehicle suspension dampers",
        ("Category:Automobile shock absorbers",),
    )
    assert not import_module.commons_candidate_relevant(
        "shock_absorber",
        "File:The shock absorber in Taipei 101.jpg",
        "Tuned mass damper in a skyscraper",
        (),
    )
    assert not import_module.commons_candidate_relevant(
        "shock_absorber",
        "File:SteelProtection absorber.jpg",
        "Fall arrest safety lanyard",
        (),
    )


def test_standardizer_composites_transparency_and_rejects_empty_output(tmp_path: Path) -> None:
    source = tmp_path / "transparent.png"
    destination = tmp_path / "standardized.jpg"
    Image.new("RGBA", (400, 400), (0, 0, 0, 0)).save(source)

    with pytest.raises(ValueError, match="standard deviation|entropy|luminance"):
        import_module.standardize_image(source, destination)
    assert not destination.exists()


def test_image_content_metrics_accept_visual_information() -> None:
    gradient = np.tile(np.arange(224, dtype=np.uint8), (224, 1))
    image = Image.fromarray(gradient, mode="L").convert("RGB")
    pixel_std, entropy, dominant_fraction = import_module.validate_standardized_image(image)
    assert pixel_std >= import_module.MIN_STANDARDIZED_PIXEL_STD
    assert entropy >= import_module.MIN_STANDARDIZED_LUMA_ENTROPY
    assert dominant_fraction <= import_module.MAX_STANDARDIZED_SINGLE_LUMA_FRACTION

