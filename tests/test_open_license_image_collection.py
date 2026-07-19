from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

import src.build_open_license_review_gallery as gallery
import src.collect_open_license_images as collector
import src.validate_open_license_dataset as validator
from src.open_license_dataset_config import (
    OPEN_LICENSE_MANIFEST_COLUMNS,
    OPEN_LICENSE_REVIEW_COLUMNS,
    OPEN_LICENSE_SEARCH_LIMIT,
    OPEN_LICENSE_SEARCH_QUERIES,
)


def image_bytes(
    *,
    size: tuple[int, int] = (640, 480),
    color: tuple[int, int, int] = (120, 80, 40),
    image_format: str = "JPEG",
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(
        buffer,
        format=image_format,
    )
    return buffer.getvalue()


def commons_page(
    *,
    page_id: int,
    title: str,
    license_name: str = "CC BY-SA 4.0",
    license_url: str = (
        "https://creativecommons.org/licenses/by-sa/4.0/"
    ),
    author: str = "Example Author",
    download_url: str | None = None,
) -> dict:
    return {
        "pageid": page_id,
        "title": f"File:{title}",
        "imageinfo": [
            {
                "url": (
                    f"https://upload.wikimedia.org/{page_id}.jpg"
                ),
                "thumburl": download_url
                or f"https://upload.wikimedia.org/thumb/{page_id}.jpg",
                "descriptionurl": (
                    f"https://commons.wikimedia.org/wiki/File:{title}"
                ),
                "mime": "image/jpeg",
                "extmetadata": {
                    "Artist": {"value": author},
                    "Credit": {"value": ""},
                    "LicenseShortName": {
                        "value": license_name
                    },
                    "LicenseUrl": {"value": license_url},
                    "AttributionRequired": {"value": "true"},
                    "UsageTerms": {"value": license_name},
                },
            }
        ],
    }


def configure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_root = tmp_path / "data" / "external" / "open_license"
    report_root = tmp_path / "reports" / "external_dataset"

    path_values = {
        "OPEN_LICENSE_IMAGES_DIRECTORY": open_root / "images",
        "OPEN_LICENSE_MANIFEST_PATH": (
            open_root / "open_license_manifest.csv"
        ),
        "OPEN_LICENSE_REVIEW_PATH": (
            open_root / "open_license_review.csv"
        ),
        "OPEN_LICENSE_ATTRIBUTION_PATH": (
            open_root / "ATTRIBUTION.md"
        ),
        "OPEN_LICENSE_COLLECTION_REPORT_PATH": (
            report_root / "open_license_collection_summary.md"
        ),
    }
    for name, value in path_values.items():
        monkeypatch.setattr(collector, name, value)

    validator_values = {
        "OPEN_LICENSE_MANIFEST_PATH": (
            open_root / "open_license_manifest.csv"
        ),
        "OPEN_LICENSE_REVIEW_PATH": (
            open_root / "open_license_review.csv"
        ),
        "OPEN_LICENSE_ATTRIBUTION_PATH": (
            open_root / "ATTRIBUTION.md"
        ),
        "OPEN_LICENSE_VALIDATION_REPORT_PATH": (
            report_root / "open_license_validation_summary.md"
        ),
    }
    for name, value in validator_values.items():
        monkeypatch.setattr(validator, name, value)

    monkeypatch.setattr(
        gallery,
        "OPEN_LICENSE_REVIEW_PATH",
        open_root / "open_license_review.csv",
    )
    monkeypatch.setattr(
        gallery,
        "OPEN_LICENSE_REVIEW_GALLERY_PATH",
        report_root / "open_license_review_gallery.html",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))



def test_expanded_search_plan_uses_commons_categories():
    assert OPEN_LICENSE_SEARCH_LIMIT >= 50

    expected_category_queries = {
        "alternator": "Automobile alternators",
        "brake_disc": "Brake disks",
        "brake_pad": "Brake pads",
        "shock_absorber": "Automobile shock absorbers",
        "coil_spring": "Coil spring automobile suspension",
        "headlight": "Automobile headlamps",
        "taillight": "Automobile rear lights",
        "oil_filter": "Automobile oil filters",
        "air_filter": "Automobile engine air filters",
    }

    for category, commons_category in (
        expected_category_queries.items()
    ):
        queries = OPEN_LICENSE_SEARCH_QUERIES[category]
        assert len(queries) >= 5
        assert any(
            "incategory:" in query
            and commons_category in query
            for query in queries
        )




def test_targeted_replacement_queries_are_automotive_specific():
    starter_queries = OPEN_LICENSE_SEARCH_QUERIES["starter"]
    brake_disc_queries = OPEN_LICENSE_SEARCH_QUERIES["brake_disc"]
    brake_pad_queries = OPEN_LICENSE_SEARCH_QUERIES["brake_pad"]

    assert any(
        'incategory:"Electric starter motors"' in query
        for query in starter_queries
    )
    assert any(
        'incategory:"Automobile disk brakes"' in query
        for query in brake_disc_queries
    )

    expected_brake_pad_files = {
        "Automobile brake pad.jpg",
        "Brake pads.JPG",
        "Brakepad.jpg",
        "Performance Disk Brake Pads.jpg",
    }
    for filename in expected_brake_pad_files:
        assert any(
            filename in query
            for query in brake_pad_queries
        )




def test_final_exact_replacement_queries_are_registered():
    starter_queries = OPEN_LICENSE_SEARCH_QUERIES["starter"]
    brake_disc_queries = OPEN_LICENSE_SEARCH_QUERIES["brake_disc"]
    brake_pad_queries = OPEN_LICENSE_SEARCH_QUERIES["brake_pad"]

    for filename in (
        "MOTOR STARTER.jpg",
        "Starter motor.JPG",
        "Motor starter.jpg",
    ):
        assert any(filename in query for query in starter_queries)

    for filename in (
        "Disc brakes.jpg",
        "Disc brake car.jpg",
        "Scheibenbremse(Kfz).JPG",
        "Hamulec tarczowy.jpg",
    ):
        assert any(filename in query for query in brake_disc_queries)

    for filename in (
        "Brake pads.JPG",
        "Performance Disk Brake Pads.jpg",
        "Bremsbeläge-abgefahren.JPG",
    ):
        assert any(filename in query for query in brake_pad_queries)




def test_last_two_queries_use_title_restriction():
    starter_queries = OPEN_LICENSE_SEARCH_QUERIES["starter"]
    brake_disc_queries = OPEN_LICENSE_SEARCH_QUERIES["brake_disc"]

    assert 'intitle:"Starter motor.JPG"' in starter_queries
    assert 'intitle:"Disc brake car.jpg"' in brake_disc_queries




def test_exact_final_replacement_titles_are_registered():
    starter_queries = OPEN_LICENSE_SEARCH_QUERIES["starter"]
    brake_disc_queries = OPEN_LICENSE_SEARCH_QUERIES["brake_disc"]

    assert 'intitle:"Automobile starter 2.JPG"' in starter_queries
    assert 'intitle:"Automobile starter.JPG"' in starter_queries
    assert 'intitle:"Brake Discs.jpg"' in brake_disc_queries
    assert 'intitle:"Disk brake dsc03682.jpg"' in brake_disc_queries



def test_license_allowlist():
    assert collector.license_is_allowed("Public domain")
    assert collector.license_is_allowed("CC0 1.0")
    assert collector.license_is_allowed("CC BY 4.0")
    assert collector.license_is_allowed("CC BY-SA 3.0")
    assert not collector.license_is_allowed("CC BY-NC 4.0")
    assert not collector.license_is_allowed("All rights reserved")


def test_candidate_parses_attribution_metadata():
    page = commons_page(
        page_id=101,
        title="Starter_motor.jpg",
    )

    candidate = collector.candidate_from_page(
        page,
        part_category="starter",
        search_query="automotive starter motor",
    )

    assert candidate is not None
    assert candidate["part_category"] == "starter"
    assert candidate["part_family"] == "electrical"
    assert candidate["author"] == "Example Author"
    assert candidate["license_short_name"] == "CC BY-SA 4.0"
    assert candidate["attribution_required"] == "yes"


def test_candidate_rejects_non_allowlisted_license():
    page = commons_page(
        page_id=102,
        title="Restricted.jpg",
        license_name="CC BY-NC 4.0",
        license_url=(
            "https://creativecommons.org/licenses/by-nc/4.0/"
        ),
    )

    assert (
        collector.candidate_from_page(
            page,
            part_category="starter",
            search_query="starter",
        )
        is None
    )


def test_collection_downloads_and_writes_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    configure_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        collector,
        "OPEN_LICENSE_SEARCH_QUERIES",
        {
            "starter": ("starter query",),
            "alternator": ("alternator query",),
        },
    )
    monkeypatch.setattr(
        collector,
        "OPEN_LICENSE_TARGET_PER_CATEGORY",
        1,
    )

    pages_by_query = {
        "starter query": [
            commons_page(
                page_id=201,
                title="Starter.jpg",
            )
        ],
        "alternator query": [
            commons_page(
                page_id=202,
                title="Alternator.jpg",
            )
        ],
    }

    def fake_request(params):
        query = str(params["gsrsearch"])
        return {
            "query": {
                "pages": pages_by_query[query]
            }
        }

    bytes_by_url = {
        "https://upload.wikimedia.org/thumb/201.jpg": image_bytes(
            color=(10, 20, 30)
        ),
        "https://upload.wikimedia.org/thumb/202.jpg": image_bytes(
            color=(30, 40, 50)
        ),
    }

    report = collector.collect_open_license_images(
        request_json=fake_request,
        download_bytes=lambda url: bytes_by_url[url],
        sleep=lambda seconds: None,
    )

    assert report["status"] == "PASS"
    assert report["added"] == 2
    manifest_rows = read_csv(
        collector.OPEN_LICENSE_MANIFEST_PATH
    )
    review_rows = read_csv(
        collector.OPEN_LICENSE_REVIEW_PATH
    )
    assert len(manifest_rows) == 2
    assert len(review_rows) == 2
    assert {
        row["operator_decision"]
        for row in review_rows
    } == {"pending"}
    assert collector.OPEN_LICENSE_ATTRIBUTION_PATH.is_file()




def test_collection_rolls_back_files_and_metadata_on_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    configure_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        collector,
        "OPEN_LICENSE_SEARCH_QUERIES",
        {"starter": ("starter query",)},
    )
    monkeypatch.setattr(
        collector,
        "OPEN_LICENSE_TARGET_PER_CATEGORY",
        1,
    )

    page = commons_page(
        page_id=250,
        title="Starter_rollback.jpg",
    )
    fake_payload = {"query": {"pages": [page]}}

    original_atomic_write_text = collector.atomic_write_text

    def fail_attribution(path, content):
        if path == collector.OPEN_LICENSE_ATTRIBUTION_PATH:
            raise RuntimeError("forced attribution write failure")
        return original_atomic_write_text(path, content)

    monkeypatch.setattr(
        collector,
        "atomic_write_text",
        fail_attribution,
    )

    with pytest.raises(
        RuntimeError,
        match="forced attribution",
    ):
        collector.collect_open_license_images(
            request_json=lambda params: fake_payload,
            download_bytes=lambda url: image_bytes(),
            sleep=lambda seconds: None,
        )

    assert not collector.OPEN_LICENSE_MANIFEST_PATH.exists()
    assert not collector.OPEN_LICENSE_REVIEW_PATH.exists()
    assert not collector.OPEN_LICENSE_ATTRIBUTION_PATH.exists()
    image_files = list(
        collector.OPEN_LICENSE_IMAGES_DIRECTORY.rglob("*")
    )
    assert not [path for path in image_files if path.is_file()]


def test_collection_preserves_review_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    configure_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        collector,
        "OPEN_LICENSE_SEARCH_QUERIES",
        {"starter": ("starter query",)},
    )
    monkeypatch.setattr(
        collector,
        "OPEN_LICENSE_TARGET_PER_CATEGORY",
        1,
    )

    page = commons_page(
        page_id=301,
        title="Starter.jpg",
    )
    fake_payload = {"query": {"pages": [page]}}

    collector.collect_open_license_images(
        request_json=lambda params: fake_payload,
        download_bytes=lambda url: image_bytes(),
        sleep=lambda seconds: None,
    )

    review_rows = read_csv(
        collector.OPEN_LICENSE_REVIEW_PATH
    )
    review_rows[0]["operator_decision"] = "approved"
    review_rows[0]["operator_notes"] = "Visually confirmed."
    collector.atomic_write_csv(
        collector.OPEN_LICENSE_REVIEW_PATH,
        OPEN_LICENSE_REVIEW_COLUMNS,
        review_rows,
    )

    report = collector.collect_open_license_images(
        request_json=lambda params: fake_payload,
        download_bytes=lambda url: image_bytes(),
        sleep=lambda seconds: None,
    )

    assert report["added"] == 0
    preserved = read_csv(
        collector.OPEN_LICENSE_REVIEW_PATH
    )[0]
    assert preserved["operator_decision"] == "approved"
    assert preserved["operator_notes"] == "Visually confirmed."



def test_rejected_candidate_is_replaced_without_redownloading_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    configure_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        collector,
        "OPEN_LICENSE_SEARCH_QUERIES",
        {"starter": ("starter query",)},
    )
    monkeypatch.setattr(
        collector,
        "OPEN_LICENSE_TARGET_PER_CATEGORY",
        1,
    )

    old_page = commons_page(
        page_id=351,
        title="Wrong_starter.jpg",
    )
    new_page = commons_page(
        page_id=352,
        title="Replacement_starter.jpg",
    )

    payload_by_run = [
        {"query": {"pages": [old_page]}},
        {"query": {"pages": [old_page, new_page]}},
    ]
    run_index = {"value": 0}

    def fake_request(params):
        return payload_by_run[run_index["value"]]

    bytes_by_url = {
        "https://upload.wikimedia.org/thumb/351.jpg": image_bytes(
            color=(10, 10, 10)
        ),
        "https://upload.wikimedia.org/thumb/352.jpg": image_bytes(
            color=(20, 20, 20)
        ),
    }

    first_report = collector.collect_open_license_images(
        request_json=fake_request,
        download_bytes=lambda url: bytes_by_url[url],
        sleep=lambda seconds: None,
    )
    assert first_report["added"] == 1

    review_rows = read_csv(
        collector.OPEN_LICENSE_REVIEW_PATH
    )
    review_rows[0]["operator_decision"] = "rejected"
    review_rows[0]["rejection_reason"] = "Wrong category."
    collector.atomic_write_csv(
        collector.OPEN_LICENSE_REVIEW_PATH,
        OPEN_LICENSE_REVIEW_COLUMNS,
        review_rows,
    )

    run_index["value"] = 1
    second_report = collector.collect_open_license_images(
        request_json=fake_request,
        download_bytes=lambda url: bytes_by_url[url],
        sleep=lambda seconds: None,
    )

    assert second_report["added"] == 1
    manifest_rows = read_csv(
        collector.OPEN_LICENSE_MANIFEST_PATH
    )
    assert [
        row["commons_page_id"]
        for row in manifest_rows
    ] == ["352", "351"] or [
        row["commons_page_id"]
        for row in manifest_rows
    ] == ["351", "352"]
    final_review = read_csv(
        collector.OPEN_LICENSE_REVIEW_PATH
    )
    decisions = {
        row["commons_title"]: row["operator_decision"]
        for row in final_review
    }
    assert decisions["File:Wrong_starter.jpg"] == "rejected"
    assert decisions["File:Replacement_starter.jpg"] == "pending"


def write_complete_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decision: str = "approved",
) -> tuple[Path, Path]:
    configure_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(
        validator,
        "OPEN_LICENSE_SEARCH_QUERIES",
        {"starter": ("starter",)},
    )
    monkeypatch.setattr(
        validator,
        "OPEN_LICENSE_TARGET_PER_CATEGORY",
        1,
    )

    image_path = (
        tmp_path
        / "data"
        / "external"
        / "open_license"
        / "images"
        / "starter"
        / "commons_starter_401.jpg"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_data = image_bytes()
    image_path.write_bytes(image_data)
    digest = hashlib.sha256(image_data).hexdigest()

    local_path = image_path.relative_to(tmp_path).as_posix()
    manifest_row = {
        column: ""
        for column in OPEN_LICENSE_MANIFEST_COLUMNS
    }
    manifest_row.update(
        {
            "asset_id": "commons_starter_401",
            "part_family": "electrical",
            "part_category": "starter",
            "search_query": "starter",
            "commons_page_id": "401",
            "commons_title": "File:Starter.jpg",
            "description_url": (
                "https://commons.wikimedia.org/wiki/File:Starter.jpg"
            ),
            "original_url": (
                "https://upload.wikimedia.org/401.jpg"
            ),
            "download_url": (
                "https://upload.wikimedia.org/thumb/401.jpg"
            ),
            "author": "Example Author",
            "license_short_name": "CC BY-SA 4.0",
            "license_url": (
                "https://creativecommons.org/licenses/by-sa/4.0/"
            ),
            "attribution_required": "yes",
            "usage_terms": "CC BY-SA 4.0",
            "local_path": local_path,
            "sha256": digest,
            "file_size_bytes": str(len(image_data)),
            "width": "640",
            "height": "480",
            "format": "JPEG",
            "downloaded_at_utc": "2026-07-19T00:00:00+00:00",
            "modifications": (
                "Wikimedia thumbnail resized to maximum "
                "1024 px width; otherwise unmodified."
            ),
        }
    )
    review_row = {
        column: ""
        for column in OPEN_LICENSE_REVIEW_COLUMNS
    }
    review_row.update(
        {
            "asset_id": "commons_starter_401",
            "part_family": "electrical",
            "part_category": "starter",
            "local_path": local_path,
            "commons_title": "File:Starter.jpg",
            "author": "Example Author",
            "license_short_name": "CC BY-SA 4.0",
            "license_url": (
                "https://creativecommons.org/licenses/by-sa/4.0/"
            ),
            "description_url": (
                "https://commons.wikimedia.org/wiki/File:Starter.jpg"
            ),
            "operator_decision": decision,
            "rejection_reason": (
                "Wrong category."
                if decision == "rejected"
                else ""
            ),
        }
    )

    collector.atomic_write_csv(
        validator.OPEN_LICENSE_MANIFEST_PATH,
        OPEN_LICENSE_MANIFEST_COLUMNS,
        [manifest_row],
    )
    collector.atomic_write_csv(
        validator.OPEN_LICENSE_REVIEW_PATH,
        OPEN_LICENSE_REVIEW_COLUMNS,
        [review_row],
    )
    validator.OPEN_LICENSE_ATTRIBUTION_PATH.write_text(
        "# Attribution\n",
        encoding="utf-8",
    )
    return image_path, validator.OPEN_LICENSE_REVIEW_PATH


def test_validator_reports_ready_after_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    write_complete_dataset(
        tmp_path,
        monkeypatch,
        decision="approved",
    )

    report = validator.validate_open_license_dataset()

    assert report["status"] == "PASS"
    assert report["readiness"] == "READY_FOR_EXTERNAL_DATASET"
    assert report["category_counts"]["starter"]["approved"] == 1


def test_validator_detects_hash_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    image_path, _ = write_complete_dataset(
        tmp_path,
        monkeypatch,
        decision="approved",
    )
    image_path.write_bytes(
        image_bytes(color=(250, 10, 10))
    )

    report = validator.validate_open_license_dataset()

    assert report["status"] == "FAIL"
    assert any(
        "SHA-256 differs" in error
        for error in report["errors"]
    )


def test_gallery_renders_source_and_license(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _, review_path = write_complete_dataset(
        tmp_path,
        monkeypatch,
        decision="approved",
    )

    result = gallery.build_gallery()
    html_text = Path(result["gallery"]).read_text(
        encoding="utf-8"
    )

    assert result["images"] == 1
    assert "commons_starter_401" in html_text
    assert "CC BY-SA 4.0" in html_text
    assert "Open Wikimedia Commons source" in html_text
