from __future__ import annotations

import ast
from typing import Callable

from src.open_license_dataset_config import (
    COMMONS_API_URL,
    OPEN_LICENSE_MANIFEST_COLUMNS,
    OPEN_LICENSE_SEARCH_LIMIT,
    OPEN_LICENSE_SEARCH_QUERIES,
    OPEN_LICENSE_REVIEW_COLUMNS,
    OPEN_LICENSE_RUNTIME_DIRECTORY,
)
from src.project_cli import COMMANDS
from src.real_dataset_config import PROJECT_ROOT
from src.validate_open_license_dataset import (
    validate_open_license_dataset,
)

REQUIRED_FILES = (
    PROJECT_ROOT / "src" / "open_license_dataset_config.py",
    PROJECT_ROOT / "src" / "collect_open_license_images.py",
    PROJECT_ROOT / "src" / "validate_open_license_dataset.py",
    PROJECT_ROOT / "src" / "build_open_license_review_gallery.py",
    PROJECT_ROOT / "src" / "verify_step_010_1.py",
    PROJECT_ROOT
    / "tests"
    / "test_open_license_image_collection.py",
    PROJECT_ROOT
    / "reports"
    / "external_dataset"
    / "step_010_1_open_license_collection.md",
)

EXPECTED_COMMANDS = {
    "collect-open-license-images": (
        "src.collect_open_license_images"
    ),
    "validate-open-license-images": (
        "src.validate_open_license_dataset"
    ),
    "build-open-license-review-gallery": (
        "src.build_open_license_review_gallery"
    ),
    "verify-step-010-1": "src.verify_step_010_1",
}

SAFE_READINESS = {
    "AWAITING_COLLECTION",
    "COLLECTION_INCOMPLETE",
    "MANUAL_REVIEW_REQUIRED",
    "READY_FOR_EXTERNAL_DATASET",
    "REPLACEMENT_IMAGES_REQUIRED",
    "VALIDATION_BLOCKED",
}


def source_contains(source: str, marker: str) -> bool:
    if marker in source:
        return True
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        marker in node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
    )


def check_structure() -> list[str]:
    return [
        f"Missing file: {path.relative_to(PROJECT_ROOT)}"
        for path in REQUIRED_FILES
        if not path.is_file()
    ]


def check_cli_and_documentation() -> list[str]:
    errors: list[str] = []
    for command, module_name in EXPECTED_COMMANDS.items():
        command_spec = COMMANDS.get(command)
        if command_spec is None:
            errors.append(f"Missing CLI command: {command}.")
        elif command_spec.module != module_name:
            errors.append(
                f"CLI command {command} uses "
                f"{command_spec.module}; expected {module_name}."
            )

    readme = (PROJECT_ROOT / "README.md").read_text(
        encoding="utf-8-sig"
    )
    protocol = (
        PROJECT_ROOT
        / "reports"
        / "real_dataset"
        / "collection_protocol.md"
    ).read_text(encoding="utf-8-sig")

    for command in EXPECTED_COMMANDS:
        command_text = f"python -m src.project_cli {command}"
        if command_text not in readme:
            errors.append(
                f"README is missing: {command_text}."
            )
        if command_text not in protocol:
            errors.append(
                f"Collection protocol is missing: {command_text}."
            )

    return errors


def check_schemas_and_boundaries() -> list[str]:
    errors: list[str] = []
    if len(OPEN_LICENSE_MANIFEST_COLUMNS) != len(
        set(OPEN_LICENSE_MANIFEST_COLUMNS)
    ):
        errors.append(
            "Open-license manifest columns are not unique."
        )
    if len(OPEN_LICENSE_REVIEW_COLUMNS) != len(
        set(OPEN_LICENSE_REVIEW_COLUMNS)
    ):
        errors.append(
            "Open-license review columns are not unique."
        )

    try:
        OPEN_LICENSE_RUNTIME_DIRECTORY.relative_to(
            PROJECT_ROOT
            / "data"
            / "external"
            / "open_license"
        )
    except ValueError:
        errors.append(
            "Open-license runtime directory is outside its "
            "dedicated dataset boundary."
        )

    gitignore = (PROJECT_ROOT / ".gitignore").read_text(
        encoding="utf-8-sig"
    )
    required_ignore_markers = (
        "data/external/open_license/runtime/",
        "data/external/open_license/.collect_tmp_*/",
    )
    for marker in required_ignore_markers:
        if marker not in gitignore:
            errors.append(
                f".gitignore is missing: {marker}."
            )

    if COMMONS_API_URL != (
        "https://commons.wikimedia.org/w/api.php"
    ):
        errors.append(
            "The collector does not use the Wikimedia Commons API."
        )

    if OPEN_LICENSE_SEARCH_LIMIT < 50:
        errors.append(
            "The Commons search limit is too small for the "
            "expanded collection pass."
        )

    category_query_markers = {
        "starter": "Electric starter motors",
        "brake_disc": "Automobile disk brakes",
        "coil_spring": "Coil spring automobile suspension",
        "headlight": "Automobile headlamps",
        "taillight": "Automobile rear lights",
        "oil_filter": "Automobile oil filters",
        "air_filter": "Automobile engine air filters",
    }
    for category, marker in category_query_markers.items():
        queries = OPEN_LICENSE_SEARCH_QUERIES.get(category, ())
        if not any(
            "incategory:" in query and marker in query
            for query in queries
        ):
            errors.append(
                f"Expanded category query is missing for {category}: "
                f"{marker}."
            )

    brake_pad_queries = OPEN_LICENSE_SEARCH_QUERIES.get(
        "brake_pad",
        (),
    )
    for filename in (
        "Automobile brake pad.jpg",
        "Brake pads.JPG",
        "Brakepad.jpg",
        "Performance Disk Brake Pads.jpg",
    ):
        if not any(
            filename in query
            for query in brake_pad_queries
        ):
            errors.append(
                f"Targeted brake-pad query is missing: {filename}."
            )

    final_exact_queries = {
        "starter": (
            "MOTOR STARTER.jpg",
            "Starter motor.JPG",
            "Motor starter.jpg",
        ),
        "brake_disc": (
            "Disc brakes.jpg",
            "Disc brake car.jpg",
            "Scheibenbremse(Kfz).JPG",
            "Hamulec tarczowy.jpg",
        ),
        "brake_pad": (
            "Brake pads.JPG",
            "Performance Disk Brake Pads.jpg",
            "Bremsbeläge-abgefahren.JPG",
        ),
    }
    for category, filenames in final_exact_queries.items():
        queries = OPEN_LICENSE_SEARCH_QUERIES.get(category, ())
        for filename in filenames:
            if not any(filename in query for query in queries):
                errors.append(
                    f"Final exact query is missing for {category}: "
                    f"{filename}."
                )

    exact_last_queries = {
        "starter": 'intitle:"Starter motor.JPG"',
        "brake_disc": 'intitle:"Disc brake car.jpg"',
    }
    for category, required_query in exact_last_queries.items():
        queries = OPEN_LICENSE_SEARCH_QUERIES.get(category, ())
        if required_query not in queries:
            errors.append(
                f"Last exact query is missing for {category}: "
                f"{required_query}."
            )

    final_replacement_titles = {
        "starter": (
            'intitle:"Automobile starter 2.JPG"',
            'intitle:"Automobile starter.JPG"',
        ),
        "brake_disc": (
            'intitle:"Brake Discs.jpg"',
            'intitle:"Disk brake dsc03682.jpg"',
        ),
    }
    for category, required_queries in final_replacement_titles.items():
        queries = OPEN_LICENSE_SEARCH_QUERIES.get(category, ())
        for required_query in required_queries:
            if required_query not in queries:
                errors.append(
                    f"Exact final replacement query is missing for "
                    f"{category}: {required_query}."
                )

    return errors


def check_collection_safeguards() -> list[str]:
    errors: list[str] = []
    collector = (
        PROJECT_ROOT
        / "src"
        / "collect_open_license_images.py"
    ).read_text(encoding="utf-8-sig")
    validator = (
        PROJECT_ROOT
        / "src"
        / "validate_open_license_dataset.py"
    ).read_text(encoding="utf-8-sig")

    collector_markers = (
        "LicenseShortName",
        "LicenseUrl",
        "AttributionRequired",
        "license_is_allowed",
        "Wikimedia thumbnail resized",
        "Refusing to overwrite",
        "restore_files",
        ".collect_tmp_",
        "operator_decision",
        'decision != "rejected"',
        "pending",
    )
    for marker in collector_markers:
        if not source_contains(collector, marker):
            errors.append(
                f"Collector safeguard marker is missing: {marker}."
            )

    validator_markers = (
        "SHA-256 differs from the manifest",
        "duplicate image content",
        "rejected row requires a reason",
        "READY_FOR_EXTERNAL_DATASET",
        "REPLACEMENT_IMAGES_REQUIRED",
    )
    for marker in validator_markers:
        if not source_contains(validator, marker):
            errors.append(
                f"Validator safeguard marker is missing: {marker}."
            )

    return errors


def check_current_state() -> list[str]:
    try:
        report = validate_open_license_dataset()
    except Exception as error:
        return [
            f"Current open-license validation failed: {error}."
        ]

    errors: list[str] = []
    readiness = report.get("readiness")
    if readiness not in SAFE_READINESS:
        errors.append(
            f"Unexpected Step 010.1 readiness: {readiness}."
        )
    if (
        readiness == "READY_FOR_EXTERNAL_DATASET"
        and report.get("status") != "PASS"
    ):
        errors.append(
            "READY_FOR_EXTERNAL_DATASET must have PASS status."
        )
    return errors


def run_check(
    name: str,
    callback: Callable[[], list[str]],
) -> tuple[str, list[str]]:
    try:
        return name, callback()
    except Exception as error:
        return name, [f"Unexpected verifier error: {error}."]


def main() -> None:
    checks = [
        run_check("structure", check_structure),
        run_check(
            "cli_and_documentation",
            check_cli_and_documentation,
        ),
        run_check(
            "schemas_and_boundaries",
            check_schemas_and_boundaries,
        ),
        run_check(
            "collection_and_license_safeguards",
            check_collection_safeguards,
        ),
        run_check("current_state", check_current_state),
    ]

    failed = False
    print("Step 010.1 verification")
    for name, errors in checks:
        status = "PASS" if not errors else "FAIL"
        print(f"- {name}: {status}")
        for error in errors:
            print(f"  - {error}")
        failed = failed or bool(errors)

    print(f"Status: {'FAIL' if failed else 'PASS'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
