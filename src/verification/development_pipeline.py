from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from src.project_cli import COMMANDS
from src.real_dataset_config import (
    IMAGE_MANIFEST_COLUMNS,
    PART_GROUP_COLUMNS,
    REAL_DATASET_CATEGORIES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
README_PATH = PROJECT_ROOT / "README.md"
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "real_dataset"
    / "collection_protocol.md"
)
REQUIREMENTS_LOCK_PATH = PROJECT_ROOT / "requirements-lock.txt"

REQUIRED_PROTOCOL_HEADINGS = (
    "## Purpose",
    "## Initial target",
    "## Categories",
    "## File naming",
    "## Directory layout",
    "## Image capture requirements",
    "## Annotation files",
    "## Description rules",
    "## Approval workflow",
    "## Leakage prevention",
    "## Validation checklist",
)

REQUIRED_README_COMMANDS = tuple(
    f"python -m src.project_cli {command_name}"
    for command_name in COMMANDS
)

FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")


def find_unclosed_markdown_fence(path: Path) -> str | None:
    open_fence: tuple[str, int, int] | None = None

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        match = FENCE_PATTERN.match(line)

        if match is None:
            continue

        marker = match.group(1)
        marker_character = marker[0]
        marker_length = len(marker)

        if open_fence is None:
            open_fence = (
                marker_character,
                marker_length,
                line_number,
            )
            continue

        open_character, open_length, _ = open_fence

        if (
            marker_character == open_character
            and marker_length >= open_length
        ):
            open_fence = None

    if open_fence is None:
        return None

    marker_character, marker_length, line_number = open_fence
    marker = marker_character * marker_length

    return (
        f"{path.relative_to(PROJECT_ROOT)} has an unclosed "
        f"'{marker}' fence opened on line {line_number}."
    )


def collect_markdown_fence_errors() -> list[str]:
    errors: list[str] = []

    for markdown_path in sorted(PROJECT_ROOT.rglob("*.md")):
        error = find_unclosed_markdown_fence(markdown_path)

        if error is not None:
            errors.append(error)

    return errors


def validate_requirements_lock_encoding() -> list[str]:
    errors: list[str] = []
    raw_content = REQUIREMENTS_LOCK_PATH.read_bytes()

    if b"\x00" in raw_content:
        errors.append(
            "requirements-lock.txt contains NUL bytes and is not plain UTF-8."
        )

    try:
        decoded_content = raw_content.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(
            "requirements-lock.txt cannot be decoded as UTF-8."
        )
        return errors

    if not decoded_content.strip():
        errors.append("requirements-lock.txt is empty.")

    return errors


def validate_cli_modules() -> list[str]:
    errors: list[str] = []

    for command_name, command_spec in COMMANDS.items():
        if importlib.util.find_spec(command_spec.module) is None:
            errors.append(
                f"CLI command '{command_name}' points to missing module "
                f"'{command_spec.module}'."
            )

    return errors


def validate_readme() -> list[str]:
    errors: list[str] = []
    readme_text = README_PATH.read_text(encoding="utf-8-sig")

    for command in REQUIRED_README_COMMANDS:
        if command not in readme_text:
            errors.append(
                f"README.md does not document command: {command}"
            )

    direct_execution_pattern = re.compile(
        r"^\s*python\s+src/",
        flags=re.MULTILINE,
    )

    if direct_execution_pattern.search(readme_text):
        errors.append(
            "README.md contains a runnable direct src command; use python -m."
        )

    return errors


def validate_collection_protocol() -> list[str]:
    errors: list[str] = []
    protocol_text = PROTOCOL_PATH.read_text(encoding="utf-8-sig")

    for heading in REQUIRED_PROTOCOL_HEADINGS:
        if heading not in protocol_text:
            errors.append(
                f"Collection protocol is missing heading: {heading}"
            )

    for category in REAL_DATASET_CATEGORIES:
        if f"- `{category}`" not in protocol_text:
            errors.append(
                f"Collection protocol is missing category: {category}"
            )

    for column_name in (
        *PART_GROUP_COLUMNS,
        *IMAGE_MANIFEST_COLUMNS,
    ):
        if f"`{column_name}`" not in protocol_text:
            errors.append(
                f"Collection protocol is missing CSV column: {column_name}"
            )

    return errors


def build_verification_report() -> dict[str, object]:
    checks = {
        "markdown_fences": collect_markdown_fence_errors(),
        "requirements_lock_encoding": (
            validate_requirements_lock_encoding()
        ),
        "cli_modules": validate_cli_modules(),
        "readme_commands": validate_readme(),
        "collection_protocol": validate_collection_protocol(),
    }

    errors = [
        error
        for check_errors in checks.values()
        for error in check_errors
    ]

    return {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    report = build_verification_report()

    print("Development pipeline verification")

    for check_name, check_errors in report["checks"].items():
        status = "PASS" if not check_errors else "FAIL"
        print(f"- {check_name}: {status}")

        for error in check_errors:
            print(f"  - {error}")

    print(f"Status: {report['status']}")

    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
