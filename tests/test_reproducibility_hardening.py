from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from src.project_cli import COMMANDS, build_parser, run_command
from src.verification.development_pipeline import (
    PROTOCOL_PATH,
    README_PATH,
    REQUIRED_PROTOCOL_HEADINGS,
    REQUIREMENTS_LOCK_PATH,
    build_verification_report,
    collect_markdown_fence_errors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_requirements_lock_is_plain_utf8() -> None:
    raw_content = REQUIREMENTS_LOCK_PATH.read_bytes()

    assert b"\x00" not in raw_content
    assert raw_content.decode("utf-8").strip()


def test_all_markdown_fences_are_balanced() -> None:
    assert collect_markdown_fence_errors() == []


def test_collection_protocol_contains_required_headings() -> None:
    protocol_text = PROTOCOL_PATH.read_text(encoding="utf-8")

    for heading in REQUIRED_PROTOCOL_HEADINGS:
        assert heading in protocol_text


def test_readme_documents_all_project_cli_commands() -> None:
    readme_text = README_PATH.read_text(encoding="utf-8")

    for command_name in COMMANDS:
        assert (
            f"python -m src.project_cli {command_name}"
            in readme_text
        )

    assert re.search(
        r"^\s*python\s+src/",
        readme_text,
        flags=re.MULTILINE,
    ) is None


def test_cli_command_modules_exist() -> None:
    for command_spec in COMMANDS.values():
        module_path = (
            PROJECT_ROOT
            / Path(*command_spec.module.split("."))
        ).with_suffix(".py")

        assert module_path.is_file()


def test_cli_loads_only_the_selected_command() -> None:
    imported_modules: list[str] = []
    executed: list[bool] = []

    def fake_importer(module_name: str) -> SimpleNamespace:
        imported_modules.append(module_name)
        return SimpleNamespace(
            __name__=module_name,
            main=lambda: executed.append(True),
        )

    run_command(
        command_name="validate-development-data",
        importer=fake_importer,
    )

    assert imported_modules == [
        "src.validate_development_dataset"
    ]
    assert executed == [True]


def test_cli_parser_accepts_registered_command() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        ["verify-development-pipeline"]
    )

    assert arguments.command_name == "verify-development-pipeline"


def test_cli_rejects_unknown_internal_command() -> None:
    with pytest.raises(ValueError, match="Unknown project command"):
        run_command("missing-command")


def test_step_008_2_verification_passes() -> None:
    report = build_verification_report()

    assert report["status"] == "PASS"
    assert report["errors"] == []
