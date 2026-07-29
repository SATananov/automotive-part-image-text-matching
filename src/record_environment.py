from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

from src.data import RESULTS_DIR

CORE_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "pillow",
    "jupyter",
    "nbformat",
    "nbclient",
    "pytest",
    "kagglehub",
)
LOCK_JSON_PATH = RESULTS_DIR / "environment_lock.json"
LOCK_TEXT_PATH = RESULTS_DIR / "environment_lock.txt"


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in CORE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def render_lock_text(python_version: str, versions: dict[str, str]) -> str:
    lines = [f"python=={python_version}"]
    lines.extend(f"{name}=={version}" for name, version in sorted(versions.items()))
    return "\n".join(lines) + "\n"


def write_environment_lock() -> dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python_version = platform.python_version()
    versions = package_versions()
    text = render_lock_text(python_version, versions)
    LOCK_TEXT_PATH.write_text(text, encoding="utf-8", newline="\n")
    summary = {
        "python_version": python_version,
        "packages": versions,
        "text_lock_sha256": hashlib.sha256(LOCK_TEXT_PATH.read_bytes()).hexdigest(),
    }
    LOCK_JSON_PATH.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def main() -> None:
    print(json.dumps(write_environment_lock(), indent=2))


if __name__ == "__main__":
    main()
