from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from src.data import PROJECT_ROOT

FORBIDDEN_PARTS = {".git", ".venv", ".cache", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}


def git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_archive(path: Path, expected_comment: bytes) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if archive.comment != expected_comment:
            raise AssertionError("ZIP comment does not match the exact Git commit")
        if len(names) != len(set(names)):
            raise AssertionError("Duplicate ZIP member names")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise AssertionError(f"Unsafe archive path: {name}")
            if FORBIDDEN_PARTS.intersection(pure.parts) or pure.suffix.lower() in FORBIDDEN_SUFFIXES:
                raise AssertionError(f"Forbidden generated content in archive: {name}")
        bad = archive.testzip()
        if bad is not None:
            raise AssertionError(f"Corrupt ZIP member: {bad}")
    return {"file_count": len(names), "archive_safety": "PASS", "integrity": "PASS"}


def create_checkpoint(output_dir: Path) -> dict[str, object]:
    if git("status", "--porcelain"):
        raise RuntimeError("Git working tree must be clean before creating a checkpoint")
    commit = git("rev-parse", "HEAD")
    short = git("rev-parse", "--short", "HEAD")
    commit_count = int(git("rev-list", "--count", "HEAD"))
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"automotive-part-image-text-matching_DATASET_V2_FINAL_{short}.zip"
    subprocess.run(
        ["git", "archive", "--format=zip", f"--output={archive_path}", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.comment = commit.encode("ascii")
    validation = validate_archive(archive_path, commit.encode("ascii"))
    digest = sha256(archive_path)
    summary = {
        "status": "PASS",
        "archive": str(archive_path),
        "sha256": digest,
        "commit": commit,
        "short_commit": short,
        "commit_count": commit_count,
        **validation,
    }
    sidecar = archive_path.with_suffix(".checkpoint.json")
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    archive_path.with_suffix(".sha256.txt").write_text(
        f"{digest}  {archive_path.name}\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_checkpoint(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
