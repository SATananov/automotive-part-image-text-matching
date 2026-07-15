from pathlib import Path


PROJECT_DIRECTORIES = (
    "app",
    "data/raw/images",
    "data/processed",
    "models",
    "notebooks",
    "reports",
    "src",
    "tests",
)


def test_project_directories_exist() -> None:
    project_root = Path(__file__).resolve().parents[1]

    for directory in PROJECT_DIRECTORIES:
        assert (project_root / directory).is_dir()
