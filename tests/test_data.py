import json
from pathlib import Path

from src.audit import file_sha256
from src.data import DATA_DIR, PROJECT_ROOT, check_split, load_split


def test_split_sizes_and_groups() -> None:
    train = load_split("train")
    validation = load_split("validation")
    assert len(train) == 180
    assert len(validation) == 60
    assert train["part_group_id"].nunique() == 60
    assert validation["part_group_id"].nunique() == 20


def test_no_train_validation_overlap() -> None:
    train = load_split("train")
    validation = load_split("validation")
    assert check_split(train, validation) == {
        "group_overlap": 0,
        "image_id_overlap": 0,
        "image_path_overlap": 0,
    }


def test_every_group_has_the_three_labels() -> None:
    for name in ("train", "validation"):
        data = load_split(name)
        counts = data.groupby("part_group_id")["label"].nunique()
        assert counts.eq(3).all()


def test_images_exist() -> None:
    for name in ("train", "validation"):
        data = load_split(name)
        assert all((PROJECT_ROOT / path).is_file() for path in data["image_path"].unique())


def test_normal_loader_refuses_test_split() -> None:
    try:
        load_split("test")
    except ValueError as error:
        assert "train and validation" in str(error)
    else:
        raise AssertionError("The normal loader should not open the test split.")


def test_test_lock_hash_matches_the_file() -> None:
    lock = json.loads((DATA_DIR / "test_lock.json").read_text(encoding="utf-8"))
    assert lock["test_locked"] is True
    assert lock["test_evaluation_permitted"] is False
    assert lock["test_sha256"] == file_sha256(DATA_DIR / "test.csv")


def test_notebook_and_source_do_not_open_test_data() -> None:
    notebook = json.loads((PROJECT_ROOT / "project.ipynb").read_text(encoding="utf-8"))
    notebook_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    source_code = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "src").glob("*.py"))
    )
    checked_code = notebook_code + "\n" + source_code
    assert 'read_csv(ROOT / "data/test.csv")' not in checked_code
    assert "load_split(\"test\")" not in checked_code
    assert "load_split('test')" not in checked_code
