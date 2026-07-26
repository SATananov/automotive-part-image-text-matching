from pathlib import Path

from src.data import PROJECT_ROOT, check_split, load_split


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
