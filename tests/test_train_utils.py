from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.train import grouped_bootstrap_interval, save_model_summary


def test_save_model_summary_uses_real_newlines(tmp_path) -> None:
    path = tmp_path / "summary.txt"
    save_model_summary(nn.Linear(2, 3), path)
    raw = path.read_bytes()
    assert b"Trainable parameters: 9\n" in raw
    assert b"Total parameters: 9\n" in raw
    assert b"\\n" not in raw


def test_grouped_bootstrap_interval_is_deterministic() -> None:
    true = np.array(["A", "B", "C", "A", "B", "C"])
    predicted = np.array(["A", "B", "A", "A", "C", "C"])
    groups = np.array(["image_1"] * 3 + ["image_2"] * 3)
    metric = lambda y_true, y_pred: float(np.mean(y_true == y_pred))
    first = grouped_bootstrap_interval(true, predicted, groups, metric, repeats=100, seed=7)
    second = grouped_bootstrap_interval(true, predicted, groups, metric, repeats=100, seed=7)
    assert first == second
    assert 0 <= first[0] <= first[1] <= 1


def test_training_requires_canonical_torch_version(monkeypatch) -> None:
    from src.train import CANONICAL_TORCH_VERSION, require_canonical_torch_version

    monkeypatch.setattr(torch, "__version__", f"{CANONICAL_TORCH_VERSION}+cpu")
    require_canonical_torch_version()

    monkeypatch.setattr(torch, "__version__", "2.11.0+cpu")
    import pytest

    with pytest.raises(RuntimeError, match="pins PyTorch"):
        require_canonical_torch_version()
