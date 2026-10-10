"""Tests for training.config -- the one training/ module with no torch
dependency, so unlike the rest of training/'s test files, these
actually execute in this environment (no pytest.importorskip needed).
"""

from __future__ import annotations

import pytest

from training.config import TrainingConfig
from training.errors import TrainingConfigValidationError


def _valid_kwargs(**overrides) -> dict:
    kwargs = dict(
        seed=0,
        learning_rate=1e-3,
        batch_size=2,
        sequence_length=8,
        max_steps=10,
        eval_interval=5,
        checkpoint_interval=5,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_config_constructs():
    config = TrainingConfig(**_valid_kwargs())
    assert config.seed == 0
    assert config.grad_clip_norm is None
    assert config.weight_decay == 0.0
    assert config.device == "cpu"


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("batch_size", 0),
        ("sequence_length", 0),
        ("max_steps", 0),
        ("eval_interval", 0),
        ("checkpoint_interval", 0),
        ("learning_rate", 0.0),
        ("learning_rate", -1e-3),
        ("weight_decay", -0.1),
        ("grad_clip_norm", 0.0),
        ("grad_clip_norm", -1.0),
    ],
)
def test_invalid_field_rejected(field, bad_value):
    with pytest.raises(TrainingConfigValidationError):
        TrainingConfig(**_valid_kwargs(**{field: bad_value}))


def test_grad_clip_norm_none_is_valid():
    config = TrainingConfig(**_valid_kwargs(grad_clip_norm=None))
    assert config.grad_clip_norm is None


def test_grad_clip_norm_positive_is_valid():
    config = TrainingConfig(**_valid_kwargs(grad_clip_norm=1.0))
    assert config.grad_clip_norm == 1.0


def test_config_is_frozen():
    config = TrainingConfig(**_valid_kwargs())
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        config.seed = 1
