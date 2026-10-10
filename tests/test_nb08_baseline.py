"""Tests for training.experiments.nb08_baseline -- no torch dependency,
these actually execute in this environment."""

from __future__ import annotations

import math

import pytest

from training.errors import DatasetError
from training.experiments.nb08_baseline import (
    unigram_baseline_loss,
    uniform_baseline_loss,
)


def test_uniform_baseline_matches_known_formula():
    result = uniform_baseline_loss(384)
    expected = math.log(384)
    assert abs(result - expected) < 1e-9


def test_uniform_baseline_rejects_invalid_vocab_size():
    with pytest.raises(DatasetError):
        uniform_baseline_loss(0)


def test_unigram_baseline_lower_than_uniform_for_skewed_distribution():
    # Token 0 dominates -- a unigram model should beat uniform-random
    # guessing on data this skewed.
    train_ids = [0] * 90 + [1] * 5 + [2] * 5
    eval_ids = [0] * 8 + [1, 2]

    uniform = uniform_baseline_loss(vocab_size=3)
    unigram = unigram_baseline_loss(train_ids, eval_ids)

    assert unigram < uniform


def test_unigram_baseline_handles_unseen_eval_token():
    train_ids = [0, 0, 0, 1, 1]
    eval_ids = [0, 1, 99]  # 99 never appears in training

    # Must not raise, must return a finite number.
    result = unigram_baseline_loss(train_ids, eval_ids)
    assert math.isfinite(result)


def test_unigram_baseline_rejects_empty_inputs():
    with pytest.raises(DatasetError):
        unigram_baseline_loss([], [1, 2])
    with pytest.raises(DatasetError):
        unigram_baseline_loss([1, 2], [])
