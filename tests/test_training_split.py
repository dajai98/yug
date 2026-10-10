"""Tests for training.split -- like training.config, this has no torch
dependency, so these tests actually execute in this environment."""

from __future__ import annotations

import pytest

from training.split import split_sentences
from training.errors import DatasetError


def test_split_is_deterministic_for_same_seed():
    sentences = [f"sentence {i}" for i in range(20)]
    train1, val1 = split_sentences(sentences, val_fraction=0.2, seed=42)
    train2, val2 = split_sentences(sentences, val_fraction=0.2, seed=42)
    assert train1 == train2
    assert val1 == val2


def test_split_differs_for_different_seeds():
    sentences = [f"sentence {i}" for i in range(20)]
    train1, val1 = split_sentences(sentences, val_fraction=0.2, seed=1)
    train2, val2 = split_sentences(sentences, val_fraction=0.2, seed=2)
    assert (train1, val1) != (train2, val2)


def test_train_and_val_are_disjoint_and_cover_everything():
    sentences = [f"sentence {i}" for i in range(40)]
    train, val = split_sentences(sentences, val_fraction=0.2, seed=1234)
    assert set(train).isdisjoint(set(val))
    assert sorted(train + val) == sorted(sentences)


def test_val_fraction_produces_expected_val_size():
    sentences = [f"sentence {i}" for i in range(40)]
    train, val = split_sentences(sentences, val_fraction=0.2, seed=1234)
    assert len(val) == 8
    assert len(train) == 32


def test_val_size_never_zero_even_for_small_fraction():
    sentences = [f"sentence {i}" for i in range(10)]
    train, val = split_sentences(sentences, val_fraction=0.01, seed=1)
    assert len(val) >= 1


def test_empty_sentences_rejected():
    with pytest.raises(DatasetError):
        split_sentences([], val_fraction=0.2, seed=1)


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.1])
def test_invalid_val_fraction_rejected(bad_fraction):
    sentences = ["a", "b", "c"]
    with pytest.raises(DatasetError):
        split_sentences(sentences, val_fraction=bad_fraction, seed=1)


def test_split_does_not_mutate_global_random_state():
    import random

    random.seed(999)
    state_before = random.getstate()

    split_sentences([f"s{i}" for i in range(10)], val_fraction=0.2, seed=42)

    assert random.getstate() == state_before
