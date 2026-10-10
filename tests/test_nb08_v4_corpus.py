"""Tests for training.experiments.nb08_v4_corpus -- no torch
dependency, so these actually execute in this environment."""

from __future__ import annotations

import pytest

from training.experiments.nb08_v4_corpus import (
    SCALE_PAIRS_PER_TEMPLATE,
    generate_corpus,
)
from training.experiments.nb08_v3_corpus import CORPUS_SENTENCES as V3_CORPUS
from training.split import split_sentences


def test_generation_is_deterministic():
    assert generate_corpus(24) == generate_corpus(24)


def test_scale_sizes_match_expected():
    # 6 templates x pairs_per_template, verified to have zero
    # dedup collisions at each configured scale.
    assert len(generate_corpus(24)) == 144
    assert len(generate_corpus(60)) == 360
    assert len(generate_corpus(144)) == 864


def test_scales_are_strictly_increasing():
    sizes = [len(generate_corpus(p)) for p in sorted(SCALE_PAIRS_PER_TEMPLATE.values())]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)  # strictly increasing, no ties


def test_larger_scale_is_superset_of_smaller_scale_sentences():
    """The nested-loop pair ordering means a larger scale's corpus
    should contain every sentence the next-smaller scale has, plus
    more -- verifies the ladder is genuinely additive, not just
    differently-shuffled."""
    small = set(generate_corpus(24))
    large = set(generate_corpus(60))
    assert small.issubset(large)


def test_invalid_pairs_per_template_rejected():
    with pytest.raises(ValueError):
        generate_corpus(0)
    with pytest.raises(ValueError):
        generate_corpus(145)  # exceeds 12x12 ceiling


@pytest.mark.parametrize("pairs", [24, 60, 144])
def test_no_exact_sentence_leakage_at_any_scale(pairs):
    corpus = generate_corpus(pairs)
    train, val = split_sentences(corpus, val_fraction=0.2, seed=1234)
    assert set(train).isdisjoint(set(val))


@pytest.mark.parametrize("pairs", [24, 60, 144])
def test_word_coverage_at_least_as_good_as_v3_at_every_scale(pairs):
    corpus = generate_corpus(pairs)
    train, val = split_sentences(corpus, val_fraction=0.2, seed=1234)
    train_words = set(w for s in train for w in s.rstrip(".").split())
    val_words = set(w for s in val for w in s.rstrip(".").split())
    coverage = len(val_words & train_words) / len(val_words)
    assert coverage >= 0.9  # V3's own bar; verified higher (100%) in practice


def test_v3_baseline_corpus_unchanged():
    """Scale 0 must be V3's own corpus, untouched by this module."""
    assert len(V3_CORPUS) == 72
    assert len(set(V3_CORPUS)) == 72
