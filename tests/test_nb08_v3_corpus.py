"""Tests for training.experiments.nb08_v3_corpus -- no torch
dependency, so these actually execute in this environment."""

from __future__ import annotations

from training.experiments.nb08_v3_corpus import (
    CORPUS_SENTENCES,
    _generate_corpus,
    _TEMPLATES,
)
from training.split import split_sentences


def test_corpus_generation_is_deterministic():
    # No seed anywhere in generation -- must be identical every call.
    assert _generate_corpus() == _generate_corpus() == CORPUS_SENTENCES


def test_corpus_has_no_duplicate_sentences():
    assert len(CORPUS_SENTENCES) == len(set(CORPUS_SENTENCES))


def test_corpus_is_substantially_larger_than_v1():
    from training.experiments.nb08_corpus import CORPUS_SENTENCES as V1_CORPUS

    v1_chars = sum(len(s) for s in V1_CORPUS)
    v3_chars = sum(len(s) for s in CORPUS_SENTENCES)
    assert v3_chars > v1_chars
    assert len(CORPUS_SENTENCES) > len(V1_CORPUS)


def test_corpus_uses_all_six_templates():
    # Every template function should have produced at least one
    # sentence that survived de-duplication.
    per_template_counts = [
        sum(1 for i in range(12) if template(i) in CORPUS_SENTENCES)
        for template in _TEMPLATES
    ]
    assert all(count > 0 for count in per_template_counts)


def test_word_level_validation_coverage_is_high():
    """The specific, evidence-driven design goal of this corpus
    (see ARCHITECTURE/NB08_DATA_REGIME_EXPERIMENT_V3.md, Section 3):
    unlike V1 (43% word coverage), nearly every validation word must
    also appear in training."""
    train, val = split_sentences(CORPUS_SENTENCES, val_fraction=0.2, seed=1234)
    train_words = set(w for s in train for w in s.rstrip(".").split())
    val_words = set(w for s in val for w in s.rstrip(".").split())

    coverage = len(val_words & train_words) / len(val_words)
    assert coverage > 0.9, (
        f"word-level validation coverage was only {coverage:.1%}, "
        f"missing: {val_words - train_words}"
    )


def test_split_produces_enough_sentences_for_both_sides():
    train, val = split_sentences(CORPUS_SENTENCES, val_fraction=0.2, seed=1234)
    assert len(train) >= 30
    assert len(val) >= 5
