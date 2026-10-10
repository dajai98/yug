"""Tests for training.dataset (NB-07).

VERIFICATION STATUS: written without a PyTorch installation available
in the development sandbox. NOT executed by the assistant. See
training/__init__.py for the full evidence trail.

All tests are skipped automatically (via the module-level
pytest.importorskip("torch") below) in any environment without
PyTorch, matching the convention already established in
tests/test_native_model.py and tests/test_native_generation.py.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.dataset import (  # noqa: E402
    encode_corpus,
    iter_batches,
    make_sequences,
    sequences_to_batch,
)
from training.errors import DatasetError  # noqa: E402


class _StubTokenizer:
    """A minimal, deterministic stand-in for
    yug.tokenizer.ByteLevelBpeTokenizer -- avoids coupling these
    dataset-pipeline tests to a real trained tokenizer, exactly as
    tests/test_native_model.py avoids coupling model tests to a real
    tokenizer via its own _PLACEHOLDER_IDENTITY convention."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 50 for c in text]


def test_encode_corpus_concatenates_in_order():
    tok = _StubTokenizer()
    ids = encode_corpus(tok, ["ab", "cd"])
    assert ids == tok.encode("ab") + tok.encode("cd")


def test_encode_corpus_rejects_empty_corpus():
    with pytest.raises(DatasetError):
        encode_corpus(_StubTokenizer(), [])


def test_make_sequences_produces_expected_chunk_count():
    token_ids = list(range(20))
    sequences = make_sequences(token_ids, sequence_length=4)
    # chunk_len = 5; 20 // 5 = 4 whole chunks, none dropped
    assert len(sequences) == 4
    assert all(len(s) == 5 for s in sequences)


def test_make_sequences_drops_incomplete_final_chunk():
    token_ids = list(range(12))  # chunk_len=5 -> 2 full chunks, 2 left over
    sequences = make_sequences(token_ids, sequence_length=4)
    assert len(sequences) == 2


def test_make_sequences_rejects_insufficient_tokens():
    with pytest.raises(DatasetError):
        make_sequences([1, 2, 3], sequence_length=10)


def test_sequences_to_batch_shapes_and_shift():
    sequences = [[1, 2, 3, 4], [5, 6, 7, 8]]
    inputs, targets = sequences_to_batch(sequences)
    assert inputs.shape == (2, 3)
    assert targets.shape == (2, 3)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    assert inputs[0].tolist() == [1, 2, 3]
    assert targets[0].tolist() == [2, 3, 4]


def test_iter_batches_groups_deterministically_no_shuffle():
    sequences = [[i, i + 1] for i in range(6)]  # sequence_length=1 shape
    batches = iter_batches(sequences, batch_size=2)
    assert len(batches) == 3
    for inputs, targets in batches:
        assert inputs.shape[0] == 2


def test_iter_batches_keeps_short_final_batch():
    sequences = [[i, i + 1] for i in range(5)]
    batches = iter_batches(sequences, batch_size=2)
    assert len(batches) == 3
    assert batches[-1][0].shape[0] == 1
