"""Deterministic eager compatibility APIs and bounded iterator equivalents.

Both paths share per-document encoding, concatenation, L+1 windowing and
batching semantics. DocumentSequences supports repeatable lazy Trainer input;
it does not provide acquisition, fitting, persistent storage or shuffling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import islice

import torch

from training.errors import DatasetError


def encode_corpus(tokenizer, corpus: list[str]) -> list[int]:
    """Encode every string in ``corpus`` and concatenate the resulting
    token ids into one flat stream, in the order given.

    No document-boundary special tokens are inserted between entries
    unless ``corpus`` itself already contains them via
    ``tokenizer.encode(..., add_bos=True, add_eos=True)`` upstream --
    this function intentionally does not invent a document-separator
    convention the tokenizer doesn't already define (per the parent
    milestone's "Do not invent special tokens" rule).

    Raises:
        DatasetError: ``corpus`` is empty, or encodes to zero tokens.
    """
    return list(iter_encoded_tokens(tokenizer, corpus))


def make_sequences(token_ids: list[int], sequence_length: int) -> list[list[int]]:
    """Split a flat token-id stream into non-overlapping chunks of
    length ``sequence_length + 1`` (the extra token gives each chunk
    both its input positions ``[0:-1]`` and target positions ``[1:]``
    once split by :func:`sequences_to_batch`).

    A final partial chunk that cannot reach ``sequence_length + 1``
    tokens is dropped rather than padded -- there is no padding
    convention defined anywhere else in the tokenizer/model contract
    for this milestone to invent one for (per "Do not invent ...
    padding" implied by the parent prompt's silence on it -- padding
    is listed only as a conditional "if introduced" concern for the
    loss function, not a requirement here).

    Raises:
        DatasetError: fewer than ``sequence_length + 1`` tokens are
            available, so zero sequences would be produced.
    """
    return list(iter_sequences(token_ids, sequence_length))


def sequences_to_batch(sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of equal-length ``sequence_length + 1`` id lists
    into ``(input_ids, target_ids)`` tensors of shape
    ``(batch_size, sequence_length)`` each.

    ``input_ids[i] = sequences[i][:-1]``, ``target_ids[i] = sequences[i][1:]``
    -- the standard causal-LM shift, matching the ``t1 from t0, t2 from
    t1, ...`` contract from the parent milestone's loss section
    verbatim. Both tensors use ``dtype=torch.long`` (required by
    :class:`torch.nn.Embedding` / :func:`torch.nn.functional.cross_entropy`).
    """
    inputs = torch.tensor([s[:-1] for s in sequences], dtype=torch.long)
    targets = torch.tensor([s[1:] for s in sequences], dtype=torch.long)
    return inputs, targets


def iter_batches(
    sequences: list[list[int]], batch_size: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Group ``sequences`` (already ``sequence_length + 1``-length
    chunks from :func:`make_sequences`) into batches of ``batch_size``,
    in the given order -- deterministic, no shuffling, per the "no
    giant dataset framework" scope limit. A caller wanting shuffled
    epochs should shuffle ``sequences`` itself, deterministically,
    before calling this (e.g. ``random.Random(seed).shuffle(sequences)``)
    -- this function does not own randomness policy.

    The final, possibly-short batch is included as-is (not dropped and
    not padded) -- this is a batching convenience, not a modeling
    decision, so there is nothing here for a caller relying on a fixed
    batch size to be surprised by beyond ordinary short-last-batch
    handling.
    """
    return list(iter_lazy_batches(sequences, batch_size))


def iter_encoded_tokens(tokenizer, corpus: Iterable[str]) -> Iterator[int]:
    """Encode one whole document at a time; no separators or cross-doc merges.

    Memory is bounded by the largest encoded document, not a fixed byte limit.
    Errors occur on iteration. Never split a document to reduce BPE memory.
    """
    documents = tokens = False
    for text in corpus:
        documents = True
        ids = tokenizer.encode(text)
        if ids:
            tokens = True
        yield from ids
        del ids
    if not documents:
        raise DatasetError("corpus must contain at least one string")
    if not tokens:
        raise DatasetError("corpus encoded to zero tokens")


def iter_sequences(token_ids: Iterable[int], sequence_length: int) -> Iterator[list[int]]:
    """Legacy non-overlapping L+1 windows, including cross-document windows."""
    if type(sequence_length) is not int or sequence_length < 1:
        raise DatasetError("sequence_length must be a positive integer")
    stream = iter(token_ids)
    size = sequence_length + 1
    produced = False
    while True:
        window = list(islice(stream, size))
        if len(window) != size:
            if not produced:
                raise DatasetError(
                    f"token stream has {len(window)} tokens, need at least "
                    f"{size} (sequence_length + 1) to build one training sequence"
                )
            return
        produced = True
        yield window


def iter_lazy_batches(sequences: Iterable[list[int]], batch_size: int):
    """At most one batch of sequence lists; preserve order, dtype and short tail."""
    if type(batch_size) is not int or batch_size < 1:
        raise DatasetError("batch_size must be a positive integer")
    stream = iter(sequences)
    while True:
        group = list(islice(stream, batch_size))
        if not group:
            return
        yield sequences_to_batch(group)


class DocumentSequences:
    """Repeatable lazy sequences from an already ordered/split document factory.

    The factory must return a fresh iterator over identical documents each time;
    the tokenizer and source must remain unchanged. No splitting, fitting,
    caching or RNG use occurs here. Each pass re-encodes the documents. An
    iterator captured by a factory is NOT repeatable and is caller misuse.
    Retained source lists still consume memory; a streaming source need not.
    """

    def __init__(self, tokenizer, documents: Callable[[], Iterable[str]], sequence_length: int):
        if not callable(documents):
            raise DatasetError("documents must be a repeatable document factory")
        if type(sequence_length) is not int or sequence_length < 1:
            raise DatasetError("sequence_length must be a positive integer")
        self.tokenizer = tokenizer
        self.documents = documents
        self.sequence_length = sequence_length

    def __iter__(self):
        return iter_sequences(iter_encoded_tokens(self.tokenizer, self.documents()),
                              self.sequence_length)
