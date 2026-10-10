"""NB-08: baselines computable without PyTorch.

Both baselines here operate on plain token-id lists (from
``training.dataset.encode_corpus`` or, since neither of these
functions needs a tensor, directly on ``tokenizer.encode()`` output) --
neither imports ``torch``, so unlike the rest of this experiment,
these ARE genuinely testable and were genuinely run in the sandbox
that wrote this milestone. See
``ARCHITECTURE/NB08_CONTROLLED_LM_EXPERIMENT.md`` for the real numbers
this produced against the actual NB-08 corpus.
"""

from __future__ import annotations

import math
from collections import Counter

from training.errors import DatasetError


def uniform_baseline_loss(vocab_size: int) -> float:
    """The expected next-token cross-entropy of a model that outputs a
    uniform distribution over the vocabulary, regardless of input --
    the textbook "worst reasonable baseline". Equal to ``ln(vocab_size)``.
    """
    if vocab_size < 1:
        raise DatasetError(f"vocab_size must be >= 1, got {vocab_size}")
    return math.log(vocab_size)


def unigram_baseline_loss(train_token_ids: list[int], eval_token_ids: list[int]) -> float:
    """The cross-entropy of a fixed unigram model (next-token
    probability = that token's overall frequency in ``train_token_ids``,
    ignoring all context) evaluated on ``eval_token_ids``.

    This is a meaningfully stronger baseline than
    :func:`uniform_baseline_loss` for real (non-uniform-frequency) text
    -- a trained model's validation loss should beat THIS, not just
    beat the uniform baseline, to be worth calling a genuine learning
    signal (per this milestone's "comparison against baseline where
    appropriate" acceptance criterion).

    Uses add-one (Laplace) smoothing over the observed training
    vocabulary so a token that appears in ``eval_token_ids`` but never
    in ``train_token_ids`` gets a finite (not infinite) loss
    contribution, rather than this function raising or returning
    ``inf`` -- consistent with real language having some genuinely
    novel tokens at evaluation time.

    Raises:
        DatasetError: either argument is empty.
    """
    if not train_token_ids:
        raise DatasetError("train_token_ids must not be empty")
    if not eval_token_ids:
        raise DatasetError("eval_token_ids must not be empty")

    counts = Counter(train_token_ids)
    vocab_seen = len(counts)
    total = len(train_token_ids)
    # Laplace smoothing: P(token) = (count + 1) / (total + vocab_seen)
    denom = total + vocab_seen

    total_neg_log_prob = 0.0
    for token_id in eval_token_ids:
        prob = (counts.get(token_id, 0) + 1) / denom
        total_neg_log_prob += -math.log(prob)

    return total_neg_log_prob / len(eval_token_ids)
