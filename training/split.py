"""NB-08: deterministic train/validation split at the sentence level.

Deliberately its own module, not part of ``training/dataset.py``: this
function has no PyTorch dependency (only Python's ``random``), and
keeping it separate from ``dataset.py`` (which imports ``torch`` at
module level) means it can actually be imported and tested in any
environment, mirroring the same reasoning already applied to
``training/config.py`` and ``training/errors.py``.

Splitting happens at the SENTENCE level, before tokenization, not at
the flat-token-stream level -- two reasons:

1. A tokenizer trained on the full corpus (train + validation text
   together) would let its learned merge vocabulary "see" validation
   text before the model ever gets evaluated on it -- a real, if
   subtle, form of leakage. Splitting sentences first, and training
   the tokenizer ONLY on the training sentences (see
   ``training/experiments/nb08_experiment.py``), avoids this: any
   byte-pair patterns unique to validation sentences are genuinely
   unseen by the tokenizer, making validation loss a more honest
   generalization signal.
2. Splitting an already-tokenized flat stream at an arbitrary token
   index risks cutting a sentence in half across the train/val
   boundary, silently leaking a is-this-sentence-continuing signal
   into whichever split got the other half. Splitting whole sentences
   avoids this entirely.
"""

from __future__ import annotations

import random

from training.errors import DatasetError


def split_sentences(
    sentences: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministically shuffle ``sentences`` (using a
    ``random.Random(seed)`` instance local to this call -- this does
    NOT touch the global ``random`` module state, so it cannot
    interfere with :func:`training.determinism.seed_everything`'s
    seeding of anything else) and split off ``val_fraction`` of them
    (rounded down, at least 1) as the validation set. The remainder is
    the training set, in shuffled order.

    Raises:
        DatasetError: ``sentences`` is empty, ``val_fraction`` is not
            in ``(0, 1)``, or the split would leave zero training
            sentences.
    """
    if not sentences:
        raise DatasetError("sentences must contain at least one entry")
    if not (0 < val_fraction < 1):
        raise DatasetError(f"val_fraction must be in (0, 1), got {val_fraction}")

    shuffled = sentences.copy()
    random.Random(seed).shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_fraction))
    if n_val >= len(shuffled):
        raise DatasetError(
            f"val_fraction={val_fraction} on {len(shuffled)} sentences would "
            f"leave zero training sentences"
        )

    val_sentences = shuffled[:n_val]
    train_sentences = shuffled[n_val:]
    return train_sentences, val_sentences
