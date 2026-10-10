"""NB-07: deterministic seed handling.

VERIFICATION STATUS: not executed -- see ``training/__init__.py``.

Deliberately narrow, per the parent milestone's "Document what is
deterministic and what is not" instruction (Section 19) rather than
"falsely claim cross-hardware determinism": this seeds Python's
``random``, PyTorch's CPU RNG, and (if available) PyTorch's CUDA RNG.
It does NOT claim bit-identical results across different PyTorch
versions, different hardware, or different CPU thread counts --
``torch.use_deterministic_algorithms`` is intentionally not forced on
here, since several of ``YugTransformer``'s existing operations may
not have deterministic CUDA implementations, and enabling that flag
would raise at runtime for such ops rather than silently doing
nothing. A caller that needs that stronger guarantee should call
``torch.use_deterministic_algorithms(True)`` itself and handle the
resulting errors, with full knowledge of which ops in the actual model
it's running against don't support it -- this module should not make
that decision silently on the caller's behalf.
"""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int) -> None:
    """Seed Python's ``random`` and PyTorch's RNGs (CPU, and CUDA if
    available) with ``seed``.

    Does not seed NumPy (this project has no NumPy dependency anywhere
    in ``yug`` or ``training`` -- adding one merely for this would
    violate the parent milestone's "Do not add unnecessary
    dependencies" rule).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
