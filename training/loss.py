"""NB-07: next-token causal language-model loss.

Training and evaluation share flattened position/class layout with explicit
ignore_index=-100; evaluation retains FP64 NLL accumulation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from training.errors import TrainingError


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy loss.

    Args:
        logits: ``(batch_size, sequence_length, vocab_size)`` --
            exactly :meth:`yug.native.model.YugTransformer.forward`'s
            output shape and dtype (no reshaping/casting is done to
            the caller's model output beyond what
            :func:`torch.nn.functional.cross_entropy` itself requires
            below).
        targets: ``(batch_size, sequence_length)`` of ``torch.long``
            token ids -- exactly :func:`training.dataset.sequences_to_batch`'s
            ``target_ids`` output. The shift from raw token ids to
            "target for position i is token i+1" already happened in
            ``training.dataset``; this function does not shift
            anything itself, it only computes cross-entropy over
            already-aligned ``(logits, targets)`` pairs.

    Returns:
        A scalar tensor: mean cross-entropy over every
        ``(batch, position)`` pair, excluding PyTorch's default ignored
        label (-100). The current dataset does not introduce padding.

    Flatten positions to (B*T, V) before cross entropy. This keeps the
    valid-target mean while avoiding the K-dimensional CUDA loss path.
    No dtype conversion, new shift, or padding convention is introduced.
    """
    flat_logits, flat_targets = _flatten_positions(logits, targets)
    return F.cross_entropy(flat_logits, flat_targets, reduction="mean", ignore_index=-100)


def _flatten_positions(logits: torch.Tensor, targets: torch.Tensor):
    # Do not silently accept mismatched B/T shapes with the same element count.
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise RuntimeError("Expected aligned logits (B,T,V) and targets (B,T)")
    return logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)


def evaluation_nll_count(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, int]:
    """Evaluation-only NLL numerator (FP64 accumulation) and exact denominator.

    Cross entropy validates non-ignored class IDs and tensor shapes, including
    for an all-ignored batch. No padding or target-shift semantics are added.
    """
    flat_logits, flat_targets = _flatten_positions(logits, targets)
    losses = F.cross_entropy(
        flat_logits, flat_targets, reduction="none", ignore_index=-100,
    )
    valid = flat_targets != -100
    count = int(valid.sum().item())
    selected = losses[valid].detach().to(torch.float64)
    if not bool(torch.isfinite(selected).all()):
        raise TrainingError("Non-finite evaluation NLL for a valid target")
    return selected.sum().item(), count
