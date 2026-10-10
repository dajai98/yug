"""NB-06: deterministic greedy autoregressive generation over token ids.

Builds directly on :class:`yug.native.model.YugTransformer`'s existing
``forward(input_ids) -> logits`` contract (NB-04/NB-05) -- no changes to
that module, no tokenizer coupling, no text decoding. Per the NB-06
roadmap entry ("deterministic first, sampling later if justified") and
Architectural Decision #27, only greedy (argmax) selection is
implemented here. Temperature/top-k/top-p/beam-search sampling,
KV-cache optimization, and any tokenizer/``BrainInterface``/adapter
integration are explicitly out of scope for this milestone.
"""

from __future__ import annotations

import torch

from yug.native.errors import NativeError
from yug.native.model import YugTransformer


@torch.no_grad()
def generate(
    model: YugTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    stop_token_id: int | None = None,
) -> torch.Tensor:
    """Deterministic greedy autoregressive generation.

    Repeatedly calls ``model(input_ids)``, selects the highest-probability
    next token at each step via ``argmax`` over the vocabulary dimension
    (never sampled), and appends it to the running sequence.

    Args:
        model: a constructed :class:`YugTransformer`. Its ``training``
            flag is saved before generation and restored afterward;
            generation itself always runs with the model in ``eval()``
            mode (dropout disabled), because greedy decoding is only
            actually deterministic if dropout is off regardless of the
            mode the caller left the model in.
        input_ids: ``[batch, seq_len]`` starting token ids. Validated by
            :meth:`YugTransformer.forward` on every call it makes (2D
            shape, ``context_length``, vocab range) -- this function
            does not duplicate that validation. Note: if
            ``max_new_tokens == 0``, ``forward`` is never called, so
            that validation does not run and ``input_ids`` is returned
            unchanged as-is (see below).
        max_new_tokens: number of tokens to generate. Must be a
            non-negative ``int``. ``0`` returns ``input_ids`` unchanged.
        stop_token_id: optional caller-supplied integer. This is an
            opaque id -- generation does not assume it corresponds to
            any particular tokenizer's EOS convention, and this module
            never imports ``yug.tokenizer``. For a batch (``batch >
            1``), the stop condition is checked across the whole batch
            at each step: generation halts for *all* sequences as soon
            as *any* sequence in the batch produces ``stop_token_id`` at
            that step. No per-sequence early stopping or padding is
            implemented -- that would be additional scope beyond NB-06.

    Returns:
        A ``[batch, seq_len + n]`` tensor of token ids on the same
        device and with the same dtype as ``input_ids``, where
        ``n <= max_new_tokens`` (fewer if ``stop_token_id`` fires
        before the bound is reached).

    Raises:
        NativeError: if ``max_new_tokens`` is not a non-negative
            ``int``, if ``stop_token_id`` is neither ``None`` nor an
            ``int``, or if ``model.forward`` rejects ``input_ids``
            (that error propagates unchanged -- it is not re-wrapped).
    """
    if not isinstance(max_new_tokens, int):
        raise NativeError(f"max_new_tokens must be an int, got {type(max_new_tokens).__name__}")
    if max_new_tokens < 0:
        raise NativeError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if stop_token_id is not None and not isinstance(stop_token_id, int):
        raise NativeError(f"stop_token_id must be an int or None, got {type(stop_token_id).__name__}")

    was_training = model.training
    model.eval()
    try:
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if stop_token_id is not None and bool(torch.any(next_token == stop_token_id)):
                break
        return generated
    finally:
        model.train(was_training)
