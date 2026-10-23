"""YUG's own decoder-only, causal, pre-norm Transformer.

Requires PyTorch (`pip install -e ".[native]"`) -- this is the one
module in `yug.native` that does. See package `__init__.py` for why
this is not re-exported there (keeps `yug.native.config` usable
without PyTorch installed).

Every architectural knob (layer count, hidden size, head count, FFN
size, vocab size, context length, normalization variant, positional
encoding, activation, bias, embedding tying) comes from a
:class:`yug.native.config.ModelConfig` instance -- nothing here is
hardcoded. The module structure is written to mirror
``yug.native.config.estimate_parameter_count()``'s documented formula
component-for-component, so the two can be directly compared (see the
NB-04 report's "Parameter Count Verification" section).

IMPORTANT VERIFICATION STATUS: this module was written without a
PyTorch installation available in the development environment (no
network access to install it). It has NOT been executed. Every design
choice below is justified by careful manual derivation against the
NB-03 parameter formula and standard, well-established Transformer
mechanics -- but "carefully reasoned" is not "verified". Do not treat
any claim about this module's runtime correctness as confirmed until
it has actually been run (see NB-04 report).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from yug.native.config import ModelConfig
from yug.native.errors import NativeError


class RMSNorm(nn.Module):
    """Root-mean-square normalization: a single learnable scale vector,
    no bias -- contributes exactly ``hidden_size`` parameters, matching
    the ``normalization == "rmsnorm"`` branch of
    ``estimate_parameter_count()``.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


def build_normalization(config: ModelConfig) -> nn.Module:
    """Return the normalization module for ``config.normalization``.

    ``"layernorm"`` uses ``nn.LayerNorm`` with its default
    ``elementwise_affine=True`` (weight + bias), contributing
    ``2 * hidden_size`` parameters -- matching the ``"layernorm"``
    branch of ``estimate_parameter_count()`` exactly.
    """
    if config.normalization == "rmsnorm":
        return RMSNorm(config.hidden_size)
    return nn.LayerNorm(config.hidden_size)


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables for positions ``0..seq_len-1``.

    Deterministic, parameter-free (no learned weights -- matches
    ``estimate_parameter_count()``'s assumption that
    ``positional_encoding == "rope"`` contributes zero learned
    positional parameters). Requires an even ``head_dim`` (rotation
    operates on adjacent coordinate pairs); this is validated at model
    construction time, not here.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [seq_len, head_dim // 2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to ``x``.

    Args:
        x: ``[batch, num_heads, seq_len, head_dim]``.
        cos, sin: ``[seq_len, head_dim // 2]``, from :func:`build_rope_cache`.
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_ = cos[None, None, :, :]
    sin_ = sin[None, None, :, :]
    rotated_1 = x1 * cos_ - x2 * sin_
    rotated_2 = x1 * sin_ + x2 * cos_
    return torch.stack([rotated_1, rotated_2], dim=-1).flatten(start_dim=-2)


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention.

    Four ``hidden_size x hidden_size`` projections (Q, K, V, O) --
    parameter count is independent of ``num_heads`` (it only changes
    how the ``hidden_size`` dimension is split into heads), matching
    ``estimate_parameter_count()``'s documented assumption exactly.
    Uses ``torch.nn.functional.scaled_dot_product_attention`` for the
    actual masked-softmax-matmul mechanics: that primitive is exactly
    the kind of mature numerical infrastructure PyTorch is authorized
    to provide (Directive Section 5) -- the architecture (which
    projections exist, how heads are formed, RoPE) is YUG's own.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            # Already guaranteed by ModelConfig.validate(), re-asserted
            # here since this module must not trust an un-validated
            # config from elsewhere.
            raise NativeError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_heads ({config.num_heads})"
            )

        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.positional_encoding = config.positional_encoding

        if self.positional_encoding == "rope" and self.head_dim % 2 != 0:
            raise NativeError(
                f"RoPE requires an even head_dim (hidden_size / num_heads); "
                f"got head_dim={self.head_dim} (hidden_size={config.hidden_size}, "
                f"num_heads={config.num_heads}). This is an additional "
                f"constraint discovered during NB-04 that ModelConfig's NB-03 "
                f"validation does not check -- see NB-04 report."
            )

        h = config.hidden_size
        self.q_proj = nn.Linear(h, h, bias=config.bias)
        self.k_proj = nn.Linear(h, h, bias=config.bias)
        self.v_proj = nn.Linear(h, h, bias=config.bias)
        self.o_proj = nn.Linear(h, h, bias=config.bias)
        # Attention-weight dropout, passed to scaled_dot_product_attention
        # directly rather than a separate nn.Dropout module -- it has no
        # learnable parameters either way, so this does not affect the
        # parameter-count formula. Respects train()/eval() automatically
        # via self.training (an nn.Module attribute).
        self.dropout_p = config.dropout

    def forward(self, x: torch.Tensor, rope_cache: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        batch, seq_len, hidden = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.positional_encoding == "rope":
            if rope_cache is None:
                raise NativeError("positional_encoding is 'rope' but no rope_cache was provided")
            cos, sin = rope_cache
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=(self.dropout_p if self.training else 0.0)
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, hidden)
        return self.o_proj(attn_out)


class FeedForward(nn.Module):
    """Position-wise feed-forward block.

    ``activation == "swiglu"`` uses a gated, 3-projection variant
    (gate, up, down) -- ``3 * hidden_size * ffn_size`` weight
    parameters. Every other supported activation (gelu/relu/silu) uses
    the standard 2-projection variant (up, down) -- ``2 * hidden_size *
    ffn_size``. Both match ``estimate_parameter_count()``'s branches
    exactly, including the bias-term counts.
    """

    _ELEMENTWISE_ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.is_gated = config.activation == "swiglu"
        h, ffn = config.hidden_size, config.ffn_size

        if self.is_gated:
            self.gate_proj = nn.Linear(h, ffn, bias=config.bias)
            self.up_proj = nn.Linear(h, ffn, bias=config.bias)
            self.down_proj = nn.Linear(ffn, h, bias=config.bias)
        else:
            self.up_proj = nn.Linear(h, ffn, bias=config.bias)
            self.down_proj = nn.Linear(ffn, h, bias=config.bias)
            self.activation = self._ELEMENTWISE_ACTIVATIONS[config.activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_gated:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.activation(self.up_proj(x)))


class TransformerBlock(nn.Module):
    """One pre-norm decoder block: ``x + Attn(Norm(x))``, then
    ``x + FFN(Norm(x))``. Two normalization instances per block,
    matching ``estimate_parameter_count()``'s ``norm_per_layer = 2 *
    norm_per_instance`` assumption."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = build_normalization(config)
        self.attn = CausalSelfAttention(config)
        self.norm2 = build_normalization(config)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, rope_cache: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_cache)
        x = x + self.ffn(self.norm2(x))
        return x


class YugTransformer(nn.Module):
    """The full model: embeddings, a stack of :class:`TransformerBlock`,
    a final normalization, and a vocabulary projection.

    Forward interface (Directive Section 14): ``forward(input_ids)``
    where ``input_ids`` is ``[batch, seq_len]`` (integer token ids) and
    the return value is ``[batch, seq_len, vocab_size]`` (float
    logits). No training-specific outputs, no hidden-state API.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.position_embedding: nn.Embedding | None
        if config.positional_encoding == "learned":
            self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        else:
            self.position_embedding = None

        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = build_normalization(config)

        # Output bias is deliberately suppressed when tied: the
        # parameter-count formula counts tied embeddings as
        # contributing exactly zero *additional* parameters (weight is
        # literally shared; there is no separate untied bias term
        # either in that case). See estimate_parameter_count()'s
        # `output_projection` branch.
        output_bias = config.bias and not config.tie_embeddings
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=output_bias)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self._uses_rope = config.positional_encoding == "rope"
        if self._uses_rope:
            head_dim = config.hidden_size // config.num_heads
            if head_dim % 2 != 0:
                raise NativeError(
                    f"RoPE requires an even head_dim (hidden_size / num_heads); "
                    f"got head_dim={head_dim}."
                )
            cos, sin = build_rope_cache(config.context_length, head_dim)
            self.register_buffer("_rope_cos", cos, persistent=False)
            self.register_buffer("_rope_sin", sin, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise NativeError(f"input_ids must be 2D [batch, seq_len], got shape {tuple(input_ids.shape)}")

        batch, seq_len = input_ids.shape

        if seq_len > self.config.context_length:
            raise NativeError(
                f"sequence length {seq_len} exceeds context_length {self.config.context_length}"
            )

        if seq_len > 0:
            min_id = int(input_ids.min())
            max_id = int(input_ids.max())
            if min_id < 0 or max_id >= self.config.vocab_size:
                raise NativeError(
                    f"token id out of range: found id(s) in [{min_id}, {max_id}], "
                    f"valid range is [0, {self.config.vocab_size - 1}]"
                )

        x = self.token_embedding(input_ids)

        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=input_ids.device)
            x = x + self.position_embedding(positions)[None, :, :]

        rope_cache = None
        if self._uses_rope:
            rope_cache = (self._rope_cos[:seq_len], self._rope_sin[:seq_len])

        for block in self.blocks:
            x = block(x, rope_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def count_parameters(self) -> int:
        """Actual instantiated parameter count (deduplicated: a tied
        weight shared by two modules is counted once, matching
        ``estimate_parameter_count()``'s intent)."""
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total
