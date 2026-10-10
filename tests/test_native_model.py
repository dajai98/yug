"""Tests for yug.native.model (NB-04).

VERIFICATION STATUS: written without a PyTorch installation available
in the development sandbox (no network to install it). These tests
have NOT been executed by the assistant. They are written to be run
by whoever has PyTorch installed -- see the NB-04 report for the exact
verification gap and what to run to close it
(`pip install -e ".[native]"` then `pytest tests/test_native_model.py -v`).

All tests are skipped automatically (via the module-level
`pytest.importorskip("torch")` below) in any environment without
PyTorch, rather than failing at collection time.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yug.native.config import ModelConfig, estimate_parameter_count  # noqa: E402
from yug.native.errors import NativeError  # noqa: E402
from yug.native.model import YugTransformer, apply_rope, build_rope_cache  # noqa: E402

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64


def _tiny_config(**overrides) -> ModelConfig:
    """A tiny config for fast test execution (Directive Section 18's
    "test a tiny configuration suitable for fast CI execution")."""
    kwargs = dict(
        vocab_size=64,
        tokenizer_identity=_PLACEHOLDER_IDENTITY,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        ffn_size=32,
        context_length=32,
        normalization="rmsnorm",
        positional_encoding="rope",
        activation="silu",
        bias=False,
        tie_embeddings=True,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


# ---- 1. construction / 2. configuration compatibility ---------------------


def test_model_constructs_from_config():
    model = YugTransformer(_tiny_config())
    assert isinstance(model, torch.nn.Module)


def test_model_construction_matches_config_dimensions():
    config = _tiny_config(hidden_size=32, num_layers=3)
    model = YugTransformer(config)
    assert len(model.blocks) == 3
    assert model.token_embedding.embedding_dim == 32


@pytest.mark.parametrize("normalization", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("activation", ["gelu", "relu", "silu", "swiglu"])
@pytest.mark.parametrize("positional_encoding", ["rope", "learned", "none"])
def test_model_constructs_for_every_config_driven_combination(normalization, activation, positional_encoding):
    config = _tiny_config(
        normalization=normalization, activation=activation, positional_encoding=positional_encoding
    )
    model = YugTransformer(config)
    assert model is not None


# ---- 3. parameter shapes / 4. embedding shape ------------------------------


def test_token_embedding_shape():
    config = _tiny_config(vocab_size=100, hidden_size=24)
    model = YugTransformer(config)
    assert model.token_embedding.weight.shape == (100, 24)


def test_learned_positional_embedding_shape():
    config = _tiny_config(positional_encoding="learned", context_length=48, hidden_size=16)
    model = YugTransformer(config)
    assert model.position_embedding.weight.shape == (48, 16)


def test_rope_and_none_have_no_position_embedding_module():
    for pe in ("rope", "none"):
        model = YugTransformer(_tiny_config(positional_encoding=pe))
        assert model.position_embedding is None


# ---- 5. attention shape / 6. causal masking / 7. multi-head reshaping -----


def test_attention_output_shape_matches_input():
    config = _tiny_config(hidden_size=16, num_heads=4)
    model = YugTransformer(config)
    x = torch.randn(2, 5, 16)
    out = model.blocks[0].attn(x, rope_cache=None if config.positional_encoding != "rope" else _rope_cache_for(config, 5))
    assert out.shape == (2, 5, 16)


def _rope_cache_for(config: ModelConfig, seq_len: int):
    head_dim = config.hidden_size // config.num_heads
    return build_rope_cache(seq_len, head_dim)


def test_causal_masking_future_tokens_do_not_affect_earlier_output():
    """Mathematical invariant, not a mask-tensor-existence check: the
    output at position 0 must be IDENTICAL whether or not later tokens
    in the sequence are changed."""
    torch.manual_seed(0)
    config = _tiny_config(positional_encoding="none")  # isolate attention causality from positional effects
    model = YugTransformer(config)
    model.eval()

    ids_a = torch.randint(0, config.vocab_size, (1, 6))
    ids_b = ids_a.clone()
    ids_b[0, 3:] = (ids_b[0, 3:] + 1) % config.vocab_size  # change everything from position 3 onward

    with torch.no_grad():
        logits_a = model(ids_a)
        logits_b = model(ids_b)

    # Positions 0..2 (strictly before the change) must be unaffected.
    assert torch.allclose(logits_a[0, :3], logits_b[0, :3], atol=1e-5)
    # Position 3 (the change point) is expected to differ.
    assert not torch.allclose(logits_a[0, 3], logits_b[0, 3], atol=1e-5)


def test_multihead_reshape_is_consistent_across_head_counts():
    """Different num_heads with the same hidden_size must both run
    without shape errors (this is a shape-consistency check, not a
    claim that outputs are equal across head counts)."""
    for num_heads in (1, 2, 4, 8):
        config = _tiny_config(hidden_size=16, num_heads=num_heads)
        model = YugTransformer(config)
        ids = torch.randint(0, config.vocab_size, (1, 5))
        logits = model(ids)
        assert logits.shape == (1, 5, config.vocab_size)


# ---- 8. RoPE dimensions -----------------------------------------------------


def test_rope_cache_shapes():
    head_dim = 8
    seq_len = 10
    cos, sin = build_rope_cache(seq_len, head_dim)
    assert cos.shape == (seq_len, head_dim // 2)
    assert sin.shape == (seq_len, head_dim // 2)


def test_rope_requires_even_head_dim():
    # hidden_size=48, num_heads=... choose an odd head_dim: 48/... can't
    # get odd head_dim divisor of common nums easily; construct directly.
    config = _tiny_config(hidden_size=48, num_heads=16, positional_encoding="rope")  # head_dim = 3 (odd)
    with pytest.raises(NativeError):
        YugTransformer(config)


def test_apply_rope_preserves_shape():
    batch, heads, seq_len, head_dim = 2, 4, 6, 8
    x = torch.randn(batch, heads, seq_len, head_dim)
    cos, sin = build_rope_cache(seq_len, head_dim)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_apply_rope_preserves_vector_norm():
    """RoPE is a rotation; it must preserve the L2 norm of each
    (pair-wise) rotated vector -- a real mathematical property, not an
    implementation detail."""
    torch.manual_seed(1)
    x = torch.randn(1, 1, 4, 8)
    cos, sin = build_rope_cache(4, 8)
    out = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-4)


# ---- 9. Transformer block output shape / 10. complete forward shape -------


def test_transformer_block_output_shape():
    config = _tiny_config()
    block = YugTransformer(config).blocks[0]
    x = torch.randn(2, 5, config.hidden_size)
    rope_cache = _rope_cache_for(config, 5) if config.positional_encoding == "rope" else None
    out = block(x, rope_cache)
    assert out.shape == x.shape


def test_full_forward_shape():
    config = _tiny_config(vocab_size=50)
    model = YugTransformer(config)
    ids = torch.randint(0, 50, (3, 7))
    logits = model(ids)
    assert logits.shape == (3, 7, 50)


# ---- 11. batch size > 1 / 12. sequence length > 1 --------------------------


@pytest.mark.parametrize("batch_size", [1, 2, 8])
def test_various_batch_sizes(batch_size):
    config = _tiny_config()
    model = YugTransformer(config)
    ids = torch.randint(0, config.vocab_size, (batch_size, 5))
    logits = model(ids)
    assert logits.shape[0] == batch_size


@pytest.mark.parametrize("seq_len", [1, 2, 10, 32])
def test_various_sequence_lengths_up_to_context_length(seq_len):
    config = _tiny_config(context_length=32)
    model = YugTransformer(config)
    ids = torch.randint(0, config.vocab_size, (1, seq_len))
    logits = model(ids)
    assert logits.shape[1] == seq_len


# ---- 13. context-length boundary / 14. overflow ----------------------------


def test_sequence_exactly_at_context_length_succeeds():
    config = _tiny_config(context_length=16)
    model = YugTransformer(config)
    ids = torch.randint(0, config.vocab_size, (1, 16))
    logits = model(ids)
    assert logits.shape[1] == 16


def test_sequence_exceeding_context_length_raises_clearly():
    config = _tiny_config(context_length=16)
    model = YugTransformer(config)
    ids = torch.randint(0, config.vocab_size, (1, 17))
    with pytest.raises(NativeError):
        model(ids)


# ---- 15. invalid token IDs --------------------------------------------------


def test_negative_token_id_raises():
    config = _tiny_config(vocab_size=32)
    model = YugTransformer(config)
    ids = torch.tensor([[0, 1, -1, 2]])
    with pytest.raises(NativeError):
        model(ids)


def test_token_id_at_or_above_vocab_size_raises():
    config = _tiny_config(vocab_size=32)
    model = YugTransformer(config)
    ids = torch.tensor([[0, 1, 32, 2]])  # 32 is out of range for vocab_size=32 (valid: 0..31)
    with pytest.raises(NativeError):
        model(ids)


def test_max_valid_token_id_succeeds():
    config = _tiny_config(vocab_size=32)
    model = YugTransformer(config)
    ids = torch.tensor([[0, 31]])
    logits = model(ids)
    assert logits.shape == (1, 2, 32)


# ---- 16. tied embeddings / 17. untied embeddings ---------------------------


def test_tied_embeddings_share_the_same_parameter_object():
    config = _tiny_config(tie_embeddings=True)
    model = YugTransformer(config)
    assert model.lm_head.weight is model.token_embedding.weight


def test_untied_embeddings_are_independent_parameters():
    config = _tiny_config(tie_embeddings=False)
    model = YugTransformer(config)
    assert model.lm_head.weight is not model.token_embedding.weight
    assert model.lm_head.weight.shape == model.token_embedding.weight.shape


def test_tied_embeddings_mutation_reflects_in_both():
    """Proves genuine tying (same tensor), not merely equal-valued
    copies: mutating one must change the other."""
    config = _tiny_config(tie_embeddings=True)
    model = YugTransformer(config)
    with torch.no_grad():
        model.token_embedding.weight[0, 0] = 12345.0
    assert model.lm_head.weight[0, 0].item() == 12345.0


def test_untied_embeddings_mutation_does_not_reflect_in_the_other():
    config = _tiny_config(tie_embeddings=False)
    model = YugTransformer(config)
    original = model.lm_head.weight[0, 0].item()
    with torch.no_grad():
        model.token_embedding.weight[0, 0] = 99999.0
    assert model.lm_head.weight[0, 0].item() == original


# ---- 18. parameter count ----------------------------------------------------


@pytest.mark.parametrize("tie_embeddings", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("activation", ["gelu", "silu", "swiglu"])
@pytest.mark.parametrize("normalization", ["rmsnorm", "layernorm"])
@pytest.mark.parametrize("positional_encoding", ["rope", "learned", "none"])
def test_actual_parameter_count_matches_formula_exactly(
    positional_encoding, normalization, activation, bias, tie_embeddings
):
    config = _tiny_config(
        positional_encoding=positional_encoding,
        normalization=normalization,
        activation=activation,
        bias=bias,
        tie_embeddings=tie_embeddings,
    )
    model = YugTransformer(config)
    actual = model.count_parameters()
    expected = estimate_parameter_count(config).total
    assert actual == expected, (
        f"MISMATCH for {positional_encoding}/{normalization}/{activation}/"
        f"bias={bias}/tied={tie_embeddings}: formula says {expected:,}, "
        f"actual instantiated model has {actual:,}. Per Directive Section 13, "
        f"this must be investigated and reported, not silently patched."
    )


# ---- 19. deterministic controlled forward ----------------------------------


def test_forward_is_deterministic_with_fixed_seed_and_eval_mode():
    config = _tiny_config(dropout=0.0)
    torch.manual_seed(42)
    model_a = YugTransformer(config)
    torch.manual_seed(42)
    model_b = YugTransformer(config)
    model_a.eval()
    model_b.eval()

    ids = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        out_a = model_a(ids)
        out_b = model_b(ids)

    assert torch.allclose(out_a, out_b)


def test_repeated_forward_same_model_same_input_is_identical():
    config = _tiny_config(dropout=0.0)
    model = YugTransformer(config)
    model.eval()
    ids = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        out1 = model(ids)
        out2 = model(ids)
    assert torch.equal(out1, out2)


# ---- 20. gradient existence / basic autograd compatibility -----------------


def test_gradients_flow_to_all_parameters():
    """Verifies differentiability only -- NOT training. No optimizer
    step is taken; this just confirms backward() populates .grad on
    every parameter, i.e. the architecture doesn't accidentally detach
    part of the graph."""
    config = _tiny_config()
    model = YugTransformer(config)
    model.train()

    ids = torch.randint(0, config.vocab_size, (2, 5))
    logits = model(ids)
    loss = logits.sum()
    loss.backward()

    missing_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing_grad, f"Parameters with no gradient (broken autograd path): {missing_grad}"


def test_gradients_flow_with_tied_embeddings():
    """A specific risk with weight tying: gradients from both the
    embedding-lookup path and the output-projection path must
    accumulate onto the single shared parameter."""
    config = _tiny_config(tie_embeddings=True)
    model = YugTransformer(config)
    model.train()

    ids = torch.randint(0, config.vocab_size, (1, 5))
    logits = model(ids)
    logits.sum().backward()

    assert model.token_embedding.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    # Since they're the same tensor, this is trivially the same object,
    # but assert it explicitly for clarity.
    assert model.token_embedding.weight.grad is model.lm_head.weight.grad
