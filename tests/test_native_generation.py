"""Tests for yug.native.generation (NB-06).

VERIFICATION STATUS: written without a PyTorch installation available
in this development sandbox (no network to install it -- same
constraint as NB-04's test_native_model.py). These tests have NOT been
executed by the assistant. They are written to be run by whoever has
PyTorch installed (`pip install -e ".[native]"` then
`pytest tests/test_native_generation.py -v`).

All tests are skipped automatically (via the module-level
`pytest.importorskip("torch")` below) in any environment without
PyTorch, rather than failing at collection time.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yug.native.config import ModelConfig  # noqa: E402
from yug.native.errors import NativeError  # noqa: E402
from yug.native.generation import generate  # noqa: E402
from yug.native.model import YugTransformer  # noqa: E402

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64


def _tiny_config(**overrides) -> ModelConfig:
    """A tiny config for fast test execution, matching
    test_native_model.py's convention."""
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


def _model(**overrides) -> YugTransformer:
    return YugTransformer(_tiny_config(**overrides))


def _prompt(batch: int, seq_len: int, vocab_size: int = 64) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, vocab_size, (batch, seq_len))


# ---- 1. deterministic greedy selection -------------------------------------


def test_generate_is_deterministic_across_repeated_calls():
    """Same model, same prompt, same call -> byte-identical output every
    time. This is the core claim of NB-06 ("deterministic first")."""
    model = _model()
    prompt = _prompt(batch=1, seq_len=4)

    out_1 = generate(model, prompt, max_new_tokens=5)
    out_2 = generate(model, prompt, max_new_tokens=5)

    assert torch.equal(out_1, out_2)


def test_generate_matches_manual_argmax_greedy_step():
    """The selected token at each step must be exactly argmax(logits),
    not merely "some" plausible token -- checked by re-deriving the
    first generated token independently via a direct forward() call."""
    model = _model()
    prompt = _prompt(batch=1, seq_len=4)

    model.eval()
    with torch.no_grad():
        expected_first_token = torch.argmax(model(prompt)[:, -1, :], dim=-1, keepdim=True)

    out = generate(model, prompt, max_new_tokens=1)

    assert torch.equal(out[:, -1:], expected_first_token)


def test_generate_never_samples_is_pure_function_of_model_and_prompt():
    """Two independently constructed generate() calls interleaved with
    unrelated random-number-generator use must not diverge -- there is
    no hidden sampling step consuming randomness."""
    model = _model()
    prompt = _prompt(batch=1, seq_len=4)

    out_1 = generate(model, prompt, max_new_tokens=4)
    torch.manual_seed(12345)
    _ = torch.rand(100)  # perturb global RNG state between calls
    out_2 = generate(model, prompt, max_new_tokens=4)

    assert torch.equal(out_1, out_2)


# ---- 2. correct number of generated tokens ---------------------------------


def test_generate_appends_exactly_max_new_tokens_when_no_stop_token():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)

    out = generate(model, prompt, max_new_tokens=6)

    assert out.shape == (1, 3 + 6)


def test_generate_preserves_original_prompt_prefix():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)

    out = generate(model, prompt, max_new_tokens=4)

    assert torch.equal(out[:, :3], prompt)


# ---- 3. max_new_tokens boundary ---------------------------------------------


def test_generate_with_zero_max_new_tokens_returns_input_unchanged():
    model = _model()
    prompt = _prompt(batch=1, seq_len=5)

    out = generate(model, prompt, max_new_tokens=0)

    assert torch.equal(out, prompt)


def test_generate_rejects_negative_max_new_tokens():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)

    with pytest.raises(NativeError):
        generate(model, prompt, max_new_tokens=-1)


def test_generate_rejects_non_int_max_new_tokens():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)

    with pytest.raises(NativeError):
        generate(model, prompt, max_new_tokens=2.5)  # type: ignore[arg-type]


# ---- 4. optional stop_token_id behavior ------------------------------------


def test_generate_stops_immediately_when_stop_token_is_generated():
    """Force the model to always predict a specific stop id, independent
    of the input, by zeroing the output projection weights and giving
    only that id's bias a large positive value -- so logit_i = bias_i
    for every i regardless of the hidden state's sign or magnitude.

    Uses tie_embeddings=False and bias=True explicitly (overriding
    _tiny_config's defaults): with tied embeddings, lm_head.weight IS
    token_embedding.weight (the same object), so zeroing it would also
    zero the embedding table and corrupt every subsequent forward pass;
    with bias=False there is no bias term to force a deterministic
    winner at all.
    """
    model = _model(tie_embeddings=False, bias=True)
    prompt = _prompt(batch=1, seq_len=3)
    stop_id = 7

    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.bias.zero_()
        model.lm_head.bias[stop_id] = 1e6

    out = generate(model, prompt, max_new_tokens=10, stop_token_id=stop_id)

    assert out.shape[1] == 3 + 1  # generated exactly one token, then stopped
    assert out[0, -1].item() == stop_id


def test_generation_module_does_not_import_tokenizer():
    """NB-06 must not import yug.tokenizer or couple stop_token_id to
    any tokenizer convention -- checked via AST inspection of actual
    import statements only. (Checking raw source text for the substring
    "yug.tokenizer" is the wrong test: this module's own docstring
    legitimately mentions "yug.tokenizer" by name when documenting that
    it is NOT imported, which would make a substring check fail on
    exactly the correct, intended code.)"""
    import ast
    import inspect

    import yug.native.generation as generation_module

    tree = ast.parse(inspect.getsource(generation_module))
    imported_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.append(node.module)

    assert not any(name.startswith("yug.tokenizer") for name in imported_names)
    assert imported_names == ["__future__", "torch", "yug.native.errors", "yug.native.model"]


# ---- 5. stop-token behavior when the token is never generated --------------


def test_generate_runs_full_max_new_tokens_when_stop_token_never_appears():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)
    # A stop id chosen from outside the model's vocabulary can never be
    # produced by argmax over that vocabulary.
    impossible_stop_id = -1

    out = generate(model, prompt, max_new_tokens=5, stop_token_id=impossible_stop_id)

    assert out.shape == (1, 3 + 5)


def test_generate_with_stop_token_id_none_behaves_like_no_stop_token():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)

    out_default = generate(model, prompt, max_new_tokens=4)
    out_explicit_none = generate(model, prompt, max_new_tokens=4, stop_token_id=None)

    assert torch.equal(out_default, out_explicit_none)


# ---- 6. input validation ----------------------------------------------------


def test_generate_rejects_invalid_stop_token_id_type():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)

    with pytest.raises(NativeError):
        generate(model, prompt, max_new_tokens=2, stop_token_id="7")  # type: ignore[arg-type]


def test_generate_propagates_forward_validation_for_wrong_input_shape():
    """generate() does not duplicate forward()'s input validation; it
    must surface the same NativeError forward() itself would raise."""
    model = _model()
    bad_prompt = torch.randint(0, 64, (2, 3, 1))  # 3D, not [batch, seq_len]

    with pytest.raises(NativeError):
        generate(model, bad_prompt, max_new_tokens=1)


def test_generate_propagates_forward_validation_for_out_of_range_token_id():
    model = _model(vocab_size=64)
    bad_prompt = torch.tensor([[0, 1, 999]])  # 999 >= vocab_size

    with pytest.raises(NativeError):
        generate(model, bad_prompt, max_new_tokens=1)


def test_generate_with_zero_max_new_tokens_skips_forward_validation():
    """Documented edge case: when max_new_tokens == 0, forward() is
    never called, so even an out-of-range prompt is returned as-is
    rather than raising."""
    model = _model(vocab_size=64)
    bad_prompt = torch.tensor([[0, 1, 999]])

    out = generate(model, bad_prompt, max_new_tokens=0)

    assert torch.equal(out, bad_prompt)


# ---- 7. output shape/type/device, consistent with YugTransformer ----------


def test_generate_output_dtype_matches_input_dtype():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3).to(torch.long)

    out = generate(model, prompt, max_new_tokens=3)

    assert out.dtype == prompt.dtype


def test_generate_output_device_matches_input_device():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)

    out = generate(model, prompt, max_new_tokens=3)

    assert out.device == prompt.device


def test_generate_output_is_2d_batch_seq():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)

    out = generate(model, prompt, max_new_tokens=3)

    assert out.dim() == 2


def test_generate_supports_batch_greater_than_one():
    model = _model()
    prompt = _prompt(batch=3, seq_len=4)

    out = generate(model, prompt, max_new_tokens=3)

    assert out.shape == (3, 4 + 3)


def test_generate_restores_model_training_mode_after_call():
    """generate() must not leave the model permanently in eval() mode
    -- it should restore whatever mode the caller had it in."""
    model = _model()
    model.train()
    prompt = _prompt(batch=1, seq_len=3)

    generate(model, prompt, max_new_tokens=2)

    assert model.training is True


def test_generate_does_not_track_gradients():
    model = _model()
    prompt = _prompt(batch=1, seq_len=3)

    out = generate(model, prompt, max_new_tokens=2)

    assert out.requires_grad is False
