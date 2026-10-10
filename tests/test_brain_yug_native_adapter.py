"""Tests for YugNativeAdapter (NB-10).

VERIFICATION STATUS: written without a PyTorch installation available
in this sandbox (no network access to install it). NOT executed by
this assistant — see ARCHITECTURE/NB10_NATIVE_ADAPTER_PLAN.md and the
NB-10 implementation report for what has and hasn't been genuinely
verified, and by whom.

Test categories (matching tests/test_brain_llama_cpp_adapter.py's own
established convention):

- UNIT TEST: adapter construction, capability/health reporting from
  configuration alone, and module importability. These never require
  PyTorch and pass regardless of whether it is installed.
- INTEGRATION TEST (torch-conditional): round-trips a tiny synthetic
  checkpoint (built with the same tools NB-07/NB-08/NB-09 already use)
  through the real checkpoint-loading and generation paths. Each such
  test calls ``pytest.importorskip("torch")`` itself, matching this
  repository's existing convention for torch-dependent tests
  (tests/test_training_evaluate.py, tests/test_training_checkpoint.py)
  rather than gating the whole file behind one module-level
  importorskip, which would also skip the UNIT tests above and defeat
  the point of the importability test.
- No REQUIRES REAL MODEL category: unlike LlamaCppAdapter's .gguf
  dependency, testing this adapter's own logic does not require a real
  NB-08 checkpoint (none exists in this repository — see
  ARCHITECTURE/NB09_MINIMAL_EVALUATION_HARNESS.md — and none is
  fabricated here; a tiny synthetic checkpoint is sufficient, per
  ARCHITECTURE/NB10_NATIVE_ADAPTER_PLAN.md Section 7).
- Existing-adapter regression is not re-tested in this file — it is
  covered by tests/test_brain_llama_cpp_adapter.py itself continuing
  to pass unmodified as part of the full suite (see the NB-10
  implementation report's full-suite result).
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

from yug.brain.adapters.yug_native import YugNativeAdapter
from yug.brain.errors import (
    EmbeddingNotSupportedError,
    GenerationError,
    ModelLoadError,
    ModelNotFoundError,
)
from yug.brain.interface import BrainInterface
from yug.brain.types import HealthState, Message, ToolDefinition
from yug.native.config import ModelConfig
from yug.tokenizer.bpe import ByteLevelBpeTokenizer

_TORCH_INSTALLED = importlib.util.find_spec("torch") is not None
_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64


class _DummyTokenizer:
    """Minimal stand-in exposing only what the adapter touches before a
    real checkpoint/tokenizer pairing is available (``.identity``)."""

    identity = _PLACEHOLDER_IDENTITY


def _tiny_model_config(
    tokenizer_identity: str = _PLACEHOLDER_IDENTITY, *, context_length: int = 16
) -> ModelConfig:
    # Same tiny values as tests/test_training_checkpoint.py's own
    # _tiny_config() — not inventing a second "what counts as tiny"
    # convention for this milestone. context_length is the one
    # parameterized field: most tests never generate, so the small
    # default of 16 (unchanged) is fine for them; the one test that
    # actually runs generation needs enough room for its formatted
    # prompt + max_new_tokens (see test_generation_is_deterministic).
    return ModelConfig(
        vocab_size=32,
        tokenizer_identity=tokenizer_identity,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        context_length=context_length,
    )


def _config_only_checkpoint(tmp_path, tokenizer_identity: str = _PLACEHOLDER_IDENTITY):
    """A checkpoint directory with only ``model_config.json`` — no
    ``state.pt``. Zero PyTorch dependency (``yug.native.config.ModelConfig``
    has none), so this fixture works even where PyTorch is not
    installed, unlike :func:`_real_checkpoint` below."""
    config = _tiny_model_config(tokenizer_identity)
    checkpoint_dir = tmp_path / "config_only_ckpt"
    checkpoint_dir.mkdir()
    config.save(checkpoint_dir / "model_config.json")
    return checkpoint_dir, config


def _real_checkpoint(tmp_path, tokenizer, *, context_length: int = 16):
    """A complete checkpoint (``model_config.json`` + ``metadata.json``
    + ``state.pt``), built via the real
    ``training.checkpoint.save_checkpoint`` — exercises the actual
    checkpoint-writing path, not a hand-rolled stand-in, exactly as
    ``tests/test_training_evaluate.py``'s own ``_write_checkpoint()``
    does for NB-09. ``context_length`` defaults to the same tiny value
    as every other test's checkpoint; only a test that actually calls
    ``generate()`` needs to override it (see
    ``test_generation_is_deterministic``).

    Uses ``ModelConfig.from_tokenizer(tokenizer, ...)`` rather than
    ``_tiny_model_config()``'s hardcoded ``vocab_size=32`` — this
    function always receives a real, already-trained tokenizer (unlike
    ``_config_only_checkpoint()``, which uses a dummy stand-in and so
    keeps using ``_tiny_model_config()``), and only
    ``from_tokenizer(...)`` guarantees ``vocab_size`` actually matches
    what that tokenizer can produce. A prior version of this fixture
    used the hardcoded value here too; it was never exercised by real
    token ids until test_generation_is_deterministic actually ran
    generation, which is what surfaced the mismatch (see this
    project's NB-10 test-failure-fix report for the diagnosis)."""
    import torch

    from training.checkpoint import save_checkpoint
    from training.config import TrainingConfig
    from yug.native.model import YugTransformer

    config = ModelConfig.from_tokenizer(
        tokenizer,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        context_length=context_length,
    )
    model = YugTransformer(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    checkpoint_dir = tmp_path / "real_ckpt"
    save_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        model_config=config,
        training_config=TrainingConfig(
            seed=0,
            learning_rate=1e-2,
            batch_size=2,
            sequence_length=4,
            max_steps=1,
            eval_interval=1,
            checkpoint_interval=1,
        ),
        step=3,
        tokenizer_identity=tokenizer.identity,
    )
    return checkpoint_dir, config


def _tiny_tokenizer() -> ByteLevelBpeTokenizer:
    return ByteLevelBpeTokenizer.train(
        ["hello world", "goodbye world", "hello there friend"], vocab_size=260
    )


# ---------------------------------------------------------------------------
# 1. BrainInterface compatibility
# ---------------------------------------------------------------------------


def test_yug_native_adapter_satisfies_brain_interface():
    adapter = YugNativeAdapter(checkpoint_dir="/nonexistent", tokenizer=_DummyTokenizer())
    assert isinstance(adapter, BrainInterface)


# ---------------------------------------------------------------------------
# 2. Importability without the native (PyTorch) dependency
# ---------------------------------------------------------------------------


def test_module_importable_without_torch_installed():
    """The adapter module itself must import cleanly whether or not
    PyTorch happens to be installed in the environment running this
    test — torch/YugTransformer/native generation are all imported
    lazily inside _load_model_sync/generate, never at module scope
    (ARCHITECTURE/NB10_NATIVE_ADAPTER_PLAN.md Section 9, Risk 2). This
    test's assertion is the import itself succeeding; only a
    module-scope torch import in yug_native.py could make it fail here,
    which is exactly the regression this test exists to catch."""
    import yug.brain.adapters.yug_native  # noqa: F401


# ---------------------------------------------------------------------------
# 3. Capabilities without a model load
# ---------------------------------------------------------------------------


def test_get_capabilities_does_not_require_a_loaded_model(tmp_path):
    checkpoint_dir, config = _config_only_checkpoint(tmp_path)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=_DummyTokenizer())

    caps = adapter.get_capabilities()

    assert caps.context_length == config.context_length
    # Unlike LlamaCppAdapter (permanently parameter_count=None pre-load),
    # this IS knowable pre-load via estimate_parameter_count.
    assert caps.parameter_count is not None
    assert caps.parameter_count > 0
    assert caps.embedding_dim is None
    assert caps.supports_tools is False
    assert caps.supports_vision is False


def test_get_capabilities_raises_honestly_without_valid_checkpoint():
    adapter = YugNativeAdapter(checkpoint_dir="/nonexistent", tokenizer=_DummyTokenizer())
    with pytest.raises(ModelNotFoundError):
        adapter.get_capabilities()


def test_model_id_defaults_to_checkpoint_directory_name(tmp_path):
    checkpoint_dir, _ = _config_only_checkpoint(tmp_path)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=_DummyTokenizer())
    assert adapter.get_capabilities().model_id == checkpoint_dir.name


# ---------------------------------------------------------------------------
# 4. Native checkpoint loading (real round trip)
# ---------------------------------------------------------------------------


def test_native_checkpoint_loading(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=tokenizer)

    model = asyncio.run(adapter._ensure_loaded())

    assert model is not None
    assert model.count_parameters() > 0


# ---------------------------------------------------------------------------
# 5. Deterministic generation
# ---------------------------------------------------------------------------


def test_generation_is_deterministic(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    # This is the one test that actually calls generate(), so it's the
    # one test that needs a context_length large enough for the
    # formatted prompt + max_new_tokens. The formatted prompt
    # ("user: hello world\nassistant:", tokenized with add_bos=True) is
    # 27 tokens with this fixture's tokenizer; max_tokens=5 below pushes
    # the longest forward pass to 32 tokens. 64 is comfortable headroom
    # above that hard minimum, not a magic number — every other test in
    # this file keeps the smaller default (16) since they never
    # generate and so never hit this limit.
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer, context_length=64)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=tokenizer)
    messages = [Message(role="user", content="hello world")]

    # Greedy decoding from this fixture's untrained checkpoint can
    # legitimately land on a token sequence that doesn't decode to
    # valid UTF-8 (see yug/brain/adapters/yug_native.py's own
    # DecodeError -> GenerationError boundary, and
    # training/experiments/nb08_experiment.py's safe_decode_generation()
    # for the same failure mode found during NB-08 V2). Determinism
    # means the SAME input produces the SAME outcome every time --
    # not that success is guaranteed. Both of the following are valid
    # deterministic outcomes; only a mismatch between the two calls,
    # or a raw (non-GenerationError) exception, is a real failure.
    def _call():
        try:
            return asyncio.run(adapter.generate(messages, max_tokens=5)), None
        except GenerationError as exc:
            return None, exc

    result_a, error_a = _call()
    result_b, error_b = _call()

    if error_a is None and error_b is None:
        assert result_a.content == result_b.content
        assert result_a.finish_reason == result_b.finish_reason
    elif error_a is not None and error_b is not None:
        assert str(error_a) == str(error_b)
    else:
        pytest.fail(
            f"Generation was not deterministic: one call succeeded and the "
            f"other raised GenerationError. Call A: "
            f"{'success' if error_a is None else f'error={error_a}'}; "
            f"Call B: {'success' if error_b is None else f'error={error_b}'}"
        )


# ---------------------------------------------------------------------------
# 6. Tokenizer identity mismatch
# ---------------------------------------------------------------------------


def test_tokenizer_identity_mismatch_raises(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer)

    other_tokenizer = ByteLevelBpeTokenizer.train(
        ["a totally different corpus with different bytes"], vocab_size=260
    )
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=other_tokenizer)

    with pytest.raises(ModelLoadError):
        asyncio.run(adapter._ensure_loaded())


# ---------------------------------------------------------------------------
# 7. Health state
# ---------------------------------------------------------------------------


def test_health_check_reports_unavailable_for_missing_checkpoint():
    adapter = YugNativeAdapter(checkpoint_dir="/nonexistent", tokenizer=_DummyTokenizer())
    status = adapter.health_check()
    assert status.state == HealthState.UNAVAILABLE
    assert status.model_loaded is False


def test_health_check_reports_degraded_before_load(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=tokenizer)

    status = adapter.health_check()

    assert status.state == HealthState.DEGRADED
    assert status.model_loaded is False


def test_health_check_reports_healthy_after_load(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=tokenizer)

    asyncio.run(adapter._ensure_loaded())
    status = adapter.health_check()

    assert status.state == HealthState.HEALTHY
    assert status.model_loaded is True


# ---------------------------------------------------------------------------
# 8. Invalid checkpoint behavior
# ---------------------------------------------------------------------------


def test_invalid_checkpoint_raises_model_not_found(tmp_path):
    adapter = YugNativeAdapter(
        checkpoint_dir=str(tmp_path / "does_not_exist"), tokenizer=_DummyTokenizer()
    )
    with pytest.raises(ModelNotFoundError):
        adapter.get_capabilities()


def test_corrupt_state_file_raises_model_load_error(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer)
    (checkpoint_dir / "state.pt").write_bytes(b"not a real checkpoint")

    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=tokenizer)

    with pytest.raises(ModelLoadError):
        asyncio.run(adapter._ensure_loaded())


# ---------------------------------------------------------------------------
# 9. Honest rejection of unsupported capabilities (tools / stop_sequences
#    / embeddings) — directly required by the adapter contract, Sections
#    4/12/13 of the architecture plan.
# ---------------------------------------------------------------------------


def test_generate_rejects_tools(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=tokenizer)
    messages = [Message(role="user", content="hi")]
    tools = [ToolDefinition(name="x", description="x")]

    with pytest.raises(GenerationError):
        asyncio.run(adapter.generate(messages, tools=tools))


def test_generate_rejects_stop_sequences(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=tokenizer)
    messages = [Message(role="user", content="hi")]

    with pytest.raises(GenerationError):
        asyncio.run(adapter.generate(messages, stop_sequences=["STOP"]))


def test_embed_raises_not_supported():
    adapter = YugNativeAdapter(checkpoint_dir="/nonexistent", tokenizer=_DummyTokenizer())
    with pytest.raises(EmbeddingNotSupportedError):
        asyncio.run(adapter.embed(["hello"]))
