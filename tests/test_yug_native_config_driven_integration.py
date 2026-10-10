"""Config-driven end-to-end integration: YugConfig -> build_core() ->
YugNativeAdapter -> real checkpoint -> YugCore.respond().

VERIFICATION STATUS: written without a PyTorch installation available
in this sandbox (no network access to install it; confirmed again for
this task). NOT executed under real pytest by this assistant -- the
one test in this file is gated behind ``pytest.importorskip("torch")``
and reports SKIPPED here, exactly like every other torch-dependent
test already in this suite (see e.g.
tests/test_runtime_yug_native_regression.py, tests/test_training_evaluate.py).
This is disclosed, not hidden: no pass is claimed for this test in
this environment.

Purpose: tests/test_runtime_yug_native_regression.py (NB-11) already
proved YugCore can drive a real YugNativeAdapter through an unmodified
BrainInterface, but its own docstring explicitly deferred the
config-driven path: "not a config-driven-selection test ... wiring the
native backend into yug.composition/BrainConfig remains explicitly
deferred." That gap is what a later session's composition/config
wiring task (yug.composition._build_yug_native_adapter,
BrainConfig.checkpoint_dir/tokenizer_path) closed. This file is the
one integration test proving that closed gap actually reaches a real
checkpoint end to end: YugConfig(brain=BrainConfig(backend="yug_native",
checkpoint_dir=..., tokenizer_path=...)) -> build_core() -> a real,
loaded YugNativeAdapter -> YugCore.respond() succeeding.

Fixtures are file-local (tiny tokenizer + a real, deterministic
checkpoint), matching this repository's existing per-file-fixture
convention rather than a shared conftest.py (same choice
tests/test_runtime_yug_native_regression.py's own docstring makes, for
the same reason). ``_real_checkpoint`` below is copied, not
re-derived, from that file's own fixture -- same tiny architecture
(hidden_size=8, num_layers=1, num_heads=2, ffn_size=16,
tie_embeddings=False), same hand-set zero-weight-plus-one-token
construction that makes greedy generation deterministically select one
known-safe ASCII token at every step (see that file for the full
derivation of why this construction, not a briefly-trained model, is
what actually guarantees a successful, deterministic response rather
than merely a likely one). No new model architecture and no training
experiment is introduced here.

The one difference from that file's own fixture: context_length is
sized to comfortably exceed build_core()'s fixed default
max_tokens=1024 (YugCore's own constructor default; build_core() does
not currently forward a max_tokens override from YugConfig, and this
task does not authorize adding one) plus this fixture's prompt length,
so respond() can actually reach a successful completion under that
un-overridable default instead of hitting YugTransformer's
context_length bound partway through generation. This is a fixture
sizing choice, not a production change.
"""

from __future__ import annotations

import pytest

from yug.composition import build_core
from yug.config.schema import BrainConfig, MemoryConfig, YugConfig

_CORPUS = ["hello world", "goodbye world", "hello there friend"]

# Prompt "user: hello world\nassistant:" tokenizes to 27 tokens with
# this fixture's tokenizer (same corpus/tokenizer as
# tests/test_runtime_yug_native_regression.py, which quotes the same
# figure). build_core()'s YugCore uses max_tokens=1024 (YugCore's own
# constructor default) with no override available through YugConfig
# today, so context_length must clear 27 + 1024 with real headroom.
_CONTEXT_LENGTH = 1200


def _tiny_tokenizer():
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    return ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=260)


def _real_checkpoint(tmp_path, tokenizer, *, context_length: int):
    """A complete checkpoint (model_config.json + metadata.json +
    state.pt), built via the real training.checkpoint.save_checkpoint
    -- copied from tests/test_runtime_yug_native_regression.py's own
    fixture (see that file for the full derivation of the hand-set,
    zero-weight-plus-one-token construction that guarantees, rather
    than merely makes likely, that greedy decoding deterministically
    produces one known-safe ASCII token at every generation step)."""
    import torch

    from training.checkpoint import save_checkpoint
    from training.config import TrainingConfig
    from training.determinism import seed_everything
    from yug.native.config import ModelConfig
    from yug.native.model import YugTransformer

    config = ModelConfig.from_tokenizer(
        tokenizer,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        context_length=context_length,
        # tie_embeddings=False: see test_runtime_yug_native_regression.py's
        # fixture -- tied embeddings would make the token_embedding
        # (all rows [1,...,1]) and lm_head (one row [1,...,1], the
        # rest zero) constructions below directly conflict.
        tie_embeddings=False,
    )

    TARGET_TOKEN_ID = 97  # ASCII 'a' -- any byte 0-127 would do.
    assert TARGET_TOKEN_ID < config.vocab_size

    seed_everything(1234)
    model = YugTransformer(config)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.token_embedding.weight.fill_(1.0)
        model.final_norm.weight.fill_(1.0)
        model.lm_head.weight[TARGET_TOKEN_ID].fill_(1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    checkpoint_dir = tmp_path / "real_ckpt"
    save_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        model_config=config,
        training_config=TrainingConfig(
            seed=1234,
            learning_rate=1e-2,
            batch_size=2,
            sequence_length=8,
            max_steps=1,
            eval_interval=1,
            checkpoint_interval=1,
        ),
        step=1,
        tokenizer_identity=tokenizer.identity,
    )
    return checkpoint_dir, config


# ---------------------------------------------------------------------------
# The one integration test this task authorizes.
# ---------------------------------------------------------------------------


def test_yugconfig_build_core_reaches_real_native_checkpoint_and_responds(tmp_path):
    pytest.importorskip("torch")

    tokenizer = _tiny_tokenizer()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer, context_length=_CONTEXT_LENGTH)

    # The real, unmodified config-driven path -- no direct
    # YugNativeAdapter(...) or YugCore(...) construction anywhere in
    # this test.
    config = YugConfig(
        brain=BrainConfig(
            backend="yug_native",
            model_id="integration-test-native",
            checkpoint_dir=str(checkpoint_dir),
            tokenizer_path=str(tokenizer_path),
        ),
        memory=MemoryConfig(storage_path=str(tmp_path / "memory.json")),
    )

    core = build_core(config)

    response = core.respond("hello world")

    # generate() can only reach a successful GenerationResult after
    # _ensure_loaded() -> _load_model_sync() -> load_model_state() has
    # genuinely succeeded against this real checkpoint directory --
    # any failure there raises a BrainError, which YugCore.respond()
    # converts to success=False (see
    # test_yugcore_reports_brain_error_from_yug_native_adapter in
    # tests/test_runtime_yug_native_regression.py for that exact,
    # already-covered failure path). So success=True here is itself
    # the proof the real checkpoint was actually loaded, not merely
    # that construction/wiring succeeded.
    assert response.success is True
    assert response.error is None
    assert isinstance(response.content, str)
    assert response.content != ""
    # Deliberately not asserting exact generated text content beyond
    # non-emptiness -- this is a wiring/integration test, not a
    # generation-quality test (same convention as
    # tests/test_runtime_yug_native_regression.py).
