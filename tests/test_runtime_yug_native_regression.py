"""NB-11: Native Brain Runtime & Composition Regression.

VERIFICATION STATUS: written without a PyTorch installation available
in this sandbox (no network access to install it). NOT executed under
real pytest by this assistant — see the NB-11 implementation report
for what has and hasn't been genuinely verified, and by whom.

Purpose (per ARCHITECTURE/NB11_NATIVE_BRAIN_RUNTIME_PLAN.md): prove
that ``yug.runtime.core.YugCore`` can drive the real (non-mock, non-
hand-rolled) ``YugNativeAdapter`` through the unmodified
``BrainInterface`` contract, with zero changes anywhere in
``yug/runtime/``, ``yug/memory/``, ``yug/tools/``, or
``yug/verification/``. This is an integration/wiring check, not a
second copy of NB-10's own generation-correctness tests (no exact
generated text is ever asserted) and not a config-driven-selection
test (``YugCore(brain=YugNativeAdapter(...), memory=WorkingMemory())``
is constructed directly in these tests — wiring the native backend
into ``yug.composition``/``BrainConfig`` remains explicitly deferred,
per the plan's Section E/N).

Fixtures are self-contained in this file (tiny tokenizer + tiny real
checkpoint via ``training.checkpoint.save_checkpoint``), matching this
repository's existing per-file-fixture convention
(``tests/test_training_evaluate.py``, ``tests/test_brain_yug_native_adapter.py``)
rather than introducing a shared ``conftest.py`` — a reasonable future
cleanup, deliberately not attempted here (see the plan's Section N).
"""

from __future__ import annotations

import pytest

from yug.brain.adapters.yug_native import YugNativeAdapter
from yug.memory.working_memory import WorkingMemory
from yug.runtime.core import YugCore

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64
_CORPUS = ["hello world", "goodbye world", "hello there friend"]


def _tiny_tokenizer():
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    return ByteLevelBpeTokenizer.train(_CORPUS, vocab_size=260)


def _real_checkpoint(tmp_path, tokenizer, *, context_length: int):
    """A complete checkpoint (``model_config.json`` + ``metadata.json``
    + ``state.pt``), built via the real
    ``training.checkpoint.save_checkpoint`` and
    ``ModelConfig.from_tokenizer(...)`` -- exercises the actual
    checkpoint-writing path, not a hand-rolled stand-in, and never
    hardcodes a vocab_size that could disagree with the tokenizer
    (Directive Section 6/10), exactly matching
    ``tests/test_brain_yug_native_adapter.py``'s own fixture (after its
    NB-10 test-failure-fix) and ``tests/test_training_evaluate.py``'s
    ``_write_checkpoint()`` for NB-09.

    This function went through two earlier, insufficient fixes before
    landing on the current approach -- recorded here because the
    reasoning is exactly why the current construction looks the way it
    does:

    1. A bare, freshly-initialized (never-trained) checkpoint: failed
       under real execution because greedy decoding from an untrained
       model is essentially arbitrary over the full 256-byte-token
       space, and roughly half of those bytes (128-255) are not valid
       standalone UTF-8 -- a degenerate repeated-highest-logit token
       (a well-known greedy-decoding failure mode) has real odds of
       landing on one of them.
    2. Seeding (``training.determinism.seed_everything``) plus 30 real
       training steps on the fixture's own corpus: closed the
       cross-test RNG-ordering nondeterminism (``YugTransformer``'s
       initial weights are drawn from PyTorch's *global* RNG, so
       leaving it unseeded meant this fixture's outcome depended on
       how much of that stream every earlier test in the same process
       had already consumed -- exactly why the failure appeared when
       this file ran alone but not in the full suite). But real
       Windows execution showed 30 steps still wasn't always enough to
       fully overcome the untrained model's tendency to land on an
       invalid byte -- "train longer and hope" doesn't give a real
       guarantee, only better odds.

    The current construction gives an actual guarantee instead of
    better odds: it hand-sets model parameters (on a real, unmodified
    ``YugTransformer``) so that greedy generation deterministically
    selects one chosen, known-safe ASCII token at every step --
    ASCII bytes are trivially valid UTF-8 alone or repeated any number
    of times.

      1. Zero every parameter. A linear layer with zero weights
         contributes exactly zero to its output regardless of its
         input -- true for every attention Q/K/V/O projection and
         every FFN layer, in every block, whatever ``num_layers`` is,
         and unaffected by RoPE (``config.positional_encoding``'s
         default): RoPE rotates Q/K, and rotating an all-zero vector
         by any angle is still the all-zero vector. So with every
         block's weights at zero, each block is an exact identity on
         the residual stream (``x = x + 0 = x``).
      2. Set ``token_embedding.weight`` so every row (every possible
         token id) is the identical constant vector ``[1, 1, ..., 1]``.
         Combined with (1), the hidden state entering ``final_norm``
         is this same constant vector at every position, regardless of
         the actual prompt tokens.
      3. Set ``final_norm.weight`` (RMSNorm's learnable per-channel
         scale -- ``yug/native/model.py``'s ``RMSNorm.forward``:
         ``normalized * weight``, ``normalized = x * rsqrt(mean(x**2)
         + eps)``) to ``[1, 1, ..., 1]``. RMSNorm of a constant
         ``[1,...,1]`` vector is itself a known, computable positive
         constant vector ``h*`` -- derived directly from the formula,
         not guessed.
      4. Set ``lm_head.weight``'s row for one chosen ASCII target
         token to ``[1, 1, ..., 1]`` (same direction as ``h*``), and
         every other row to zero. ``lm_head`` has no bias, so
         ``logits[target] = h* . [1,...,1]`` (a known positive
         constant) and ``logits[j] = h* . 0 = 0`` for every other
         token. ``logits[target] > 0 = logits[j]`` for all ``j !=
         target`` is a strict inequality -- not a tie, not dependent
         on argmax's tie-breaking behavior -- so greedy decoding
         selects ``target`` at every step, unconditionally.

    ``seed_everything`` is kept as defense-in-depth for anything not
    explicitly overridden above, but is no longer load-bearing for
    correctness the way it was in the previous (training-based) fix.
    """
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
        # ModelConfig defaults to tie_embeddings=True, under which
        # lm_head.weight IS token_embedding.weight -- the literal same
        # tensor, not just an equal one. The construction below sets
        # every embedding row to [1,...,1] AND separately needs
        # lm_head's rows to differ (one row [1,...,1], every other
        # row zero) -- with tied embeddings those two requirements
        # directly conflict (whichever is set last would silently win
        # for both), which would put every vocab logit back into a
        # tie. Untying gives lm_head its own independent parameter.
        tie_embeddings=False,
    )

    # Follow-up to the seed+training fix (which real Windows execution
    # showed was still insufficient -- 30 steps was not always enough
    # to fully overcome an untrained model's tendency to land on an
    # invalid-UTF-8 byte under greedy decoding). This fixture no
    # longer trains, or otherwise hopes for a favorable outcome: it
    # hand-constructs model parameters that make greedy generation
    # ALWAYS select one chosen, known-safe ASCII token, by direct
    # mathematical guarantee -- not by chance, seed choice, or amount
    # of training. seed_everything is kept as defense-in-depth for
    # anything not explicitly overridden below, but is no longer load-
    # bearing for correctness.
    #
    # Construction (real YugTransformer, real ModelConfig -- only the
    # parameter *values* are hand-set, not the architecture):
    #   1. Zero every parameter in the model. Any linear layer fed a
    #      zero-weight transform contributes exactly zero to its
    #      output, REGARDLESS of what its input is -- this holds for
    #      every attention Q/K/V/O projection and every FFN layer in
    #      every block, whatever their number, and is unaffected by
    #      RoPE (config.positional_encoding's default): RoPE rotates
    #      Q/K, and rotating an all-zero vector by any angle is still
    #      the all-zero vector. So with every block's weights at zero,
    #      each block is an exact identity on the residual stream
    #      (`x = x + 0 = x`), regardless of num_layers or num_heads.
    #   2. Set token_embedding.weight so EVERY row (every possible
    #      input token id) is the identical constant vector [1, 1, ...,
    #      1]. Combined with (1), the hidden state entering final_norm
    #      is therefore this exact same constant vector, for every
    #      position, regardless of the actual prompt tokens.
    #   3. Set final_norm.weight (RMSNorm's learnable per-channel
    #      scale -- see yug/native/model.py's RMSNorm.forward:
    #      `normalized * weight`, normalized = x * rsqrt(mean(x**2) +
    #      eps)) to [1, 1, ..., 1]. RMSNorm of a constant [1,...,1]
    #      vector is itself a known, computable positive constant
    #      vector `h*` -- no guessing required, it follows directly
    #      from the formula above.
    #   4. Set lm_head.weight's row for one chosen target token id
    #      (an ASCII byte -- every ASCII byte is trivially valid
    #      UTF-8, alone or repeated any number of times) to [1, 1, ...,
    #      1] (same direction as h*), and every other row to zero.
    #      lm_head has no bias (see yug/native/model.py), so
    #      logits[target] = h* . [1,...,1] = (a known positive
    #      constant), and logits[j] = h* . 0 = 0 for every other
    #      token j. logits[target] > 0 = logits[j] for all j != target
    #      is then a strict, unconditional inequality -- not a tie,
    #      not dependent on argmax's tie-breaking behavior -- so
    #      greedy (argmax) generation selects `target` at every single
    #      step, deterministically, regardless of prompt content, RNG
    #      state, PyTorch version, or hardware.
    TARGET_TOKEN_ID = 97  # ASCII 'a' -- any byte 0-127 would do; picked
    # for readability of the resulting decoded content ("aaaa...").
    assert TARGET_TOKEN_ID < config.vocab_size  # always true (base bytes are 0-255)

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
# Test 1 — real native brain through YugCore
# ---------------------------------------------------------------------------


def test_yugcore_responds_using_yug_native_adapter(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    # Same sizing reasoning as NB-10's test_generation_is_deterministic:
    # the formatted prompt ("user: hello world\nassistant:") is 27
    # tokens with this fixture's tokenizer; context_length=64 gives
    # comfortable headroom above the ~32-token hard minimum for a
    # small max_tokens generation.
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer, context_length=64)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=tokenizer)
    # max_tokens overridden down from YugCore's default of 1024 -- this
    # is a constructor parameter YugCore already exposes for exactly
    # this purpose; not a production change.
    core = YugCore(adapter, WorkingMemory(), max_tokens=5)

    response = core.respond("hello world")

    assert response.success is True
    assert isinstance(response.content, str)
    # Deliberately NOT asserting exact generated text (Directive
    # Section 6, item 9) -- this is a wiring test, not a
    # generation-quality test.


# ---------------------------------------------------------------------------
# Test 2 — history persistence across two real calls
# ---------------------------------------------------------------------------


def test_yugcore_persists_history_with_yug_native_adapter(tmp_path):
    pytest.importorskip("torch")
    tokenizer = _tiny_tokenizer()
    # This test's second prompt includes the first turn's user message,
    # the first turn's (unpredictable, but max_tokens=3-bounded)
    # generated reply, and the second short user message -- generously
    # sized well above any plausible accumulated length rather than
    # computed to an exact minimum, since the generated text's exact
    # token count from an untrained checkpoint isn't knowable without
    # actually running it.
    checkpoint_dir, _ = _real_checkpoint(tmp_path, tokenizer, context_length=128)
    adapter = YugNativeAdapter(checkpoint_dir=str(checkpoint_dir), tokenizer=tokenizer)
    memory = WorkingMemory()
    core = YugCore(adapter, memory, max_tokens=3)

    first = core.respond("hello world")
    assert first.success is True
    assert memory.contains("conversation_history") is True
    history_after_first = memory.get("conversation_history")
    assert len(history_after_first) == 2  # user, assistant

    second = core.respond("hi")

    assert second.success is True
    history_after_second = memory.get("conversation_history")
    assert len(history_after_second) == 4  # user, assistant, user, assistant
    # The first turn's content is still there, unchanged, under the
    # existing history/memory mechanism -- observable through the
    # existing public API (WorkingMemory.get), not private state
    # (Directive Section 7).
    assert history_after_second[0] == history_after_first[0]
    assert history_after_second[1] == history_after_first[1]
    assert history_after_second[2]["role"] == "user"
    assert history_after_second[2]["content"] == "hi"


# ---------------------------------------------------------------------------
# Test 3 — real BrainError propagation through the existing runtime path
# ---------------------------------------------------------------------------


def test_yugcore_reports_brain_error_from_yug_native_adapter(tmp_path):
    # Deliberately NOT gated behind pytest.importorskip("torch"): a
    # missing checkpoint_dir raises a BrainError subclass either way
    # (ModelNotFoundError if the deferred torch/model imports succeed
    # and self._model_config is None; ModelLoadError if those deferred
    # imports themselves fail because torch isn't installed) -- both
    # are BrainError, both are handled identically by
    # YugCore.respond()'s existing `except BrainError` branch, which is
    # exactly the property this test exists to check. This lets the
    # test run -- and this assistant verify it -- in a torch-less
    # sandbox too, not only in the real PyTorch environment.
    tokenizer = _tiny_tokenizer()
    adapter = YugNativeAdapter(
        checkpoint_dir=str(tmp_path / "does_not_exist"), tokenizer=tokenizer
    )
    core = YugCore(adapter, WorkingMemory())

    response = core.respond("this will fail")

    assert response.success is False
    assert response.error is not None
    assert response.content == ""
