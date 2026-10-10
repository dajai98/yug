"""Tests for training.evaluate (NB-09).

VERIFICATION STATUS: written without a PyTorch installation available
in this sandbox (no network access to install it). NOT executed by
this assistant -- see training/__init__.py and
ARCHITECTURE/NB09_MINIMAL_EVALUATION_HARNESS.md for what has and
hasn't been genuinely verified, and by whom.
"""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from training.checkpoint import save_checkpoint  # noqa: E402
from training.config import TrainingConfig  # noqa: E402
from training.evaluate import (  # noqa: E402
    EvaluationError,
    evaluate_checkpoint,
    perplexity_from_loss,
)
from yug.native.config import ModelConfig  # noqa: E402
from yug.native.errors import ModelConfigSerializationError  # noqa: E402
from yug.native.model import YugTransformer  # noqa: E402
from yug.tokenizer.bpe import ByteLevelBpeTokenizer  # noqa: E402

_TRAIN_CORPUS = ["hello world", "goodbye world", "hello there friend"]
_EVAL_SENTENCES = ["hello world again, my friend"]


def _tiny_tokenizer() -> ByteLevelBpeTokenizer:
    # vocab_size=260 keeps this well above the 258-token minimum (256
    # base bytes + "bos"/"eos") while still allowing a couple of
    # merges -- large enough to be a real tokenizer, small enough to
    # stay a fast, deterministic test fixture.
    return ByteLevelBpeTokenizer.train(_TRAIN_CORPUS, vocab_size=260)


def _tiny_model_config(tokenizer: ByteLevelBpeTokenizer) -> ModelConfig:
    return ModelConfig.from_tokenizer(
        tokenizer,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        context_length=32,
    )


def _tiny_training_config() -> TrainingConfig:
    return TrainingConfig(
        seed=0,
        learning_rate=1e-2,
        batch_size=2,
        sequence_length=4,
        max_steps=1,
        eval_interval=1,
        checkpoint_interval=1,
    )


def _write_checkpoint(tmp_path, tokenizer: ByteLevelBpeTokenizer, step: int = 7):
    """Build a tiny YugTransformer and save it via the existing (NB-07)
    checkpoint writer -- this is deliberately the real
    training.checkpoint.save_checkpoint path, not a hand-rolled
    stand-in, so test_evaluate_checkpoint_round_trip genuinely
    exercises the real checkpoint-loading path (Directive Section 11,
    Test 2)."""
    model_config = _tiny_model_config(tokenizer)
    model = YugTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    checkpoint_dir = tmp_path / "ckpt"
    save_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        training_config=_tiny_training_config(),
        step=step,
        tokenizer_identity=tokenizer.identity,
    )
    return checkpoint_dir, model_config


# ---------------------------------------------------------------------------
# Test 1: perplexity correctness against known synthetic cases.
# ---------------------------------------------------------------------------


def test_perplexity_equals_exp_loss_on_known_case():
    # Known, hand-computable cases -- not merely "the function returns
    # some finite number" (Directive Section 7 / Section 11 Test 1).
    assert abs(perplexity_from_loss(0.0) - 1.0) < 1e-9
    assert abs(perplexity_from_loss(math.log(4.0)) - 4.0) < 1e-9
    assert abs(perplexity_from_loss(2.0) - math.exp(2.0)) < 1e-9


# ---------------------------------------------------------------------------
# Test 2: checkpoint round trip through the real checkpoint-loading path.
# ---------------------------------------------------------------------------


def test_evaluate_checkpoint_round_trip(tmp_path):
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _write_checkpoint(tmp_path, tokenizer)

    report = evaluate_checkpoint(
        checkpoint_dir,
        tokenizer,
        _EVAL_SENTENCES,
        sequence_length=4,
        batch_size=2,
        seed=0,
    )

    assert math.isfinite(report.loss)
    assert math.isfinite(report.perplexity)
    assert report.perplexity > 0.0
    # perplexity must actually be exp(loss) for THIS report, not just
    # independently finite.
    assert abs(report.perplexity - math.exp(report.loss)) < 1e-6


# ---------------------------------------------------------------------------
# Test 3: determinism (no generation randomness involved).
# ---------------------------------------------------------------------------


def test_evaluate_is_deterministic(tmp_path):
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _write_checkpoint(tmp_path, tokenizer)

    report_a = evaluate_checkpoint(
        checkpoint_dir, tokenizer, _EVAL_SENTENCES,
        sequence_length=4, batch_size=2, seed=0,
    )
    report_b = evaluate_checkpoint(
        checkpoint_dir, tokenizer, _EVAL_SENTENCES,
        sequence_length=4, batch_size=2, seed=0,
    )

    assert abs(report_a.loss - report_b.loss) < 1e-9
    assert abs(report_a.perplexity - report_b.perplexity) < 1e-9
    assert report_a.to_dict() == report_b.to_dict()


# ---------------------------------------------------------------------------
# Test 4: identity/configuration metadata is present and correct.
# ---------------------------------------------------------------------------


def test_evaluate_reports_identity_fields(tmp_path):
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, model_config = _write_checkpoint(tmp_path, tokenizer, step=7)

    report = evaluate_checkpoint(
        checkpoint_dir, tokenizer, _EVAL_SENTENCES,
        sequence_length=4, batch_size=2, seed=123,
    )

    assert report.checkpoint_step == 7
    assert report.model_identity == model_config.identity
    assert report.tokenizer_identity == tokenizer.identity
    assert report.model_parameter_count > 0
    assert report.seed == 123
    assert report.sequence_length == 4
    assert report.batch_size == 2
    assert report.eval_token_count > 0
    assert report.eval_sequence_count > 0

    # Machine-readable JSON (Directive Section 6) carries the same
    # identity fields, not a lossy/re-derived subset of them.
    parsed = json.loads(report.to_json())
    assert parsed["checkpoint_step"] == 7
    assert parsed["model_identity"] == model_config.identity
    assert parsed["tokenizer_identity"] == tokenizer.identity

    # Human-readable report (Directive Section 6) is non-empty and
    # mentions the same checkpoint step -- not asserting exact
    # formatting, just that it exists and isn't vacuous.
    human = report.to_human_readable()
    assert "7" in human
    assert "perplexity" in human.lower()


# ---------------------------------------------------------------------------
# Test 5: invalid/missing checkpoint fails clearly, no fake metric.
# ---------------------------------------------------------------------------


def test_evaluate_rejects_missing_checkpoint(tmp_path):
    tokenizer = _tiny_tokenizer()
    with pytest.raises(ModelConfigSerializationError):
        evaluate_checkpoint(
            tmp_path / "does_not_exist",
            tokenizer,
            _EVAL_SENTENCES,
            sequence_length=4,
            batch_size=2,
        )


# ---------------------------------------------------------------------------
# Extra: tokenizer-identity mismatch is a distinct, evaluation-specific
# failure mode this module itself introduces (Directive Section 5's
# "tokenizer identity" requirement) -- covered directly rather than
# left implicit.
# ---------------------------------------------------------------------------


def test_evaluate_rejects_tokenizer_identity_mismatch(tmp_path):
    tokenizer = _tiny_tokenizer()
    checkpoint_dir, _ = _write_checkpoint(tmp_path, tokenizer)

    other_tokenizer = ByteLevelBpeTokenizer.train(
        ["a totally different corpus with different bytes"], vocab_size=260
    )

    with pytest.raises(EvaluationError):
        evaluate_checkpoint(
            checkpoint_dir, other_tokenizer, _EVAL_SENTENCES,
            sequence_length=4, batch_size=2,
        )


def test_p0_exact_target_count_excludes_window_starts_and_tail(tmp_path):
    tokenizer = ByteLevelBpeTokenizer.train([], vocab_size=258)
    checkpoint_dir, _ = _write_checkpoint(tmp_path, tokenizer)
    report = evaluate_checkpoint(
        checkpoint_dir, tokenizer, ["abcdefghijkl"], sequence_length=4, batch_size=2,
    )
    assert report.eval_token_count == 8


def test_p0_uneven_batches_match_independent_global_nll(tmp_path):
    from training.dataset import encode_corpus, make_sequences, sequences_to_batch
    from yug.native.checkpoint_io import load_model_state

    tokenizer = _tiny_tokenizer()
    checkpoint_dir, config = _write_checkpoint(tmp_path, tokenizer)
    texts = ["abcdefghijklmnopq"]
    sequences = make_sequences(encode_corpus(tokenizer, texts), 4)
    assert len(sequences) == 3
    model = YugTransformer(config)
    model.load_state_dict(load_model_state(checkpoint_dir))
    model.eval()
    x, y = sequences_to_batch(sequences)
    with torch.no_grad():
        log_probs = model(x).log_softmax(-1)
        nll = -log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    expected = nll.double().sum().item() / y.numel()
    report = evaluate_checkpoint(
        checkpoint_dir, tokenizer, texts, sequence_length=4, batch_size=2,
    )
    assert report.loss == pytest.approx(expected, abs=1e-6)
