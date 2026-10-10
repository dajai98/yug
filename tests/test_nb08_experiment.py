"""Tests for training.experiments.nb08_experiment (NB-08).

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See training/__init__.py and
ARCHITECTURE/NB08_CONTROLLED_LM_EXPERIMENT.md.

These deliberately do NOT re-test what NB-07's own tests already cover
(loss shape/finiteness, checkpoint round-trip mechanics, device
handling) -- per this milestone's own "cover the experiment
infrastructure, not merely duplicate existing NB-07 tests" instruction.
What's tested here is specific to NB-08: the real corpus, the real
split, and the full pipeline wired together.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.dataset import encode_corpus, make_sequences  # noqa: E402
from training.experiments.nb08_corpus import CORPUS_SENTENCES  # noqa: E402
from training.experiments.nb08_experiment import (  # noqa: E402
    SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_VOCAB_SIZE,
    VAL_FRACTION,
    run_experiment,
)
from training.split import split_sentences  # noqa: E402
from yug.tokenizer.bpe import ByteLevelBpeTokenizer  # noqa: E402


# -- Dataset correctness (Section 11.A) --------------------------------


def test_corpus_has_substantial_sentence_variety():
    # Not a single repeated pattern (unlike NB-07's toy corpus) --
    # every sentence should be distinct.
    assert len(CORPUS_SENTENCES) == len(set(CORPUS_SENTENCES))
    assert len(CORPUS_SENTENCES) >= 30


def test_train_val_split_disjoint_for_experiment_config():
    train_sentences, val_sentences = split_sentences(
        CORPUS_SENTENCES, val_fraction=VAL_FRACTION, seed=SEED
    )
    assert set(train_sentences).isdisjoint(set(val_sentences))
    assert len(train_sentences) > 0
    assert len(val_sentences) > 0


def test_tokenizer_trained_only_on_train_split_still_encodes_val_split():
    train_sentences, val_sentences = split_sentences(
        CORPUS_SENTENCES, val_fraction=VAL_FRACTION, seed=SEED
    )
    tokenizer = ByteLevelBpeTokenizer.train(
        corpus=train_sentences, vocab_size=TOKENIZER_VOCAB_SIZE, special_tokens=("bos", "eos")
    )
    val_ids = encode_corpus(tokenizer, val_sentences)
    assert len(val_ids) >= SEQUENCE_LENGTH + 1  # can form at least one sequence


def test_both_splits_yield_valid_sequences_no_error():
    train_sentences, val_sentences = split_sentences(
        CORPUS_SENTENCES, val_fraction=VAL_FRACTION, seed=SEED
    )
    tokenizer = ByteLevelBpeTokenizer.train(
        corpus=train_sentences, vocab_size=TOKENIZER_VOCAB_SIZE, special_tokens=("bos", "eos")
    )
    train_ids = encode_corpus(tokenizer, train_sentences)
    val_ids = encode_corpus(tokenizer, val_sentences)

    train_sequences = make_sequences(train_ids, sequence_length=SEQUENCE_LENGTH)
    val_sequences = make_sequences(val_ids, sequence_length=SEQUENCE_LENGTH)

    assert len(train_sequences) > 0
    assert len(val_sequences) > 0
    assert all(len(s) == SEQUENCE_LENGTH + 1 for s in train_sequences)
    assert all(len(s) == SEQUENCE_LENGTH + 1 for s in val_sequences)


# -- Full pipeline (Sections 11.B/C/D/14, run at reduced scale) --------


def test_safe_decode_generation_handles_decode_error(monkeypatch):
    """Regression test for the real crash found on the actual NB-08 V2
    real-environment run: DecodeError must not propagate out of the
    experiment's generation check (Directive Section 14: "Generation
    is supplementary evidence only")."""
    from training.experiments.nb08_experiment import safe_decode_generation
    from yug.tokenizer.errors import DecodeError

    class _RaisingTokenizer:
        def decode(self, ids):
            raise DecodeError("simulated: not valid UTF-8")

    result = safe_decode_generation(_RaisingTokenizer(), [1, 2, 3])
    assert isinstance(result, str)
    assert "DecodeError" in result


def test_safe_decode_generation_passes_through_on_success():
    from training.experiments.nb08_experiment import safe_decode_generation

    class _WorkingTokenizer:
        def decode(self, ids):
            return "hello"

    assert safe_decode_generation(_WorkingTokenizer(), [1, 2, 3]) == "hello"


def test_run_experiment_end_to_end_smoke(tmp_path, monkeypatch):
    """Runs the real run_experiment() pipeline, but with max_steps
    patched down for fast test execution -- this is a pipeline
    correctness smoke test, not a reproduction of the full documented
    experiment (see ARCHITECTURE/NB08_CONTROLLED_LM_EXPERIMENT.md for
    the full-scale run)."""
    import training.experiments.nb08_experiment as nb08

    original_build = nb08.build_training_config

    def _fast_config():
        cfg = original_build()
        from training.config import TrainingConfig

        return TrainingConfig(
            **{**cfg.__dict__, "max_steps": 10, "eval_interval": 5, "checkpoint_interval": 10}
        )

    monkeypatch.setattr(nb08, "build_training_config", _fast_config)

    results = run_experiment(checkpoint_dir=str(tmp_path / "ckpt"))

    assert results["train_sentences"] > 0
    assert results["val_sentences"] > 0
    assert results["parameter_count"] > 0
    assert torch.isfinite(torch.tensor(results["initial_train_loss"]))
    assert torch.isfinite(torch.tensor(results["final_val_loss"]))
    assert torch.isfinite(torch.tensor(results["final_train_loss_full_eval"]))
    # These two must be directly comparable (same measurement method --
    # see the NB-08 failure analysis, Section D): a full-corpus,
    # eval-mode pass over the training sequences must score at least
    # as well post-training as it did before any training happened.
    assert results["final_train_loss_full_eval"] < results["initial_train_loss"]
    assert len(results["training_losses"]) == 10
    assert all(torch.isfinite(torch.tensor(loss)) for loss in results["training_losses"])
    assert isinstance(results["generated_text"], str)


def test_checkpoint_is_created_by_experiment(tmp_path, monkeypatch):
    import training.experiments.nb08_experiment as nb08
    from training.config import TrainingConfig

    def _fast_config():
        return TrainingConfig(
            seed=SEED,
            learning_rate=3e-3,
            batch_size=4,
            sequence_length=SEQUENCE_LENGTH,
            max_steps=5,
            eval_interval=5,
            checkpoint_interval=5,
            device="cpu",
        )

    monkeypatch.setattr(nb08, "build_training_config", _fast_config)

    checkpoint_dir = tmp_path / "ckpt"
    run_experiment(checkpoint_dir=str(checkpoint_dir))

    assert (checkpoint_dir / "model_config.json").exists()
    assert (checkpoint_dir / "metadata.json").exists()
    assert (checkpoint_dir / "state.pt").exists()
