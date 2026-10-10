"""Tests for training.experiments.nb08_experiment_v3 (NB-08 V3).

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See training/__init__.py and
ARCHITECTURE/NB08_DATA_REGIME_EXPERIMENT_V3.md.

Deliberately does not re-test what test_nb08_v3_corpus.py (torch-free,
already genuinely executed), test_nb08_experiment.py, or
test_nb08_experiment_v2.py already cover.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.experiments.nb08_experiment_v2 import (  # noqa: E402
    V2_FFN_SIZE,
    V2_HIDDEN_SIZE,
    V2_NUM_HEADS,
    V2_NUM_LAYERS,
)
from training.experiments.nb08_experiment_v3 import run_experiment_v3  # noqa: E402
from yug.native.config import ModelConfig  # noqa: E402
from yug.native.model import YugTransformer  # noqa: E402


def test_v3_uses_identical_architecture_to_v2():
    """Regression guard for this milestone's core premise: V3 must
    change ONLY the corpus, not the model."""
    config = ModelConfig(
        vocab_size=384,
        tokenizer_identity="sha256:" + "0" * 64,
        hidden_size=V2_HIDDEN_SIZE,
        num_layers=V2_NUM_LAYERS,
        num_heads=V2_NUM_HEADS,
        ffn_size=V2_FFN_SIZE,
        context_length=32,
    )
    model = YugTransformer(config)
    assert model.count_parameters() == 55536  # same as V2's verified count


def test_run_experiment_v3_end_to_end_smoke(tmp_path, monkeypatch):
    """Runs the real V3 pipeline with max_steps patched down for fast
    test execution."""
    import training.experiments.nb08_experiment_v3 as nb08_v3

    def _fast_config():
        from training.config import TrainingConfig

        return TrainingConfig(
            seed=1234,
            learning_rate=3e-3,
            batch_size=4,
            sequence_length=16,
            max_steps=10,
            eval_interval=5,
            checkpoint_interval=10,
            grad_clip_norm=1.0,
            device="cpu",
        )

    monkeypatch.setattr(nb08_v3, "build_training_config", _fast_config)
    monkeypatch.setattr(nb08_v3, "CURVE_CHECKPOINTS", [0, 5, 10])

    results = run_experiment_v3(checkpoint_dir=str(tmp_path / "ckpt"))

    assert results["parameter_count"] == 55536
    assert results["num_steps"] == 10
    assert len(results["training_curve"]) == 3
    assert results["best_val_step"] in (0, 5, 10)
    assert torch.isfinite(torch.tensor(results["best_val_loss"]))
    for step, train_loss, val_loss in results["training_curve"]:
        assert torch.isfinite(torch.tensor(train_loss))
        assert torch.isfinite(torch.tensor(val_loss))
    assert isinstance(results["generated_text"], str)


def test_v3_checkpoint_created(tmp_path, monkeypatch):
    import training.experiments.nb08_experiment_v3 as nb08_v3
    from training.config import TrainingConfig

    def _fast_config():
        return TrainingConfig(
            seed=1234,
            learning_rate=3e-3,
            batch_size=4,
            sequence_length=16,
            max_steps=5,
            eval_interval=5,
            checkpoint_interval=5,
            device="cpu",
        )

    monkeypatch.setattr(nb08_v3, "build_training_config", _fast_config)
    monkeypatch.setattr(nb08_v3, "CURVE_CHECKPOINTS", [0, 5])

    checkpoint_dir = tmp_path / "ckpt"
    run_experiment_v3(checkpoint_dir=str(checkpoint_dir))

    assert (checkpoint_dir / "model_config.json").exists()
    assert (checkpoint_dir / "metadata.json").exists()
    assert (checkpoint_dir / "state.pt").exists()


def test_v3_tokenizer_identity_differs_from_v1_v2_but_vocab_size_matches():
    """Documents and enforces the expected, declared side effect
    (Directive Section 6): a new corpus necessarily produces a new
    tokenizer identity, but the vocab_size hyperparameter must remain
    unchanged."""
    from training.experiments.nb08_experiment import TOKENIZER_VOCAB_SIZE
    from training.experiments.nb08_v3_corpus import CORPUS_SENTENCES
    from training.split import split_sentences
    from yug.tokenizer.bpe import ByteLevelBpeTokenizer

    train, _ = split_sentences(CORPUS_SENTENCES, val_fraction=0.2, seed=1234)
    tokenizer = ByteLevelBpeTokenizer.train(
        corpus=train, vocab_size=TOKENIZER_VOCAB_SIZE, special_tokens=("bos", "eos")
    )

    assert tokenizer.vocab_size == 384  # same hyperparameter as V1/V2
    assert tokenizer.identity != (
        "sha256:ac40c48e125c1fd077563e116c813a71c10d6b8da65c916c45e1f5aec757a26e"
    )  # necessarily different -- different corpus
