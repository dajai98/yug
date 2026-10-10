"""Tests for training.experiments.nb08_experiment_v4 (NB-08 V4).

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See training/__init__.py and
ARCHITECTURE/NB08_DATA_SCALE_EXPERIMENT_V4.md.
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
from training.experiments.nb08_experiment_v4 import (  # noqa: E402
    SCALE_CORPORA,
    build_locked_tokenizer,
    run_single_scale,
)
from yug.native.config import ModelConfig  # noqa: E402
from yug.native.model import YugTransformer  # noqa: E402


def test_locked_tokenizer_matches_v3_real_run_identity():
    """Regression guard for this milestone's tokenizer-lock
    requirement: this must reproduce V3's actual verified identity."""
    tokenizer = build_locked_tokenizer()
    assert tokenizer.identity == (
        "sha256:076c103ee402a41947e7911ee9f2908b00ac0af57b09e7c5c44180a4abf741e7"
    )
    assert tokenizer.vocab_size == 384


def test_all_four_scales_present():
    assert set(SCALE_CORPORA.keys()) == {
        "scale_0_v3_baseline",
        "scale_1",
        "scale_2",
        "scale_3",
    }


def test_v4_uses_identical_architecture_to_v2_v3():
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
    assert model.count_parameters() == 55536


def test_run_single_scale_smoke(tmp_path, monkeypatch):
    """Runs one real scale through the real pipeline with max_steps
    patched down for fast test execution."""
    import training.experiments.nb08_experiment_v4 as nb08_v4

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

    monkeypatch.setattr(nb08_v4, "build_training_config", _fast_config)
    monkeypatch.setattr(nb08_v4, "CURVE_CHECKPOINTS", [0, 5, 10])

    tokenizer = build_locked_tokenizer()
    results = run_single_scale(
        "scale_0_v3_baseline",
        SCALE_CORPORA["scale_0_v3_baseline"],
        tokenizer,
        str(tmp_path / "ckpt"),
    )

    assert results["parameter_count"] == 55536
    assert results["train_val_sentence_intersection"] == 0
    assert results["num_steps"] == 10
    assert torch.isfinite(torch.tensor(results["best_val_loss"]))
    assert isinstance(results["generated_text"], str)


def test_run_all_scales_executes_every_scale(tmp_path, monkeypatch):
    """Directive Section 26: a multi-scale runner must execute ALL
    planned scales, never a subset."""
    import training.experiments.nb08_experiment_v4 as nb08_v4

    def _fast_config():
        from training.config import TrainingConfig

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

    monkeypatch.setattr(nb08_v4, "build_training_config", _fast_config)
    monkeypatch.setattr(nb08_v4, "CURVE_CHECKPOINTS", [0, 5])

    all_results = nb08_v4.run_all_scales(str(tmp_path))

    assert set(all_results.keys()) == set(SCALE_CORPORA.keys())
    for scale_name, results in all_results.items():
        assert results["num_steps"] == 5
        assert results["parameter_count"] == 55536
