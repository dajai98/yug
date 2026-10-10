"""Tests for training.experiments.nb08_experiment_v2 (NB-08 V2).

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See training/__init__.py and
ARCHITECTURE/NB08_CORRECTIVE_EXPERIMENT_V2.md.

Deliberately does not re-test what test_nb08_experiment.py (V1) or
NB-07's own tests already cover -- these are specific to V2's own new
surface: the smaller model configuration and the training-curve
capture.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.experiments.nb08_experiment import TOKENIZER_VOCAB_SIZE  # noqa: E402
from training.experiments.nb08_experiment_v2 import (  # noqa: E402
    CURVE_CHECKPOINTS,
    V2_FFN_SIZE,
    V2_HIDDEN_SIZE,
    V2_NUM_HEADS,
    V2_NUM_LAYERS,
    run_experiment_v2,
)
from yug.native.config import ModelConfig  # noqa: E402
from yug.native.model import YugTransformer  # noqa: E402


def test_v2_config_is_substantially_smaller_than_v1():
    """Regression guard for this milestone's core premise: V2 must
    actually be a smaller model, not accidentally the same or larger."""
    v1_params = 221760  # the real, reported V1 parameter count
    config = ModelConfig(
        vocab_size=TOKENIZER_VOCAB_SIZE,
        tokenizer_identity="sha256:" + "0" * 64,
        hidden_size=V2_HIDDEN_SIZE,
        num_layers=V2_NUM_LAYERS,
        num_heads=V2_NUM_HEADS,
        ffn_size=V2_FFN_SIZE,
        context_length=32,
    )
    model = YugTransformer(config)
    v2_params = model.count_parameters()

    assert v2_params < v1_params
    # Sanity check against this module's own analytically-derived
    # figure (55,536) -- if this drifts, either the architecture or
    # the derivation changed, and that's worth knowing explicitly
    # rather than silently.
    assert v2_params == 55536


def test_v2_head_dim_is_even_for_rope():
    assert (V2_HIDDEN_SIZE // V2_NUM_HEADS) % 2 == 0


def test_run_experiment_v2_end_to_end_smoke(tmp_path, monkeypatch):
    """Runs the real V2 pipeline with max_steps patched down for fast
    test execution -- a pipeline-correctness smoke test, not a
    reproduction of the full documented experiment."""
    import training.experiments.nb08_experiment_v2 as nb08_v2

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

    monkeypatch.setattr(nb08_v2, "build_training_config", _fast_config)
    monkeypatch.setattr(nb08_v2, "CURVE_CHECKPOINTS", [0, 5, 10])

    results = run_experiment_v2(checkpoint_dir=str(tmp_path / "ckpt"))

    assert results["parameter_count"] < 221760
    assert results["num_steps"] == 10
    assert len(results["training_curve"]) == 3  # steps 0, 5, 10
    assert results["training_curve"][0][0] == 0
    assert results["training_curve"][-1][0] == 10
    for step, train_loss, val_loss in results["training_curve"]:
        assert torch.isfinite(torch.tensor(train_loss))
        assert torch.isfinite(torch.tensor(val_loss))
    assert isinstance(results["generated_text"], str)


def test_v2_checkpoint_created(tmp_path, monkeypatch):
    import training.experiments.nb08_experiment_v2 as nb08_v2

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

    monkeypatch.setattr(nb08_v2, "build_training_config", _fast_config)
    monkeypatch.setattr(nb08_v2, "CURVE_CHECKPOINTS", [0, 5])

    checkpoint_dir = tmp_path / "ckpt"
    run_experiment_v2(checkpoint_dir=str(checkpoint_dir))

    assert (checkpoint_dir / "model_config.json").exists()
    assert (checkpoint_dir / "metadata.json").exists()
    assert (checkpoint_dir / "state.pt").exists()
