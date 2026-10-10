"""Tests for training.checkpoint (NB-07).

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See training/__init__.py.
"""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from training.config import TrainingConfig  # noqa: E402
from training.errors import CheckpointError  # noqa: E402
from yug.native.config import ModelConfig  # noqa: E402
from yug.native.model import YugTransformer  # noqa: E402

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64


def _tiny_config(**overrides) -> ModelConfig:
    kwargs = dict(
        vocab_size=32,
        tokenizer_identity=_PLACEHOLDER_IDENTITY,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        ffn_size=16,
        context_length=16,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        seed=0,
        learning_rate=1e-2,
        batch_size=2,
        sequence_length=4,
        max_steps=10,
        eval_interval=5,
        checkpoint_interval=5,
    )


def test_save_then_load_restores_model_state_exactly(tmp_path):
    model_config = _tiny_config()
    model = YugTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    save_checkpoint(
        tmp_path / "ckpt",
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        training_config=_training_config(),
        step=3,
        tokenizer_identity=_PLACEHOLDER_IDENTITY,
    )

    # Mutate the live model so we can tell load actually restored it,
    # not merely that it was already correct.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    loaded_config = ModelConfig.load(tmp_path / "ckpt" / "model_config.json")
    restored_model = YugTransformer(loaded_config)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-2)

    metadata = load_checkpoint(
        tmp_path / "ckpt", model=restored_model, optimizer=restored_optimizer
    )

    assert metadata["step"] == 3
    assert metadata["tokenizer_identity"] == _PLACEHOLDER_IDENTITY
    for name, tensor in restored_model.state_dict().items():
        assert torch.equal(tensor, original_state[name])


def test_mismatched_tokenizer_identity_rejected(tmp_path):
    model_config = _tiny_config()
    model = YugTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    with pytest.raises(CheckpointError):
        save_checkpoint(
            tmp_path / "ckpt",
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            training_config=_training_config(),
            step=0,
            tokenizer_identity="sha256:" + "1" * 64,  # deliberately mismatched
        )


def test_load_missing_checkpoint_raises_checkpoint_error(tmp_path):
    model_config = _tiny_config()
    model = YugTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "does_not_exist", model=model, optimizer=optimizer)


def test_random_state_round_trips(tmp_path):
    model_config = _tiny_config()
    model = YugTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    random.seed(42)
    torch.manual_seed(42)
    random.random()  # advance state so it's not the fresh-seed state
    torch.rand(1)

    expected_random_state = random.getstate()
    expected_torch_state = torch.get_rng_state().clone()

    save_checkpoint(
        tmp_path / "ckpt",
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        training_config=_training_config(),
        step=0,
        tokenizer_identity=_PLACEHOLDER_IDENTITY,
    )

    # Disturb both RNG states before reloading.
    random.seed(999)
    torch.manual_seed(999)

    load_checkpoint(
        tmp_path / "ckpt", model=model, optimizer=optimizer, restore_rng_state=True
    )

    assert random.getstate() == expected_random_state
    assert torch.equal(torch.get_rng_state(), expected_torch_state)
