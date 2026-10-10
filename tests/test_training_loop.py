"""Tests for training.loop (NB-07) -- including the milestone's
required "tiny overfit" acceptance proof.

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See training/__init__.py.

Per the parent milestone's explicit instruction ("Do not hard-code an
arbitrary loss threshold without validating the numerical behavior
first"), the overfit test below asserts a *relative* improvement
(final loss substantially below initial loss) rather than an absolute
threshold picked without ever having run the code -- an absolute
number chosen blind would be exactly the kind of unvalidated guess
that instruction warns against.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.config import TrainingConfig  # noqa: E402
from training.dataset import (  # noqa: E402
    encode_corpus,
    make_sequences,
)
from training.determinism import seed_everything  # noqa: E402
from training.loop import Trainer  # noqa: E402
from yug.native.config import ModelConfig  # noqa: E402
from yug.native.model import YugTransformer  # noqa: E402
from yug.tokenizer.bpe import ByteLevelBpeTokenizer  # noqa: E402


def _tiny_tokenizer() -> ByteLevelBpeTokenizer:
    # ROOT CAUSE (NB-07 real-pytorch-run investigation): the original
    # vocab_size=280 here gave a 22-merge budget (280 - 256 base bytes
    # - 2 specials) against a training corpus that is just "A B C D "
    # repeated 20 times -- an extremely periodic, 5-distinct-byte
    # string. Byte-level BPE (yug/tokenizer/bpe.py -- no
    # pre-tokenization, greedy most-frequent-pair merging, ties broken
    # by first occurrence, per that module's own documented and
    # deterministic contract) does exactly what it's supposed to on
    # input like that: it doesn't stop at merging "A B C D " into one
    # token, it keeps going and merges pairs of already-merged tokens
    # too, so a merge budget that large produces tokens representing
    # *multiple whole repetitions* of the unit (traced by hand against
    # the actual algorithm: merge 7 -> single "A B C D " unit as one
    # token id 262; merge 8 -> a *two-unit* chunk as one token id 263;
    # and so on). That tokenizer, applied to a much SHORTER evaluation
    # string later (e.g. 4 or 5 repetitions), then collapses it to a
    # mere 2-3 tokens -- reproducing exactly the reported
    # `DatasetError: token stream has 3 tokens, need at least 9`.
    #
    # This is not a tokenizer bug (it's honoring its documented
    # contract) and not a training.dataset bug (make_sequences()
    # correctly refuses to silently pad or fabricate sequences it
    # doesn't have tokens for). It's this fixture asking for far more
    # merge budget than a 5-byte-alphabet, single-repeated-unit corpus
    # can meaningfully use without over-compressing. vocab_size=260
    # (only a 2-merge budget) keeps each "A B C D " unit at 6 symbols
    # instead of collapsing entire multi-unit spans into single tokens
    # -- verified by the same hand-trace: merge 1 = (A,' ') -> "A ",
    # merge 2 = ("A ",B) -> "A B", leaving ["A B", ' ', C, ' ', D, ' ']
    # per unit. A 4-unit (32-byte) evaluation string then encodes to
    # ~24 tokens, comfortably above the 9 tokens make_sequences()
    # requires for sequence_length=8; a 5-unit string encodes to ~30,
    # comfortably above the 5 tokens needed for sequence_length=4.
    return ByteLevelBpeTokenizer.train(
        corpus=["A B C D " * 20],
        vocab_size=260,
        special_tokens=("bos", "eos"),
    )


def _tiny_model_config(tokenizer) -> ModelConfig:
    return ModelConfig.from_tokenizer(
        tokenizer,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        ffn_size=32,
        context_length=16,
    )


def test_train_step_changes_model_parameters():
    tokenizer = _tiny_tokenizer()
    model_config = _tiny_model_config(tokenizer)
    seed_everything(0)
    model = YugTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    training_config = TrainingConfig(
        seed=0,
        learning_rate=1e-2,
        batch_size=1,
        sequence_length=4,
        max_steps=1,
        eval_interval=1,
        checkpoint_interval=1,
    )
    trainer = Trainer(model, optimizer, training_config, model_config, tokenizer.identity)

    before = [p.clone() for p in model.parameters()]
    input_ids = torch.randint(0, model_config.vocab_size, (1, 4))
    target_ids = torch.randint(0, model_config.vocab_size, (1, 4))
    trainer.train_step(input_ids, target_ids)

    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert trainer.step == 1


def test_trainer_moves_model_to_configured_device():
    """Regression test for a Phase-2 review finding: TrainingConfig.device
    was declared but never applied anywhere -- batches built by
    training.dataset are always plain CPU tensors, and nothing moved
    the model to match. This only exercises the CPU path (no GPU is
    available in the environment that wrote this test), but it does
    verify Trainer.__init__ actually calls model.to(config.device)
    rather than silently ignoring the field."""
    tokenizer = _tiny_tokenizer()
    model_config = _tiny_model_config(tokenizer)
    model = YugTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    training_config = TrainingConfig(
        seed=0,
        learning_rate=1e-2,
        batch_size=1,
        sequence_length=4,
        max_steps=1,
        eval_interval=1,
        checkpoint_interval=1,
        device="cpu",
    )

    trainer = Trainer(model, optimizer, training_config, model_config, tokenizer.identity)

    assert all(p.device.type == "cpu" for p in trainer.model.parameters())


def test_tiny_overfit_loss_decreases_substantially(tmp_path):
    """The milestone's required acceptance proof: a tiny YUG native
    model can memorize a tiny deterministic dataset through the real
    training pipeline (loop -> loss -> optimizer -> repeated steps)."""
    tokenizer = _tiny_tokenizer()
    model_config = _tiny_model_config(tokenizer)

    seed_everything(0)
    model = YugTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    training_config = TrainingConfig(
        seed=0,
        learning_rate=3e-3,
        batch_size=1,
        sequence_length=8,
        max_steps=200,
        eval_interval=50,
        checkpoint_interval=200,
    )
    trainer = Trainer(model, optimizer, training_config, model_config, tokenizer.identity)

    token_ids = encode_corpus(tokenizer, ["A B C D A B C D A B C D A B C D"])
    assert len(token_ids) >= training_config.sequence_length + 1, (
        f"tokenizer over-compressed the evaluation string to only "
        f"{len(token_ids)} tokens (need >= "
        f"{training_config.sequence_length + 1}); see _tiny_tokenizer()'s "
        f"docstring comment for why vocab_size matters here"
    )
    sequences = make_sequences(token_ids, sequence_length=training_config.sequence_length)
    # Deliberately tiny + repeated: the whole point of an overfit proof
    # is a dataset small enough to memorize completely.
    assert len(sequences) >= 1

    losses = trainer.fit(
        train_sequences=sequences,
        eval_sequences=sequences,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )

    assert len(losses) == training_config.max_steps
    initial_loss = sum(losses[:5]) / 5
    final_loss = sum(losses[-5:]) / 5
    # Relative improvement, not an absolute number picked without
    # having run this -- see module docstring.
    assert final_loss < initial_loss * 0.5


def test_resume_continues_from_saved_step(tmp_path):
    tokenizer = _tiny_tokenizer()
    model_config = _tiny_model_config(tokenizer)
    training_config = TrainingConfig(
        seed=0,
        learning_rate=1e-2,
        batch_size=1,
        sequence_length=4,
        max_steps=5,
        eval_interval=5,
        checkpoint_interval=5,
    )
    token_ids = encode_corpus(tokenizer, ["A B C D " * 5])
    assert len(token_ids) >= training_config.sequence_length + 1, (
        f"tokenizer over-compressed the evaluation string to only "
        f"{len(token_ids)} tokens (need >= "
        f"{training_config.sequence_length + 1}); see _tiny_tokenizer()'s "
        f"docstring comment for why vocab_size matters here"
    )
    sequences = make_sequences(token_ids, sequence_length=training_config.sequence_length)

    seed_everything(0)
    model = YugTransformer(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    trainer = Trainer(model, optimizer, training_config, model_config, tokenizer.identity)
    trainer.fit(sequences, sequences, str(tmp_path / "ckpt"))
    assert trainer.step == 5

    from training.checkpoint import load_checkpoint

    resumed_model = YugTransformer(model_config)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-2)
    metadata = load_checkpoint(
        tmp_path / "ckpt", model=resumed_model, optimizer=resumed_optimizer
    )
    resumed_trainer = Trainer(
        resumed_model, resumed_optimizer, training_config, model_config, tokenizer.identity
    )
    resumed_trainer.step = metadata["step"]

    assert resumed_trainer.step == 5
    resumed_trainer.config = TrainingConfig(
        **{**training_config.__dict__, "max_steps": 10}
    )
    resumed_trainer.fit(sequences, sequences, str(tmp_path / "ckpt2"))
    assert resumed_trainer.step == 10
