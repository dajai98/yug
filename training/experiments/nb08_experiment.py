"""NB-08: Controlled Small-Scale Language-Modeling Experiment -- entry point.

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See ``training/__init__.py`` and
``ARCHITECTURE/NB08_CONTROLLED_LM_EXPERIMENT.md``.

Run with:

    python -m training.experiments.nb08_experiment

or import :func:`run_experiment` directly (used by
``tests/test_nb08_experiment.py``).

Pipeline (reuses every existing NB-07 component; nothing here
duplicates model/training logic -- see this milestone's own
"architectural principle" requirement):

    training.experiments.nb08_corpus.CORPUS_SENTENCES
        -> training.split.split_sentences        (train/val sentences)
        -> yug.tokenizer.bpe.ByteLevelBpeTokenizer.train  (on TRAIN sentences only)
        -> training.dataset.encode_corpus          (both splits, separately)
        -> training.dataset.make_sequences          (both splits, separately)
        -> yug.native.config.ModelConfig.from_tokenizer
        -> yug.native.model.YugTransformer
        -> training.determinism.seed_everything
        -> training.loop.Trainer.fit
        -> training.checkpoint (via Trainer.fit's own checkpoint_interval)
        -> yug.native.generation.generate            (qualitative check only)
"""

from __future__ import annotations

import torch

from training.config import TrainingConfig
from training.dataset import encode_corpus, iter_batches, make_sequences
from training.determinism import seed_everything
from training.experiments.nb08_baseline import (
    unigram_baseline_loss,
    uniform_baseline_loss,
)
from training.experiments.nb08_corpus import CORPUS_SENTENCES
from training.loop import Trainer
from training.split import split_sentences
from yug.native.config import ModelConfig
from yug.native.generation import generate
from yug.native.model import YugTransformer
from yug.tokenizer.bpe import ByteLevelBpeTokenizer
from yug.tokenizer.errors import DecodeError

# All of these were verified against the REAL corpus/tokenizer in the
# sandbox that wrote this experiment (no torch needed for that part --
# see ARCHITECTURE/NB08_CONTROLLED_LM_EXPERIMENT.md for the exact
# numbers): 32 train / 8 val sentences, 463 train tokens / 152 val
# tokens at this vocab_size, comfortable margin above
# SEQUENCE_LENGTH + 1 = 17 for both splits.
SEED = 1234
VAL_FRACTION = 0.2
TOKENIZER_VOCAB_SIZE = 384
SEQUENCE_LENGTH = 16
CONTEXT_LENGTH = 32  # headroom above SEQUENCE_LENGTH, for the generation check


def build_training_config() -> TrainingConfig:
    return TrainingConfig(
        seed=SEED,
        learning_rate=3e-3,
        batch_size=4,
        sequence_length=SEQUENCE_LENGTH,
        max_steps=500,
        eval_interval=50,
        checkpoint_interval=500,
        grad_clip_norm=1.0,
        device="cpu",
    )


def safe_decode_generation(tokenizer: ByteLevelBpeTokenizer, token_ids: list[int]) -> str:
    """Decode a model-generated token-id sequence, tolerating the
    tokenizer's own documented failure mode for this exact situation.

    Found during the NB-08 V2 real-environment run: greedy decoding
    from a small/undertrained model has no notion of "valid UTF-8
    byte-boundary" -- it can legitimately land on a token sequence
    whose underlying bytes do not reconstitute valid UTF-8 (e.g. a
    token that represents only the middle byte of a multi-byte
    character, followed by a token that doesn't complete it).
    ``ByteLevelBpeTokenizer.decode()`` correctly and deliberately
    raises ``DecodeError`` for this rather than silently returning
    corrupted text (see its own docstring/implementation) -- that is
    correct, intentional tokenizer behavior, not a bug to work around
    in the tokenizer itself. The bug was here: this experiment's
    generation check is explicitly "supplementary evidence only" (this
    milestone's own Section 14), so a failure in it must never prevent
    the primary evidence (loss curves, baselines, checkpoint) from
    being computed and reported.
    """
    try:
        return tokenizer.decode(token_ids)
    except DecodeError as exc:
        return f"[DecodeError: generated token ids did not decode to valid UTF-8: {exc}]"


def run_experiment(checkpoint_dir: str) -> dict:
    """Run the full NB-08 pipeline once and return a results dict
    (all the fields required by this milestone's "[VERIFIED BY
    EXPERIMENT]" report section). Does not print anything -- the
    ``__main__`` block below handles reporting; this function is also
    what ``tests/test_nb08_experiment.py`` calls directly.
    """
    train_sentences, val_sentences = split_sentences(
        CORPUS_SENTENCES, val_fraction=VAL_FRACTION, seed=SEED
    )

    tokenizer = ByteLevelBpeTokenizer.train(
        corpus=train_sentences,
        vocab_size=TOKENIZER_VOCAB_SIZE,
        special_tokens=("bos", "eos"),
    )

    train_token_ids = encode_corpus(tokenizer, train_sentences)
    val_token_ids = encode_corpus(tokenizer, val_sentences)

    train_sequences = make_sequences(train_token_ids, sequence_length=SEQUENCE_LENGTH)
    val_sequences = make_sequences(val_token_ids, sequence_length=SEQUENCE_LENGTH)

    model_config = ModelConfig.from_tokenizer(
        tokenizer,
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        ffn_size=256,
        context_length=CONTEXT_LENGTH,
    )

    training_config = build_training_config()

    seed_everything(SEED)
    model = YugTransformer(model_config)
    parameter_count = model.count_parameters()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    trainer = Trainer(model, optimizer, training_config, model_config, tokenizer.identity)

    initial_train_loss = trainer.evaluate(
        iter_batches(train_sequences, training_config.batch_size)
    )

    losses = trainer.fit(
        train_sequences=train_sequences,
        eval_sequences=val_sequences,
        checkpoint_dir=checkpoint_dir,
    )

    final_val_loss = trainer.evaluate(iter_batches(val_sequences, training_config.batch_size))
    # Directly comparable to initial_train_loss and final_val_loss --
    # all three computed the exact same way (Trainer.evaluate(): full
    # pass over the named sequences, eval mode, no gradient) rather
    # than mixing this with the raw per-step training-mode losses in
    # `losses` (each computed on a single mini-batch, in train() mode,
    # during optimization) -- a real methodological inconsistency
    # found during the NB-08 failure analysis: those two numbers are
    # not measuring the same thing, even though both were previously
    # reported as "train loss". See
    # ARCHITECTURE/NB08_FAILURE_ANALYSIS.md, Section D.
    final_train_loss_full_eval = trainer.evaluate(
        iter_batches(train_sequences, training_config.batch_size)
    )

    uniform_loss = uniform_baseline_loss(model_config.vocab_size)
    unigram_loss = unigram_baseline_loss(train_token_ids, val_token_ids)

    # Qualitative generation check only -- not an NB-10 adapter
    # integration. A short, arbitrary in-vocabulary prompt; sample
    # output is recorded, not judged for "intelligence" (explicitly
    # out of scope -- see this module's own docstring and the
    # milestone prompt's Section 13).
    prompt_ids = torch.tensor([encode_corpus(tokenizer, ["the"])], dtype=torch.long)
    generated_ids = generate(model, prompt_ids, max_new_tokens=10)
    generated_text = safe_decode_generation(tokenizer, generated_ids[0].tolist())

    return {
        "train_sentences": len(train_sentences),
        "val_sentences": len(val_sentences),
        "train_tokens": len(train_token_ids),
        "val_tokens": len(val_token_ids),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_identity": tokenizer.identity,
        "parameter_count": parameter_count,
        "seed": SEED,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": sum(losses[-5:]) / min(5, len(losses)),
        "final_train_loss_full_eval": final_train_loss_full_eval,
        "training_losses": losses,
        "final_val_loss": final_val_loss,
        "uniform_baseline_loss": uniform_loss,
        "unigram_baseline_loss": unigram_loss,
        "num_steps": len(losses),
        "checkpoint_dir": checkpoint_dir,
        "generated_text": generated_text,
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        results = run_experiment(checkpoint_dir=tmp + "/nb08_checkpoint")
        for key, value in results.items():
            if key == "training_losses":
                continue  # too long for a one-line print
            print(f"{key}: {value}")
