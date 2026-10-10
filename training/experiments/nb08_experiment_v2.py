"""NB-08 V2: capacity-only corrective experiment.

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See ``training/__init__.py`` and
``ARCHITECTURE/NB08_CORRECTIVE_EXPERIMENT_V2.md``.

Single-variable design (Directive Section 5: "do not simultaneously
change... unless a change is explicitly necessary"): this reuses V1's
exact corpus, split, seed, tokenizer configuration, sequence length,
and training configuration (learning rate, batch size, steps, grad
clip) UNCHANGED -- imported from ``nb08_experiment``, not duplicated.
The ONLY changed variable is model capacity (hidden_size 64->48,
num_layers 4->2, ffn_size 256->96 -- ~221,760 -> ~55,536 parameters,
derived analytically from the exact formula in
``yug.native.model.YugTransformer``'s own module docstring/comments,
matching ``yug.native.config.estimate_parameter_count()``).
Regularization (dropout=0.0, weight_decay=0.0) is deliberately left
identical to V1 -- Directive Section 7 explicitly says "prefer testing
capacity first", and changing both capacity and regularization in one
experiment would make it impossible to attribute any observed effect
to either one alone.

Unlike ``nb08_experiment.py``, this module does NOT use
``Trainer.fit()`` -- it calls ``Trainer.train_step()`` and
``Trainer.evaluate()`` directly (both already-tested, UNMODIFIED
public methods; ``training/loop.py`` itself is not touched anywhere in
this milestone) in a small custom loop, solely so the train/validation
loss trajectory at specific step checkpoints (Directive Section 12)
can be captured -- ``Trainer.fit()`` computes this internally during
its periodic-eval calls but does not currently return it. Reusing the
lower-level primitives instead of extending ``fit()`` keeps NB-07's
existing, real-machine-verified test suite completely unaffected by
this milestone.
"""

from __future__ import annotations

import torch

from training.checkpoint import save_checkpoint
from training.dataset import encode_corpus, iter_batches, make_sequences
from training.determinism import seed_everything
from training.experiments.nb08_baseline import (
    unigram_baseline_loss,
    uniform_baseline_loss,
)
from training.experiments.nb08_corpus import CORPUS_SENTENCES
from training.experiments.nb08_experiment import (
    CONTEXT_LENGTH,
    SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_VOCAB_SIZE,
    VAL_FRACTION,
    build_training_config,
    safe_decode_generation,
)
from training.loop import Trainer
from training.split import split_sentences
from yug.native.config import ModelConfig
from yug.native.generation import generate
from yug.native.model import YugTransformer
from yug.tokenizer.bpe import ByteLevelBpeTokenizer

# The training curve is captured at exactly these steps (Directive
# Section 12's own example points; unchanged since V1/V2 both use
# max_steps=500, so no adjustment is needed).
CURVE_CHECKPOINTS = [0, 50, 100, 200, 300, 500]

# Derived analytically (see this module's docstring) from the exact
# formula embedded in yug.native.model's own comments -- NOT a guess:
# vocab_size (384, fixed, same tokenizer as V1) x hidden_size, tied
# embeddings, plus num_layers x (2*hidden_size [two RMSNorm instances]
# + 4*hidden_size^2 [Q/K/V/O projections, bias=False] +
# 2*hidden_size*ffn_size [silu, 2-projection FFN, bias=False]), plus
# one final hidden_size-sized RMSNorm. = 55,536 parameters -- to be
# confirmed by YugTransformer.count_parameters() when actually run.
V2_HIDDEN_SIZE = 48
V2_NUM_LAYERS = 2
V2_NUM_HEADS = 4  # head_dim = 48/4 = 12, even -> RoPE-compatible
V2_FFN_SIZE = 96


def run_experiment_v2(checkpoint_dir: str) -> dict:
    """Run the NB-08 V2 capacity-only experiment and return a results
    dict, including a ``training_curve`` field:
    ``list[tuple[step, train_loss_full_eval, val_loss]]`` at
    :data:`CURVE_CHECKPOINTS`.
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
        hidden_size=V2_HIDDEN_SIZE,
        num_layers=V2_NUM_LAYERS,
        num_heads=V2_NUM_HEADS,
        ffn_size=V2_FFN_SIZE,
        context_length=CONTEXT_LENGTH,
    )

    training_config = build_training_config()  # UNCHANGED from V1, deliberately

    seed_everything(SEED)
    model = YugTransformer(model_config)
    parameter_count = model.count_parameters()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    trainer = Trainer(model, optimizer, training_config, model_config, tokenizer.identity)

    train_batches = iter_batches(train_sequences, training_config.batch_size)
    val_batches = iter_batches(val_sequences, training_config.batch_size)

    def _current_train_val_loss() -> tuple[float, float]:
        # Same evaluate() methodology for both, so the two numbers at
        # any given checkpoint are directly comparable to each other
        # (the exact comparability issue found and fixed in V1 -- see
        # ARCHITECTURE/NB08_FAILURE_ANALYSIS.md, Section D/G).
        return trainer.evaluate(train_batches), trainer.evaluate(val_batches)

    training_curve: list[tuple[int, float, float]] = []
    initial_train_loss, initial_val_loss = _current_train_val_loss()
    training_curve.append((0, initial_train_loss, initial_val_loss))

    batch_idx = 0
    while trainer.step < training_config.max_steps:
        input_ids, target_ids = train_batches[batch_idx % len(train_batches)]
        trainer.train_step(input_ids, target_ids)
        batch_idx += 1

        if trainer.step in CURVE_CHECKPOINTS:
            train_loss, val_loss = _current_train_val_loss()
            training_curve.append((trainer.step, train_loss, val_loss))

    final_train_loss_full_eval, final_val_loss = _current_train_val_loss()

    save_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        training_config=training_config,
        step=trainer.step,
        tokenizer_identity=tokenizer.identity,
    )

    uniform_loss = uniform_baseline_loss(model_config.vocab_size)
    unigram_loss = unigram_baseline_loss(train_token_ids, val_token_ids)

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
        "final_train_loss_full_eval": final_train_loss_full_eval,
        "final_val_loss": final_val_loss,
        "training_curve": training_curve,
        "uniform_baseline_loss": uniform_loss,
        "unigram_baseline_loss": unigram_loss,
        "num_steps": trainer.step,
        "checkpoint_dir": checkpoint_dir,
        "generated_text": generated_text,
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        results = run_experiment_v2(checkpoint_dir=tmp + "/nb08_v2_checkpoint")
        for key, value in results.items():
            print(f"{key}: {value}")
