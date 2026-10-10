"""NB-08 V3: data-regime-only corrective experiment.

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See ``training/__init__.py`` and
``ARCHITECTURE/NB08_DATA_REGIME_EXPERIMENT_V3.md``.

Single-variable design, same principle as V2: everything reused
UNCHANGED from V2 except the corpus. Model architecture
(hidden_size=48, num_layers=2, num_heads=4, ffn_size=96 -> 55,536
parameters, same as V2), training configuration (seed, learning rate,
batch size, steps, grad clip), sequence length, and tokenizer
vocab_size hyperparameter (384) are all identical to V2 -- imported
from ``nb08_experiment``/``nb08_experiment_v2``, not duplicated.

The ONLY changed variable is the corpus: ``nb08_v3_corpus.py`` (72
deterministic, template-generated sentences, word-coverage-verified --
see that module's docstring) replaces ``nb08_corpus.py`` (40
sentences). The tokenizer is necessarily retrained (different corpus
-> different identity hash), but its algorithm/hyperparameter
(``ByteLevelBpeTokenizer``, ``vocab_size=384``) is unchanged -- this is
documented explicitly, per this milestone's own requirement, as a
necessary and expected side effect of the declared data-regime change,
not an independent, hidden variable.
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
from training.experiments.nb08_experiment import (
    CONTEXT_LENGTH,
    SEED,
    SEQUENCE_LENGTH,
    TOKENIZER_VOCAB_SIZE,
    VAL_FRACTION,
    build_training_config,
    safe_decode_generation,
)
from training.experiments.nb08_experiment_v2 import (
    CURVE_CHECKPOINTS,
    V2_FFN_SIZE,
    V2_HIDDEN_SIZE,
    V2_NUM_HEADS,
    V2_NUM_LAYERS,
)
from training.experiments.nb08_v3_corpus import CORPUS_SENTENCES
from training.loop import Trainer
from training.split import split_sentences
from yug.native.config import ModelConfig
from yug.native.generation import generate
from yug.native.model import YugTransformer
from yug.tokenizer.bpe import ByteLevelBpeTokenizer


def run_experiment_v3(checkpoint_dir: str) -> dict:
    """Run the NB-08 V3 data-regime-only experiment. Same structure as
    :func:`training.experiments.nb08_experiment_v2.run_experiment_v2`
    (training-curve capture via direct ``Trainer.train_step()``/
    ``evaluate()`` calls, not ``Trainer.fit()`` -- see that module's
    docstring for why), applied to the V3 corpus instead of V1's.
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

    training_config = build_training_config()  # UNCHANGED from V1/V2, deliberately

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

    best_val_loss = min(val for _, _, val in training_curve)
    best_val_step = min(training_curve, key=lambda t: t[2])[0]

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
        "best_val_loss": best_val_loss,
        "best_val_step": best_val_step,
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
        results = run_experiment_v3(checkpoint_dir=tmp + "/nb08_v3_checkpoint")
        for key, value in results.items():
            print(f"{key}: {value}")
