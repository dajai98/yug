"""NB-08 V4: multi-scale, data-scale-only experiment.

VERIFICATION STATUS: written without a PyTorch installation available.
NOT executed by the assistant. See ``training/__init__.py`` and
``ARCHITECTURE/NB08_DATA_SCALE_EXPERIMENT_V4.md``.

Single independent variable across 4 scales (Directive Section 8):
training-data scale only. LOCKED, identical across every scale:

- Tokenizer: the exact ``ByteLevelBpeTokenizer`` instance trained once
  on V3's own training split (``vocab_size=384``) -- reused UNCHANGED
  (never retrained) for every scale, verified in this sandbox to
  reproduce V3's real-run identity hash exactly
  (``sha256:076c103ee402...``). Byte-level BPE can encode any input
  text using its existing merges plus a guaranteed fallback to the 256
  base-byte ids for anything unmerged -- there is no "out of
  vocabulary" failure mode to hit here, which is exactly what makes
  this lock possible without any incompatibility (see this milestone's
  own Section 10 stop-condition, which does not trigger).
- Model architecture: hidden_size=48, num_layers=2, num_heads=4,
  ffn_size=96 -> 55,536 parameters, identical to V2/V3.
- Training configuration: seed 1234, AdamW lr=3e-3, batch_size=4,
  sequence_length=16, max_steps=500 for every scale (Directive Section
  15, Option A -- fixed compute/step budget across scales, the
  standard way to isolate a pure data-scaling effect without
  confounding it with a training-budget change; a larger dataset under
  a fixed step count sees more unique data and fewer repetitions per
  epoch, which is exactly the comparison this experiment needs).
- Evaluation methodology: ``Trainer.evaluate()`` for every reported
  loss, never a raw per-step training-mode loss.
- Curve checkpoints: [0, 50, 100, 200, 300, 500], identical to V1/V2/V3.

The corpus is the only thing that changes between scales:
Scale 0 = V3's own 72-sentence corpus, unmodified.
Scale 1/2/3 = ``nb08_v4_corpus.generate_corpus()`` at increasing
``pairs_per_template`` (24/60/144 -- see that module for why V3's own
generator cannot itself produce these larger scales).
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
from training.experiments.nb08_v3_corpus import CORPUS_SENTENCES as V3_CORPUS_SENTENCES
from training.experiments.nb08_v4_corpus import SCALE_PAIRS_PER_TEMPLATE, generate_corpus
from training.loop import Trainer
from training.split import split_sentences
from yug.native.config import ModelConfig
from yug.native.generation import generate
from yug.native.model import YugTransformer
from yug.tokenizer.bpe import ByteLevelBpeTokenizer

SCALE_CORPORA = {
    "scale_0_v3_baseline": V3_CORPUS_SENTENCES,
    **{name: generate_corpus(pairs) for name, pairs in SCALE_PAIRS_PER_TEMPLATE.items()},
}


def build_locked_tokenizer() -> ByteLevelBpeTokenizer:
    """The one tokenizer used, unchanged, across every scale --
    trained on V3's own training split, exactly reproducing V3's real
    verified identity hash. Not retrained per scale (Directive Section
    10: "Do NOT retrain it. Do NOT silently regenerate it.")."""
    v3_train, _ = split_sentences(V3_CORPUS_SENTENCES, val_fraction=VAL_FRACTION, seed=SEED)
    return ByteLevelBpeTokenizer.train(
        corpus=v3_train, vocab_size=TOKENIZER_VOCAB_SIZE, special_tokens=("bos", "eos")
    )


def run_single_scale(
    scale_name: str, corpus_sentences: list[str], tokenizer: ByteLevelBpeTokenizer, checkpoint_dir: str
) -> dict:
    """Run the full pipeline for one data scale, using the shared,
    LOCKED ``tokenizer`` (never retrained here)."""
    train_sentences, val_sentences = split_sentences(
        corpus_sentences, val_fraction=VAL_FRACTION, seed=SEED
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

    training_config = build_training_config()  # UNCHANGED across every scale

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

    train_words = set(w for s in train_sentences for w in s.rstrip(".").split())
    val_words = set(w for s in val_sentences for w in s.rstrip(".").split())
    word_coverage = len(val_words & train_words) / len(val_words) if val_words else 0.0
    sentence_intersection = len(set(train_sentences) & set(val_sentences))

    return {
        "scale_name": scale_name,
        "train_sentences": len(train_sentences),
        "val_sentences": len(val_sentences),
        "unique_train_sentences": len(set(train_sentences)),
        "unique_val_sentences": len(set(val_sentences)),
        "train_val_sentence_intersection": sentence_intersection,
        "train_tokens": len(train_token_ids),
        "val_tokens": len(val_token_ids),
        "train_word_vocabulary": len(train_words),
        "validation_word_vocabulary": len(val_words),
        "word_coverage": word_coverage,
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
        "best_model_minus_unigram": best_val_loss - unigram_loss,
        "final_model_minus_unigram": final_val_loss - unigram_loss,
        "num_steps": trainer.step,
        "checkpoint_dir": checkpoint_dir,
        "generated_text": generated_text,
    }


def run_all_scales(checkpoint_root: str) -> dict[str, dict]:
    """Run every scale in :data:`SCALE_CORPORA`, in order, using one
    shared LOCKED tokenizer (built once, reused for all scales -- see
    :func:`build_locked_tokenizer`). Returns ``{scale_name: results}``.

    Per Directive Section 26 ("If the experiment is implemented as a
    multi-scale runner, it must execute ALL planned scales"), this
    function always runs every scale in ``SCALE_CORPORA``; there is no
    partial-run option.
    """
    tokenizer = build_locked_tokenizer()
    results = {}
    for scale_name, corpus_sentences in SCALE_CORPORA.items():
        results[scale_name] = run_single_scale(
            scale_name, corpus_sentences, tokenizer, f"{checkpoint_root}/{scale_name}"
        )
    return results


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        all_results = run_all_scales(checkpoint_root=tmp)
        for scale_name, results in all_results.items():
            print(f"=== {scale_name} ===")
            for key, value in results.items():
                print(f"  {key}: {value}")
            print()
