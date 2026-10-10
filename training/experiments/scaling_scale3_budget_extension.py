"""YUG Scale-3 Training-Budget Extension.

Chief-architect-authorized controlled experiment. NOT a milestone: no
NB-13 is assigned, ``ARCHITECTURE/POST_NB07_ROADMAP.md`` is not
touched by this file or its execution. Investigates whether NB-08
V4's Scale-3 result (``best_val_loss=1.4878207333385944`` at step
500 -- the *last* evaluated checkpoint) reflects genuine convergence
or premature stopping.

VERIFICATION STATUS: written and syntax-checked without a PyTorch
installation available in this sandbox (no network access to install
one). NOT executed by this assistant. See
``ARCHITECTURE/SCALING_SCALE3_TRAINING_BUDGET.md`` for what has and
hasn't been verified, and by whom.

FROZEN CONTROLS -- reused via import, never duplicated or modified:
  - scale_3 corpus: training.experiments.nb08_experiment_v4.SCALE_CORPORA["scale_3"]
  - locked tokenizer: training.experiments.nb08_experiment_v4.build_locked_tokenizer()
  - architecture: V2_HIDDEN_SIZE=48, V2_NUM_LAYERS=2, V2_NUM_HEADS=4, V2_FFN_SIZE=96
  - seed: SEED=1234 (training.experiments.nb08_experiment)
  - train/val split: VAL_FRACTION=0.2, seed=SEED (training.split.split_sentences)
  - sequence_length=16 (SEQUENCE_LENGTH), context_length=32 (CONTEXT_LENGTH)
  - optimizer: AdamW, lr=3e-3, weight_decay=0.0 (TrainingConfig's own
    default -- build_training_config() never overrides it), grad_clip_norm=1.0,
    batch_size=4 -- all copied verbatim from
    training.experiments.nb08_experiment.build_training_config()

CHANGED -- the sole independent variable:
  - max_steps: 500 -> 5000
  - checkpoint_interval: 500 -> 5000, kept equal to max_steps, matching
    build_training_config()'s own max_steps==checkpoint_interval
    relationship. Not functionally consulted by this script's manual
    training loop (which checkpoints at explicit, named steps instead
    -- see RETAINED_CHECKPOINTS below), changed here only so the
    TrainingConfig object itself doesn't carry a stale, inconsistent
    value -- not a silent, unexplained change.
  - checkpoint/evaluation schedule extended: the original
    [0, 50, 100, 200, 300, 500] (training.experiments.nb08_experiment_v2's
    CURVE_CHECKPOINTS, reused as literal step values, not by importing
    that frozen constant -- this script defines its own list so NB-08's
    own module is never touched) plus the newly-authorized
    [1000, 1500, 2000, 3000, 4000, 5000]. Preserving the original six
    points (rather than starting a fresh, disjoint schedule at 500)
    keeps the full curve comparable end-to-end against the frozen
    NB-08 V4 record, not just its tail.

WHY THIS ISN'T A CALL TO run_single_scale(): that function hardcodes
``training_config = build_training_config()`` internally with no
parameter to override ``max_steps`` -- there is no way to get a
5000-step run out of it without either modifying NB-08 V4 (forbidden
-- it is frozen) or reimplementing the loop. This script does the
latter, but not as a parallel invention: every primitive it calls
(``split_sentences``, ``encode_corpus``, ``make_sequences``,
``ModelConfig.from_tokenizer``, ``Trainer``, ``save_checkpoint``,
``uniform_baseline_loss``/``unigram_baseline_loss``,
``generate``/``safe_decode_generation``) is the exact same building
block ``run_single_scale()`` itself is built from, imported unmodified,
in the same order, with the same arguments except where explicitly
noted above.

Usage:
    python -m training.experiments.scaling_scale3_budget_extension
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

EXTENDED_CURVE_CHECKPOINTS = [0, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]
# Full model+optimizer checkpoints are only retained (saved to disk,
# integrity-checked, reloaded through NB-09's real evaluator) at these
# steps -- the dense early points above are curve data only (cheap:
# an in-memory loss number), not separately persisted artifacts.
RETAINED_CHECKPOINTS = [500, 1000, 1500, 2000, 3000, 4000, 5000]
# A small, FIXED prompt set for cross-checkpoint comparability
# (Directive: "use a small FIXED prompt set"). Deliberately a set of
# exactly one: "the" is run_single_scale()'s own exact prompt --
# reusing it, rather than inventing new prompts, keeps every sample
# directly comparable to the existing frozen NB-08 V4 generation
# record, and avoids treating this as a benchmark-design exercise.
GENERATION_CHECKPOINTS = [500, 1000, 2000, 3000, 5000]
FIXED_PROMPT = "the"
MAX_STEPS = 5000


def _check_state_dict_finite(state_dict) -> tuple[bool, str]:
    """Checks every tensor in a raw model state_dict for NaN/Inf --
    mirrors training/experiments/nb12_checkpoint_closure.py's own
    helper of the same purpose (not duplicated logic in spirit, the
    identical check, kept local here only because importing a
    function from another experiment script would be an odd coupling
    for two independent, standalone experiments)."""
    import torch

    bad = [name for name, t in state_dict.items() if not torch.isfinite(t).all()]
    if bad:
        return False, f"NaN/Inf found in checkpoint parameters: {bad}"
    return True, "all checkpoint parameters finite"


def run() -> dict:
    """Run the complete training-budget-extension experiment once and
    return a structured result dict (also used to write
    ARCHITECTURE/SCALING_SCALE3_TRAINING_BUDGET.md). Hard-stop
    conditions set ``stop_reason`` and break the training loop early;
    they do not raise -- only a genuinely unexpected crash in this
    orchestration code itself would propagate as a real exception.
    """
    import torch

    from training.checkpoint import save_checkpoint
    from training.config import TrainingConfig
    from training.dataset import encode_corpus, iter_batches, make_sequences
    from training.determinism import seed_everything
    from training.evaluate import evaluate_checkpoint
    from training.experiments.nb08_baseline import (
        unigram_baseline_loss,
        uniform_baseline_loss,
    )
    from training.experiments.nb08_experiment import (
        CONTEXT_LENGTH,
        SEED,
        SEQUENCE_LENGTH,
        VAL_FRACTION,
        safe_decode_generation,
    )
    from training.experiments.nb08_experiment_v2 import (
        V2_FFN_SIZE,
        V2_HIDDEN_SIZE,
        V2_NUM_HEADS,
        V2_NUM_LAYERS,
    )
    from training.experiments.nb08_experiment_v4 import SCALE_CORPORA, build_locked_tokenizer
    from training.loop import Trainer
    from training.split import split_sentences
    from yug.native.config import ModelConfig
    from yug.native.generation import generate
    from yug.native.model import YugTransformer

    result: dict = {
        "experiment": "scale_3_training_budget_extension",
        "not_a_milestone": True,
        "nb13_assigned": False,
        "stop_reason": None,
    }

    # -- Frozen controls: identical to run_single_scale("scale_3", ...) --
    tokenizer = build_locked_tokenizer()
    corpus = SCALE_CORPORA["scale_3"]
    train_sentences, val_sentences = split_sentences(corpus, val_fraction=VAL_FRACTION, seed=SEED)
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

    # -- The one authorized change: max_steps (see module docstring). --
    training_config = TrainingConfig(
        seed=SEED,
        learning_rate=3e-3,
        batch_size=4,
        sequence_length=SEQUENCE_LENGTH,
        max_steps=MAX_STEPS,
        eval_interval=50,
        checkpoint_interval=MAX_STEPS,
        grad_clip_norm=1.0,
        device="cpu",
    )

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

    def _current_losses() -> tuple[float, float]:
        return trainer.evaluate(train_batches), trainer.evaluate(val_batches)

    checkpoint_root = Path(tempfile.mkdtemp(prefix="scaling_scale3_"))  # never inside the repo

    training_curve: list[dict] = []
    checkpoints_saved: dict[int, dict] = {}
    generation_samples: dict[int, str] = {}
    stop_reason: str | None = None

    initial_train_loss, initial_val_loss = _current_losses()
    training_curve.append({"step": 0, "train_loss": initial_train_loss, "val_loss": initial_val_loss})
    if not (math.isfinite(initial_train_loss) and math.isfinite(initial_val_loss)):
        stop_reason = "non-finite loss at step 0 (before any training)"

    batch_idx = 0
    while stop_reason is None and trainer.step < training_config.max_steps:
        input_ids, target_ids = train_batches[batch_idx % len(train_batches)]
        step_train_loss = trainer.train_step(input_ids, target_ids)
        batch_idx += 1

        if not math.isfinite(step_train_loss):
            stop_reason = f"non-finite training-step loss at step {trainer.step}: {step_train_loss}"
            break

        if trainer.step not in EXTENDED_CURVE_CHECKPOINTS:
            continue

        train_loss, val_loss = _current_losses()
        training_curve.append({"step": trainer.step, "train_loss": train_loss, "val_loss": val_loss})

        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            stop_reason = f"non-finite loss at step {trainer.step}: train={train_loss} val={val_loss}"
            break

        if trainer.step in RETAINED_CHECKPOINTS:
            ckpt_dir = checkpoint_root / f"step_{trainer.step}"
            save_checkpoint(
                ckpt_dir,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                training_config=training_config,
                step=trainer.step,
                tokenizer_identity=tokenizer.identity,
            )
            raw_state = torch.load(ckpt_dir / "state.pt", map_location="cpu")
            state_ok, state_detail = _check_state_dict_finite(raw_state["model"])

            eval_ok = True
            eval_detail = None
            eval_loss = eval_ppl = None
            evaluation_report = None
            try:
                eval_report = evaluate_checkpoint(
                    str(ckpt_dir),
                    tokenizer,
                    val_sentences,
                    sequence_length=SEQUENCE_LENGTH,
                    batch_size=training_config.batch_size,
                    seed=SEED,
                    data_provenance={
                        "corpus_identity": None, "split_name": "validation",
                        "split_method": "training.split.split_sentences",
                        "split_seed": SEED, "split_fraction": VAL_FRACTION,
                    },
                )
                evaluation_report = eval_report.to_dict()
                eval_loss, eval_ppl = eval_report.loss, eval_report.perplexity
                if not (math.isfinite(eval_loss) and math.isfinite(eval_ppl)):
                    eval_ok = False
                    eval_detail = f"non-finite: loss={eval_loss} perplexity={eval_ppl}"
            except Exception as exc:  # noqa: BLE001 - record, don't crash the experiment
                eval_ok = False
                eval_detail = f"{type(exc).__name__}: {exc}"

            checkpoints_saved[trainer.step] = {
                "checkpoint_dir": str(ckpt_dir),
                "state_finite": state_ok,
                "state_detail": state_detail,
                "evaluation_report": evaluation_report if eval_ok else None,
                "reload_eval_success": eval_ok,
                "reload_eval_loss": eval_loss,
                "reload_eval_perplexity": eval_ppl,
                "reload_eval_detail": eval_detail,
            }
            if not state_ok or not eval_ok:
                stop_reason = (
                    f"checkpoint integrity/reload-evaluation failure at step "
                    f"{trainer.step}: state_ok={state_ok}, eval_ok={eval_ok}"
                )

        if trainer.step in GENERATION_CHECKPOINTS:
            prompt_ids = torch.tensor([encode_corpus(tokenizer, [FIXED_PROMPT])], dtype=torch.long)
            generated_ids = generate(model, prompt_ids, max_new_tokens=10)
            generation_samples[trainer.step] = safe_decode_generation(
                tokenizer, generated_ids[0].tolist()
            )

        # Hard stop: sustained overfitting -- val loss rising AND train
        # loss falling across the last 4 recorded curve points (not a
        # single blip; requires a consistent trend across 3 consecutive
        # transitions before stopping, per "do not stop merely because
        # validation loss improves slowly" / "additional checkpoints no
        # longer answer the convergence question").
        if stop_reason is None and len(training_curve) >= 4:
            last4 = training_curve[-4:]
            val_rising = all(last4[i]["val_loss"] < last4[i + 1]["val_loss"] for i in range(3))
            train_falling = all(last4[i]["train_loss"] > last4[i + 1]["train_loss"] for i in range(3))
            if val_rising and train_falling:
                stop_reason = (
                    f"sustained overfitting detected: val loss rose and train "
                    f"loss fell across the last 4 recorded checkpoints "
                    f"(steps {[c['step'] for c in last4]})"
                )

    final_train_loss_full_eval, final_val_loss = _current_losses()

    best_point = min(training_curve, key=lambda c: c["val_loss"])
    best_val_loss = best_point["val_loss"]
    best_val_step = best_point["step"]

    uniform_loss = uniform_baseline_loss(model_config.vocab_size)
    unigram_loss = unigram_baseline_loss(train_token_ids, val_token_ids)

    result.update(
        {
            "scale_name": "scale_3",
            "tokenizer_identity": tokenizer.identity,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "parameter_count": parameter_count,
            "seed": SEED,
            "model_config": {
                "hidden_size": V2_HIDDEN_SIZE,
                "num_layers": V2_NUM_LAYERS,
                "num_heads": V2_NUM_HEADS,
                "ffn_size": V2_FFN_SIZE,
                "context_length": CONTEXT_LENGTH,
                "sequence_length": SEQUENCE_LENGTH,
            },
            "training_config": {
                "learning_rate": training_config.learning_rate,
                "batch_size": training_config.batch_size,
                "max_steps": training_config.max_steps,
                "grad_clip_norm": training_config.grad_clip_norm,
                "weight_decay": training_config.weight_decay,
            },
            "train_sentences": len(train_sentences),
            "val_sentences": len(val_sentences),
            "train_tokens": len(train_token_ids),
            "val_tokens": len(val_token_ids),
            "initial_train_loss": initial_train_loss,
            "final_train_loss_full_eval": final_train_loss_full_eval,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "best_val_step": best_val_step,
            "uniform_baseline_loss": uniform_loss,
            "unigram_baseline_loss": unigram_loss,
            "training_curve": training_curve,
            "checkpoints_saved": checkpoints_saved,
            "generation_samples": generation_samples,
            "num_steps_completed": trainer.step,
            "reached_max_steps": trainer.step >= MAX_STEPS and stop_reason is None,
            "stop_reason": stop_reason,
        }
    )

    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(outcome, indent=2, default=str))
    sys.exit(0 if outcome["stop_reason"] is None or "overfitting" in (outcome["stop_reason"] or "") else 1)
