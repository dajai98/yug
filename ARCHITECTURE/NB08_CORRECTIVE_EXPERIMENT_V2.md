# NB-08 — Corrective Experiment V2: Capacity-Only Intervention

**Status: BLOCKED (execution). Design, implementation, and static
verification complete; no PyTorch available in this sandbox to
actually run it.**

## Executive Summary

V2 tests exactly one hypothesis: that V1's catastrophic
validation-loss gap was caused by model capacity vastly exceeding
available training data, not by an implementation bug, a leaked split,
or a tokenizer problem (all directly ruled out in
`ARCHITECTURE/NB08_FAILURE_ANALYSIS.md`). V2 changes **only** model
capacity — same corpus, same split, same seed, same tokenizer, same
training configuration as V1 — so that any change in outcome can be
attributed to capacity alone.

## V1 Failure (preserved, unchanged)

```
parameter_count: 221760
train_tokens: 463    val_tokens: 152

initial_train_loss: 56.32944924490793
final_train_loss_full_eval: 0.00653181677937807
final_val_loss: 15.98197317123413

uniform_baseline_loss: 5.950642552587727
unigram_baseline_loss: 4.701993336814625

Verdict: FAIL — EXPERIMENT DESIGN
Root cause: ~479 parameters per training token, dropout=0.0,
weight_decay=0.0, 500 unregularized optimization steps.
```

## Hypothesis

If capacity was the primary cause, a model with substantially fewer
parameters (~50k–100k, down from 221,760), trained identically
otherwise, should show a smaller train/validation gap and — the
concrete, falsifiable claim — a validation loss closer to (ideally
below) the unigram baseline of 4.7020, without requiring any change to
data, split, or regularization.

If validation loss remains far above baseline even at much lower
capacity, that would point away from "just capacity" and toward either
a more fundamental limitation of this tiny corpus, or a limitation of
the current architecture/training recipe that regularization (not
tested here, deliberately — see Limitations) might address instead.

## Experimental Design

**[VERIFIED IN REPOSITORY]** — everything below except model
architecture is imported, unmodified, from `nb08_experiment.py`:
corpus (`nb08_corpus.CORPUS_SENTENCES`, 40 sentences), split
(`split_sentences`, `seed=1234`, `val_fraction=0.2`, giving the same
32 train / 8 val sentences), tokenizer configuration
(`TOKENIZER_VOCAB_SIZE=384`, trained only on the 32 training sentences,
same leakage-avoidance reasoning as V1), sequence length (16), and
training configuration (`build_training_config()` — learning rate
3e-3, batch size 4, max_steps 500, eval_interval 50,
checkpoint_interval 500, grad_clip_norm 1.0, device cpu — all
identical to V1, unchanged).

**Changed (the one variable)** — model capacity, derived analytically
from the exact parameter-count formula embedded in
`yug/native/model.py`'s own module comments (matching
`yug.native.config.estimate_parameter_count()`):

```
V1: hidden_size=64, num_layers=4, num_heads=4, ffn_size=256  -> 221,760 params
V2: hidden_size=48, num_layers=2, num_heads=4, ffn_size=96   ->  55,536 params (derived, not yet executed)
```

`head_dim = 48/4 = 12`, even — RoPE-compatible, confirmed against
`YugTransformer`'s own validation logic. `vocab_size=384` is identical
to V1 (same tokenizer), so the embedding table itself (`384×48 =
18,432`, tied with the output head) is the single largest component,
same as in V1's architecture.

**Regularization**: deliberately **unchanged** from V1 (`dropout=0.0`,
`weight_decay=0.0`) — per this milestone's own "prefer testing capacity
first" instruction and to preserve single-variable interpretability.
Combining a capacity change with a regularization change in one
experiment would make it impossible to attribute any observed effect
to either change alone.

## Training Curve

**[UNKNOWN]** — requires execution. Captured at steps `[0, 50, 100,
200, 300, 500]` via `training.experiments.nb08_experiment_v2`'s own
loop (calls `Trainer.train_step()`/`Trainer.evaluate()` directly,
`training/loop.py` itself untouched — see the module's docstring for
why this approach was chosen over extending `Trainer.fit()`).

## Baseline Comparison

**[VERIFIED IN REPOSITORY]** — identical to V1, since the corpus,
split, and tokenizer are unchanged: `uniform_baseline_loss = 5.9506`,
`unigram_baseline_loss = 4.7020`. V2's `final_val_loss` (once measured)
must beat 4.7020 to count as genuine evidence of learning beyond word
frequency — the same bar V1 failed to clear.

## Generalization Analysis

**[UNKNOWN]** — pending execution. To be filled in with the actual
training curve and final numbers.

## V1 vs V2

| Metric | NB-08 V1 | NB-08 V2 |
| --- | ---: | ---: |
| Parameters | 221760 | 55536 (derived, unexecuted) |
| Train tokens | 463 | 463 (unchanged corpus/split) |
| Val tokens | 152 | 152 (unchanged corpus/split) |
| Initial train loss | 56.3294 | UNKNOWN |
| Final train loss (full eval) | 0.00653 | UNKNOWN |
| Final val loss | 15.9820 | UNKNOWN |
| Unigram baseline | 4.7020 | 4.7020 (unchanged) |

## Limitations

- Corpus remains tiny (40 sentences) — even a "successful" V2 would
  only show that capacity reduction helps at this scale, not that the
  architecture generalizes well in general.
- Regularization was deliberately not tested this round — if V2 still
  fails, the natural next single-variable experiment is
  dropout/weight_decay at V1's or V2's capacity, not a combined
  capacity+regularization change (which this experiment's own
  single-variable principle argues against doing in one step).
- The training-curve capture required bypassing `Trainer.fit()`'s
  convenience wrapper in favor of its lower-level methods — functionally
  equivalent (same `train_step`/`evaluate` calls, same batch-cycling
  logic `Trainer.fit()` itself uses), but not literally
  `Trainer.fit()` itself, worth noting for anyone comparing the two
  experiment scripts side by side.

## Decision

**BLOCKED**

Same environment constraint as every milestone before it in this
project: no PyTorch, no real pytest, in this sandbox. The experiment
is fully designed, implemented, and statically verified (syntax-checked,
architecture-derived parameter count math double-checked, a real test
bug in the assistant's own test file found and fixed — see below); it
has never been run.

**Exact commands for the real environment:**

```bash
python -m pytest tests/test_nb08_experiment_v2.py -q
python -m training.experiments.nb08_experiment_v2
python -m pytest tests -q
python -m compileall -q yug training tests
```

Once run, replace every `[UNKNOWN]` above with the real numbers before
this document can carry a PASS/FAIL-GENERALIZATION verdict.
