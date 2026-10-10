# NB-08 — Controlled Small-Scale Language-Modeling Experiment

**Status: PARTIALLY VERIFIED. Dataset/tokenizer/split/baseline stages
[VERIFIED BY TEST] (genuinely executed, no PyTorch needed). Model
training/validation/checkpoint/generation stages [UNKNOWN] — written,
not executed (no PyTorch available in the sandbox that wrote this).**

## 1. Objective

Answer the question NB-07 deliberately left open: can YUG's native
Transformer learn something beyond pure memorization of a single
repeated toy pattern, in a controlled, reproducible, measurable
experiment?

## 2. Hypothesis

Training on a small but lexically/structurally varied corpus, with a
genuine held-out validation split, will produce a validation loss that
beats both a uniform baseline and a unigram-frequency baseline — a
concrete, falsifiable claim, not "the model seems to learn something."

## 3. Dataset

`training/experiments/nb08_corpus.py` — 40 original, self-contained
English sentences, written for this experiment (no external source, no
copyright question, no network dependency). Deliberately varied in
vocabulary and structure — the opposite of NB-07's single-repeated-unit
"A B C D" corpus, which is exactly what made NB-07's tokenizer-fixture
bug possible (see the NB-07 root-cause investigation in this
project's history).

**[VERIFIED BY TEST]** raw corpus size: 1481 bytes / 1481 characters,
40 sentences, all distinct.

## 4. Train/Validation Split

`training/split.py`'s `split_sentences()` — a new, small, torch-free
module (mirroring `training/config.py`/`errors.py`'s existing
precedent of keeping torch-free code separately testable). Splits at
the **sentence** level, before tokenization, specifically to let the
tokenizer be trained only on the training sentences (see Section 5) —
this is a deliberate leakage-avoidance decision, documented in the
module's own docstring.

**[VERIFIED BY TEST]** (11 real, executed tests in
`tests/test_training_split.py`): deterministic for a fixed seed,
different for different seeds, train/val disjoint and covering all 40
sentences, does not mutate global `random` state (so it cannot
interfere with `training.determinism.seed_everything`).

With `val_fraction=0.2`, `seed=1234`: **32 train sentences, 8
validation sentences** — computed by actually running the real
function, not estimated.

## 5. Tokenizer

`yug.tokenizer.bpe.ByteLevelBpeTokenizer` (existing, NB-02) — no
tokenizer was created or modified for this experiment. Trained **only
on the 32 training sentences** (not the full 40), specifically to keep
validation genuinely held-out at the tokenizer level, not just the
model level (see `training/split.py`'s docstring for the full
reasoning).

**[VERIFIED BY TEST]** (real tokenizer, run directly in the sandbox
that wrote this document — the tokenizer module has zero PyTorch
dependency, confirmed by inspecting its imports):

```
vocab_size:        384  (256 base bytes + 126 merges + 2 special tokens)
tokenizer identity: sha256:ac40c48e125c1fd077563e116c813a71c10d6b8da65c916c45e1f5aec757a26e
train tokens:       463
val tokens:         152
```

Both comfortably exceed `SEQUENCE_LENGTH + 1 = 17` — the exact failure
mode NB-07's root-cause investigation uncovered (over-compression from
too large a merge budget relative to corpus repetitiveness) was
checked for directly rather than assumed away: at `vocab_size=384`,
merge budget saturates well before over-compressing this genuinely
varied corpus (confirmed empirically: `vocab_size=448` and `512`
produced the *same* total token count as `384`, meaning no further
useful merges were available — the corpus isn't repetitive enough to
over-compress the way NB-07's was).

## 6. Model Configuration

`yug.native.config.ModelConfig.from_tokenizer(tokenizer, ...)` (existing,
NB-03) — vocabulary size and tokenizer identity pulled directly from
the trained tokenizer, not hand-copied.

```
vocab_size:        384        (from tokenizer)
hidden_size:        64
num_layers:          4
num_heads:            4
ffn_size:           256
context_length:      32        (headroom above sequence_length=16, for the generation check)
normalization:  rmsnorm        (ModelConfig default)
positional_encoding: rope      (ModelConfig default)
activation:        silu        (ModelConfig default)
bias:              False        (ModelConfig default)
tie_embeddings:     True        (ModelConfig default)
```

**Parameter count: [UNKNOWN]** — requires constructing `YugTransformer`,
which requires PyTorch. Not fabricated.

## 7. Training Configuration

```
seed:                1234   (same seed as the train/val split, for full-pipeline reproducibility)
learning_rate:      3e-3
batch_size:            4
sequence_length:      16
max_steps:           500
eval_interval:        50
checkpoint_interval:  500
grad_clip_norm:      1.0
weight_decay:        0.0
device:              cpu
```

`batch_size=4` against 27 training sequences (`463 // 17`) and 8
validation sequences (`152 // 17`) gives ~7 batches per training epoch
and 2 validation batches — small numbers by design, appropriate for a
CPU-development-scale experiment, not tuned for benchmark performance.

## 8. Determinism Configuration

`training.determinism.seed_everything(1234)`, called once, immediately
before model construction (matching the documented requirement that
seeding must precede construction to affect weight initialization).
Per that module's own documented scope limit: this seeds Python's
`random` and PyTorch's CPU/CUDA RNGs, but does **not** force
`torch.use_deterministic_algorithms(True)` — no claim of bit-exact
cross-hardware reproducibility is made here.

## 9. Baseline

`training/experiments/nb08_baseline.py` — two torch-free baselines,
**[VERIFIED BY TEST]** (5 real, executed tests in
`tests/test_nb08_baseline.py`), computed against the actual corpus:

```
uniform baseline loss (ln(384)):                5.9506
unigram baseline loss (train freq -> val):       4.7020
unigram baseline loss (train freq -> train, sanity check): 4.7187
```

The unigram baseline beats the uniform baseline, as expected for
real (non-uniform-frequency) text — this validates the baseline
module itself makes sense before it's used to judge the trained
model. **A trained model's validation loss must beat 4.7020 (the
unigram baseline), not merely 5.9506 (uniform), to count as genuine
evidence of learning beyond word frequency** — this is the acceptance
threshold this experiment will be judged against, decided from real
computed numbers, not guessed in advance.

## 10. Training Results

**[UNKNOWN]** — requires PyTorch execution. Not fabricated.

## 11. Validation Results

**[UNKNOWN]** — requires PyTorch execution. Not fabricated.

## 12. Checkpoint Result

**[UNKNOWN]** — requires PyTorch execution. The pipeline calls the
existing `training.checkpoint` mechanism (via `Trainer.fit`'s own
`checkpoint_interval`) unmodified; no new checkpoint logic was written
for this experiment.

## 13. Generation Result

**[UNKNOWN]** — requires PyTorch execution. The pipeline includes a
qualitative-only check (`yug.native.generation.generate`, unmodified,
prompted with the single word "the", 10 new tokens, greedy decoding)
— its output will be recorded but not judged for "intelligence," per
this milestone's explicit scope limit.

## 14. Interpretation

Cannot be written honestly until Sections 10–13 have real numbers.
Placeholder, to be filled in after real execution:

> [PENDING] Given final validation loss of ___, compared to the
> unigram baseline of 4.7020: [beats / does not beat] the baseline by
> ___. Training loss trajectory: ___. Verdict: ___.

## 15. Limitations

- Corpus is small (40 sentences, ~1.5KB) — this experiment answers
  "does the pipeline show a generalization signal at all," not
  "how good is YUG's native brain at language modeling" — the latter
  question is explicitly out of scope (reserved for later,
  larger-scale, evidence-justified milestones per
  `ARCHITECTURE/POST_NB07_ROADMAP.md`'s deferred tracks).
- No cross-hardware determinism claim (Section 8).
- The generation check is qualitative only, not a quantitative metric.
- Baseline computations use Laplace (add-one) smoothing for unseen
  tokens — a standard, simple choice, not extensively justified beyond
  "handles zero-frequency tokens without dividing by zero."

## 16. Acceptance Criteria

Per Section 11 of the NB-08 implementation prompt:
- **A. Dataset correctness** — [VERIFIED BY TEST], Section 4/5 above.
- **B. Model correctness** — [UNKNOWN], pending execution.
- **C. Training correctness** — [UNKNOWN], pending execution.
- **D. Generalization signal** — [UNKNOWN]; the threshold to judge it
  against (unigram baseline = 4.7020) is real and computed now, not
  invented after seeing results, per the "do not invent a universal
  numerical threshold before observing the experiment" instruction —
  this baseline **is** the observation the threshold is derived from,
  computed honestly before any model has been trained.
- **E. Reproducibility** — seed and full configuration recorded above;
  actual repeated-run consistency [UNKNOWN], pending execution.

## 17. Final Verdict

**BLOCKED** (for the execution-dependent portions) — same environment
constraint as every prior milestone in this project: no PyTorch, no
real pytest, no network, in the sandbox that implemented this. The
dataset/tokenizer/split/baseline infrastructure is genuinely complete
and verified. The experiment itself has never been run.

**Exact commands to run in a real PyTorch + pytest environment:**

```bash
python -m pytest tests/test_training_split.py tests/test_nb08_baseline.py -q   # already known-good
python -m pytest tests/test_nb08_experiment.py -q
python -m training.experiments.nb08_experiment    # the full, real-scale (max_steps=500) run
python -m pytest tests -q
python -m compileall -q yug training tests
```

The full-scale run's printed output should be transcribed back into
Sections 10–14 above, replacing every `[UNKNOWN]`/`[PENDING]` with the
real numbers, before NB-08 can be marked PASS.
