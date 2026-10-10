# NB-08 — Failure Analysis: Memorization Without Generalization

**Status: FAIL — EXPERIMENT DESIGN (not an implementation bug).**

## A. Executive Summary

The reported NB-08 result (train loss 56.33 → 0.0101; validation loss
15.98, worse than the unigram baseline's 4.70) is **not caused by a
bug** in the tokenizer, dataset pipeline, model architecture, loss
function, or training loop. Every one of those was checked directly
against real evidence and found correct. The cause is a genuine,
severe **capacity/data mismatch with zero active regularization**:
221,760 trainable parameters against only 463 training tokens (~479
parameters per training token), with `dropout=0.0` and
`weight_decay=0.0` throughout. A model this oversized, trained this
long, with nothing to discourage memorization, memorized the specific
training sequences rather than learning generalizable structure — and
because it memorized with high confidence, it makes sharp, wrong
predictions on validation sequences instead of hedging the way even a
crude unigram-frequency model does. This is why validation loss is
*worse* than the naive baseline, not merely "not as good as hoped."

While investigating, one real (if minor, non-root-cause) methodological
issue in the assistant's own `nb08_experiment.py` was found and fixed:
`initial_train_loss` and `final_val_loss` were computed via
`Trainer.evaluate()` (full-corpus, eval-mode, no-grad), but
`final_train_loss` was the average of raw per-step training-mode
losses — not the same measurement, despite both being labeled "train
loss." A `final_train_loss_full_eval` field was added, computed the
same way as the other two, for a genuinely apples-to-apples comparison
in future runs.

## B. Evidence

Exact reported numbers, treated as authoritative per instruction (not
re-derived, not reinterpreted):

```
train_sentences: 32          val_sentences: 8
train_tokens: 463             val_tokens: 152
tokenizer_vocab_size: 384
parameter_count: 221760
seed: 1234

initial_train_loss: 56.32944924490793
final_train_loss:    0.010122571072133723   (raw per-step training-mode losses, see Section D)
final_val_loss:      15.98197317123413

uniform_baseline_loss:  5.950642552587727
unigram_baseline_loss:  4.701993336814625

num_steps: 500
generated_text: "theared in the greenmoran  flow"
```

Full suite (real pytest, reported): `1114 passed, 2 skipped in 26.09s`.

## C. Data Path Audit

**[VERIFIED BY TEST]**, real execution in this investigation (the
tokenizer and `training.split` have zero PyTorch dependency, confirmed
by inspecting their imports — this was genuinely run, not guessed):

```
sentence -> ByteLevelBpeTokenizer.encode() -> flat token-id list
          -> training.dataset.encode_corpus (concatenates per-sentence, unchanged)
          -> training.dataset.make_sequences (chunks into sequence_length+1 pieces, unchanged)
          -> training.dataset.sequences_to_batch (input=seq[:-1], target=seq[1:])
```

**[VERIFIED IN REPOSITORY]** — `sequences_to_batch`'s shift logic
(`training/dataset.py`) was already reviewed line-by-line during NB-07
and matches the standard causal-LM shift exactly: `input_ids[i] =
sequences[i][:-1]`, `target_ids[i] = sequences[i][1:]`. Not modified
this turn; re-confirmed by re-reading the source, not re-executed
(would need torch).

**[VERIFIED BY TEST]** vocabulary validity, run directly this turn:
every validation token id is in `[0, 384)` — confirmed by exhaustive
check (`max val id: 380 < vocab_size: 384`), not sampled.

## D. Validation Path Audit

**[VERIFIED IN REPOSITORY]** `Trainer.evaluate()` (`training/loop.py`):
sets `model.eval()`, runs `torch.no_grad()`, computes
`causal_lm_loss()` per batch, averages across batches, restores the
model's prior training-mode afterward. Symmetric with how training
batches are constructed — no separate/divergent code path for
validation versus training data; the same `causal_lm_loss` function is
used for both, unchanged.

**[VERIFIED IN REPOSITORY]** — a genuine measurement inconsistency
found this turn: `initial_train_loss` (computed via `evaluate()`, full
corpus, eval mode) and `final_val_loss` (same) are directly comparable
to each other, but `final_train_loss` (the average of the last 5 raw
per-step `train_step()` losses, each computed on one mini-batch, in
`train()` mode, during active optimization) is **not** the same
measurement. This does not explain the reported *validation* failure
(nothing about it affects `final_val_loss`'s correctness), but it did
mean "initial train loss" and "final train loss" were never really
comparable to each other in the way the report implied. **Fixed** —
see Section G.

**[VERIFIED IN REPOSITORY]** — `Trainer.evaluate()`'s `total_loss /
len(batches)` is a mean of per-batch means, which is only exactly
equal to a token-weighted mean when all batches are the same size.
Checked for this specific run: `val_tokens=152`, `sequence_length=16`
→ 8 validation sequences → `batch_size=4` → exactly 2 equal-sized
batches. **This is NOT a contributing factor in the reported result**
(the two batches are equal, so the averaging is exact here) — noted as
a latent, general-case imprecision, not fixed this turn per the
minimality rule (it did not cause or contribute to the observed
failure, so changing it now would be an unrelated modification).

**[VERIFIED IN REPOSITORY]** — no dropout-like or batch-statistics
layer exists in `YugTransformer` that would behave differently between
`train()`/`eval()` mode *except* attention dropout itself
(`yug/native/model.py`: `dropout_p=(self.dropout_p if self.training
else 0.0)` — correctly gated on training mode). Since
`ModelConfig.dropout` defaults to `0.0` (confirmed:
`yug/native/config.py` line 63) and NB-08's experiment never overrode
it, **dropout was 0.0 throughout — train and eval mode are behaviorally
identical for this model**, so no train/eval-mode discrepancy exists to
explain the gap. This directly rules out hypothesis #7 in the original
investigation list (train/eval behavior difference).

## E. Root Cause Analysis

```
ROOT CAUSE
  Severe capacity/data mismatch with zero active regularization:
  221,760 parameters vs. 463 training tokens (~479 params/token),
  dropout=0.0, weight_decay=0.0, 500 optimization steps. Sufficient,
  on its own, to drive training loss to near-zero via memorization
  of specific training sequences rather than general structure.

CONTRIBUTING FACTOR
  Tiny validation set (152 tokens / 8 sentences) -- not large enough
  to average out noise in any single evaluation, though this affects
  the PRECISION of the validation-loss estimate, not its catastrophic
  MAGNITUDE (15.98 vs. baseline 4.70 is far too large a gap to be
  explained by sampling noise alone on 152 tokens).

NOT A CAUSE (checked directly, this turn)
  - Tokenizer/vocabulary mismatch -- ruled out: tokenizer identity
    matches exactly across train/val; 97.3% of validation token TYPES
    and 98.7% of validation token OCCURRENCES already appear in the
    training vocabulary (computed by actually running the real
    tokenizer against the real corpus this turn, not estimated).
  - Causal masking bug -- ruled out: `yug/native/model.py` delegates
    masking to `torch.nn.functional.scaled_dot_product_attention(...,
    is_causal=True)`, PyTorch's own built-in, well-tested primitive,
    not a hand-rolled mask.
  - Target-shift bug -- ruled out: `sequences_to_batch`'s shift logic
    was independently reviewed in NB-07 and matches the standard
    causal-LM convention exactly.
  - Loss-calculation bug -- ruled out: `causal_lm_loss` was unit-tested
    in NB-07 (`test_training_loss.py`, including a "perfect prediction
    -> near-zero loss" case and a "uniform logits -> ln(vocab_size)"
    case) and those tests are part of the reported "1114 passed" run.
  - Train/eval-mode behavioral difference -- ruled out: dropout=0.0
    throughout, confirmed from `ModelConfig`'s actual default and this
    experiment's own (unmodified) configuration.
  - Batch-averaging imprecision in `evaluate()` -- ruled out for THIS
    run specifically: validation batches are exactly equal-sized (2
    batches of 4), so mean-of-means equals the token-weighted mean
    exactly here (see Section D).

UNKNOWN
  Whether reducing model capacity and/or enabling
  dropout/weight_decay would bring validation loss below the unigram
  baseline on this same corpus -- this requires an actual corrective
  experiment (Section F, Experiment E) that has not been run.
```

## F. Diagnostic Experiments

Only genuinely torch-free experiments were executed this turn.
Torch-dependent experiments are designed here, ready to run, but their
results are honestly `[UNKNOWN]` — not fabricated.

### Executed this turn (torch-free)

**Question**: Do train and validation use exactly the same tokenizer,
and is every validation token id valid?
**Method**: Re-ran `ByteLevelBpeTokenizer.train()` on the real training
split and compared its `.identity` against the reported hash; checked
`0 <= id < vocab_size` for every validation token.
**Result**: Identity matches exactly (`sha256:ac40c48e1...`); all
validation ids valid, max id 380 < vocab_size 384.
**Interpretation**: Rules out tokenizer mismatch and vocabulary
indexing bugs.

**Question**: How much lexical/vocabulary overlap exists between train
and validation?
**Method**: Computed set overlap of unique token ids and
occurrence-weighted overlap, using the real tokenizer and corpus.
**Result**: 97.3% of validation token types, 98.7% of validation token
occurrences, already appear in the training vocabulary.
**Interpretation**: The failure is not "the model has never seen these
tokens" — it has seen almost all of them. The failure is at the
sequence/context level: specific continuations in validation sentences
were never seen during training, and an overparameterized,
unregularized model answers confidently and wrongly rather than
hedging.

### Designed, not executed (require PyTorch)

**Experiment A — train-set evaluation** (partially redundant with the
`final_train_loss_full_eval` fix in Section G, which supersedes a
separate one-off script for this).
Question: is the reported near-zero training loss consistent under
the same evaluation methodology used for validation?
Method: `Trainer.evaluate()` over all training sequences, post-training.
Result: **[UNKNOWN]** — requires re-running the experiment with the
Section G fix in place.

**Experiment E — capacity reduction / regularization**
Question: does a substantially smaller model, and/or nonzero
dropout/weight_decay, reduce the train/validation gap on this same
corpus and split?
Method: re-run `training.experiments.nb08_experiment` with (for
example) `hidden_size` reduced from 64 to 16-32, `num_layers` reduced
from 4 to 1-2, and/or `TrainingConfig(dropout>0, weight_decay>0)` —
holding the corpus, split, seed, and evaluation methodology fixed, so
only capacity/regularization changes.
Result: **[UNKNOWN]** — not run. This is the recommended next
corrective experiment (Section G), not yet authorized to implement per
this milestone's hard stop.

**Experiment F — training-duration curve**
Question: at what point does the train/validation gap open up — is it
present from early steps (suggesting a fundamental generalization
failure) or does it emerge gradually (consistent with progressive
overfitting)?
Method: evaluate train and validation loss at steps 0, 50, 100, 200,
300, 500 (matching `training_config.eval_interval=50`, which already
computes this during `fit()` — the data exists in principle but is
currently discarded; `Trainer.evaluate()`'s return values during
`fit()`'s periodic evaluation calls are not currently captured/returned
by `fit()`).
Result: **[UNKNOWN]** — not run, and `Trainer.fit()` does not currently
expose this trajectory (a real, minor limitation worth noting; not
fixed this turn as it's not required to explain the current failure,
only to visualize it more precisely).

## G. Corrective Action

**Implemented this turn** (justified by Section D's evidence, not by a
desire to improve the metric):
- `training/experiments/nb08_experiment.py`: added
  `final_train_loss_full_eval`, computed via the same
  `Trainer.evaluate()` path as `initial_train_loss`/`final_val_loss`,
  so all three are genuinely comparable.
- `tests/test_nb08_experiment.py`: added an assertion that
  `final_train_loss_full_eval < initial_train_loss` — a real
  regression test for the new metric, not yet run (no PyTorch here).

**Recommended, not implemented** (per this milestone's hard stop):
Experiment E above — re-run the same corpus/split with reduced model
capacity and/or nonzero dropout/weight_decay, and compare
`final_val_loss` against the same 4.7020 unigram baseline. This is the
minimum scientifically valid next step; it does not require a larger
corpus, a different corpus, or advancing to NB-09.

**Explicitly not done**, per Section 9's prohibitions: no dataset
changes, no baseline changes, no leaking validation into training, no
blind hyperparameter search, no test weakening.

## H. NB-08 Status

**FAIL — EXPERIMENT DESIGN**

Not an implementation bug (Section E rules out every code-level
hypothesis with direct evidence). Not simply "needs more data" either
— the specific, evidenced cause is capacity/regularization choices in
the experiment configuration itself, which is exactly the kind of
finding this milestone was designed to surface. Training loss
decreasing is explicitly insufficient for PASS per this milestone's own
rule; validation loss must beat the unigram baseline, and it does not.
