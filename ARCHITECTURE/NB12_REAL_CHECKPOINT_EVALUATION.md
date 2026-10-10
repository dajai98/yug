# NB-12 — Real-Checkpoint Evaluation & Adapter Exercise: Evidence Report

Status: **PASS.** This document has four parts, in chronological
order: (1) the original real execution and its genuine FAIL, (2) the
investigation and root-cause classification, (3) the authorized,
narrowly-scoped driver remediation, (4) the final real rerun and its
PASS. Nothing here is fabricated or reinterpreted — every metric in
Parts 1 and 4 is recorded verbatim as reported from real execution.

---

## PART 1 — Original Real Execution (Historical Evidence, Preserved)

**[VERIFIED BY ACTUAL RUN — project-owner-supplied, real Windows/PyTorch
environment.]** Recorded here verbatim, not overwritten by the
remediation below — this is what actually happened on the first real
run:

```
scale_3 training completed successfully
parameter_count = 55,536
initial_train_loss = 45.157485871803104
final_train_loss_full_eval = 1.186492787571404
best_val_loss = 1.4878207333385944
best_val_step = 500
uniform_baseline_loss = 5.950642552587727
unigram_baseline_loss = 4.494875493507538
evaluation loss = 1.4878207333385944
perplexity = 4.427436433350047
checkpoint_integrity = all checkpoint parameters finite
adapter_load = SUCCESS
YugCore.respond("Hello.") FAILED with:
  GenerationError: Generation failed: sequence length 33 exceeds context_length 32
```

**Original verdict: FAIL.** Correctly classified as FAIL, not RESULT/
OBSERVATION, because this is *not* the permitted `DecodeError ->
GenerationError` boundary — the milestone's own classification logic
worked exactly as designed here.

## PART 2 — Investigation & Root Cause

**Classification: B — an incorrect NB-12 driver invocation.** Nothing
in production code (`YugTransformer.forward()`, `native_generate()`,
`YugNativeAdapter`, `YugCore.respond()`) is defective; every one of
them behaved exactly as designed and previously verified.

**Root cause**: `training/experiments/nb12_checkpoint_closure.py`
constructed `YugCore(adapter, WorkingMemory())` without overriding
`max_tokens`, silently inheriting `YugCore.__init__`'s default of
**1024** — against NB-08 V4's deliberately tiny
`CONTEXT_LENGTH = 32` (`training/experiments/nb08_experiment.py`,
commented *"headroom above SEQUENCE_LENGTH, for the generation
check"* — never intended to support open-ended generation).

**Exact reproduced trace** (rebuilt the real locked tokenizer
torch-free in this sandbox and encoded the real formatted prompt):

```
prompt = "user: Hello.\nassistant:"
prompt_ids (add_bos=True) = 21 tokens   [VERIFIED BY ACTUAL RUN, this sandbox]
context_length = 32
```

`native_generate()`'s loop calls `model(generated)` *before* appending
each new token, so the forward-pass length at the start of iteration
*k* is `21 + (k-1)`, which must be `<= 32`: iterations 1-12 succeed
(length reaches 32 exactly at iteration 12's start), producing a 12th
new token that brings `generated` to length 33. Iteration 13's forward
call then sees `seq_len=33 > 32` and raises — arithmetically,
precisely matching the real reported error, not a coincidence.

This was not a newly-discovered integration gap: NB-10's and NB-11's
own test fixtures already correctly handled this exact risk class
(explicitly sizing `context_length` with headroom and passing small
`max_tokens` values, with their own comments explaining why). NB-12's
driver simply didn't re-apply that already-established discipline.

**Risk to NB-08/09/10/11 evidence: none.** Part 1's training/
checkpoint/evaluation results (`parameter_count`, all losses,
baselines, `evaluate_checkpoint()`'s finite loss/perplexity) were
already complete before `YugCore.respond()` was ever called, and are
entirely unaffected by this driver-level bug.

## PART 3 — Applied Remediation & Re-Verification Status

**Exact one-line remediation applied**, in
`training/experiments/nb12_checkpoint_closure.py`, nothing else:

```diff
- core = YugCore(adapter, WorkingMemory())
+ core = YugCore(adapter, WorkingMemory(), max_tokens=8)
```

`21 + 8 = 29`, safely inside `context_length=32` with margin. No other
line in this file changed. No file outside
`training/experiments/nb12_checkpoint_closure.py` was modified.

### Fresh execution attempt (sandbox)

**[VERIFIED BY ACTUAL RUN, this sandbox]**: `python -m
training.experiments.nb12_checkpoint_closure` was attempted
immediately after applying the remediation, in this assistant's own
sandbox. Result: same standing constraint as every prior milestone
this session — `ModuleNotFoundError: No module named 'torch'`. This
sandbox has no PyTorch and no network access to install it, and so
could not itself produce a fresh result.

## PART 4 — Final Real Execution (Post-Remediation, PASS)

**[VERIFIED BY ACTUAL RUN — project-owner-supplied, real Windows/PyTorch
environment.]** NB-12 was rerun cleanly from the beginning with the
remediated driver (`max_tokens=8`), using the same authorized
`scale_3` / `run_single_scale()` / `build_locked_tokenizer()` /
`evaluate_checkpoint()` path — no values manually substituted:

```
scale_name: scale_3
parameter_count: 55,536
initial_train_loss: 45.157485871803104
final_train_loss_full_eval: 1.186492787571404
final_val_loss: 1.4878207333385944
best_val_loss: 1.4878207333385944
best_val_step: 500
uniform_baseline_loss: 5.950642552587727
unigram_baseline_loss: 4.494875493507538
checkpoint_integrity: all checkpoint parameters finite
evaluation loss: 1.4878207333385944
evaluation perplexity: 4.427436433350047
adapter_load: SUCCESS
respond_success: true
respond_content: "a small child is bright.the app"
respond_error: null
```

**Training evidence check** (NB-08 V4's own condition,
Acceptance Criterion #5):
```
best_val_loss (1.4878...) < initial_train_loss (45.157...)   -> TRUE
best_val_loss (1.4878...) < uniform_baseline_loss (5.9506...) -> TRUE
best_val_loss (1.4878...) < unigram_baseline_loss (4.4949...) -> TRUE
```
All three hold — the same result as the original (failed-only-at-the-
respond-step) run; training/checkpoint/evaluation were never in
question, only the downstream `max_tokens` misconfiguration was.

**Response classification**: `respond_success = true`,
`respond_content` is non-empty (`"a small child is bright.the app"`),
`respond_error = null`. Per Acceptance Criterion #11, a successful
non-empty `Response` is the normal successful runtime outcome — this
is **not** the permitted `DecodeError -> GenerationError` boundary (no
error occurred at all), so it is classified plainly as **SUCCESS**,
not RESULT/OBSERVATION. No claim is made about the *quality* of the
generated text beyond it being real, decoded, non-empty output from
the real trained checkpoint.

### Regression (real, project-owner-supplied)

```
python -m pytest tests/test_training_evaluate.py -q          -> 6 passed
python -m pytest tests/test_brain_yug_native_adapter.py -q   -> 16 passed
python -m pytest tests/test_composition.py -q                -> 6 passed
python -m pytest tests -q                                    -> 615 passed, 1 skipped
```
Known skip: `tests/test_brain_llama_cpp_adapter.py` (requires
`YUG_TEST_MODEL_PATH`) — pre-existing, unrelated to NB-12. No new
skip, no failure, no regression against NB-11's own closing baseline
(also 615 passed, 1 skipped).

### Dependency boundary (final)

```
grep -rn "^import training\|^from training" yug/
```
Result: **zero matches.**

### Scope audit (final)

Files changed across the full NB-12 arc: `training/experiments/
nb12_checkpoint_closure.py` (created, then one line remediated) and
this evidence document. No production file in `yug/` or `training/`
(other than the new driver itself) was ever modified. No test file was
modified. No prior milestone's evidence (NB-08/09/10/11) was altered.

## Final Verdict

```
PASS
```

Chain of custody, in order: (1) original real execution — genuine
FAIL, caused by the driver inheriting `YugCore`'s `max_tokens=1024`
default against the experimental model's `context_length=32`;
(2) investigation — Classification B, incorrect driver invocation, no
production defect found; (3) authorized, narrowly-scoped remediation —
one line, `max_tokens=8`; (4) clean rerun from scratch — real
training, real checkpoint, real evaluation, real adapter load, real
`YugCore.respond()` — all succeeded; (5) full regression — 615 passed,
1 skipped, zero new failures, boundary clean. This is the first time
in this repository's history that a genuinely-trained native
checkpoint has passed through the complete
training → checkpoint → evaluation → adapter → runtime chain.

