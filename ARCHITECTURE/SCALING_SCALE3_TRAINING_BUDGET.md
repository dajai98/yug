# YUG — Scale-3 Training-Budget Extension (Controlled Experiment)

**This is an experiment, not a milestone.** No `NB-13` is assigned.
`ARCHITECTURE/POST_NB07_ROADMAP.md` is not amended by this document or
its execution. NB-08 through NB-12 remain frozen and unmodified.

Status: **BLOCKED — no real training execution has occurred.** This
document records what was and was not verified in the sandbox that
implemented this experiment's driver.

## 1. Objective

Determine whether NB-08 V4's Scale-3 result
(`best_val_loss=1.4878207333385944` at `best_val_step=500` — the
*last* evaluated checkpoint) reflects genuine convergence or premature
stopping, by extending the training budget to 5,000 steps while
holding every other variable fixed.

## 2. Frozen Controls

`[VERIFIED IN REPOSITORY]` and, for the non-training portion,
`[VERIFIED BY ACTUAL RUN]` in this sandbox (tokenizer/corpus/split
computation needs no PyTorch and was actually executed — see §4):

| Control | Value | Verification |
|---|---|---|
| Corpus | `SCALE_CORPORA["scale_3"]`, 864 sentences (6 templates × 144 pairs, the generator's own combinatorial ceiling) | `[VERIFIED BY ACTUAL RUN]` |
| Tokenizer | locked, `vocab_size=384`, identity `sha256:076c103ee...` | `[VERIFIED BY ACTUAL RUN]` — see §4, matches the project's original V3 identity exactly |
| Train/val split | 692 / 172 sentences, `VAL_FRACTION=0.2`, `seed=1234` | `[VERIFIED BY ACTUAL RUN]` |
| Architecture | `hidden_size=48, num_layers=2, num_heads=4, ffn_size=96` | `[VERIFIED IN REPOSITORY]`, reused unmodified |
| `sequence_length` / `context_length` | 16 / 32 | `[VERIFIED IN REPOSITORY]`, reused unmodified |
| Optimizer | AdamW, `lr=3e-3`, `weight_decay=0.0` (default), `grad_clip_norm=1.0`, `batch_size=4` | `[VERIFIED IN REPOSITORY]`, copied verbatim from `build_training_config()` |
| Seed | 1234, applied via `seed_everything` before model construction | `[VERIFIED IN REPOSITORY]` |

## 3. Exact Change

```
max_steps: 500 -> 5000
checkpoint_interval: 500 -> 5000 (kept equal to max_steps, matching
  build_training_config()'s own convention; not functionally used by
  this script's manual loop, changed only for internal consistency)
```

Checkpoint/evaluation schedule: original `[0, 50, 100, 200, 300, 500]`
preserved, extended with the authorized
`[1000, 1500, 2000, 3000, 4000, 5000]` — full curve, not a disjoint
tail, for direct comparability with the frozen NB-08 V4 record.

`run_single_scale()` was **not** called and **not** modified — it
hardcodes `build_training_config()` internally with no step-count
override. This experiment reuses every underlying primitive
`run_single_scale()` itself is built from (`split_sentences`,
`encode_corpus`, `make_sequences`, `ModelConfig.from_tokenizer`,
`Trainer`, `save_checkpoint`, `uniform_baseline_loss`/
`unigram_baseline_loss`, `generate`/`safe_decode_generation`),
imported unmodified — see
`training/experiments/scaling_scale3_budget_extension.py`'s own
docstring for the full reasoning.

## 4. Execution Environment

**[VERIFIED BY ACTUAL RUN, this sandbox]**: Python 3.12.3, Linux
(container). PyTorch: **not installed, no network access to install
it** — the same disclosed, standing constraint present at every prior
milestone/experiment this session. `python3 -m
training.experiments.scaling_scale3_budget_extension` fails cleanly at
`import torch` inside `run()` — the exact point real training would
begin.

As a partial, real (not simulated) check, the torch-free portion of
this experiment's frozen controls was independently reproduced in this
sandbox:
```
tokenizer.identity: sha256:076c103ee402a41947e7911ee9f2908b00ac0af57b09e7c5c44180a4abf741e7
tokenizer.vocab_size: 384
scale_3 corpus size (sentences, pre-split): 864
train_sentences: 692
val_sentences: 172
```
The tokenizer identity matches the project's original, long-standing
V3 identity exactly, confirming this experiment's corpus/tokenizer/
split setup is correct and consistent with every prior milestone. This
does **not** substitute for real training execution.

## 5. Training Curve

**[UNKNOWN — not established]**. No training occurred. No curve data
exists.

## 6. Checkpoint Results

**[UNKNOWN — not established]**. No checkpoints were saved. `[500,
1000, 1500, 2000, 3000, 4000, 5000]` integrity/reload-evaluation
results are all unrecorded.

## 7. Generation Samples

**[UNKNOWN — not established]**. No generation occurred at any
checkpoint.

## 8. Checkpoint Integrity

**[UNKNOWN — not established]**. The driver's integrity checks
(`_check_state_dict_finite`, reload via `evaluate_checkpoint()`) are
implemented and syntax-verified but never executed against real
tensors.

## 9. Regression Results

`[VERIFIED BY ACTUAL RUN, this sandbox]`: the repository's own
`run_tests_shim.py` (not equivalent to real pytest) shows
`passed=406 failed=0 skipped=11 errors=0 collection_errors=11` —
identical to the pre-experiment baseline; the new script is not a test
file and is not collected. Real pytest was not run (not installed in
this sandbox). `grep -rn "^import training\|^from training" yug/` →
zero matches, re-confirmed after adding this script.

## 10. Convergence Analysis

**[UNKNOWN — not established]**. None of the seven analysis questions
(did validation loss continue improving past step 500; where did
improvement diminish; was step 500 premature; did validation loss ever
rise while training loss fell; what is the best checkpoint; is there
enough evidence to treat this as converged; what uncertainty remains)
can be honestly answered without real execution.

## 11. Classification

```
E. FAILED / BLOCKED
```

Not a judgment about the model — a statement about the absence of any
real evidence. Per this experiment's own governing rule ("if real
execution cannot be performed because PyTorch/environment is
unavailable, classify the experiment BLOCKED"), this is the only
honest classification available from this sandbox.

## 12. Remaining Uncertainty

Everything in §5–§10. The single, central question this experiment
exists to answer — whether Scale 3's `best_val_step=500=max_steps`
reflects a still-improving trend or coincidence — remains exactly as
open as it was before this document was written.

## 13. Recommendation for Chief Review

Run `python -m training.experiments.scaling_scale3_budget_extension`
in the project's real Windows/PyTorch environment, then:
```
grep -rn "^import training\|^from training" yug/
```
(or the PowerShell `Select-String` equivalent), and send back the
printed JSON output and exit code verbatim. This document's §5–§11
will be replaced with the real training curve, checkpoint results,
generation samples, integrity results, and an evidence-based
classification (A–E) once that happens — not before. No aspect of
NB-08–NB-12's frozen evidence is at risk from this experiment; it
executes entirely in a temporary directory outside the repository and
touches no frozen file.
