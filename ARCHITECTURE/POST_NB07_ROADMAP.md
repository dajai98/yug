# YUG — Post-NB-07 Roadmap (Authoritative)

**Status: authoritative for future implementation unless explicitly amended.**
**Supersedes: any milestone sequencing implied by older architecture documents where they conflict with this one.**

This document was produced by a fresh, evidence-based audit of the
repository as it actually exists, not by continuing prior milestone
numbering mechanically. Where the repository's own numbering (NB-01
through NB-07, as defined in `ARCHITECTURE/NATIVE_BRAIN_ARCHITECTURE.md`
and `ARCHITECTURE/NB07_ARCHITECTURE_AMENDMENT.md`) is sound, it is kept
unchanged. New milestones (NB-08 onward) are numbered fresh from here.

---

## 1. Current Verified State

**[VERIFIED BY TEST — reported from the real development environment;
not independently re-executed by this assistant in its own sandbox,
which has no PyTorch]**:

```
python -m pytest tests/test_training_loop.py -q   → 4 passed
python -m pytest tests -q                          → 1070 passed, 2 skipped, 0 failed
```

This is accepted as authoritative per explicit instruction. NB-07 is
**COMPLETE** and will not be reopened absent a genuine regression
discovered by a later audit.

**[VERIFIED IN REPOSITORY]** — fresh inspection this turn confirmed:
- `yug/` contains 9 subsystems: `brain` (interface + `llama_cpp`/`mock`
  adapters + manifest), `composition.py`, `config`, `memory`, `native`
  (config/model/generation/errors), `runtime`, `tokenizer`, `tools`
  (registry/executor/permissions/sandbox/validation + 4 builtin tools),
  `verification` (engine/schema/math).
- `training/` contains 7 modules: `config`, `errors`, `dataset`, `loss`,
  `determinism`, `checkpoint`, `loop` — all present, none missing.
- No `agent`/`planner`/`state_machine` package exists anywhere.
- No `YugNativeAdapter` class exists anywhere (`grep` for the class
  name across `yug/`, zero matches).
- `yug/config/schema.py`'s `_VALID_BRAIN_BACKENDS` is still exactly
  `{"llama_cpp", "mock"}` — no native/own-model backend has been added
  to the composition root yet, correctly, per its own prior scope
  limit ("our own model does not exist yet").

## 2. Subsystem Classification

| Subsystem | Actual implementation | Tests | Status |
|---|---|---|---|
| Core runtime | `yug/runtime/core.py` — `YugCore`, brain/memory/tools/verification wiring, no brain→tool autonomy | `test_runtime_*.py`, `test_composition.py` | **COMPLETE** (M1 scope) |
| Brain interface | `yug/brain/interface.py` — 4 abstract methods | exercised via both adapters' tests | **COMPLETE** |
| External brain adapter | `LlamaCppAdapter` — full, lazy-loaded, typed errors | `test_brain_llama_cpp_adapter.py` (9/10; 1 skipped — needs a real `.gguf` model) | **COMPLETE**, unverified against a real model |
| Tokenizer | `yug/tokenizer/bpe.py` — byte-level BPE, no pre-tokenization | `test_tokenizer_*.py` | **COMPLETE** |
| Model configuration | `yug/native/config.py` — `ModelConfig`, JSON save/load, tokenizer-compat check | `test_native_config_*.py` | **COMPLETE** |
| Native Transformer | `yug/native/model.py` — `YugTransformer` | `test_native_model.py` (31 tests) | **COMPLETE**, [VERIFIED BY TEST — real environment, per above] |
| Forward pass | same file | same | **COMPLETE** |
| Generation | `yug/native/generation.py` — greedy deterministic decode | `test_native_generation.py` (22 tests) | **COMPLETE**, [VERIFIED BY TEST — real environment, per above] |
| Dataset pipeline | `training/dataset.py` | `test_training_dataset.py` (8 tests) | **COMPLETE** |
| Training configuration | `training/config.py` | `test_training_config.py` (14 tests, incl. this assistant's own genuinely-executed run) | **COMPLETE** |
| Loss | `training/loss.py` | `test_training_loss.py` (4 tests) | **COMPLETE** |
| Determinism | `training/determinism.py` | exercised via `test_training_loop.py` | **COMPLETE** (documented scope: CPU+CUDA RNG seeding only, no cross-hardware claim) |
| Checkpointing | `training/checkpoint.py` | `test_training_checkpoint.py` (4 tests) | **COMPLETE** |
| Training loop | `training/loop.py` (incl. this turn's device-handling fix and the tokenizer-fixture fix from the prior turn) | `test_training_loop.py` (4 tests) | **COMPLETE** |
| Memory | `yug/memory/` — working memory + persistence | `test_memory_*.py` | **COMPLETE** (M1 scope; no episodic memory yet) |
| Tools | `yug/tools/` — registry, executor, permissions, sandbox, 4 builtins | `test_tools_*.py` | **COMPLETE** (M1 scope) |
| Verification | `yug/verification/` — schema + math verifiers, engine | `test_verification_*.py` | **COMPLETE** (M1 scope) |
| Agent runtime | **does not exist** | none | **NOT STARTED** — by design, per progressive-autonomy boundary |

## 3. What NB-07 Actually Left Behind

| # | Question | Answer |
|---|---|---|
| 1 | Can YUG tokenize training data? | **[VERIFIED BY TEST]** yes |
| 2 | Can it create training sequences? | **[VERIFIED BY TEST]** yes |
| 3 | Can it construct the native Transformer? | **[VERIFIED BY TEST]** yes |
| 4 | Can it perform a forward pass? | **[VERIFIED BY TEST]** yes |
| 5 | Can it calculate training loss? | **[VERIFIED BY TEST]** yes |
| 6 | Can it backpropagate? | **[VERIFIED BY TEST]** yes |
| 7 | Can it update weights? | **[VERIFIED BY TEST]** yes |
| 8 | Can it save checkpoints? | **[VERIFIED BY TEST]** yes |
| 9 | Can it resume training? | **[VERIFIED BY TEST]** yes |
| 10 | Can it demonstrate tiny-corpus overfitting? | **[VERIFIED BY TEST]** yes (a single repeated 4-token pattern) |
| 11 | Can it evaluate a trained model? | **[VERIFIED BY TEST]** `Trainer.evaluate()` computes no-grad loss over held-out batches; **[UNKNOWN]** whether this has been exercised on genuinely held-out (non-training) data rather than the same tiny sequences used for training |
| 12 | Can it generate text from the trained native model? | **[VERIFIED BY TEST]** `generation.py`'s forward path executes; **[UNKNOWN]** whether generation has ever been run using a checkpoint produced by `training/`, specifically (as opposed to a freshly-initialized model) |
| 13 | Can a trained native model be exposed through the existing `BrainInterface`? | **[VERIFIED IN REPOSITORY]** **NO** — no `YugNativeAdapter` exists (confirmed by `grep`, zero matches); this is the largest concrete gap this roadmap addresses first |

## 4. Architectural Boundaries (preserved, unchanged by this roadmap)

All `[DECISION]`, none altered by this document:

- `[DECISION]` YUG is a system, not merely a model.
- `[DECISION]` The native YUG brain is the long-term foundation brain.
- `[DECISION]` `BrainInterface` remains replaceable.
- `[DECISION]` Memory remains external to model weights.
- `[DECISION]` Tools remain permissioned.
- `[DECISION]` Brain output must never automatically execute tools.
- `[DECISION]` Verification remains explicit, caller-driven, never automatic.
- `[DECISION]` Progressive autonomy — no agent/planner until justified.
- `[DECISION]` Local-first architecture.
- `[DECISION]` Production `yug/` must not depend on `training/` — **[VERIFIED IN REPOSITORY]** re-confirmed this turn, zero matches.
- `[DECISION]` `training/` may depend on `yug/` — **[VERIFIED IN REPOSITORY]** re-confirmed, `training/errors.py` imports `yug.utils.errors`.
- `[DECISION]` No premature distributed training / giant training framework.
- `[DECISION]` No premature multimodal architecture.
- `[DECISION]` No RLHF/DPO/MoE without a later, separately justified milestone.
- `[DECISION]` `LlamaCppAdapter` is not deleted merely because the native brain exists.
- `[DECISION]` No dependencies added without justification.

## 5. Milestone Dependency Graph

```
NB-07 (complete)
   │
   ▼
NB-08 — Controlled Small-Scale LM Experiment
   │
   ▼
NB-09 — Minimal Evaluation Harness
   │
   ▼
NB-10 — YugNativeAdapter (BrainInterface Integration)
   │
   ▼
NB-11 — Native Brain Runtime & Composition Regression Validation
   │
   ▼
NB-12 — Real-Checkpoint Evaluation & Adapter Exercise
   │
   ▼
[Deferred track: further brain-engineering scaling]
[Deferred track: agent/planner — blocked on its own separate justification]
```

NB-08 and NB-09 are Brain Engineering. NB-10 sits at the integration
boundary (Brain Engineering work, System Integration consequence).
NB-11 is pure Cognitive/System Integration (regression-only, no new
brain capability). NB-12 is Brain Engineering evidence-gathering
(reuses NB-08 V4's pipeline unmodified; no new brain capability
either). Deliberately not parallelized with an agent/planner
track — nothing justifies starting that yet (Section 8's gate).

## 6. Milestone Definitions

### NB-08 — Controlled Small-Scale Language-Modeling Experiment

**Objective**: Prove the training pipeline built in NB-07 works on a
small but genuinely varied text corpus, not just a single repeated
4-token pattern — i.e. move past pure memorization toward an actual
(if tiny) language-modeling sanity check.

**Why it exists**: NB-07's own acceptance criterion was deliberately
narrow ("tiny overfit", a single repeated pattern) — precisely so the
pipeline's correctness could be isolated from the harder question of
whether it generalizes at all. That harder question is still
completely **[UNKNOWN]** and is the entire point of this milestone.

**Repository prerequisites**: `training/` (complete, NB-07), a real
text corpus to be selected as part of this milestone's own scope (not
pre-chosen here — selecting it prematurely would be exactly the kind
of "invent the answer before doing the work" this roadmap's own STEP 6
warns against).

**Implementation scope**: corpus selection/preparation (small,
locally-available, license-clean text — explicitly NOT a giant/
web-scale dataset), a held-out split, running `Trainer.fit()` against
train+held-out data, tracking both loss curves.

**Explicit non-goals**: no new model architecture, no scaling beyond
what NB-07's tiny configs already used unless evidence from this
milestone itself justifies it, no evaluation-harness productionization
(that's NB-09), no `BrainInterface` integration (that's NB-10).

**Tests required**: held-out split correctness (no train/eval overlap),
train-loss-decreases-on-real-corpus, held-out-loss does not diverge
catastrophically (a specific, evidence-derived threshold — not
invented in advance, matching NB-07's own "don't hardcode a blind
threshold" precedent).

**Acceptance criteria**: train loss decreases; held-out loss is
computed and finite; the run is reproducible under `determinism.py`'s
documented guarantees.

**Exit criteria**: all required tests pass under real execution
(explicitly not the custom shim); results are recorded in a milestone
report with actual numbers (loss curves, corpus size, steps).

**Dependencies**: NB-07.

**Risks**: a real corpus may expose training instabilities NB-07's toy
data never could (loss spikes, NaN, gradient explosion) — this is the
milestone's actual purpose, not a failure mode to avoid.

**Evidence required before advancing**: a genuine train/held-out loss
report, real-execution test results.

---

### NB-09 — Minimal Evaluation Harness

**Objective**: A standalone, repeatable way to measure a checkpoint's
language-modeling quality (loss, perplexity) independent of any
specific training run.

**Why it exists**: `Trainer.evaluate()` (NB-07) computes eval loss
*during* a training run only — there is no standalone tool to load an
arbitrary checkpoint and report its quality after the fact. NB-10
(adapter integration) needs *something* to decide whether a given
checkpoint is worth exposing at all; this milestone is that something.

**Repository prerequisites**: NB-08's held-out corpus/methodology,
`training/checkpoint.py` (complete, NB-07).

**Implementation scope**: a small evaluation entry point (loads a
checkpoint via `training.checkpoint.load_checkpoint`, runs no-grad
forward passes over a held-out set, reports loss and perplexity).

**Explicit non-goals**: no large/standardized public benchmark suite,
no leaderboard-style tracking infrastructure, no automated
model-comparison tooling.

**Tests required**: perplexity computed correctly from loss (the
well-known `exp(loss)` relationship — verify against a known-loss
synthetic case, not just "it runs"), checkpoint-loading path exercised
end-to-end.

**Acceptance criteria**: given a checkpoint from NB-08, the harness
reports a finite, correctly-computed perplexity.

**Exit criteria**: real-execution tests pass; a report exists showing
this run against NB-08's checkpoint.

**Dependencies**: NB-08.

**Risks**: none beyond ordinary implementation risk — this is a small,
self-contained utility.

**Evidence required before advancing**: an actual perplexity number
for an actual checkpoint, from actual execution.

---

### NB-10 — YugNativeAdapter: BrainInterface Integration

**Objective**: Close the exact gap identified in Section 3, Q13 — make
a trained native checkpoint reachable through `BrainInterface`,
alongside `MockBrainAdapter` and `LlamaCppAdapter`.

**Why it exists**: this was originally scoped (informally, as
"NB-06.5") before the project's from-scratch training decision was
made — at that time it was correctly blocked, because there was no
training infrastructure yet to produce anything worth exposing. That
blocker is now resolved by NB-07/08/09; this milestone is the deferred
follow-through, done in the right order this time (after there's
something worth exposing, not before).

**Repository prerequisites**: NB-08 (a real, non-toy checkpoint),
NB-09 (a way to know whether that checkpoint is any good), the
existing `BrainInterface` contract and `LlamaCppAdapter` as the
reference implementation to mirror.

**Implementation scope**:
- `yug/brain/adapters/yug_native.py` — `YugNativeAdapter`, implementing
  all 4 `BrainInterface` methods, delegating to `yug.native.generation`
  + a checkpoint loaded via `training.checkpoint.load_checkpoint` (this
  is the one place `yug/` reads from `training/`'s *checkpoint format*,
  not its training logic — the checkpoint file format itself, not the
  `training` package, is the interchange boundary; the adapter should
  depend on the file format, ideally via a thin loader that does not
  import `training.loop`/`training.dataset` at all).
- One new entry in `yug/composition.py`'s `_BACKEND_FACTORIES` and in
  `yug/config/schema.py`'s `_VALID_BRAIN_BACKENDS` (e.g. `"yug_native"`)
  — the exact one-line change anticipated back when the composition
  root milestone deliberately deferred it ("our own model does not
  exist yet").

**Explicit non-goals**: no removal of `LlamaCppAdapter` (per the
standing `[DECISION]` above), no claim that native-brain output is
competitive with the external adapter's quality, no agent-level
tool-calling from the native brain.

**Tests required**: `YugNativeAdapter` satisfies `BrainInterface`
(mirroring `test_llama_cpp_adapter_satisfies_brain_interface`), health
check reflects real checkpoint-loaded state, generation executes
end-to-end from a real NB-08 checkpoint, composition-root dispatch
test for the new backend value.

**Acceptance criteria**: `build_brain(BrainConfig(backend="yug_native", ...))`
returns a working `YugNativeAdapter` that can `generate()` from a real
checkpoint.

**Exit criteria**: real-execution tests pass; `yug/runtime/core.py`
remains untouched (confirms the whole point of `BrainInterface` — a
third backend option requires zero runtime changes).

**Dependencies**: NB-08, NB-09.

**Risks**: checkpoint-format assumptions baked into `training/checkpoint.py`
during NB-07 may need a read-only, training-independent loading path
carved out for this adapter to use without pulling in `training/`'s
full dependency surface — worth watching, not yet confirmed as an
actual problem.

**Evidence required before advancing**: a real generation output from
a real NB-08 checkpoint, through `BrainInterface`, through
`composition.py` — the complete path.

---

### NB-11 — Native Brain Runtime & Composition Regression Validation

**Objective**: Confirm that adding a third, real (non-mock) brain
backend genuinely required zero changes to `runtime/`, `memory/`,
`tools/`, or `verification/` — i.e. prove the isolation this whole
project has maintained since the composition-root milestone actually
holds up in practice, not just in intent.

**Why it exists**: this is a Cognitive/System Integration concern, not
Brain Engineering — it validates the system around the brain, not the
brain itself. Deliberately kept separate from NB-10 so the two
concerns (does the adapter work / did adding it break isolation) are
independently, unambiguously testable.

**Repository prerequisites**: NB-10.

**Implementation scope**: no new production code expected; if this
milestone finds it needs any, that finding itself is the important
result and should be reported before making the change.

**Explicit non-goals**: no new brain capability, no agent/planner
work.

**Tests required**: full regression suite; a static-inspection test
(mirroring `test_yugcore_does_not_import_concerete_adapters` from the
composition-root milestone) confirming `runtime/core.py` still imports
nothing adapter-specific after NB-10.

**Acceptance criteria**: 0 regressions across the full suite; the
isolation check passes.

**Exit criteria**: real-execution full-suite run, explicitly compared
against NB-10's baseline.

**Dependencies**: NB-10.

**Risks**: none anticipated; this is a validation milestone by design.

**Evidence required before advancing**: a clean full-suite diff.

---

### NB-12 — Real-Checkpoint Evaluation & Adapter Exercise

**Objective**: Using training.experiments.nb08_experiment_v4's existing,
unmodified `run_single_scale()` at `scale_3` and
`build_locked_tokenizer()`, produce one genuinely-trained native
checkpoint, evaluate it via NB-09's `evaluate_checkpoint()`, and load
and exercise it once via `YugNativeAdapter` and `YugCore.respond()`.

**Why it exists**: Promoted from the "evaluation maturity" deferred
track. NB-09, NB-10, and NB-11 each left the same evidence gap open:
no genuinely-trained native checkpoint has yet been evaluated or
exercised through the real native inference/runtime chain.

**Repository prerequisites**: NB-07, NB-08 (V4), NB-09, NB-10, NB-11.

**Implementation scope**:
- Import and call the existing
  `training.experiments.nb08_experiment_v4.run_single_scale()`.
- Use exactly `scale_3`.
- Use the existing `SCALE_CORPORA["scale_3"]`.
- Use the existing `build_locked_tokenizer()`.
- Produce one real checkpoint using the existing training pipeline.
- Run `evaluate_checkpoint()` against that same checkpoint.
- Load that same checkpoint through `YugNativeAdapter`.
- Construct `YugCore` and call `respond()` once.
- Record the complete real execution evidence.

No training logic may be reimplemented.

**Explicit non-goals**:
- Scaling beyond NB-08 V4.
- Running `run_all_scales()`.
- New evaluation metrics.
- Evaluation-framework redesign.
- Tokenizer redesign.
- Checkpoint-format redesign.
- BrainInterface changes.
- YugNativeAdapter changes.
- YugTransformer changes.
- Native generation changes.
- Composition-root/config wiring.
- Agent/planner work.
- Memory work.
- Tool work.
- Verification-subsystem work.
- RLHF/DPO/MoE work.
- Multimodal training.
- Distributed training.
- Any unrelated refactoring.

**Required infrastructure reuse**:
- `training.experiments.nb08_experiment_v4.run_single_scale`
- `training.experiments.nb08_experiment_v4.build_locked_tokenizer`
- `training.experiments.nb08_experiment_v4.SCALE_CORPORA["scale_3"]`
- `training.evaluate.evaluate_checkpoint`
- `YugNativeAdapter`
- `YugCore`

No replacement implementation of these components is permitted.

**Required evidence**:

1. A real checkpoint is produced by the existing NB-08 V4
   `run_single_scale()` pipeline.
2. The returned training/evaluation metrics are recorded verbatim.
3. Reported loss values are finite.
4. No NaN or Inf exists in the saved checkpoint state.
5. The training result demonstrates actual learning by satisfying the
   existing NB-08 V4 evidence condition:
   `best_val_loss < initial_train_loss`.
6. `best_val_loss` is compared against the existing
   `uniform_baseline_loss` and `unigram_baseline_loss`.
7. Tokenizer identity, seed, parameter count, training step count,
   checkpoint location, and relevant model configuration are recorded.
8. `evaluate_checkpoint()` successfully evaluates the resulting real
   checkpoint and produces finite loss/perplexity.
9. `YugNativeAdapter` successfully loads the same real checkpoint
   without exception.
10. After successful loading, `YugCore.respond()` is called once.
11. A successful non-empty `Response` is the normal successful runtime
    outcome.
12. A `GenerationError` caused specifically by the tokenizer's
    `DecodeError` may be recorded as an expected model-level result,
    but MUST NOT be silently classified as equivalent to successful
    generation.
13. Any other `BrainError`, checkpoint-loading error, configuration
    mismatch, tokenizer mismatch, unexpected exception, or runtime
    infrastructure failure is a FAIL.
14. `yug/` continues to contain zero imports from `training/`.
15. No production code is modified.

**Failure semantics**:

### PASS

All required infrastructure and evidence criteria are satisfied.

### RESULT / OBSERVATION

The real pipeline successfully executes, but model quality or
generation quality is poor.

Examples include:
- high loss
- poor generated text
- unexpected but specifically permitted
  `DecodeError → GenerationError`

A RESULT must still be recorded honestly and must not be represented
as successful generation.

### FAIL

A required infrastructure or evidence criterion cannot be established.

Examples:
- checkpoint cannot be loaded
- evaluation cannot run
- tokenizer/configuration mismatch
- unexpected runtime exception
- non-finite metrics
- NaN/Inf checkpoint weights
- training evidence does not satisfy the stated NB-08 V4 condition
- `yug → training` dependency appears
- production code is changed

### BLOCKED

A prerequisite required to execute the milestone is unavailable.

**Tests**:

No new pytest test file is required.

The existing test suite remains the regression safety net.

The NB-12 execution driver itself must perform the required milestone
checks and fail loudly on infrastructure failures.

Do not increase the test count merely for cosmetic coverage.

**Deliverables**:

- A thin NB-12 execution driver that imports and reuses the existing
  NB-08 V4 experiment functions rather than reimplementing training.
- `ARCHITECTURE/NB12_REAL_CHECKPOINT_EVALUATION.md` containing the
  actual execution evidence.

The checkpoint binary itself is not a roadmap deliverable and must not
be committed merely to satisfy this milestone.

**Exit criteria**:

Real execution evidence is recorded in
`ARCHITECTURE/NB12_REAL_CHECKPOINT_EVALUATION.md`, including:

- complete `run_single_scale()` return data
- training configuration/provenance
- checkpoint integrity result
- `evaluate_checkpoint()` result
- adapter loading result
- classified `YugCore.respond()` result
- PASS / RESULT / FAIL / BLOCKED determination

The result must be based on real execution, not a custom shim.

**Evidence required before advancing**:

The completed NB-12 architecture/evidence document with real execution
results must exist and satisfy the milestone's acceptance criteria.

**Architectural constraints**:

- YUG remains a SYSTEM, not merely a model.
- `BrainInterface` remains inference-only.
- `YugNativeAdapter` remains replaceable.
- `LlamaCppAdapter` remains.
- `yug/` must not import from `training/`.
- `training/` may import from `yug/`.
- Memory remains external.
- Tools remain separate.
- Verification remains separate.
- No premature RLHF/DPO/MoE/multimodal architecture.

---

## 7. Deferred Tracks (no milestone number assigned yet)

Per Section 5's "no vague milestones" rule, these are named as tracks,
not numbered milestones, because none has enough repository evidence
yet to define concrete scope/acceptance criteria:

- **Scaling experiments** — beyond NB-08's small corpus, pending
  NB-08/09's actual results justifying a next step.
- **RLHF / DPO / MoE / multimodal training** — explicitly out of scope
  until a separately-justified future milestone; nothing in the
  current roadmap moves toward these.
- **Agent / planner runtime** — blocked on its own justification,
  unrelated to brain-engineering progress; adding a native brain
  option (NB-10) does not by itself justify starting this.
- **Episodic memory** — noted as an M1-scope gap in an earlier audit,
  unrelated to this roadmap's brain-engineering focus; not scheduled
  here.

(Evaluation maturity was promoted to NB-12 — see Section 6 — and is no
longer listed here.)

## 8. Advancement Gates

No milestone in Section 6 begins before its listed dependencies show
"Evidence required before advancing" satisfied by **real execution**,
not the custom shim, and not static inspection alone where the
milestone's own definition calls for execution. Each milestone's own
"Exit criteria" is the literal gate for starting the next.

## 9. Known Unknowns

- Whether `Trainer.evaluate()` has ever been run against genuinely
  held-out data (Section 3, Q11) — **[UNKNOWN]**, motivates NB-08's
  held-out-split requirement being explicit rather than assumed.
- Whether generation has ever been run from a `training/`-produced
  checkpoint specifically (Section 3, Q12) — **[UNKNOWN]**, motivates
  NB-10 rather than assuming NB-07's success implies this works too.
- Real-model verification of `LlamaCppAdapter` (a pre-existing, older
  open item from before NB-07) remains **[UNKNOWN]** and is unrelated
  to this roadmap's track — not scheduled here, not forgotten either.

## 10. Rules for Amending This Roadmap

- A milestone's scope, acceptance criteria, or ordering may only be
  changed with repository evidence justifying the change, recorded in
  this file (matching the precedent set by
  `NB07_ARCHITECTURE_AMENDMENT.md`) — not silently rewritten.
- New milestones are appended, not inserted with reused numbers.
- If a milestone's "Exit criteria" turns out to have been declared met
  without real execution, that milestone is reopened, not
  reinterpreted.
- Deferred tracks (Section 7) are promoted to numbered milestones only
  once a specific prior milestone's results provide the evidence this
  document currently lacks to scope them concretely.
