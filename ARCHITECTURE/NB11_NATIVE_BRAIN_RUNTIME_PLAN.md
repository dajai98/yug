# NB-11 — Native Brain Runtime & Composition Regression: Architecture Plan

Status: **PLANNING ONLY. NO CODE HAS BEEN WRITTEN FOR NB-11.** Per this
milestone's authorization, this document is the sole output of this
session's NB-11 work. It does not authorize implementation.

Authorized by: `YUG — NB-10 FINAL CLOSURE + NB-11 PLANNING AUTHORIZATION`.
Grounded in `ARCHITECTURE/POST_NB07_ROADMAP.md` §6, NB-11 — but this
plan does **not** treat that definition as automatically correct
(per this milestone's own instruction to perform an adversarial
review first); Section C explains exactly where and why it diverges.

---

## A. Current-state audit

`[VERIFIED IN REPOSITORY]`, re-inspected this session:

- `yug/composition.py` ("M1 composition root") already exists.
  `_BACKEND_FACTORIES` currently registers exactly two backends:
  `"mock"` → `MockBrainAdapter`, `"llama_cpp"` → `LlamaCppAdapter`.
  **`YugNativeAdapter` is not registered.** `build_brain(brain_config)`
  raises `ConfigValidationError` for any unrecognized backend name —
  there is no silent fallback.
- `yug/config/schema.py`'s `_VALID_BRAIN_BACKENDS` is
  `frozenset({"llama_cpp", "mock"})` — same gap, consistently.
- `BrainConfig` (`yug/config/schema.py`) fields: `backend: str`,
  `model_id: str`, `model_path: str | None`, `context_length: int`,
  `n_threads: int | None`. **None of these can express what
  `YugNativeAdapter.__init__` actually needs**: a `checkpoint_dir`
  (a directory, not `model_path`'s single-file shape) and, critically,
  a **tokenizer object** — `BrainConfig` has no field of any kind for
  specifying or loading a tokenizer, and no existing code path loads a
  `ByteLevelBpeTokenizer` from configuration. `context_length` and
  `n_threads` are also not meaningful for this adapter as currently
  defined (`context_length` is authoritatively read from the
  checkpoint's own `model_config.json`; `n_threads` is a
  llama.cpp-specific concept).
- `yug/runtime/core.py` (`YugCore`) imports only `BrainInterface`,
  `Message`, `BrainError` from the brain package — zero concrete
  adapter references, `[VERIFIED IN REPOSITORY]` by direct inspection
  of its import list, not just by trusting its docstring.
- `grep -rn "LlamaCppAdapter\|MockBrainAdapter\|YugNativeAdapter\|llama_cpp\b" yug/runtime/ yug/memory/ yug/tools/ yug/verification/`
  → exactly one hit, a docstring mention in `yug/memory/__init__.py`
  (not an import, not a behavioral branch — already confirmed
  harmless during NB-10's own audit). No code-level coupling anywhere
  in the four subsystems the roadmap names.
- `YugCore.__init__(brain: BrainInterface, memory: WorkingMemory, executor=None, verification_engine=None, ...)`
  — `executor`/`verification_engine` are optional; `build_core()`'s own
  docstring confirms wiring them in was deliberately deferred to a
  separate decision, unrelated to brain backend choice.
- `YugCore.respond()` is a **synchronous** method that internally does
  `asyncio.run(self._brain.generate(messages, temperature=..., max_tokens=...))`
  — it never passes `tools` or `stop_sequences`, so
  `YugNativeAdapter`'s honest-rejection paths for those are never
  triggered by ordinary `respond()` usage. `asyncio.run()` wrapping a
  coroutine that itself calls `asyncio.to_thread(...)` internally
  (`YugNativeAdapter.generate()`'s own implementation) is a standard,
  supported pattern — no nested-event-loop conflict.
- `tests/test_runtime_core.py` already constructs `YugCore` with
  hand-rolled `BrainInterface` doubles (`_EchoBrain`, `_FailingBrain`)
  in addition to `MockBrainAdapter` — i.e. the repository already has
  standing, pre-NB-10 evidence that `YugCore` works with arbitrary
  `BrainInterface` implementations, not just the two registered
  adapters. NB-11's job is to add one more, *real* (not hand-rolled)
  confirmation using the actual `YugNativeAdapter`.
- `tests/test_composition.py::test_yugcore_does_not_import_concrete_adapters`
  already exists (from the prior "composition-root milestone"). It
  does a static-source-inspection check:
  `assert "LlamaCppAdapter" not in source`,
  `assert "MockBrainAdapter" not in source`,
  `assert "llama_cpp" not in source`,
  `assert "BrainInterface" in source`. **It does not check for
  `"YugNativeAdapter"`** — it predates NB-10 and was never updated,
  because updating it wasn't in NB-10's authorized scope.

`[VERIFIED BY ACTUAL RUN]` (project-owner-supplied, this session):
NB-10's real Windows/PyTorch verification — `16 passed` (dedicated
suite), `612 passed, 1 skipped` (full suite). This is NB-11's
regression baseline (Section B).

`[INFERENCE]`: nothing currently exercises `YugCore.respond()` with
`YugNativeAdapter` as the brain. This is the actual integration gap
(Section C) — not a missing production feature, a missing **test**.

## B. NB-10 closure evidence

Recorded as project-owner-supplied `[VERIFIED BY ACTUAL RUN]`, real
Windows/PyTorch environment:

```
python -m pytest tests/test_brain_yug_native_adapter.py -q
16 passed in 8.23s

python -m pytest tests -q
612 passed, 1 skipped in 18.91s
```

The single skip (`tests/test_brain_llama_cpp_adapter.py`, requires
`YUG_TEST_MODEL_PATH`) is pre-existing and unrelated to NB-10. This is
NB-11's baseline for Section J's regression-diff requirement.

## C. Problem statement

**Adversarial review finding, not assumed from the roadmap**:
`POST_NB07_ROADMAP.md`'s NB-11 definition says *"Confirm that adding a
third, real (non-mock) brain backend genuinely required zero changes
to `runtime/`, `memory/`, `tools/`, or `verification/`"* and
*"Implementation scope: no new production code expected."* Taken
literally against Section A's evidence, this is only achievable if
"adding a third backend" means *"constructing `YugCore` directly with
a `YugNativeAdapter` instance,"* **not** *"making it reachable through
`YugConfig`/`composition.py`'s factory."* The latter would require new
`BrainConfig` fields (at minimum, something that resolves to a
tokenizer instance) and a non-trivial factory function (unlike the
existing two backends' one-line lambdas) — genuinely new production
code, and a real, undesigned decision (how should configuration
specify a tokenizer?), not a mechanical extension.

This plan adopts the former reading, for two reasons: (1) it is the
reading under which the roadmap's own "no new production code"
constraint is actually true against the real repository, rather than
silently false; (2) `composition.py`'s own docstring frames itself as
the one place *allowed* to know about concrete adapters — a change
there (if one were needed) would not violate the isolation property
NB-11 exists to check, but Section A shows it isn't needed for the
isolation property itself, only for end-user config-file selection,
which is a separate concern this plan explicitly defers (Section E).

**The actual, narrow problem NB-11 solves**: nothing has yet exercised
`YugCore`'s runtime path (`respond()`, its `WorkingMemory`
persistence, its `BrainError`-vs-unexpected-exception handling) with a
real, loaded `YugNativeAdapter` as the brain, and the static isolation
check that is supposed to catch a future accidental coupling doesn't
yet know `YugNativeAdapter` exists.

## D. Scope

1. One new integration test exercising `YugCore.respond()` with a real
   (synthetic-checkpoint) `YugNativeAdapter` — request → generation →
   `Response.success=True` → history persisted correctly.
2. Extend the existing static isolation test
   (`tests/test_composition.py::test_yugcore_does_not_import_concrete_adapters`)
   with one additional assertion: `"YugNativeAdapter" not in source`.
3. Full regression suite run, diffed against Section B's baseline.

That is the complete scope. No production code.

## E. Non-goals

- Wiring `YugNativeAdapter` into `yug/composition.py`'s
  `_BACKEND_FACTORIES` / `yug/config/schema.py`'s
  `_VALID_BRAIN_BACKENDS` / extending `BrainConfig` — genuinely
  deferred (Section N), not silently dropped: it requires a real
  design decision (how configuration specifies a tokenizer) that this
  narrow validation milestone should not rush.
- Any new brain capability, agent/planner work, memory system changes,
  tool/verification changes (per the roadmap's own explicit non-goals,
  confirmed still correct by Section A's evidence).
- A dependency-injection framework, plugin registry, or any expansion
  of the selection mechanism beyond what already exists — Section L
  explains why the existing plain-dict factory remains sufficient and
  why this milestone doesn't even need to touch it.
- Streaming, sampling, tool execution through the native brain, or any
  other NB-10-scope item already explicitly closed.

## F. Architecture

No new composition path. NB-11 tests the *existing* one directly, with
`YugNativeAdapter` substituted in by hand (test code), not by
extending `composition.py`:

```
YugCore(brain=YugNativeAdapter(...), memory=WorkingMemory())
              │
              └── unchanged: YugCore only ever sees `BrainInterface`
```

`yug/composition.py` remains exactly as-is; this diagram intentionally
does **not** include it, because NB-11 does not go through it.

## G. Interface contracts

Frozen, unmodified by NB-11 (all re-confirmed by Section A's direct
inspection, not assumed): `BrainInterface`, `YugCore.__init__`/
`respond()`'s signature, `WorkingMemory`, `ToolExecutor`,
`VerificationEngine`, `YugNativeAdapter`'s public contract (NB-10,
closed), `yug.native.*` (NB-04–NB-10, closed), `training/checkpoint.py`
(NB-07/NB-10, closed), `training/evaluate.py` (NB-09, closed).

## H. Proposed files

```
NEW:
  tests/test_runtime_yug_native_regression.py

MODIFIED:
  tests/test_composition.py    (one assertion line added to an
                                 existing test)

Potentially:
  ARCHITECTURE/NB11_NATIVE_BRAIN_RUNTIME_PLAN.md   (this document,
                                 to record implementation-completion
                                 evidence once run for real)
```

No production file in `yug/` or `training/` is touched.

## I. Test plan

1. `test_yugcore_responds_using_yug_native_adapter` (new file) — build
   a tiny synthetic checkpoint (same fixture pattern as
   `tests/test_brain_yug_native_adapter.py`'s `_real_checkpoint`, kept
   self-contained in the new file per this repository's existing
   convention of per-file fixtures rather than a shared `conftest.py`
   — see Section N), construct `YugCore(YugNativeAdapter(...), WorkingMemory())`,
   call `core.respond("hello world")`, assert `response.success is True`
   and `response.content` is a string (not asserting exact generated
   text — this is an integration/wiring check, not a re-test of
   NB-10's own generation-correctness tests).
2. `test_yugcore_persists_history_with_yug_native_adapter` — two
   sequential `respond()` calls through the same `YugCore`, confirm
   the second call's prompt reflects the first turn (history
   round-trip through a real, non-mock brain).
3. `test_yugcore_reports_brain_error_from_yug_native_adapter` — force
   a `YugNativeAdapter` failure (e.g. point it at a missing checkpoint)
   and confirm `respond()` returns `Response.success=False` with a
   populated `error` field, never raising — exercises the exact
   `except BrainError` branch in `respond()` with a real adapter
   instead of `_FailingBrain`.
4. `tests/test_composition.py::test_yugcore_does_not_import_concrete_adapters`
   — add `assert "YugNativeAdapter" not in source`, in the same
   function, alongside the existing two assertions.
5. Full suite regression (`python -m pytest tests -q`), diffed against
   Section B's `612 passed, 1 skipped` baseline — expect
   `612 + (new test count) passed, 1 skipped`, zero failures.

## J. Acceptance criteria

- All new/modified tests pass under real pytest + real PyTorch
  (`[VERIFIED BY ACTUAL RUN]` required — this sandbox cannot execute
  them, per this project's standing, disclosed constraint since NB-04).
- Full suite shows zero regressions against the `612 passed, 1 skipped`
  baseline (i.e. exactly the baseline plus the new tests, nothing
  newly broken).
- The extended isolation test passes, confirming
  `yug/runtime/core.py`'s source still contains no `YugNativeAdapter`
  reference.
- No production file changed.

## K. Risks

- **Low**: modifying an existing test from a prior, already-closed
  milestone (`tests/test_composition.py`) rather than confined to
  purely new files — flagged explicitly per this project's convention
  of disclosing any touch to prior-milestone artifacts, even a
  one-line, clearly-in-spirit addition. Not expected to be
  controversial (the test's own stated purpose already covers exactly
  this case), but not silently done either.
- **Low**: `test_yugcore_reports_brain_error_from_yug_native_adapter`
  needs a way to force `YugNativeAdapter` into a failure state — using
  a missing/invalid `checkpoint_dir` (already a tested `ModelLoadError`/
  `ModelNotFoundError` path from NB-10) is the smallest, already-proven
  way to do this; no new failure-injection mechanism needed.
- **None anticipated beyond these** — consistent with the roadmap's own
  "this is a validation milestone by design" framing, which Section A's
  evidence supports: the isolation property already holds structurally
  (pre-dating NB-10, per `_EchoBrain`/`_FailingBrain`'s existence), so
  NB-11 is confirming an existing property with a real backend, not
  discovering or building a new one.

## L. Alternatives considered

- **No factory/selector**: not applicable — one already exists
  (`yug/composition.py`) and already works for two backends. The
  question is only whether to *extend* it now.
- **Extend the existing minimal factory now** (add `"native"` to
  `_BACKEND_FACTORIES` + `_VALID_BRAIN_BACKENDS` + new `BrainConfig`
  fields): rejected for NB-11 specifically — Section C explains this
  requires a real, undesigned decision (tokenizer specification via
  config) that a narrow regression-validation milestone shouldn't rush
  into as a side effect. Worth doing eventually (Section N), under its
  own explicit authorization, once that design question has an answer.
- **Registry/plugin architecture**: firmly premature — the existing
  plain-dict factory handles two backends with zero friction and a
  third would be one more entry once the config-shape question is
  resolved; nothing here justifies discovery/plugin machinery, and
  introducing one would itself be exactly the kind of premature system
  this project's stated priorities warn against.
- **Conclusion**: the existing minimal factory is sufficient in
  principle; NB-11 doesn't need to touch it at all, and extending it
  is correctly deferred, not because the mechanism is inadequate but
  because the *config schema* for the native backend hasn't been
  designed yet.

## M. Exit criteria

NB-11 may be marked `PASS` when, in a real PyTorch environment:

1. `tests/test_runtime_yug_native_regression.py`'s 3 tests pass.
2. The extended `test_yugcore_does_not_import_concrete_adapters`
   passes.
3. Full suite: `612 + 4` (3 new + 1 extended-in-place, so `+3` net new
   test functions) `passed, 1 skipped`, zero failures — exact number
   to be confirmed against whatever the real run reports, not
   pre-committed here as fact.
4. No production file changed (verified by diff/checksum, as done at
   every prior milestone this session).

## N. Deferred work

Explicitly NOT part of NB-11, recorded so it isn't silently lost:

- Wiring `YugNativeAdapter` into `yug/composition.py` /
  `yug/config/schema.py` for end-user, config-file-driven selection —
  blocked on deciding how `BrainConfig` should specify a tokenizer
  (a path to a saved tokenizer file? a checkpoint-relative default?
  this needs its own small design pass, not a rushed field addition).
- Extracting shared test fixtures (tiny tokenizer/checkpoint builders)
  currently duplicated across `tests/test_training_evaluate.py`,
  `tests/test_brain_yug_native_adapter.py`, and this milestone's new
  file into a shared `conftest.py` — a reasonable future cleanup, not
  attempted here to avoid an unrelated refactor riding along with a
  validation milestone.
- Everything the roadmap and this session's authorization already
  name as out of scope: agent runtime, planner, memory system changes,
  event bus, tool execution expansion, sampling, streaming, multimodal,
  distributed inference, model routing, RLHF/DPO, MoE, cloud inference,
  retraining orchestration, advanced sandboxing, autonomous task
  execution.

---

## NB-11 READINESS DECISION

**NB-11 READY FOR IMPLEMENTATION**

Every question this planning task asked has a concrete,
repository-verified answer (Section A), the scope is a strictly
bounded 3-new-tests-plus-1-line-extension change (Section D/H), zero
production code is required once "adding a third backend" is read as
Section C justifies, and the one deliberate divergence from the
roadmap's literal wording is disclosed and reasoned, not silently
decided. The only genuine open design question found (native-backend
config-schema wiring, Section E/L/N) is correctly scoped *out* of
NB-11 rather than blocking it.
