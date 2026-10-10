# NB-10 — YugNativeAdapter: Architecture & Implementation-Readiness Plan

Status: **PLANNING ONLY. NO CODE HAS BEEN WRITTEN FOR NB-10.** This
document is the sole output of the NB-10 planning task. It does not
authorize implementation; a separate authorization is required before
any of Section 11's steps may be carried out.

Authorized by: `YUG — NB-10 ARCHITECTURE PLANNING ONLY` (this session).
Grounded in `ARCHITECTURE/POST_NB07_ROADMAP.md` §6, NB-10.

---

## 1. Current repository state

`[VERIFIED IN REPOSITORY]` only, re-inspected this session:

- `yug/brain/interface.py` — `BrainInterface(ABC)`, 4 abstract methods:
  `generate`, `embed`, `get_capabilities`, `health_check`. Unchanged
  since NB-01.
- `yug/brain/types.py` — `Message`, `ToolDefinition`, `GenerationResult`,
  `Embedding`, `ModelCapabilities`, `HealthState`, `HealthStatus`. All
  frozen dataclasses / a `str` enum.
- `yug/brain/errors.py` — `BrainError` and 5 subclasses:
  `ModelLoadError`, `ModelNotFoundError(ModelLoadError)`,
  `GenerationError`, `EmbeddingNotSupportedError`,
  `BrainUnavailableError`.
- `yug/brain/adapters/llama_cpp.py` (224 lines) — the only existing
  real-backend adapter. Lazy model load (`_ensure_loaded`, first call
  to `generate`/`embed` only); `llama_cpp` imported inside
  `_load_model_sync`, not at module scope, so the module itself always
  imports cleanly regardless of whether the optional dependency is
  installed; `get_capabilities()`/`health_check()` answer from
  configuration/filesystem state, never forcing a load;
  backend exceptions are caught and re-raised as `BrainError`
  subclasses, never silently swallowed or faked.
- `yug/brain/adapters/mock.py` (98 lines) — dependency-free reference
  implementation, always `HEALTHY`, deterministic synthetic output.
- `yug/native/config.py` — `ModelConfig` (`.save()`/`.load()`/
  `.identity`, JSON only, **zero PyTorch dependency**, confirmed via
  `pyproject.toml`'s own comment: *"`yug/native/config.py` (NB-03) has
  zero PyTorch dependency and stays that way; only `yug/native/model.py`
  (NB-04) requires it."* Also defines `estimate_parameter_count(config)
  -> ParameterEstimate` (`.total`, `.breakdown`, `.basis`) —
  deterministic, config-only, no torch needed, no loaded model needed.
- `yug/native/model.py` — `YugTransformer(nn.Module)`, imports `torch`
  at module scope (unconditional). `forward(input_ids) -> logits`
  (validates 2D shape, `context_length`, vocab range — raises
  `yug.native.errors.NativeError`). `count_parameters()` returns the
  real, measured count once instantiated.
- `yug/native/generation.py` (NB-06) — `generate(model, input_ids,
  max_new_tokens, stop_token_id=None) -> Tensor`. **Deterministic
  greedy (argmax) only** — no temperature, no top-k/top-p, no
  sampling of any kind. `stop_token_id` is a single optional `int`,
  not a list of strings. Imports only `torch`, `yug.native.errors`,
  `yug.native.model` — no tokenizer, no `BrainInterface`, no `training`.
- `yug/tokenizer/bpe.py` — `ByteLevelBpeTokenizer`, `.identity`
  (sha256), `.encode(text, add_bos=..., add_eos=...) -> list[int]`,
  `.decode(ids) -> str`, `.save()`/`.load()`.
- `training/checkpoint.py` (NB-07) — `save_checkpoint`/`load_checkpoint`.
  Format: `model_config.json` (via `ModelConfig.save()`),
  `metadata.json` (plain JSON: step, `tokenizer_identity`,
  `training_config`, RNG state), `state.pt` (`torch.save({"model":
  model.state_dict(), "optimizer": optimizer.state_dict()}, ...)`).
  `load_checkpoint(directory, *, model, optimizer, restore_rng_state=True)`
  **requires both an already-constructed model and optimizer**; it
  restores both, plus (by default) RNG state.
- `training/evaluate.py` (NB-09) — precedent for the
  checkpoint→config→model→forward-pass pattern this milestone will
  reuse; already solves tokenizer-identity verification and the
  throwaway-optimizer problem, but does so *from inside `training/`*,
  where importing `training.checkpoint` is unproblematic.
- `pyproject.toml` — `torch>=2.0` is the `native` extra, not core.
  Base package (Phase 0 CLI, `BrainInterface`, `MockBrainAdapter`,
  `LlamaCppAdapter`, tokenizer, `ModelConfig`) has zero PyTorch
  dependency. Only importing `yug.native.model` (or a module that
  imports it) pulls PyTorch in.
- Dependency boundary, re-verified this session: `grep -rn "^import
  training\|^from training" yug/` → zero matches. Holds.
- No `YugNativeAdapter` exists anywhere (`grep -rl "YugNativeAdapter"
  .` → zero matches, verified again this session).
- Existing adapter test conventions
  (`tests/test_brain_llama_cpp_adapter.py`): three explicit categories
  — UNIT TEST (no dependency needed), INTEGRATION TEST
  (dependency-conditional, tests honest failure), REQUIRES REAL MODEL
  (skipped unless an env var points at a real file).

## 2. Existing architecture (current dependency flow)

```
BrainInterface (yug/brain/interface.py)
   ├── MockBrainAdapter        (yug/brain/adapters/mock.py)
   └── LlamaCppAdapter         (yug/brain/adapters/llama_cpp.py)
                                    │
                                    └── llama_cpp (lazy-imported, optional)

yug/native/config.py   (ModelConfig, estimate_parameter_count — no torch)
yug/native/model.py    (YugTransformer — torch, no BrainInterface, no training)
yug/native/generation.py (generate() — torch, no tokenizer, no BrainInterface)
yug/tokenizer/bpe.py   (ByteLevelBpeTokenizer — no torch, no BrainInterface)

training/checkpoint.py (save_checkpoint/load_checkpoint — imports yug.*,
                         never imported by yug/*)
```

**`yug/native/*` and `yug/brain/*` do not currently know about each
other at all** — `[VERIFIED IN REPOSITORY]`, zero cross-imports either
direction. NB-10 is the first thing that connects them.

## 3. NB-10 objective

Implement `YugNativeAdapter`, a `BrainInterface` implementation under
`yug/brain/adapters/yug_native.py`, that loads a native YUG checkpoint
(`ModelConfig` + `YugTransformer` weights + a matching tokenizer) and
answers `generate()`/`get_capabilities()`/`health_check()` using the
existing NB-04–NB-07 native stack — reusing, not duplicating, every
piece of that stack. `embed()` is not supported (the native model has
no embedding head) and must say so honestly via
`EmbeddingNotSupportedError`, exactly as the interface contract
requires for a backend without that capability.

## 4. Proposed adapter contract

`[PROPOSAL]` — mirrors `LlamaCppAdapter`'s established shape:

```python
class YugNativeAdapter(BrainInterface):
    def __init__(
        self,
        checkpoint_dir: str,
        tokenizer: ByteLevelBpeTokenizer,   # already loaded by caller
        model_id: str | None = None,
        device: str = "cpu",
        max_new_tokens_default: int = 256,
    ) -> None:
        ...  # records config only; NO torch/model import at module
             # scope (see Section 9, Risk 2) and NO load yet.

    async def generate(self, messages, tools=None, temperature=0.7,
                        max_tokens=1024, stop_sequences=None) -> GenerationResult: ...
    async def embed(self, texts: list[str]) -> list[Embedding]:
        raise EmbeddingNotSupportedError(...)   # always, unconditionally
    def get_capabilities(self) -> ModelCapabilities: ...
    def health_check(self) -> HealthStatus: ...
```

Per-method responsibilities:

- **`__init__`**: records `checkpoint_dir`, `tokenizer`, `device`
  config only. Does **not** import `torch` or `yug.native.model` at
  this point (Section 9, Risk 2) — but **may** eagerly call
  `ModelConfig.load(checkpoint_dir / "model_config.json")`, since that
  specific call has zero torch dependency (`[VERIFIED IN REPOSITORY]`,
  Section 1) and lets `get_capabilities()` answer immediately without
  a full load, matching `LlamaCppAdapter`'s "must not require a loaded
  model" contract — and improving on it, since a real
  `parameter_count` (via `estimate_parameter_count`) is available
  before any weights are loaded, unlike `LlamaCppAdapter`'s permanent
  `parameter_count=None`.
- **`generate()`**: on first call, lazily constructs `YugTransformer`
  from the already-loaded `ModelConfig`, loads its weights (Section 6
  — this is the one genuinely open question in this plan), moves it to
  `device`. Verifies `tokenizer.identity == model_config.tokenizer_identity`
  before use (exact pattern already proven in `training/evaluate.py`).
  Encodes `messages` into a single prompt string (concatenation
  strategy is an adapter-internal detail, not a `BrainInterface`
  concern — proposed: join `f"{role}: {content}"` lines, ending with an
  `"assistant:"` cue, but this is cosmetic and can be revisited without
  touching the interface), tokenizes via `tokenizer.encode(..., add_bos=True)`,
  calls `yug.native.generation.generate(model, input_ids, max_new_tokens=max_tokens)`
  **unmodified**, decodes the newly generated suffix via
  `tokenizer.decode(...)`, wraps it in a `GenerationResult`.
  - `temperature`: `[PROPOSAL]` ignored — NB-06's `generate()` is
    argmax-only with no sampling path at all (`[VERIFIED IN
    REPOSITORY]`, Section 1). Documented as a `known_limitations`
    entry on `get_capabilities()`, exactly as `LlamaCppAdapter`
    documents its own unsupported features, rather than raising for
    every call (default `temperature=0.7` would make that the common
    case, not an edge case).
  - `stop_sequences`: `[PROPOSAL]` any non-empty value raises
    `GenerationError` (matching `LlamaCppAdapter`'s exact precedent for
    unsupported `tools`: *"be explicit rather than silently ignoring
    caller intent"*). NB-06's `generate()` only accepts a single opaque
    `stop_token_id: int`, not a list of strings; silently mapping or
    truncating caller-supplied stop strings would risk exactly the
    kind of fabricated-conformance this project's error-handling
    philosophy exists to prevent.
  - `tools`: `[PROPOSAL]` non-empty raises `GenerationError`,
    identical to `LlamaCppAdapter`'s existing behavior — no new
    decision needed here, just consistency.
- **`embed()`**: always raises `EmbeddingNotSupportedError` — the
  native model has no embedding head or pooling mechanism of any kind
  (`[VERIFIED IN REPOSITORY]`: `yug/native/model.py` defines no such
  method).
- **`get_capabilities()`**: `model_id` (config path stem or explicit
  override, matching `LlamaCppAdapter`'s convention), `context_length`
  (from `ModelConfig`, no load needed), `supports_tools=False`,
  `supports_vision=False`, `embedding_dim=None`, `parameter_count`
  from `estimate_parameter_count(model_config).total` before load, or
  the model's own `count_parameters()` once actually loaded (both are
  documented to agree — `[VERIFIED IN REPOSITORY]`, `count_parameters()`'s
  own docstring states it matches `estimate_parameter_count()`'s
  intent), `known_limitations` listing: no tool calling, no vision, no
  embeddings, temperature/sampling ignored (greedy only), stop_sequences
  unsupported.
- **`health_check()`**: mirrors `LlamaCppAdapter`'s three-state pattern
  exactly — `UNAVAILABLE` if `torch` isn't importable or
  `checkpoint_dir` doesn't contain a valid `model_config.json`;
  `DEGRADED` if config is valid but weights haven't been loaded yet;
  `HEALTHY` only once a real load has actually succeeded. Never
  reports `HEALTHY` without having checked, per `HealthStatus`'s own
  documented contract.

## 5. Integration flow

```
checkpoint_dir/model_config.json
        ↓  ModelConfig.load()            [yug/native/config.py, no torch]
   ModelConfig
        ↓  YugTransformer(config)        [yug/native/model.py, torch]
   YugTransformer (freshly initialized weights)
        ↓  <load real weights>           [Section 6 — open question]
   YugTransformer (real weights)
        ↓  tokenizer.encode(prompt)      [yug/tokenizer/bpe.py]
   input_ids
        ↓  yug.native.generation.generate(model, input_ids, ...)  [unmodified NB-06]
   output token ids
        ↓  tokenizer.decode(new_ids)
   text
        ↓  wrap
   GenerationResult
```

## 6. Reuse map

| Component | Reused as-is? | Notes |
|---|---|---|
| `BrainInterface` | Yes, unmodified | No incompatibility found requiring a change (Section 1 checks all pass) |
| `ModelConfig.load/.save/.identity` | Yes, unmodified | Zero-torch, safe to call eagerly |
| `estimate_parameter_count` | Yes, unmodified | Pre-load capability reporting |
| `YugTransformer` | Yes, unmodified | Constructed, not subclassed |
| `yug.native.generation.generate` | Yes, unmodified | Called directly; no wrapping logic that changes its behavior |
| `ByteLevelBpeTokenizer` | Yes, unmodified | Caller-supplied instance, identity-checked |
| `training.checkpoint.load_checkpoint` | **Open question — see Section 9, Risk 1** | Cannot be imported from `yug/` without reversing the frozen `training → yug` boundary |

## 7. Test plan

`[PROPOSAL]`, following the existing three-category convention
(`tests/test_brain_llama_cpp_adapter.py`):

**Required (minimal, justified individually):**
1. `test_yug_native_adapter_satisfies_brain_interface` — UNIT, no torch
   needed beyond `isinstance` check against the ABC.
2. `test_module_importable_without_torch_installed` — UNIT. Only
   meaningful, and only necessary, if Section 9 Risk 2 (lazy-import
   deferral) is adopted; if it is not adopted this test would fail by
   design and should be dropped, not weakened.
3. `test_get_capabilities_does_not_require_a_loaded_model` — UNIT.
   Verifies `context_length`/`parameter_count` are answerable from
   `model_config.json` alone (mirrors the `LlamaCppAdapter` test of
   the same name, but this adapter should assert `parameter_count is
   not None`, an improvement over the sibling adapter).
4. `test_native_checkpoint_loading` — INTEGRATION. Round-trips a tiny
   `YugTransformer` through `save_checkpoint` then the adapter, exactly
   as `tests/test_training_evaluate.py`'s
   `test_evaluate_checkpoint_round_trip` already does for NB-09 — same
   fixture pattern, reused.
5. `test_generation_is_deterministic` — INTEGRATION. Same
   prompt/checkpoint, two calls, identical `GenerationResult.content`
   (dropout is off in eval, per NB-06/NB-09 precedent already
   established).
6. `test_tokenizer_identity_mismatch_raises` — INTEGRATION. Mirrors
   `training/evaluate.py`'s existing, tested behavior.
7. `test_health_check_reports_actual_state` — UNIT/INTEGRATION mix,
   three sub-cases mirroring `LlamaCppAdapter`'s own health-check test
   suite (`UNAVAILABLE` on missing/invalid checkpoint dir, `DEGRADED`
   before first load, `HEALTHY` after).
8. `test_invalid_checkpoint_raises_model_load_error` — INTEGRATION.
   Missing/corrupt `model_config.json` or `state.pt`.
9. `test_llama_cpp_adapter_still_passes` — REGRESSION. Not a new test;
   confirms the existing `tests/test_brain_llama_cpp_adapter.py` suite
   is unaffected by NB-10's changes (no shared state, no modified
   files under `yug/brain/`).

**Deliberately not proposed:** a `REQUIRES REAL MODEL`-style test
gated on a real NB-08 checkpoint. Per NB-09's closure audit, no such
checkpoint currently exists in this repository, and this plan does not
propose fabricating one merely to give NB-10 a fixture — tests 4–6, 8
above use a tiny synthetic checkpoint built in-test (identical pattern
to NB-09), which is sufficient to verify the adapter's own logic
without needing a trained model.

## 8. Non-goals

Explicitly out of scope for NB-10 (per this task's authorization and
`ARCHITECTURE/POST_NB07_ROADMAP.md`'s own NB-10 definition): sampling
(temperature/top-k/top-p), multi-token/string `stop_sequences`,
embeddings, tool execution, KV-cache/generation-speed optimization,
batch generation (`BrainInterface.generate` is single-conversation),
GPU/CUDA-specific code paths beyond an optional `device` string,
memory, planning, verification, autonomy, any second
tokenizer/checkpoint-loader implementation, and any change to
`BrainInterface`, `YugTransformer`, `yug.native.generation`, NB-08, or
NB-09.

## 9. Risks

**Risk 1 — `[VERIFIED IN REPOSITORY]`, genuinely unresolved, blocking:**
`training.checkpoint.load_checkpoint()` is the only existing code that
reads `state.pt`, but it lives in `training/`, and `yug/` must never
import `training/` (`[VERIFIED IN REPOSITORY]`, re-checked Section 1,
zero exceptions currently exist). `YugNativeAdapter` lives under
`yug/brain/adapters/` and needs to read that same `state.pt` file for
production inference. This is a direct conflict between two explicit,
independently-stated constraints for this milestone: *"reuse existing
infrastructure... do not create a second checkpoint loader"* vs. the
frozen one-way dependency boundary. Two ways to resolve it, presented
without a unilateral choice being made here:

  - **Option A** `[PROPOSAL]`: extract the minimal, read-only part of
    checkpoint reading (open `state.pt`, return `state["model"]`) into
    a new, tiny module under `yug/` (e.g.
    `yug/native/checkpoint_io.py`), with zero optimizer/RNG-state
    concepts (those are training-only). `training/checkpoint.py`'s
    `load_checkpoint` would then be modified to call this shared
    function internally instead of duplicating the `torch.load` line,
    guaranteeing exactly one place knows the on-disk layout. This
    requires touching an existing NB-07 file (`training/checkpoint.py`)
    — a small, additive change, but one that needs explicit sign-off
    before NB-10 implementation begins, per this project's own "stop
    and report an unavoidable change" convention.
  - **Option B** `[PROPOSAL]`: `YugNativeAdapter` reads `state.pt`
    directly (`torch.load(...)["model"]`, then
    `model.load_state_dict(...)`) without touching `training/` at all.
    Two lines of logic, needing only the same two facts
    `training/checkpoint.py`'s own docstring already documents as a
    stable, explicit format contract (file name, dict key). Zero
    modification to existing training code. The one risk is a
    values-judgment about whether this counts as "a second checkpoint
    loader" in the sense NB-10's authorization meant to forbid — this
    plan's reading is that the forbidden thing is a second *save/load-
    with-training-semantics* implementation (optimizer state, RNG
    resumption, versioning), not a minimal, inference-only reader of an
    already-documented file format, but that reading is not this
    document's to finalize unilaterally.

  **This plan recommends Option B as the smaller, boundary-respecting
  default, with Option A as the more DRY long-term alternative** — but
  flags this explicitly as the one decision that must be made before
  implementation, rather than deciding it here.

**Risk 2 — `[VERIFIED IN REPOSITORY]`, non-blocking, addressed by
design in Section 4:** `yug/native/model.py` and
`yug/native/generation.py` import `torch` unconditionally at module
scope (unlike `llama_cpp`, which `LlamaCppAdapter` imports lazily
inside `_load_model_sync` specifically so its own module stays
importable without the optional dependency). If `yug_native.py`
imports `yug.native.model` at module scope, importing
`yug.brain.adapters.yug_native` would require the `native` extra
installed, breaking the project's established "only importing that
specific submodule pulls PyTorch in" design goal
(`[VERIFIED IN REPOSITORY]`, `pyproject.toml`'s own comment). Proposed
fix: defer those imports to inside the adapter's lazy-load method,
exactly mirroring `LlamaCppAdapter`'s existing pattern for `llama_cpp`.
Not a blocker — a design detail with clear precedent to follow.

**Risk 3 — `[INFERENCE]`, low severity:** the message-list-to-prompt-
string concatenation strategy in `generate()` is not specified by any
existing document (no chat template exists anywhere in the repository
for the native model — `[VERIFIED IN REPOSITORY]`, `grep -r "chat_template"
yug/` → zero matches). This is a genuinely new design decision NB-10
introduces, not a reuse of anything existing. Low severity because it's
easily changed later without touching `BrainInterface` or the native
stack — it's purely internal to the adapter.

## 10. Exit criteria

NB-10 may be declared complete when, in an environment with the
`native` extra installed:

1. All Section 7 tests pass under real `pytest` (not the shim) —
   `[VERIFIED BY ACTUAL RUN]` required, matching this project's
   standing evidence bar.
2. `tests/test_brain_llama_cpp_adapter.py` and
   `tests/test_brain_mock_adapter.py` still pass unchanged (regression
   check).
3. `grep -rn "^import training\|^from training" yug/` still returns
   zero matches.
4. `YugNativeAdapter` can be constructed, report capabilities, and run
   `health_check()` without the `native` extra installed, failing only
   `generate()`/`embed()` with a clear `ModelLoadError`/`BrainError` —
   i.e. Risk 2 is resolved, not merely noted.
5. Risk 1 has been explicitly decided (Option A or B, or a third option)
   by whoever authorizes implementation — not left implicit in the code.
6. `ARCHITECTURE/NB09_MINIMAL_EVALUATION_HARNESS.md`'s open item (no
   real NB-08 checkpoint) remains untouched by this milestone — NB-10
   does not require a trained checkpoint to be considered complete
   (Section 7's synthetic-checkpoint tests are sufficient), consistent
   with this project's decision not to manufacture one just to close a
   checkbox.

## 11. Implementation sequence

`[PROPOSAL]`, smallest sequential order, assuming Risk 1 is resolved
first:

1. Resolve Risk 1 (explicit decision, outside this document).
2. If Option A: small, isolated PR to `training/checkpoint.py`
   extracting the shared reader, verified against NB-09's existing
   tests (regression check) before touching anything else.
3. `yug/brain/adapters/yug_native.py`: constructor + `get_capabilities()`
   + `health_check()` only (all answerable pre-load, per Section 4) —
   smallest independently-testable slice first.
4. Lazy load path (checkpoint → config → model → weights), with Risk 2's
   deferred-import pattern from the start, not retrofitted.
5. `generate()`, reusing `yug.native.generation.generate` unmodified.
6. `embed()` (one line — always raises).
7. Full Section 7 test suite.
8. Regression run of the full existing suite.

---

## NB-10 IMPLEMENTATION READINESS:

**BLOCKED**

Not blocked on missing information about the existing codebase — every
question this task asked (BrainInterface contract, native model
integration, generation path, checkpoint path, LlamaCppAdapter
coexistence, device handling, error boundaries) has a concrete,
repository-verified answer above. Blocked specifically on **Risk 1**
(Section 9): implementing `YugNativeAdapter`'s checkpoint loading today
would require silently violating one of two explicit, independently-
stated constraints for this milestone (reuse the existing checkpoint
loader vs. never let `yug/` import `training/`). That is a one-decision
blocker, not a broad architectural gap — once Risk 1 is decided, this
plan's assessment is that implementation could begin immediately
against Section 11's sequence.

---

## Implementation status (added post-authorization; plan above unchanged)

Risk 1 was resolved as **Option A** (explicit authorization: `YUG — NB-10
IMPLEMENTATION AUTHORIZATION`): `yug/native/checkpoint_io.py` now holds
the single, authoritative, inference-only reader of the model-state
portion of `state.pt`; `training/checkpoint.py` was modified to call it
instead of independently reading that same entry, preserving all
optimizer/RNG/training-resume behavior unchanged.

Full implementation evidence, exact test/verification results, and
final disposition are recorded in the separate **NB-10 IMPLEMENTATION
REPORT** produced this session — this document is not duplicated with
that evidence to avoid two sources of truth drifting apart.
