# YUG — Project Handoff

**Purpose:** orientation for a fresh conversation/engineer picking up this
project with no prior context. This is a summary and pointer document; the
canonical, authoritative record is
`docs/architecture/NATIVE_BRAIN_ARCHITECTURE.md` — read that in full before
making any architectural decision. This handoff does not supersede it.

**Generated from:** direct inspection of the repository and the architecture
document as they exist right now. Where this document says `[VERIFIED]`, it
was checked this session (file exists, test suite run, dependency import
attempted, etc.) — not carried over from memory.

---

## 1. What YUG is

`[VERIFIED — from architecture doc]` YUG is a personal, local-first,
increasingly agentic AI system. Its explicit long-term goal is to own its
own brain (tokenizer, model architecture, training pipeline, checkpoint
format, inference runtime) rather than being a wrapper around an external
LLM. External models (e.g. via llama.cpp) are permanently classified as
optional baselines/development tools — never YUG's architectural identity.

---

## 2. Repository structure (verified this session)

```
yug/
  __init__.py, __main__.py           Phase 0: CLI, --version
  config/                            Phase 0: YugConfig (logging, paths, brain backend selection)
  utils/                             Phase 0: shared types, error hierarchy, structured JSON logging
  brain/
    interface.py                     BrainInterface (ABC): generate/embed/get_capabilities/health_check
    types.py, errors.py, manifest.py Message/GenerationResult/etc., BrainError hierarchy, ModelManifest
    adapters/
      mock.py                        MockBrainAdapter — dependency-free, deterministic
      llama_cpp.py                   LlamaCppAdapter — external baseline, lazy-imports llama_cpp
    model_zoo/manifests.json         empty array — no model bundled
  tokenizer/
    bpe.py, errors.py                ByteLevelBpeTokenizer — YUG-owned, no external tokenizer dep
  native/
    config.py, errors.py             ModelConfig, estimate_parameter_count() — zero PyTorch dependency
    model.py                         YugTransformer — requires PyTorch (optional extra)
tests/                                184 test functions across 18 files (see §7)
docs/architecture/
  NATIVE_BRAIN_ARCHITECTURE.md        THE canonical architecture record — read this fully
pyproject.toml
```

`training/` does **not** exist yet — by design (see §5 boundaries).

---

## 3. Completed milestones

| Milestone | Status | Deliverable |
|---|---|---|
| Phase 0 | `[VERIFIED]` Complete | Config, logging, types, error hierarchy, CLI |
| Phase 1 | `[VERIFIED]` Complete | `BrainInterface`, `MockBrainAdapter`, `LlamaCppAdapter`, `ModelManifest` |
| NB-01 | `[VERIFIED]` Complete | `docs/architecture/NATIVE_BRAIN_ARCHITECTURE.md` itself |
| NB-02 | `[VERIFIED]` Complete | `ByteLevelBpeTokenizer` (`yug/tokenizer/bpe.py`) |
| NB-03 | `[VERIFIED]` Complete | `ModelConfig` + `estimate_parameter_count()` (`yug/native/config.py`) |
| NB-04 | `[VERIFIED — code complete]` / `[UNVERIFIED — execution]` | `YugTransformer` (`yug/native/model.py`) — see §6, critical caveat |
| NB-05 onward | Not started | See roadmap in the architecture doc, §"ROADMAP" |

---

## 4. Key architectural decisions already made

Full list with reasoning: architecture doc, "ARCHITECTURAL DECISIONS"
section (26 items). Highlights most likely to matter immediately:

- `BrainInterface` is frozen, inference-time-only. Never add training
  concepts to it. A future native brain implements the same four methods
  any adapter does — the interface itself does not change.
- `LlamaCppAdapter` is permanently an external-baseline adapter — never
  promote it, never remove it without cause, never base YUG's own
  checkpoint/tokenizer/architecture on it or on GGUF.
- Tokenizer: byte-level BPE, no `<unk>` (structurally unnecessary), no
  word-level pre-tokenization regex, JSON serialization with a `sha256`
  identity hash, vocabulary size is a `train()`-time parameter (not
  frozen).
- Model architecture family: decoder-only, causal, pre-norm Transformer
  (frozen as of NB-04). Exact hyperparameters are **not** frozen (see §5).
- `yug/native/` is the canonical home for YUG-owned model config,
  architecture, and inference runtime. `training/` (not yet created) is
  strictly separate — production code must never import from it.
- PyTorch is authorized as the numerical/autograd substrate, added as an
  **optional** dependency (`pip install -e ".[native]"`), not core —
  `yug.native.config` and the package's base import surface stay
  PyTorch-free; only importing `yug.native.model` pulls PyTorch in.
- Checkpoints will be a directory (`config.json`, `tokenizer/`,
  `weights.safetensors`, `metadata.json`), never pickle, never GGUF —
  exact metadata schema still open.
- Each NB milestone requires its own explicit authorization before
  implementation begins — do not proceed past what's authorized.

---

## 5. Open decisions (genuinely unresolved — do not silently pick one)

`[from architecture doc, "OPEN DECISIONS" — unchanged as of this handoff]`

- Exact production tokenizer vocabulary size (proposed range 4K–8K, not fixed).
- Whether a `pad` special token (or others beyond `bos`/`eos`) belongs in
  the production special-token set.
- Exact model hyperparameters (layer count, hidden size, head count, FFN
  size, context length, dropout, normalization variant).
- Exact checkpoint metadata schema.
- Target training hardware (CPU-only vs. GPU access) — materially affects
  realistic timeline for training milestones.
- **Whether `yug/native/model.py` actually runs correctly** — see §6.
- Whether residual-branch dropout should be added (currently only
  attention-weight dropout is wired in).

---

## 6. Critical caveat: NB-04 is unexecuted

`[VERIFIED]` As of this handoff, PyTorch is **not installed** in this
development sandbox and **cannot be installed** (no network access — `pip
install torch` fails with no matching distribution found).

This means `yug/native/model.py` (`YugTransformer` and all its components:
`RMSNorm`, RoPE helpers, `CausalSelfAttention`, `FeedForward`,
`TransformerBlock`) and its entire test file (`tests/test_native_model.py`,
31 test functions, several parametrized) have **never been executed**.
They are syntactically valid (`py_compile` and `ast.parse` both pass) and
were carefully hand-derived to match `estimate_parameter_count()`'s formula
exactly (verified by hand for two representative configurations — see the
NB-04 implementation report / architecture doc Decision #26) — but this is
categorically weaker than a real test run.

**Before treating NB-04 as genuinely done, someone needs to, in an
environment with network access:**
```
pip install -e ".[native]"
pytest tests/test_native_model.py -v
```
and report back what actually happens — the same pattern already used
successfully once in this project (a real Windows-side `subprocess`/stdin
bug was found and fixed this way in Phase 0). Do not assume NB-04 passes
until this happens.

---

## 7. Tests

`[VERIFIED — run this session]`

```
169 passed, 0 failed, 0 errored, 2 skipped, 171 total
```
(counting only tests runnable without PyTorch, via a custom pytest-shim
harness — see below). The 2 skips: one pre-existing real-llama.cpp-model
integration test (needs `YUG_TEST_MODEL_PATH` + `llama-cpp-python`), and
the entire `test_native_model.py` module (needs PyTorch).

**Important environment note:** the real `pytest` executable has never
been successfully run by the assistant in this sandbox either — no network
access to `pip install pytest`. All "tests pass" claims in this project's
history (Phase 0 through NB-04) were verified via a custom lightweight
pytest-compatible shim (`fake_pytest.py` + `run_tests.py`, not part of the
repository) that imports the actual, unmodified test files and executes
them directly, supporting `pytest.raises`, `tmp_path`, `capsys`,
`pytest.mark.parametrize`, and `pytest.importorskip`. This gives real
confidence for pure-Python code (Phase 0–NB-03, all genuinely exercised
this way) but is not a substitute for an actual `pytest` run — the person
running this repository outside this sandbox should run real `pytest -v`
and `ruff check .` themselves and treat that as authoritative. (This has
already happened once successfully for Phase 0, on a Windows machine with
real network access, which is how the WinError 6 bug in §6-adjacent
history was found and fixed.)

Test file count: 18 files, 184 base test functions (several files use
`pytest.mark.parametrize`, so actual executed test count is higher — see
the `run_tests.py` output for exact expansion).

---

## 8. Dependencies

`[VERIFIED]`, from `pyproject.toml`:
- Core: `pyyaml>=6.0` only.
- `dev` extra: `pytest>=7.4`, `ruff>=0.4`.
- `native` extra: `torch>=2.0` (added in NB-04; optional, not core — see §4).
- `llama-cpp-python` is referenced by `LlamaCppAdapter` but is **not** a
  listed dependency anywhere (lazy-imported, genuinely optional).

None of `pytest`, `ruff`, or `torch` are installed in this development
sandbox (no network access throughout this project's history so far).

---

## 9. Constraints / invariants that must not be silently violated

`[VERIFIED — from architecture doc]`
- Never add training concepts (gradients, optimizer, loss, dataset
  handling) to `BrainInterface`.
- Never make `LlamaCppAdapter`, llama.cpp, or GGUF architecturally
  load-bearing for YUG's own brain.
- Never let production code (`yug/`) import from `training/` (once it
  exists) — the dependency direction is one-way.
- Never use pickle for tokenizer, model config, or (eventual) checkpoint
  serialization — JSON/safetensors only, for arbitrary-code-execution
  safety reasons, stated explicitly each time.
- Never add a `"yug_native"` (or similar) placeholder value to
  `BrainConfig.backend` before `YugNativeAdapter` actually exists.
- Never freeze a production hyperparameter, vocabulary size, or special-
  token set "because implementation is inconvenient" — these are
  deliberately left open (§5) until their dependent milestones justify
  fixing them.
- Never proceed to a new NB milestone without explicit authorization.
- Distinguish `[VERIFIED]` from `[INFERENCE]`/`[PROPOSAL]`/`[DECISION]` in
  all reporting — this has been the norm throughout the project and
  should continue.

---

## 10. Known risks (carried forward, unresolved)

- **NB-04 execution gap** (§6) — the most urgent.
- No network access in the development sandbox has been a recurring
  constraint (blocked `pytest`/`ruff` installation since Phase 0, and now
  blocks PyTorch/`torch` for NB-04) — expect this to keep mattering for
  NB-05 onward (all of which need PyTorch) unless the environment changes.
- RoPE requires an even `head_dim` (`hidden_size / num_heads`) — a
  constraint NB-03's `ModelConfig` validation does not check; currently
  only enforced at `YugTransformer` construction time (`NativeError`), not
  at `ModelConfig` construction time.
- Whitespace-heavy text likely compresses somewhat less efficiently under
  the tokenizer's no-pretokenization-regex design — accepted trade-off,
  not yet measured.
- Package-boundary drift risk: `training/` doesn't exist yet, so its
  import-boundary rule (§9) has never actually been tested against real
  code that could violate it.

---

## 11. Exact next step

`[DECISION REQUIRED]`: run NB-04's tests for real (§6) before deciding
whether to proceed to NB-05. If NB-04 is confirmed working:
**NB-05 — Forward Pass** is next per the roadmap, though much of its
scope (input IDs → embeddings → blocks → logits, with correct shapes) has
already been implemented and written up as tests in NB-04 — the person
picking this up should check the architecture doc's roadmap table for the
current authoritative status before assuming NB-05 starts from zero.

Do not implement anything without explicit authorization for that specific
milestone, per Architectural Decision #9.
