# YUG — FINAL ARCHITECTURE DECISION REPORT

**Status of this document:** a single-file synthesis of the project's
verified state, frozen decisions, and open items as of this package's
creation. The living, continuously-updated canonical record is
`ARCHITECTURE/NATIVE_BRAIN_ARCHITECTURE.md` in this same package — if the
two ever disagree in the future, that file is authoritative, not this one.
This report exists so a single Markdown file captures the complete picture
without requiring anyone to reconstruct it from conversation history.

**Classification tags used throughout, consistently with the rest of this
project:** `[VERIFIED]` (directly confirmed from repository files, test
runs, or command output), `[DECISION]` (explicitly frozen, with reasoning),
`[PROPOSAL]` (suggested, not frozen), `[INFERENCE]` (reasonable conclusion
from verified evidence), `[UNKNOWN]`/`[BLOCKED]` (genuinely unresolved or
unverifiable in the current environment).

---

## 1. Executive Summary

`[VERIFIED]` YUG is a personal, local-first AI system under active
construction. It has progressed through Phase 0 (foundation
infrastructure), Phase 1 (a replaceable brain-interface abstraction with
two working adapters), and NB-01 through NB-04 of a native-brain track
(architecture specification → tokenizer → model configuration → Transformer
implementation).

`[VERIFIED]` The most important single fact for anyone continuing this
work: **NB-04's model code has never been executed.** It was written
without PyTorch available in the development environment (no network
access to install it) and is syntactically valid but functionally
unverified. This is flagged repeatedly in this report rather than once,
because it is the single most consequential caveat on everything described
below as "implemented."

`[VERIFIED]` Everything from Phase 0 through NB-03 (pure Python, no
PyTorch dependency) has been genuinely executed and passes — 169 tests,
via a documented verification methodology (§6) that substitutes for real
`pytest` (also unavailable in this sandbox) with high fidelity for
straightforward, non-parametrized-fixture code paths.

---

## 2. Verified Repository State

`[VERIFIED]`, directly inspected, current file count: 48 tracked files
(excluding `__pycache__`/`.git`).

```
yug/
  __init__.py, __main__.py            Phase 0: package version, CLI (--version)
  config/                             Phase 0: YugConfig — logging, paths, brain backend selection
    __init__.py, schema.py, loader.py, defaults.yaml
  utils/                              Phase 0: shared types, error hierarchy, structured JSON logging
    __init__.py, types.py, errors.py, logging.py
  brain/                              Phase 1: replaceable brain interface + two adapters
    __init__.py, interface.py, types.py, errors.py, manifest.py
    adapters/
      __init__.py, mock.py            MockBrainAdapter — dependency-free, deterministic
      llama_cpp.py                    LlamaCppAdapter — EXTERNAL BASELINE ONLY, lazy-imports llama_cpp
    model_zoo/manifests.json          empty JSON array — no model bundled, none fabricated
  tokenizer/                          NB-02: YUG-owned byte-level BPE tokenizer
    __init__.py, bpe.py, errors.py
  native/                             NB-03/NB-04: YUG-owned model configuration and architecture
    __init__.py, config.py, errors.py   NB-03 — zero PyTorch dependency
    model.py                            NB-04 — requires PyTorch (optional extra); UNEXECUTED (see §1, §6, §9)
tests/                                 18 files, 184 base test functions (more after parametrize expansion)
docs/architecture/
  NATIVE_BRAIN_ARCHITECTURE.md         the canonical, continuously-updated architecture record
  YUG_HANDOFF.md                       a prior conversation-handoff summary
pyproject.toml, .gitignore
```

`[VERIFIED]` `training/` does not exist. This is by design (§8) — nothing
requiring it has been authorized yet.

---

## 3. Completed Milestones

| Milestone | Status | What it delivered |
|---|---|---|
| Phase 0 | `[VERIFIED]` Complete, tested | Config system, structured JSON logging, shared types, error hierarchy, CLI (`python -m yug --version`) |
| Phase 1 | `[VERIFIED]` Complete, tested | `BrainInterface` (frozen ABC: `generate`/`embed`/`get_capabilities`/`health_check`), `MockBrainAdapter`, `LlamaCppAdapter`, `ModelManifest` |
| NB-01 | `[VERIFIED]` Complete | `NATIVE_BRAIN_ARCHITECTURE.md` — the architecture specification itself |
| NB-02 | `[VERIFIED]` Complete, tested | `ByteLevelBpeTokenizer` — YUG-owned tokenizer, no external dependency, 48 tests |
| NB-03 | `[VERIFIED]` Complete, tested | `ModelConfig` + `estimate_parameter_count()` — config abstraction, 61 tests, zero PyTorch |
| NB-04 | `[VERIFIED — code complete]` / `[BLOCKED — execution]` | `YugTransformer` — full config-driven Transformer implementation, 31 tests written, **none executed** (§1, §9) |
| NB-05 onward | Not started | Forward-pass proof, generation, loss, training core, dataset pipeline, checkpoint I/O, `YugNativeAdapter`, benchmarking, scaling — see roadmap in `NATIVE_BRAIN_ARCHITECTURE.md` |

---

## 4. Architectural Decisions (Frozen)

`[DECISION]`, full reasoning for each lives in `NATIVE_BRAIN_ARCHITECTURE.md`
("ARCHITECTURAL DECISIONS", 26 items); summarized here:

1. **YUG is a system, not a model.** The brain is replaceable; memory,
   tools, permissions, verification live outside it.
2. **`BrainInterface` is frozen, inference-time-only.** `generate()`,
   `embed()`, `get_capabilities()`, `health_check()`. Never add training
   concepts (gradients, optimizer, loss, dataset handling) to it.
3. **External models are permanently baselines, never YUG's identity.**
   `LlamaCppAdapter` is classified as an EXTERNAL BASELINE ADAPTER —
   usable for development/benchmarking/comparison, never promoted, never
   the basis for YUG's own tokenizer/architecture/checkpoint format, never
   made a mandatory dependency, never tied to GGUF.
4. **Tokenizer:** byte-level BPE (256 base bytes + learned merges), pure
   stdlib implementation, no external tokenizer library/pretrained
   vocabulary. No `<unk>` token (structurally unnecessary — every byte is
   already representable). No word-level pre-tokenization regex (simpler,
   fully deterministic, avoids importing an external scheme). JSON
   serialization (never pickle) with a `sha256` identity hash. Vocabulary
   size is a `train()`-time parameter, never a frozen constant.
5. **Model configuration (`ModelConfig`):** a frozen, validated dataclass
   bound to a specific tokenizer via `tokenizer_identity`
   (`ModelConfig.from_tokenizer()` pulls this from a real tokenizer
   instance — no config can silently invent a vocab size). JSON
   serialization with a `sha256` identity hash covering all 14 semantic
   fields (excludes only the format-version field, matching the
   tokenizer's own convention). `load()` re-validates and re-verifies the
   identity hash, rejecting tampered files.
6. **Parameter-count estimation is a pure function of `ModelConfig`,
   documented as exact arithmetic under stated architectural assumptions**
   — never a measurement of real weights until a real model exists to
   measure.
7. **Model architecture family: decoder-only, causal, pre-norm
   Transformer** (frozen at NB-04). Exact hyperparameters remain open
   (§5) — the implementation is generic over any valid `ModelConfig`.
8. **`yug/native/model.py` implements every `ModelConfig`-validated option**
   for normalization (rmsnorm/layernorm), positional encoding
   (rope/learned/none), activation (gelu/relu/silu/swiglu), bias, and
   embedding tying — config-driven, not a single hardcoded choice.
9. **Causal masking uses PyTorch's built-in
   `scaled_dot_product_attention(..., is_causal=True)`** — mature
   numerical infrastructure PyTorch is authorized to provide; the
   architecture itself (which projections exist, RoPE application, block
   composition) remains YUG's own.
10. **PyTorch is an optional dependency** (`pip install -e ".[native]"`),
    not core. `yug.native.config` and the package's base import surface
    stay PyTorch-free; only importing `yug.native.model` requires it.
11. **`yug/native/` is the canonical home for YUG-owned model config,
    architecture, and inference runtime.** `training/` (not yet created)
    is strictly separate: datasets, optimizers, losses, training loops.
    Production code must never import from `training/`; `training/` may
    import from `yug/`.
12. **Checkpoints will be a directory** (`config.json`, `tokenizer/`,
    `weights.safetensors`, `metadata.json`), never pickle (arbitrary-code-
    execution risk), never GGUF. Exact metadata schema is not yet defined
    — deliberately deferred until the training pipeline exists to specify
    real requirements.
13. **Each NB milestone requires its own explicit authorization** before
    implementation begins. Nothing beyond what's listed in §3 is
    authorized.
14. **A RoPE-specific constraint (even `head_dim`) was discovered during
    NB-04** that `ModelConfig`'s NB-03 validation does not check. Handled
    by adding a check at `YugTransformer` construction time, not by
    modifying `ModelConfig` — `ModelConfig` remains exactly as NB-03 left
    it.

---

## 5. Open Decisions (Genuinely Unresolved)

`[UNKNOWN]`/`[PROPOSAL — not frozen]`, from `NATIVE_BRAIN_ARCHITECTURE.md`,
unchanged as of this report:

- Exact production tokenizer vocabulary size (proposed range 4K–8K, not fixed).
- Whether a `pad` special token (beyond the prototype's `bos`/`eos`)
  belongs in the production special-token set.
- Exact model hyperparameters — layer count, hidden size, head count, FFN
  size, context length, dropout, normalization variant. (NB-03's report
  calculated seven illustrative candidate configurations spanning ~4.3M to
  ~42M parameters, none recommended as final.)
- Exact checkpoint metadata schema.
- Target training hardware (CPU-only vs. GPU access) — materially affects
  realistic timeline for all training-related milestones.
- **Whether `yug/native/model.py` actually runs correctly** — see §1, §9.
  This is the most urgent open item.
- Whether residual-branch dropout should be added (currently only
  attention-weight dropout is wired in, via `scaled_dot_product_attention`'s
  `dropout_p` argument).

---

## 6. Test Status & Verification Methodology

`[VERIFIED]`, run fresh for this report:
```
169 passed, 0 failed, 0 errored, 2 skipped, 171 total
```
The 2 skips: one pre-existing test requiring a real local GGUF model file
plus `llama-cpp-python` (neither present here), and the entire
`test_native_model.py` module (requires PyTorch, not present here).

`[VERIFIED]` **Neither the real `pytest` executable nor `ruff` has ever
been successfully run by the assistant in this development sandbox** — no
network access to `pip install` them, throughout this project's entire
history. All "tests pass" claims for Phase 0 through NB-03 were verified
via a custom, purpose-built pytest-compatible shim (not part of this
repository/package — a development-only tool) that imports the actual,
unmodified test files and executes them directly, supporting
`pytest.raises`, `tmp_path`, `capsys`, `pytest.mark.parametrize`, and
`pytest.importorskip`. This gives real, meaningful confidence for
pure-Python code — every test in Phase 0 through NB-03 has genuinely been
exercised this way, including deliberately-induced failure cases that were
caught and fixed during development (e.g. a Windows-specific `subprocess`
stdin-handling bug found via a real `pytest` run on the user's own
machine, and two test-authoring bugs caught by this shim during NB-03).

`[BLOCKED]` This same substitute verification has **no PyTorch to fall
back on** for NB-04 — `test_native_model.py`'s 31 test functions
(construction across all config-enum combinations; shape checks; a
mathematical causal-masking invariant test — not merely checking a mask
tensor exists; RoPE norm-preservation; context-length boundaries;
invalid-token-id handling; tied/untied weight-sharing identity and
mutation; parameter-count-vs-formula comparison across 36 config
combinations; determinism; gradient-flow checks) are syntactically valid
(`py_compile`/`ast.parse` both pass) but **have not been run even once**.

`[VERIFIED — by hand, not by execution]` As partial mitigation, the
parameter-count formula was manually traced module-by-module against two
representative configurations (a minimal-options config and a
maximal-complexity config using every non-default option: SwiGLU
activation, LayerNorm, learned positional encoding, bias enabled, untied
embeddings) and matched the formula's output exactly in both cases (5,200
and 8,192 parameters respectively). This is real evidence, not a
substitute for execution.

**Recommended immediate action for whoever has network access:**
```
pip install -e ".[dev]" && pytest -v && ruff check .          # Phase 0 - NB-03, expected clean
pip install -e ".[native]" && pytest tests/test_native_model.py -v   # NB-04, status currently unknown
```

---

## 7. Dependencies

`[VERIFIED]`, from `pyproject.toml`:

| Group | Packages | Status in this sandbox |
|---|---|---|
| Core (`dependencies`) | `pyyaml>=6.0` | Installed, working |
| `dev` extra | `pytest>=7.4`, `ruff>=0.4` | **Not installed** (no network) |
| `native` extra | `torch>=2.0` | **Not installed** (no network) |
| Referenced but not declared | `llama-cpp-python` (used by `LlamaCppAdapter`, lazily imported) | Not installed; genuinely optional by design |

No Hugging Face Transformers, tiktoken, SentencePiece, or LangChain
anywhere in this project.

---

## 8. Constraints / Invariants (Never Silently Violate)

`[DECISION]`, restated for a fresh reader:
- Never add training concepts to `BrainInterface`.
- Never make `LlamaCppAdapter`/llama.cpp/GGUF architecturally load-bearing
  for YUG's own brain.
- Never let production code (`yug/`) import from `training/` (once it
  exists).
- Never use pickle for tokenizer, model config, or checkpoint
  serialization.
- Never add a `"yug_native"` (or similar) placeholder backend value before
  `YugNativeAdapter` actually exists.
- Never freeze a production hyperparameter, vocabulary size, or
  special-token set merely because implementation is inconvenient — these
  are deliberately open (§5) pending their dependent milestones.
- Never proceed to a new NB milestone without explicit authorization.
- Always distinguish `[VERIFIED]` from `[INFERENCE]`/`[PROPOSAL]`/
  `[DECISION]` in any report — this has been the norm throughout and
  should continue for whoever picks this up next.

---

## 9. Known Risks

- **NB-04 execution gap (§1, §6)** — highest priority, unresolved.
- **No network access in the development sandbox** has been a persistent,
  recurring constraint (blocked `pytest`/`ruff` since Phase 0, now blocks
  PyTorch for NB-04 and everything downstream of it that needs PyTorch).
- RoPE's even-`head_dim` requirement is enforced only at model-construction
  time, not at `ModelConfig` construction time — a `ModelConfig` can exist
  validly and only fail later if paired with RoPE and an incompatible
  head/hidden-size combination.
- Tokenizer's no-pretokenization-regex design likely compresses
  whitespace-heavy text somewhat less efficiently than a word-aware
  scheme — accepted trade-off, not yet measured.
- `training/`'s import-boundary rule has never been tested against real
  code, since `training/` doesn't exist yet.

---

## 10. Exact Next Step

`[DECISION REQUIRED]`: get NB-04 actually executed (§1, §6) — this is the
single highest-priority action, ahead of any new implementation work.
Once confirmed (pass, or pass-after-fixes), the next roadmap item is
**NB-05 — Forward Pass**, though much of its nominal scope was already
implemented and tested as part of NB-04; check
`NATIVE_BRAIN_ARCHITECTURE.md`'s roadmap table for the authoritative
status before assuming NB-05 starts from zero. Do not implement anything
further without explicit authorization for that specific milestone
(Architectural Decision, §4 item 13).

---

## Appendix: How to Use This Package

```
YUG_FINAL/
├── README.md                                          start here
├── REPORT/
│   └── YUG_FINAL_ARCHITECTURE_DECISION_REPORT.md       this file
├── ARCHITECTURE/
│   ├── NATIVE_BRAIN_ARCHITECTURE.md                    canonical, living architecture record
│   └── YUG_HANDOFF.md                                  prior conversation-handoff summary
└── SOURCE/
    └── yug/, tests/, pyproject.toml, TEST_STATUS.md     the complete package + test suite + verification status
```

Note: an earlier revision of this package kept the test suite mirrored
under a separate top-level `TESTS/` directory. That was merged into
`SOURCE/tests/` (and `TESTS/TEST_STATUS.md` into `SOURCE/TEST_STATUS.md`)
because `TESTS/` and `tests/` (the historical development-tree name)
collide on case-insensitive filesystems (Windows, default macOS),
silently duplicating every test file on those platforms.

To continue development: read `ARCHITECTURE/NATIVE_BRAIN_ARCHITECTURE.md`
in full, then this report, then `SOURCE/TEST_STATUS.md` before assuming any
test result. Do not skip straight to `SOURCE/` and start editing.
