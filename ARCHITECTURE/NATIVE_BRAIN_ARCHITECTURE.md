# YUG Native Brain Architecture

**Status:** NB-01 (Architecture Specification) — no native-brain code exists yet.
**Audience:** any engineer joining this project later, with no other context.
**Scope:** what YUG's own brain will eventually be, how it relates to
`BrainInterface` and to external baseline models, and what is/isn't
decided yet. This document contains no implementation.

---

## VERIFIED CURRENT STATE

As of this document's creation, directly confirmed by inspecting the repository:

- `yug/brain/interface.py` defines `BrainInterface(ABC)` with four methods:
  `generate()`, `embed()`, `get_capabilities()`, `health_check()`. All are
  inference-time only — no tokenizer, gradient, optimizer, loss, or
  checkpoint-training concept appears anywhere in it.
- `yug/brain/adapters/mock.py` — `MockBrainAdapter`, a dependency-free,
  deterministic implementation of `BrainInterface`.
- `yug/brain/adapters/llama_cpp.py` — `LlamaCppAdapter`, an implementation
  of `BrainInterface` backed by the external `llama-cpp-python` package.
  The import is lazy (inside a method, not module scope); `llama-cpp-python`
  is **not** listed in `pyproject.toml` and is not installed in the current
  development environment.
- `yug/brain/types.py` — `Message`, `ToolDefinition`, `GenerationResult`,
  `Embedding`, `ModelCapabilities`, `HealthState`, `HealthStatus`.
- `yug/brain/errors.py` — `BrainError` hierarchy.
- `yug/brain/manifest.py` — `ModelManifest` dataclass and a JSON
  loader/registry. `yug/brain/model_zoo/manifests.json` currently contains
  an empty array (`[]`) — no model is bundled.
- `yug/config/schema.py` — `BrainConfig` with fields `backend` (validated
  against `{"llama_cpp", "mock"}`), `model_id`, `model_path`,
  `context_length`, `n_threads`.
- `pyproject.toml` runtime dependencies: `["pyyaml>=6.0"]` only. No
  PyTorch, no tensor library, no ML framework of any kind.
- No file, class, or reference to a tokenizer, vocabulary, native model
  architecture, training loop, optimizer, loss function, dataset pipeline,
  checkpoint format, or native inference runtime exists anywhere in this
  repository (confirmed by content search, not by absence-of-memory).
- No `docs/` directory existed before this document; no documentation
  convention was already established for this project to follow.
- Phase 0 infrastructure (`yug/config/`, `yug/utils/`) and the existing
  test suite (`tests/`, 12 files) are unrelated to and unaffected by
  anything in this document.

---

## ARCHITECTURAL DECISIONS

These have been explicitly stated by the Chief Architect and are treated
as fixed for planning purposes, not open for silent reinterpretation:

1. YUG will eventually own its tokenizer, model architecture, model
   configuration, training pipeline, dataset/evaluation pipeline,
   checkpoint format, native inference runtime, and native `BrainInterface`
   adapter (`YugNativeAdapter`).
2. External pretrained models (e.g. via `LlamaCppAdapter`) are permanently
   classified as **external baseline / development adapters** — never the
   YUG brain, never architecturally load-bearing for YUG's identity.
3. `BrainInterface` is frozen as an inference-time-only contract. Training
   concepts (gradients, optimizer, loss, dataset handling) must never be
   added to it. A native brain implements the same four methods any other
   adapter does — the interface does not change to accommodate it.
4. `LlamaCppAdapter` is not to be removed, expanded in role, or used as
   the basis for YUG's checkpoint format, tokenizer, or architecture.
   GGUF is explicitly rejected as YUG's native checkpoint format.
5. Training is architecturally separate from production. Production code
   may depend on the tokenizer, native model definitions, inference
   runtime, and checkpoint loading. Production must never depend on the
   training loop, optimizer, dataset pipeline, or training-only evaluation
   code.
6. PyTorch is the proposed (not yet installed) numerical/training
   substrate: it supplies tensors, autograd, and kernel dispatch; YUG
   supplies the architecture, configuration, tokenizer, training
   orchestration, dataset design, checkpoint structure, evaluation, and
   generation policy. A custom tensor/autograd engine is explicitly out
   of scope unless a future decision authorizes it.
7. The first native model is a small (~10M–50M parameter) proof-of-capability
   brain. Its purpose is to prove the pipeline (tokenizer → model →
   training → checkpoint → reload → generation) works end to end — it is
   explicitly not expected to reach general conversational competence.
8. The tiny-corpus overfit test is a hard gate: if the pipeline cannot
   memorize a small deterministic dataset, scaling to real training data
   does not proceed.
9. Each NB milestone (below) requires its own explicit authorization.
   Nothing beyond NB-01 is authorized by this document.
10. **(NB-02)** The tokenizer algorithm is byte-level BPE (256 base
    byte tokens + learned merges), implemented natively in
    `yug/tokenizer/bpe.py` with no external tokenizer library and no
    pretrained vocabulary/merge table. Reason: this was the standing
    proposal, and NB-02's authorization directed implementing it if no
    blocker was found; none was.
11. **(NB-02)** No `<unk>` token is ever defined. Reason: byte-level
    BPE's base vocabulary already covers every possible byte, so every
    valid (and even arbitrary) byte sequence is representable without
    a fallback token — this is a structural property of the algorithm,
    independent of the (still open) production vocabulary-size
    decision, so it was safe to freeze now rather than defer.
12. **(NB-02)** Special tokens are a caller-provided, named set at
    tokenizer-training time (`train(corpus, vocab_size, special_tokens=(...))`),
    not a hardcoded list. This defers the *final* special-token set
    (see Open Decisions) without blocking implementation.
13. **(NB-02)** Tokenizer serialization is JSON (never pickle, for the
    same arbitrary-code-execution reason already stated for model
    checkpoints), with a `sha256` identity hash computed over the
    canonical (sorted-key, compact) JSON of `{algorithm, vocab_size,
    merges, special_tokens}`. `load()` recomputes and verifies this
    hash, rejecting a tampered or corrupted file rather than silently
    loading it.
14. **(NB-02)** Merges operate directly on the raw UTF-8 byte stream,
    with no word-level pre-tokenization regex (unlike GPT-2-style
    tokenizers). Reason: simpler, fully deterministic, and avoids
    reproducing a pretokenization scheme designed for a different,
    external tokenizer. Accepted trade-off: likely somewhat lower
    compression efficiency on whitespace-heavy text than a word-aware
    scheme — acceptable for a correctness-first prototype.
15. **(NB-02)** Vocabulary size remains a `train()`-time parameter, not
    a frozen constant anywhere in the tokenizer implementation — this
    NB-02 work does not resolve the open production vocabulary-size
    decision (see Open Decisions), it just ensures nothing hardcodes
    a guess at it.
16. **(NB-03 authorization)** `yug/native/` is approved as the canonical
    production package for YUG-owned native model code: model
    configuration, model architecture, and inference runtime
    components. It explicitly excludes training loops, datasets,
    optimizers, and training-only evaluation code, which remain in the
    top-level `training/` package. Dependency direction: `training/`
    may import from `yug.tokenizer` and `yug.native`; `yug/` production
    code must never import from `training/`. This closes the package-
    layout item that was previously `TBD — DECISION REQUIRED`.
17. **(NB-03)** `ModelConfig` (`yug/native/config.py`) is a frozen
    dataclass with 14 fields (see table in the NB-03 report), validated
    on construction, requiring a `tokenizer_identity` bound to a real
    tokenizer's `.identity` hash (via `ModelConfig.from_tokenizer()`)
    rather than allowing a config to exist unbound to any tokenizer.
    No production hyperparameter values are frozen — every field is
    supplied by the caller.
18. **(NB-03)** No `<unk>`-equivalent gap exists for model config
    fields, but the same "don't add fields speculatively" discipline
    applied: `bias` and `tie_embeddings` were included because they
    measurably change parameter count and are simple, well-established
    binary choices; a `config_format_version` field was included for
    serialization forward-compatibility, following the identical
    pattern already used by the tokenizer.
19. **(NB-03)** Model config serialization is JSON (never pickle, same
    reasoning as the tokenizer and the checkpoint proposal), with a
    `sha256` identity hash over all 14 semantic fields (excluding only
    the serialization format-version field, matching the tokenizer's
    own convention of excluding its format-version field from its
    hash). `load()` re-validates the config AND re-verifies the
    identity hash, rejecting tampered or invalid files.
20. **(NB-03)** Parameter-count estimation
    (`estimate_parameter_count()`) is a pure function of `ModelConfig`,
    documented as exact arithmetic under explicitly stated standard
    pre-norm-Transformer assumptions — not a measurement of real
    weights (none exist; NB-04/05 have not been implemented). No
    specific hyperparameter combination is recommended as final; seven
    illustrative candidates were calculated (see NB-03 report) spanning
    both inside and outside the 10–50M target range, precisely to show
    the sizing space rather than to freeze a choice.
21. **(NB-04)** The architecture family — decoder-only, causal,
    pre-norm Transformer — is promoted from `[PROPOSAL]` to
    `[DECISION]` via this milestone's explicit authorization (same
    pattern as NB-02's tokenizer-algorithm promotion). This does **not**
    freeze exact hyperparameters (layer/hidden/head counts etc.),
    which remain genuinely open (see Open Decisions) — `YugTransformer`
    is generic over any valid `ModelConfig`.
22. **(NB-04)** `yug/native/model.py` implements every
    `ModelConfig`-validated option for `normalization` (rmsnorm,
    layernorm), `positional_encoding` (rope, learned, none),
    `activation` (gelu, relu, silu, swiglu), `bias`, and
    `tie_embeddings` — config-driven, not a hardcoded single choice.
    This fulfills NB-03's existing validated-enum contract rather than
    resolving which single value is the eventual production choice.
23. **(NB-04)** PyTorch is added as an **optional** dependency
    (`pip install -e ".[native]"`), not a core dependency — `yug/native/`
    as a whole is production code, but only `yug.native.model` actually
    requires PyTorch; `yug.native.config` (NB-03) has zero PyTorch
    dependency and `yug/native/__init__.py` does not re-export
    `yug.native.model`, so `import yug.native` stays PyTorch-free.
    Rationale: the base package (Phase 0 CLI, `BrainInterface`,
    `MockBrainAdapter`, `LlamaCppAdapter` baseline, tokenizer, model
    configuration) has no need for a large ML dependency unless the
    native model is actually being used.
24. **(NB-04)** Causal masking uses PyTorch's built-in
    `scaled_dot_product_attention(..., is_causal=True)` rather than a
    hand-rolled mask tensor — this is exactly the "mature numerical
    infrastructure" PyTorch is authorized to provide; the architecture
    (which projections exist, how RoPE is applied, block composition)
    remains YUG's own.
25. **(NB-04)** A constraint not anticipated by NB-03's `ModelConfig`
    validation was discovered during implementation: RoPE requires an
    even `head_dim` (`hidden_size / num_heads`). Rather than modifying
    NB-03's `ModelConfig.validate()`, this check was added at model
    construction time in `yug/native/model.py` (raises `NativeError`).
    `ModelConfig` itself is unmodified.
26. **(NB-04) — VERIFICATION STATUS, IMPORTANT:** `yug/native/model.py`
    and `tests/test_native_model.py` were written without a working
    PyTorch installation in the development sandbox (no network access
    to install it — same constraint noted since Phase 0's
    pytest/ruff situation, but here it blocks executing the deliverable
    itself, not just the test runner). Correctness is supported by
    manual, by-hand parameter-count derivation matching
    `estimate_parameter_count()`'s formula exactly for two
    representative configurations (see NB-04 report), and by careful
    use of standard, well-established PyTorch APIs — but **none of this
    code has actually been run**. This is a materially different, and
    weaker, verification status than every prior NB milestone in this
    project, and must not be read as equivalent to them.
27. **(NB-06 authorization)** NB-06 is scoped exactly as: deterministic
    greedy autoregressive generation over token ids, built on
    `YugTransformer.forward()`. Required behavior: accept starting
    token ids and a `max_new_tokens` bound; optionally accept a
    caller-supplied integer `stop_token_id`; repeatedly call `forward()`
    and select the next token via `argmax` (no sampling); append it;
    stop at `max_new_tokens` or immediately if the generated token
    equals `stop_token_id`. Explicitly excluded from NB-06: temperature/
    top-k/top-p/beam-search sampling, KV-cache optimization (unless a
    future authoritative document requires it), training/loss/optimizer/
    dataset/checkpoint work, `YugNativeAdapter`, `BrainInterface`
    integration, and text decoding. `stop_token_id` is an opaque
    caller-provided integer only — NB-06 does not import
    `yug.tokenizer` and does not assume any particular id is an EOS
    convention; that coupling is deferred to NB-17 (decoded-text smoke
    test).

---

## PROPOSALS

Ideas under consideration, not yet frozen — implementation must not treat
these as decided:

- ~~Tokenizer scheme: byte-level BPE~~ — **resolved, see Architectural
  Decision #10.** Vocabulary size for the eventual production tokenizer
  remains **TBD — DECISION REQUIRED** (see Open Decisions); NB-02
  implemented `train()` with `vocab_size` as a parameter rather than
  guessing a number.
- **Special tokens:** NB-02 implemented `bos`/`eos` as the prototype
  default set (see Architectural Decision #12) and confirmed `<unk>` is
  never needed (Decision #11). Whether a `pad` token (or others) belongs
  in the *production* special-token set is still **TBD — DECISION
  REQUIRED**, deferred to when the training/batching strategy (NB-08) is
  decided.
- **Model architecture:** decoder-only Transformer, pre-norm, RoPE
  positional encoding, config-driven (no hardcoded layer/head/hidden
  counts in architecture code). Exact hyperparameters (layer count,
  hidden size, head count, FFN size, context length, dropout,
  normalization variant) are **TBD — DECISION REQUIRED**, to be set once
  the tokenizer's vocabulary size is fixed.
- **Checkpoint layout:** a directory, not a single file —
  `config.json`, `tokenizer/`, `weights.safetensors`, `metadata.json`.
  Safetensors is proposed over pickle-based serialization specifically
  because pickle can execute arbitrary code on load and safetensors
  cannot — a concrete security property, not a stylistic preference.
  Exact metadata fields are **TBD — DECISION REQUIRED**, to be derived
  from what the model/training pipeline actually needs once they exist.
- **Training specifics:** next-token cross-entropy loss, AdamW (or
  another standard PyTorch-provided optimizer). No distributed training,
  RLHF, preference optimization, mixture-of-experts, online learning, or
  speculative decoding during the first milestone.
- **Package layout:**
  ```
  yug/
    tokenizer/          <- production + training, byte-level BPE
    native/             <- production + training, model config + architecture
    brain/adapters/
      yug_native.py     <- production only, implements BrainInterface

  training/             <- training-only: datasets, preprocessing, losses,
                            optimizers, trainers, evaluation, checkpointing
  ```
  This is a proposal for how Section "BOUNDARIES" below would be enforced
  structurally (an import from `training/` appearing in `yug/` would be a
  visible, reviewable mistake), not yet created.

None of the above has been implemented. No tokenizer, model, or training
code exists as of this document.

---

## NON-GOALS (at this stage)

Explicitly out of scope until a future milestone authorizes otherwise:

- A general-purpose, ChatGPT-class native model. The first brain is a
  pipeline proof, not a capability target.
- Distributed training, reinforcement learning, RLHF, preference
  optimization, mixture-of-experts, online/continual learning, or
  speculative decoding.
- A custom tensor/autograd framework.
- A sophisticated model zoo, download manager, or fallback-model chain.
- Any change to `BrainInterface`, `BrainConfig`, `ModelManifest`, or
  `LlamaCppAdapter` code. (This document is documentation-only; see
  "Files modified" in the implementation report accompanying it.)
- Adding a `"yug_native"` (or any other) placeholder value to
  `BrainConfig.backend` before `YugNativeAdapter` actually exists.
- Any dataset-engineering platform beyond the minimum needed to overfit
  a tiny controlled corpus (NB-09/NB-10).

---

## OWNERSHIP

| Concern | Owned by | Notes |
|---|---|---|
| Tensor ops, autograd, CPU/GPU kernel dispatch | PyTorch (proposed) | Infrastructure only; not YUG's architectural identity |
| Safe weight serialization format | safetensors (proposed) | Infrastructure |
| Model architecture (module structure, forward pass) | YUG | `yug/native/` |
| Model configuration (hyperparameters as data) | YUG | `yug/native/` |
| Tokenizer (vocabulary, encode/decode, serialization) | YUG | `yug/tokenizer/` |
| Training orchestration (loop, scheduling) | YUG | `training/` |
| Dataset curation/preprocessing | YUG | `training/` |
| Checkpoint format/structure | YUG | directory format, see Proposals |
| Evaluation criteria | YUG | `training/` (and reused by production for e.g. reload tests) |
| Generation/decoding policy | YUG | `yug/native/` |
| `BrainInterface` contract | YUG | already exists, frozen (see Architectural Decisions) |
| External baseline inference (llama.cpp) | External (`llama-cpp-python`) | Isolated in `LlamaCppAdapter`; optional, never mandatory |

---

## BOUNDARIES

**Production vs. training**, restated precisely:

- Production (`yug/`) may import: `yug.tokenizer`, `yug.native`,
  `yug.brain.*` (including a future `yug.brain.adapters.yug_native`),
  checkpoint-loading code.
- Production must **never** import: anything under `training/` — no
  optimizer, no loss function, no dataset pipeline, no training-only
  evaluation code.
- `training/` may import from `yug/` (tokenizer, native model
  definitions) — the dependency direction is one-way.
- This boundary is intended to be structural (a separate top-level
  package), not merely a naming convention, specifically so an accidental
  violation is a visible, reviewable import statement rather than a
  hidden coupling.

---

## ROADMAP

| Milestone | Deliverable | Status |
|---|---|---|
| NB-01 | This document | **Complete** (this document) |
| NB-02 | YUG Tokenizer (`yug/tokenizer/`), fully tested, no model/training code | **Complete** — `ByteLevelBpeTokenizer`, 48 tests, no PyTorch, no external tokenizer dependency |
| NB-03 | Model configuration abstraction | **Complete** — `ModelConfig` (`yug/native/config.py`), `estimate_parameter_count()`, 61 tests, no PyTorch, no architecture hyperparameters frozen |
| NB-04 | Native Transformer foundation (config-driven) | **Code complete, execution UNVERIFIED** — `YugTransformer` (`yug/native/model.py`), 40+ parametrized tests written; PyTorch unavailable in the development sandbox (no network), so nothing has actually been run. See Architectural Decision #26. |
| NB-05 | Forward pass (input IDs → embeddings → blocks → logits) | Not started |
| NB-06 | Generation (deterministic first, sampling later if justified) | **Code complete, execution UNVERIFIED** — `generate()` (`yug/native/generation.py`), 22 tests written (see Architectural Decision #27); no PyTorch in the development sandbox, so nothing has actually been run. Same verification gap as NB-04 originally had. |
| NB-07 | Loss (next-token cross-entropy) | Not started |
| NB-08 | Training core (optimizer, gradient handling, training step/loop) | Not started |
| NB-09 | Tiny overfit experiment (hard gate before any further training) | Not started |
| NB-10 | Dataset/evaluation foundation | Not started |
| NB-11 | Checkpoint writer | Not started |
| NB-12 | Checkpoint loader (config + tokenizer + weights + metadata, compatibility-checked) | Not started |
| NB-13 | Reload equivalence (save → reload → deterministic comparison) | Not started |
| NB-14 | Native runtime (execution separated from training) | Not started |
| NB-15 | `YugNativeAdapter` (`yug/brain/adapters/yug_native.py`) | Not started |
| NB-16 | `BrainInterface` integration (adapter conforms to existing interface, unmodified) | Not started |
| NB-17 | Native brain smoke test (input → BrainInterface → YugNativeAdapter → generation → decoded text) | Not started |
| NB-18 | External baseline comparison (native vs. `LlamaCppAdapter`) | Not started |
| NB-19 | Architecture review | Not started |
| NB-20 | Scaling decision (evidence-gated, not automatic) | Not started |

Each milestone requires its own explicit authorization before implementation begins (Architectural Decision #9).

---

## OPEN DECISIONS

Recorded here rather than silently resolved:

- **TBD — DECISION REQUIRED:** exact *production* tokenizer vocabulary
  size (proposed range 4K–8K; final value depends on the model parameter
  budget, not yet fixed). NB-02's `train()` API accepts `vocab_size` as
  a parameter for exactly this reason — no number is hardcoded.
- ~~whether `<unk>` is needed~~ — **resolved: no, see Architectural
  Decision #11.**
- **TBD — DECISION REQUIRED:** whether a `pad` special token (or others
  beyond the NB-02 prototype's `bos`/`eos`) belongs in the production
  special-token set — deferred to the training/batching strategy (NB-08).
- **TBD — DECISION REQUIRED:** exact model hyperparameters (layer count,
  hidden size, head count, FFN size, context length, dropout,
  normalization variant) — deferred until vocabulary size is fixed.
- **TBD — DECISION REQUIRED:** exact checkpoint metadata schema — deferred
  until the model/training pipeline exists and its actual requirements
  are known, per the directive's explicit instruction not to overengineer
  this before the model exists.
- **TBD — DECISION REQUIRED:** target hardware for training (CPU-only vs.
  GPU access) — materially affects realistic iteration speed and timeline
  for NB-08/NB-09, and has not been specified.
- ~~package layout approval~~ — **resolved, see Architectural Decision #16.**
- **NEW (NB-04):** whether `yug/native/model.py`'s PyTorch code actually
  runs correctly — genuinely unverified (no PyTorch in this development
  sandbox). Needs to be run in an environment with PyTorch installed
  before NB-04 can be considered truly complete, not just code-complete.
- **NEW (NB-04):** dropout is wired into attention-weight dropout via
  `scaled_dot_product_attention`'s `dropout_p` argument, but residual-
  branch dropout (a common addition in many Transformer implementations)
  was deliberately not added, to keep NB-04 minimal — worth reconsidering
  once real training (NB-08+) reveals whether it's needed.

---

## REVISION NOTE

This document supersedes no prior document (none existed). It should be
updated, not silently replaced, as each NB milestone completes — each
update should move a roadmap row from "Not started" to a real status and
move resolved "Open Decisions" into "Architectural Decisions."
