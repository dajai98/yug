# YUG MASTER

**Last verified:** 2026-11-07 (Asia/Riyadh). **Checkout:** `C:\YUG\yug1`. This is a human-readable state and navigation document, not an approval. Paths below are repository-relative unless explicitly described as external. Recheck live records before an irreversible operation.

## Current State and Next Action

| Field | Current state |
|---|---|
| CURRENT PHASE | Phase 3 — production corpus development and controlled expansion |
| CURRENT STATUS | **OPEN / PARTIAL.** First Stride item admitted; historical attrs prospective-use authority is separate; 2B corpus and YUG-100M training are not complete. |
| VERIFIED PRE-CHECKPOINT HEAD | `3b19950a18550a18cb5084d1ac783542fc579373`; run `git rev-parse HEAD` for the current checkpoint. |
| CURRENT CORPUS | 175 items; 417,768 train and 20,616 validation tokens; ledger head `sha256:66daed652ebf2d1d3cf6f3210bf650bfad2fc46b842a81aa158613792fee14d0` |
| CURRENT BOUNDARY | The uninterrupted synthetic coordinator path passed. New live sources still require independent rights, use, ownership, and trusted-input decisions. |
| NEXT ACTION | Use YUG's validated coordinator on genuinely new candidate sources under exact human and external approvals. Stop unnecessary pipeline engineering. |

## 1. Purpose of This Document

This is the **one human-readable entry point** for YUG's purpose, current state, decisions, and evidence locations. It helps an engineer find the actual record and choose the next step. It is not cryptographic, legal, rights, corpus, ledger, or training authority. Canonical JSON/JSONL records, independently retained pins, verified bytes, the corpus ledger, signatures, timestamps, and Git history retain their own distinct roles. **If this Markdown conflicts with a machine-authoritative record for that record's scope, the machine record wins.** A local hash or seal alone proves neither independent approval nor use permission.

## 2. YUG Identity

**VISION —** YUG is a private, native-trained, specialized game-development AI system, not a general chatbot. Its long-term aim spans planning and design; software engineering; Unity and Unreal Engine; 3D workflows, rigging, animation and environments; shaders/VFX and audio; gameplay AI and physics; UI/UX and multiplayer; optimization, testing, build/release and live operations; tool orchestration; cross-domain consequence-aware planning; and verified execution on real projects. This is a target capability list, not a claim of delivery.

**CURRENTLY IMPLEMENTED —** The repository contains a native tokenizer/transformer baseline, model/training utilities, a bounded agent runtime, and extensive corpus discovery, review, admission, accounting, evidence and export software. The actual admitted corpus is small. See [the live packages](yug/), [training modules](training/), [frozen model record](training/resources/phase1/model_initialization_gate_v1/freeze.json), and [corpus specification](training/specifications/production_corpus_v1.json).

**NOT YET PROVEN —** A completed 2B approved corpus, a production-trained YUG-100M checkpoint, comprehensive game-development competence, reliable real-project execution across those domains, and the later model sizes. Training utilities and synthetic tests do not establish those outcomes.

## 3. Core Engineering Principles

- Evidence first; **measure → understand → decide**. Do not manufacture progress or authority.
- Fail closed when rights, use, ownership, provenance, protected-reference, trust, or accounting inputs are absent or inconsistent.
- Keep scope exact. Do not silently change frozen specifications, broaden a file grant, or treat a review cache as admission.
- Make operations deterministic and reproducible where possible. Preserve immutable identities and external pins.
- Provenance, licensing, attribution and protected evaluation material matter at every corpus transition.
- **Prove → earn → reinvest → scale → prove again.** Scale only after a bounded result survives independent verification.

## 4. Codex vs YUG — Non-Negotiable Rule

**Codex is NOT YUG's operational pipeline.** Codex develops, implements, fixes, tests, audits and validates YUG. YUG's own modules and commands must execute repeated operational work. Before asking Codex to perform such work manually, ask: **“Does YUG already implement this?”** If yes, use or test YUG. If no, prove the gap and implement the missing capability in YUG rather than making Codex the permanent operator. Human approval boundaries remain human even when preparation and verification are automated.

## 5. Development / Scaling Roadmap

**PLANNED, CONDITIONAL:** YUG-100M → YUG-500M → YUG 1–3B → possibly 7B → potentially ≥100B, only if measured capability, technical viability and financial sustainability justify each step. These later sizes are not approved commitments. The frozen immediate model is YUG-100M; its production training is not authorized. [Core v2 research](docs/research/YUG_EXTERNAL_SYSTEMS_RESEARCH.md) is a candidate ledger, not an implementation or grant.

## 6. Frozen YUG-100M Architecture

The [model config](training/resources/phase1/model_initialization_gate_v1/model_config.json), [freeze record](training/resources/phase1/model_initialization_gate_v1/freeze.json), and [native implementation](yug/native/model.py) agree on this baseline:

| Property | Frozen value |
|---|---:|
| Exact parameter target | 100,951,680 |
| Vocabulary / context | 4,096 / 2,048 |
| Hidden size / layers / heads / head dimension | 640 / 20 / 10 / 64 |
| FFN size | 2,560 |
| Normalization | RMSNorm, ε = 1e-6 |
| Position | RoPE, base 10,000; no scaling |
| MLP | SiLU, standard two-projection up/down feed-forward network (not gated SwiGLU) |
| Embeddings / bias / dropout | Tied / no bias / 0.0 |

The exact tokenizer identity is `sha256:0b9b2bb9efe58902e3cf3dcb359cf174a25f3d7bf603b14bd3fe8e2d7fec0507`, in the [frozen tokenizer JSON](training/resources/phase1/bpe_fitter_optimization_gate_v1/comparison/run-4096-0/tokenizer.json). The model-config identity is `sha256:c268d611800b51bc73aeafcfbadc8f3032f7db74c077d43cfbc56bbccce3cb26`. The freeze record explicitly says YUG-100M training is not authorized; later corpus activity does not itself reverse that decision.

## 7. Frozen 2B Corpus Plan

The [production corpus specification](training/specifications/production_corpus_v1.json), identity `sha256:f2df6bd444ddab6a529f77c6fdda189177b9949fff7089571fd3d777b12a2aa3`, fixes **2,000,000,000 clean training tokens** across 16 owners. Validation tokens do not fill that target.

| Category | Target | Category | Target |
|---|---:|---|---:|
| code | 440M | csharp_gamedev | 260M |
| software_engineering | 240M | general_english | 240M |
| educational_reference | 160M | nepali | 160M |
| software_docs | 120M | graphics | 100M |
| math_science | 80M | algorithms | 60M |
| linux_systems | 40M | security | 20M |
| instruction | 30M | reasoning | 20M |
| tool_use | 10M | conversation | 20M |

The spec also freezes checkpoints at 100M, 250M, 500M, 1B, 1.5B and 2B, with source/group caps and one budget owner per item. Its `acquisition_started=false` is a **stale planning flag relative to the later Stride ledger commit**; use the live ledger and production receipt for present accounting. Its `training_authorized=false` remains a relevant training-readiness boundary.

## 8. Phase Status

| Phase | Status | Purpose and evidence | Remaining work |
|---|---|---|---|
| Phase 1 | **CLOSED** software/freeze milestone | [Model freeze](training/resources/phase1/model_initialization_gate_v1/freeze.json), [tokenizer](training/resources/phase1/bpe_fitter_optimization_gate_v1/comparison/run-4096-0/tokenizer.json), [Phase 1 evidence index](docs/evidence/phase1/README.md) | Production corpus and training are later gates, not reasons to reopen the freeze. |
| Phase 2 | **CLOSED** corpus-gate implementation | [Production build](training/production_build.py), [acquisition gates](training/production_acquisition.py), [policy](docs/phase2/) and tests | Exact independent source/use inputs and scale validation remain operational work. |
| Phase 3 | **OPEN / PARTIAL** | [Ledger](artifacts/corpus_ledger_v1/ledger.sqlite), [Stride progress](artifacts/phase3_stride_manual_index_production_v1/progress.json), [attrs links](artifacts/phase3_attrs_historical_authority_link_v1/links.jsonl), live modules/tests | Approved corpus growth, external preservation, and final dataset identity; synthetic coordinator integration is validated. |
| Phase 3.5 | **PLANNED** | [Dated audit roadmap](docs/PROJECT_AUDIT/YUG_FULL_PROJECT_AUDIT_V1.md) and [research](docs/research/YUG_EXTERNAL_SYSTEMS_RESEARCH.md) | Audit measured failures after Phase 3 evidence is stable; no completion claim. |
| Core v2 | **RESEARCH / NOT FROZEN** | [External-systems research](docs/research/YUG_EXTERNAL_SYSTEMS_RESEARCH.md) | Choose a justified scope; no broad implementation authorization. |
| YUG-100M training | **NOT AUTHORIZED / NOT STARTED as a production run** | [Freeze](training/resources/phase1/model_initialization_gate_v1/freeze.json), [corpus spec](training/specifications/production_corpus_v1.json) | Complete and approve dataset, readiness, hardware/evaluation evidence and separate run authorization. Test/experimental training utilities are not this run. |

## 9. Current Phase 3 Corpus State

**VERIFIED from a read-only ledger query on 2026-11-07:** [ledger SQLite](artifacts/corpus_ledger_v1/ledger.sqlite) SHA256 `85e4618fc44cfd3a4b90dd1d6c810789edf50ee834a2fdf9718694669bd75647`; chained head `sha256:66daed652ebf2d1d3cf6f3210bf650bfad2fc46b842a81aa158613792fee14d0`. It contains **175 item-evidence rows, 417,768 training tokens, 20,616 validation tokens, nine source IDs**. Training volume is **0.0208884%** of 2B. This is accounting, not blanket permission to train.

| Source ID | Split | Items | Tokens | Current interpretation |
|---|---|---:|---:|---|
| `gh-tokio-rs-tokio` | train | 121 | 347,154 | Historical; no blanket prospective-use approval inferred. |
| `gh-python-attrs-attrs` | train | 16 | 17,598 | Historical; exact prospective training-use linkage was reported verified separately. No new admission. |
| `gh-spf13-cobra` | train | 4 | 17,501 | Historical; further use decision/NOTICE review needed. |
| `gh-hynek-svcs` | train | 8 | 23,884 | Historical; prospective review remains blocked. |
| `gh-hynek-doc2dash` | train | 9 | 9,926 | Historical; prospective review remains blocked. |
| `gh-psf-requests` | train | 1 | 1,342 | Historical; separate prospective decision needed. |
| `gh-stride3d-stride-docs` | train | 1 | 363 | First controlled native Phase 3 admission; [production progress](artifacts/phase3_stride_manual_index_production_v1/progress.json) says `COMPLETE`, one `ACCEPTED`, non-diagnostic. |
| `gh-hynek-stamina` | validation | 11 | 15,612 | Historical; no prospective evaluation permission inferred. |
| `gh-hynek-pem` | validation | 4 | 5,004 | Historical; no prospective evaluation permission inferred. |

The historical attrs scope is revision `644b4e165bfbeee7e127de6fcbda08b64014316f`; see its [prospective decision](artifacts/phase3_attrs_prospective_training_use_v1/decision.json), [16 links](artifacts/phase3_attrs_historical_authority_link_v1/links.jsonl), [production evidence manifest](artifacts/phase3_attrs_historical_production_evidence_v1/manifest.json), and [separate training-use hook](training/historical_authority_link.py). The prior exact 16-item run reported `TRAINING_AUTHORIZED` for prospective existing-item use with zero accounting delta; this document did not rerun that externally pinned decision. The [retention candidate](artifacts/phase3_attrs_historical_authority_link_retention_v1/candidate.json) itself says `authority=NONE` and `grants_training=false`; it is not the authorization hook's result and must not be promoted into one. Neither attrs use nor Stride admission authorizes a YUG-100M training run.

## 10. Phase 3 Corpus Pipeline

The stages below describe **existing YUG code**. “Automatic” means the module can run the stage when exact inputs are supplied; it does not imply autonomous rights approval or production authority.

| Stage | Live module(s) | Status / operator boundary |
|---|---|---|
| Candidate selection and seed | [Candidate registry](training/specifications/phase3_source_candidate_registry_v1.json), [discovery](training/source_discovery.py) | **PARTIAL:** seed choice and relevance require review; catalog entries are not approved sources. |
| Discovery → immutable revision | [source_discovery.py](training/source_discovery.py), [source_expansion.py](training/source_expansion.py) | **AUTOMATIC** bounded metadata/revision resolution after explicit seeds; fail-closed license states. |
| Git preparation → exact file inventory/hash | [source_prepare.py](training/source_prepare.py) | **AUTOMATIC** bounded cache/inventory and offline replay; fetch must be explicit. |
| License/rights evidence → exact-file clearance | [license_evidence.py](training/license_evidence.py), [source_approval.py](training/source_approval.py), [source_first_clearance.py](training/source_first_clearance.py), [source_revision_supersession.py](training/source_revision_supersession.py), [clearance_evidence.py](training/clearance_evidence.py), [source_grant_proposals.py](training/source_grant_proposals.py) | **PARTIAL:** machine checks and proposal preparation exist; ambiguous ownership/rights and final exact grants require independent decisions. A proposal is not a grant. |
| Bounded project-publication candidates | [publication_clearance.py](training/publication_clearance.py), [source_first_clearance.py](training/source_first_clearance.py) | **PARTIAL:** YUG prepares sealed, path-specific Markdown, RST and first-party code dossiers and deterministic batches of at most 64 files. `ROUTINE_CANDIDATE` is review material with `authority=NONE` and `training_allowed=false`; exact human scope/file decisions, external pins and separate training-use approval remain required. |
| Prospective use and ownership | [corpus_gate.py](training/corpus_gate.py), [category_ownership.py](training/category_ownership.py), [approval governance](training/approval_governance.py) | **PARTIAL:** exact receipts/pins are verified; human/external approval supplies authority. |
| Admission/preflight | [source_admission.py](training/source_admission.py), [production_build.py](training/production_build.py) | **AUTOMATIC after approval:** exact cache/grant match and compiled plan; no implied approval from discovery. |
| Acquisition/extraction/normalization | [production_build.py](training/production_build.py), [production_acquisition.py](training/production_acquisition.py) | **AUTOMATIC for supported routes:** bounded acquisition and content normalization; no universal arbitrary-format extractor. |
| Screening, holdouts, exact/near dedup, quality/language | [content_screening.py](training/content_screening.py), [protected_references.py](training/protected_references.py), [production_dedup.py](training/production_dedup.py), [quality_filtering.py](training/quality_filtering.py), [language_validation.py](training/language_validation.py) | **AUTOMATIC gates** with explicit REVIEW/EXCLUDED states and documented detector limits. [Protected inventory](artifacts/phase3_protected_references_v1/manifest.json) currently has 155 protected records. |
| Tokenizer, category/split, accounting/ledger | [frozen tokenizer](yug/tokenizer/bpe.py), [production acquisition](training/production_acquisition.py), [category ownership](training/category_ownership.py), [cumulative accounting](training/cumulative_accounting.py) | **AUTOMATIC after exact pins:** frozen tokenizer, one owner/split, reservations and chained commits. |
| NOTICE, export, token shards, replay/readback | [notice_attribution.py](training/notice_attribution.py), [production_acquisition.py](training/production_acquisition.py), [token_shards.py](training/token_shards.py) | **AUTOMATIC after gate success:** preserves provenance and supports verified replay. |

For datasets, [dataset discovery](training/dataset_discovery.py), [transport](training/dataset_transport.py), [rights scope](training/dataset_rights_scope.py), and [clearance](training/dataset_clearance.py) form a **separate** path. Git-source grants never transfer to a dataset by analogy.

## 11. Corpus Workflow Orchestrator

[training/corpus_workflow.py](training/corpus_workflow.py) binds explicit seeds and input hashes, invokes existing discovery/cache/review/proposal modules, and stops at `HUMAN_REVIEW` with `training_allowed=false`. `verify` checks sealed request, catalog, cache, file bytes and review artifacts. Completed `prepare --resume` verifies and returns the same sealed report without rewriting a bound cache status; incomplete work can continue through the normal resume path.

The publication-clearance handoff also prepares and verifies zero-authority candidate dossiers at this boundary. With a 256-commit history bound, retained-cache diagnostics found 27 routine candidates, 401 REVIEW and 213 EXCLUDE among 641 pytest files; Click had 9 routine candidates, 69 REVIEW and 101 EXCLUDE among 179 files. These counts are mechanical review states, not source clearance or training permission. Shallow-boundary-only or missing exact-path history and ambiguous publication evidence remain REVIEW.

`resume` requires an independently pinned approved-input transport, exact approved registry/spec/plan, references, ownership/governance/head pins and use receipts. It first runs source admission and production-plan preflight. An explicit second externally pinned compiled-plan identity plus `--execute-production` is required before it hands control to the existing `production_build.execute`; it does not implement another acquisition or accounting engine. Review-only output cannot grant itself use authority.

**Validation boundary:** All **12 workflow tests passed**. A local-Git fixture exercises real discovery -> cache -> review/proposals -> `HUMAN_REVIEW` -> verify/resume. The new uninterrupted synthetic test exercises approved `corpus_workflow.resume` -> real source admission/preflight -> `production_build.execute` -> acquisition/screening/dedup/quality/tokenization/accounting -> NOTICE/export -> token shards -> `TokenShardStream` readback, with an isolated test ledger and no mocked production handoff. This supported synthetic path is **PASS**; it does not establish independent live-source rights or 2B-scale operation.

## 12. Current Pipeline Capability Matrix

**Updated 2026-11-07 from the prior 40-capability assessment:** **37 PASS, 3 PARTIAL, 0 BLOCKED, 0 FAIL**. The coordinator integration moved from PARTIAL to PASS after the uninterrupted synthetic test. The three remaining partial areas are independent rights/clearance decisions, prospective training-use decisions, and ownership/external authority pins. See [workflow tests](tests/test_corpus_workflow.py), [source/admission tests](tests/test_source_admission.py), [production tests](tests/test_production_build.py), [use-gate tests](tests/test_corpus_gate.py), [evidence tests](tests/test_evidence_bundle.py), and [historical-link tests](tests/test_historical_authority_link.py). Test fixtures demonstrate behavior; they do not approve live sources.

Focused coordinator test: **1 passed**. All workflow tests: **12 passed**. Affected admission/production/export/accounting suite: **219 passed**. Current full suite: **2,236 passed, 2 skipped, 1 warning** (Torch could not initialize NumPy because NumPy is absent); focused publication/workflow tests: **40 passed**. Ruff passed on the five touched publication/workflow Python files. These results validate the synthetic integration path and candidate preparation, not live source rights or production training authority.

## 13. Human Decision Boundaries

Independent humans/external channels must supply exact rights judgments where policy requires them; prospective training/evaluation use decisions; ownership and production-plan approvals; current trust-pin/status custody; offline production signing when required; and trusted TSA timestamping when required. YUG can prepare, bind, verify and reject inputs. It must not synthesize those decisions from a license label, a locally resealed JSON file, a cache, or an old ledger row. A verifier can prove a receipt matches its scope without issuing the receipt's authority.

## 14. Evidence / Trust Architecture

Conceptual lineage: **source → revision → exact file hashes → saved license/rights evidence → separate rights and use decisions → screening/dedup → evidence manifest → detached signature → RFC 3161 timestamp → independently retained pins/evidence → admission → cumulative ledger → dataset identity → eventual model lineage**. Exact operational ordering may interleave packaging and preflight; no evidence bundle bypasses the corpus gate.

[Evidence bundle](training/evidence_bundle.py) and [trust verification](training/evidence_trust.py) validate canonical identities, saved bytes, a detached Ed25519 signature, an externally pinned ACTIVE public trust record, and production timestamp requirements. The verifier reports **`authority=NONE`** even when signature and timestamp verify. SHA256 establishes byte integrity against a trusted identity; it does **not** independently establish time, ownership, rights, or training permission. Trusted time requires the RFC 3161 chain and matching imprint. The separate [historical use hook](training/historical_authority_link.py) combines item/link/store/evidence/external-retention checks for existing-item prospective use without changing the evidence verifier's zero-authority result.

The [Stride manifest](artifacts/phase3_stride_manual_index_evidence_v1/manifest.json) and [attrs manifest](artifacts/phase3_attrs_historical_production_evidence_v1/manifest.json) each have retained signature and timestamp artifacts beside them. Previous production verification reported both evidence sets verified; this document checked the files and sealed manifest identities, **not** a fresh cryptographic/TSA replay. Their respective manifest identities are `sha256:db8391ca9cb3037183ea76f9fef770535e0b1ca690f5fc7737724c0b93822247` and `sha256:5df3803156c79fa34fff2ea60742fc0ef8d48c6963739a38ce5d8974c8b402c1`.

## 15. Key Governance

**KEY-001 — HISTORICAL / NON-PRODUCTION:** ineligible for retroactive production activation. See [key ceremony status](docs/phase3/EVIDENCE_SIGNING_CEREMONY_V1.md); dated templates and readiness records confer no trust.

**KEY-002 — ACTIVE public trust pin, bounded purpose:** [activation decision](artifacts/phase3_evidence_key_002_post_generation_v1/activation_decision_v1.json) identity `sha256:b4100758d623ce9d70e3834fe8890a4b6cbe5099a56af063c066fa279f343b86`; [ACTIVE pin](artifacts/phase3_evidence_key_002_post_generation_v1/active_trust_pin_pending_external_retention_v1.json) identity `sha256:5002095d341ee82ef19a4720932bfe5472ba39f3ec296325f7b3336638abec63`; [external-retention record](artifacts/phase3_evidence_key_002_post_generation_v1/active_pin_external_retention_v1.json) identity `sha256:c9f51034d78a676d2cd894979693ee99812235bede43f23844852514bd7de40d`. All three local records passed canonical-seal checks. The retained record says both owner-attested external channels are confirmed; this file does not independently inspect those channels. Public PEM SHA256 is `sha256:ba99da010ce81cfd802f4d2eb469e358a9dc70771a47dc0946b84b11a36cfca3`; pin interval is 2026-11-01 through 2028-11-01 UTC. Check current external key status at use time.

The activation decision is **key activation only**: it explicitly grants no source rights, intended use, corpus admission or training. Its `trusted_timestamp_status=MISSING` describes that activation record's time; later Stride/attrs timestamp artifacts must be assessed under their own exact manifests. No private-key location, bytes, backup contents or passphrase belongs in this document.

## 16. Important Completed Milestones

| Milestone | Evidence and scope |
|---|---|
| Phase 1 freeze and Phase 2 corpus-control software closure | [Freeze](training/resources/phase1/model_initialization_gate_v1/freeze.json), [production spec](training/specifications/production_corpus_v1.json), [gate implementation](training/production_acquisition.py), tests and dated [audit](docs/PROJECT_AUDIT/YUG_FULL_PROJECT_AUDIT_V1.md). |
| Protected-reference inventory and cumulative governance | [155-record manifest](artifacts/phase3_protected_references_v1/manifest.json), [ledger](artifacts/corpus_ledger_v1/ledger.sqlite), exact external approval inputs. |
| Evidence/trust machinery and KEY-002 activation | [Evidence verifier](training/evidence_bundle.py), [ACTIVE pin](artifacts/phase3_evidence_key_002_post_generation_v1/active_trust_pin_pending_external_retention_v1.json); activation does not grant data use. |
| First controlled Phase 3 admission | [Stride production progress](artifacts/phase3_stride_manual_index_production_v1/progress.json): one accepted file, 363 training tokens. |
| Historical attrs prospective-use path | [16 links](artifacts/phase3_attrs_historical_authority_link_v1/links.jsonl), [decision](artifacts/phase3_attrs_prospective_training_use_v1/decision.json), [training-use hook](training/historical_authority_link.py); no new accounting. |
| Multi-source production and new coordinator | [Build tests](tests/test_production_build.py), [workflow code](training/corpus_workflow.py), [workflow tests](tests/test_corpus_workflow.py). Uninterrupted synthetic approved-resume through export/shard readback passed; Git history records the software checkpoint. |
| Capability validation | Updated 40-capability assessment: 37 PASS / 3 PARTIAL; current full suite 2,236 passed, 2 skipped, one warning on 2026-11-07. |

## 17. Current Known Limitations

- Phase 3 is **not complete**: 417,768 train tokens are only 0.0208884% of the 2B target; most new candidate files still need exact review and decisions.
- No production YUG-100M training run has started or been authorized; no production-trained YUG model checkpoint is established here.
- Rights, use, ownership, current external trust and TSA inputs retain human/external boundaries. Ignored `artifacts/` inputs need exact external preservation for clean-checkout reproduction.
- Extraction supports specified source/file routes; there is no universal arbitrary-format extractor.
- Review/preparation can isolate a bad source. A fatal bound-input failure in the approved production shared build stops that build; it is not per-source failure isolation.
- The uninterrupted synthetic `corpus_workflow.resume` through export/shard readback now passes. Live-source rights and scale remain separate limits. The full-suite NumPy warning is an environment limitation, not an observed test failure.
- Dated narrative documents retain old status statements. Use the register below to identify their current machine-authoritative replacements.

## Known Historical / Stale Narrative Records

| Narrative record | Historical or stale status statement | Current authoritative replacement | Preserve old file? |
|---|---|---|---|
| [Full-project audit, 2026-11-01](docs/PROJECT_AUDIT/YUG_FULL_PROJECT_AUDIT_V1.md) | KEY-002 was not active; Stride was not yet admitted; corpus had 174 items and 417,405 train tokens. | [ACTIVE KEY-002 pin](artifacts/phase3_evidence_key_002_post_generation_v1/active_trust_pin_pending_external_retention_v1.json), [Stride production progress](artifacts/phase3_stride_manual_index_production_v1/progress.json), and the [live ledger](artifacts/corpus_ledger_v1/ledger.sqlite): 175 items, 417,768 train tokens. | **Yes**; dated audit evidence. |
| [Evidence signing ceremony](docs/phase3/EVIDENCE_SIGNING_CEREMONY_V1.md) | Its 2026-11-04 status says no production RFC 3161 response was requested and no attrs production evidence was signed. | [Stride evidence](artifacts/phase3_stride_manual_index_evidence_v1/manifest.json) and [attrs evidence](artifacts/phase3_attrs_historical_production_evidence_v1/manifest.json) have later retained detached signature and timestamp artifacts; assess verification against their exact manifests and trust anchors. | **Yes**; ceremony design and dated status. |
| [Project-authorized permissive profile](docs/phase3/PROJECT_AUTHORIZED_PERMISSIVE_V1.md) | Its Stride paragraph says the live admission decision is pending and no production corpus item was admitted. | [Stride production progress](artifacts/phase3_stride_manual_index_production_v1/progress.json) records one accepted file and 363 train tokens; the [ledger](artifacts/corpus_ledger_v1/ledger.sqlite) accounts for it. | **Yes**; clearance profile remains useful, status sentence is stale. |
| [Git checkpoint manifest, 2026-10-29](docs/phase3/GIT_CHECKPOINT_MANIFEST_V1.md) | Proposes a 49-file staging set from HEAD `40421b...`. | The verified baseline before the coordinator checkpoint was `3b19950a18550a18cb5084d1ac783542fc579373`; current Git history controls the checkpoint identity and file set. | **Yes**; historical checkpoint proposal. |
| [Phase 2 baseline gap audit, 2026-10-23](docs/phase2/PERMANENT_CORPUS_GATE_BASELINE.md) | Describes then-partial or missing corpus controls at HEAD `e283917...`. | Current [production gate](training/production_acquisition.py), [cumulative accounting](training/cumulative_accounting.py), and their tests describe the later implementation; exact live authority still comes from current receipts and ledger. | **Yes**; dated gap assessment. |

The frozen [corpus specification](training/specifications/production_corpus_v1.json) is a machine-readable plan, not a narrative document. Preserve its `acquisition_started=false` field as historical planning state; the later Stride receipt and live ledger establish present acquisition/accounting. Do not rewrite the spec to make an old flag appear current.

## 18. Current Highest-Priority Work

Use the validated `corpus_workflow.resume` -> `production_build` -> export -> token shards -> readback path on genuinely new candidates, with exact human/external approvals. Stop broad pipeline redesign; synthetic validation does not itself approve a live source.

## 19. Git / Software Checkpoint

**VERIFIED 2026-11-07:** The baseline before the corpus-workflow checkpoint was HEAD `3b19950a18550a18cb5084d1ac783542fc579373` (`phase3: checkpoint verified production authority pipeline`). The latest full suite reported **2,196 passed, 2 skipped, one NumPy warning**; **11 workflow tests passed**. [training/corpus_workflow.py](training/corpus_workflow.py) and [tests/test_corpus_workflow.py](tests/test_corpus_workflow.py) passed uninterrupted synthetic integration. Consult `git log` and `git status` for the current checkpoint and working-tree state. The separate audit report and six `docs/phase3/EVIDENCE_KEY_00*.json` files are outside this workflow checkpoint. This document grants no Git, corpus, or training authority.

## 20. Critical Repository Paths

| Area | Entry points |
|---|---|
| Frozen spec and resources | [production_corpus_v1.json](training/specifications/production_corpus_v1.json), [model freeze](training/resources/phase1/model_initialization_gate_v1/freeze.json), [tokenizer JSON](training/resources/phase1/bpe_fitter_optimization_gate_v1/comparison/run-4096-0/tokenizer.json) |
| Discovery and source cache | [source_discovery.py](training/source_discovery.py), [source_prepare.py](training/source_prepare.py), [license_evidence.py](training/license_evidence.py) |
| Rights, exact scope and proposal | [source_approval.py](training/source_approval.py), [source_first_clearance.py](training/source_first_clearance.py), [source_revision_supersession.py](training/source_revision_supersession.py), [source_grant_proposals.py](training/source_grant_proposals.py), [clearance_evidence.py](training/clearance_evidence.py) |
| Admission and orchestration | [source_admission.py](training/source_admission.py), [corpus_workflow.py](training/corpus_workflow.py), [production_build.py](training/production_build.py) |
| Production gates | [production_acquisition.py](training/production_acquisition.py), [content_screening.py](training/content_screening.py), [production_dedup.py](training/production_dedup.py), [quality_filtering.py](training/quality_filtering.py), [protected_references.py](training/protected_references.py), [category_ownership.py](training/category_ownership.py), [corpus_gate.py](training/corpus_gate.py), [language_validation.py](training/language_validation.py) |
| Evidence and accounting | [evidence_bundle.py](training/evidence_bundle.py), [evidence_trust.py](training/evidence_trust.py), [cumulative_accounting.py](training/cumulative_accounting.py), [historical_authority_link.py](training/historical_authority_link.py) |
| Export and training utilities | [notice_attribution.py](training/notice_attribution.py), [token_shards.py](training/token_shards.py), [native model](yug/native/model.py), [training loop](training/loop.py) |
| Retained state | [corpus ledger](artifacts/corpus_ledger_v1/ledger.sqlite), [protected-reference inventory](artifacts/phase3_protected_references_v1/manifest.json), [Stride production](artifacts/phase3_stride_manual_index_production_v1/progress.json) |

## 21. Authority Hierarchy

Precedence is **scope-specific**, not a license to turn every JSON file into approval:

1. For authority and corpus state, independently trusted exact pins/decisions, verified canonical evidence, protected-reference inputs and the live chained ledger control their specific claims. A signature or timestamp proves only what its schema and trust checks establish.
2. For implemented behavior and frozen software contracts, the current Git-visible code, frozen tracked specifications/resources and deterministic retained outputs control. Git history identifies reviewed software checkpoints; uncommitted code must be labeled as such.
3. Current tests and dated verification reports show what passed under their exact inputs and environment; they do not issue rights or training authority.
4. This master file summarizes and links those records. Older narrative docs and audits are historical evidence for their dates, not current machine state.

If records disagree, stop the dependent operation and resolve the exact scope and identity. Do not choose a newer-looking local file as an external trust pin.

## 22. Rules for Updating YUG_MASTER.md

Update this document when a phase changes state, a frozen spec changes through its approved process, a material corpus checkpoint lands, a major pipeline capability or governance decision changes, a software checkpoint becomes canonical, or the immediate next action changes. Do not edit it for each temporary artifact or tiny test. Every meaningful update records **last verified date, Git HEAD, relevant authority paths, what was actually rechecked, and what remains a dated report**. A master-file edit never mutates the underlying authority.

**Anti-circularity rule:** before creating a new subsystem, workflow, pipeline or governance layer, (1) search this file, (2) inspect its referenced live implementation, (3) search the repository, (4) determine whether the capability already exists, and (5) build only after a concrete gap is proven. Prefer testing or fixing the existing YUG path.

## 23. Historical Changelog

| Milestone | Short record |
|---|---|
| Phase 1 | Native tokenizer/model baseline and YUG-100M architecture frozen. |
| Phase 2 | Corpus-control software gates closed as an implementation milestone. |
| 2026-11-01 audit | [Dated full-project assessment](docs/PROJECT_AUDIT/YUG_FULL_PROJECT_AUDIT_V1.md); later operational events supersede its status sentences. |
| Phase 3 first production | One exact Stride Docs file admitted; corpus rose to 175 items and 417,768 train tokens. |
| Phase 3 historical use | Exact attrs prospective-use linkage and separate production evidence verified in prior workflow; no new corpus item. |
| 2026-11-06 validation | 40 capabilities assessed; 36 PASS / 4 PARTIAL; coordinator completed-review resume bug fixed, with integration test still outstanding. |
| 2026-11-07 coordinator validation | One uninterrupted synthetic approved resume through real production export and token-shard readback passed; 11 workflow tests and 2,196 full-suite tests passed. |
