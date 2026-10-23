# YUG-100M Production Corpus Specification v1

Date: 2026-10-09 (Asia/Riyadh).

Status: specification, registry, offline validation and checkpoint-receipt accounting implemented. **Controlled production acquisition is BLOCKED.** This is not an operational 2B acquisition/build coordinator, and no actual production corpus is claimed complete. The executable checks and remaining runtime work are distinguished below.

The latest owner instruction records training readiness as CLOSED. This milestone does not reopen that decision or reinterpret earlier local readiness reports. No training was performed or authorized here.

## Scope and repository evidence

Inspected training/production_corpus.py, training/token_shards.py, training/corpus_prep.py, existing representative normalization, the Stage 1C policy and source modules, both retained license/policy artifacts, frozen sample source_registry.json/acquisition.json, model_config.json and freeze.json. Existing production preparation/shard manifests establish their actual behavior. The evidence paths and SHA256 hashes are pinned in the new specification.

Existing behavior retained:
- Production preparation streams text to disk, retains provenance and applies quality filtering/global exact dedup within that preparation job. It still retains document metadata in memory.
- Existing token shards are uint32 little-endian, use the frozen tokenizer, preserve per-document concatenation without separators, verify checksums and support document-boundary tokenization resume.
- Preparation itself requires a fresh directory after an interrupted preparation; it is not resumable acquisition/preparation.
- Existing splitting is document-based (optionally category-stratified), not source-origin-group based.
- Existing normalization offers a structure-preserving route for code/markdown, but its category dispatcher recognizes the historical mini-corpus categories. Applying its prose default to new code categories would be wrong. No existing normalizer was changed.
- No existing global near-duplicate index, production per-source token quota coordinator or complete per-file production acquisition plan was found in the inspected pipeline.

No model, tokenizer, training, checkpoint, loss, old policy or existing data-pipeline file was edited. There is no Git repository metadata in this workspace; the preserved-source evidence hashes provide the comparison record.

## Exact files added

- training/production_spec.py: specification validation, path/status admission checks, dry-run planning and checkpoint receipt accounting; command-line entry point. No network or training calls.
- training/experiments/production_corpus_spec.py: deterministic builder from retained evidence plus owner-requested candidate discovery names.
- training/specifications/production_corpus_v1.json: frozen category totals, internal 3D targets, caps, source/quality/accounting/build contracts and evidence hashes.
- training/specifications/source_registry_v1.json: source candidate registry with historical status/provenance retained.
- tests/test_production_spec.py: 58 new test cases.
- artifacts/production_corpus_spec_v1/dry_run.json
- artifacts/production_corpus_spec_v1/REPORT.md
- artifacts/production_corpus_spec_v1/verification.json

## Exact allocation

All counts are clean accepted unique **training** tokens measured by the frozen 4096 tokenizer. Validation is tracked separately and never fills the training quota. Raw downloads, rejected material, duplicate content and repeated epochs do not count. This interpretation follows the 2B training-token target; no production validation percentage was invented.

| Category | Total tokens | Internal 3D tokens | Remaining non-3D target |
|---|---:|---:|---:|
| Programming code | 440,000,000 | 10,000,000 | 430,000,000 |
| Unity / C# / game development | 260,000,000 | 50,000,000 | 210,000,000 |
| Software engineering / debugging / testing | 240,000,000 | 0 | 240,000,000 |
| General high-quality text | 240,000,000 | 0 | 240,000,000 |
| Educational / reference | 160,000,000 | 0 | 160,000,000 |
| Nepali + English multilingual | 160,000,000 | 0 | 160,000,000 |
| Technical docs | 120,000,000 | 10,000,000 | 110,000,000 |
| Graphics / shaders / GPU | 100,000,000 | 15,000,000 | 85,000,000 |
| Math / science / geometry | 80,000,000 | 15,000,000 | 65,000,000 |
| Algorithms / data structures | 60,000,000 | 0 | 60,000,000 |
| Linux / systems / tools | 40,000,000 | 0 | 40,000,000 |
| Security | 20,000,000 | 0 | 20,000,000 |
| Instruction | 30,000,000 | 0 | 30,000,000 |
| Reasoning | 20,000,000 | 0 | 20,000,000 |
| Tool-use | 10,000,000 | 0 | 10,000,000 |
| Conversation | 20,000,000 | 0 | 20,000,000 |
| TOTAL | **2,000,000,000** | **100,000,000** | **1,900,000,000** |

No seventeenth category. No top-level rebalance. Source category is a planning default, not an instruction to count every document under multiple categories. A cleared file manifest fixes exactly one owner among allowed_categories, its path and content type. Repository/domain caps aggregate across source views, and known aliases share split groups. Unverified canonical origins block acquisition.

## Internal 3D plan

| Topic | Tokens | Owner |
|---|---:|---|
| Blender / modeling workflows | 20,000,000 | Unity / C# / game development |
| Topology / meshes / retopology | 15,000,000 | Math / science / geometry |
| Rigging / skeletons / skinning / weights | 15,000,000 | Unity / C# / game development |
| Animation / IK / FK / retargeting | 15,000,000 | Unity / C# / game development |
| UV / materials / PBR | 10,000,000 | Graphics / shaders / GPU |
| Blender Python / procedural tooling | 10,000,000 | Programming code |
| Asset formats / interchange / import-export | 10,000,000 | Technical docs |
| Optimization / LOD / production meshes | 5,000,000 | Graphics / shaders / GPU |

The 20M lower end of the requested 20-25M workflow range makes the eight targets total exactly 100M. This is an internal planning allocation, not demonstrated source yield.

The same 100M has a separate primary-form view: authoritative docs/spec/API 30M; real implementation 25M; workflows/tutorials 20M; troubleshooting 10M; procedural tasks 10M; instruction/Q&A 5M. These views are not additive. One primary topic and one primary form per admitted document prevent overlap inflation. Mixed-topic documents can carry other descriptive tags but receive only one quota credit. The initial boundary is whole documents; no unapproved token truncation or duplicate padding is used to force exact totals. A residual quota deficit must be reported.

These targets prepare education, reasoning, scripting, troubleshooting and future tool control. They do not claim native 3D generation.

## Source registry and policy

**174 registry entries: 49 inherited source records, 114 new candidate/discovery records, 11 explicit exclusion classes.** They are not 174 rights-cleared downloadable datasets.

| Production status | Entries |
|---|---:|
| ACCEPT | 16 |
| CONDITIONAL | 31 |
| REVIEW | 111 |
| REJECT | 16 |

ACCEPT entries carry retained Stage 1C acceptance; none receives blanket production file clearance. New discovery names intentionally have no invented URL, revision or verified license. Candidates that require component/license handling are labeled CONDITIONAL; all remain ineligible without clearance. These classifications are workflow states, not new legal findings.

Historical CC BY-SA records (Wikimedia/Nepali Wikipedia) and MS-PL MonoGame are CONDITIONAL under the new production policy; prior status and original evidence remain recorded. Existing Stage 1C files and their allowlist remain untouched. Older restrictive language/policy reasons remain historical provenance, not a newly imposed production category mix.

The inventory covers the named programming, game, graphics, algorithms, math, systems, security, instruction/reasoning/tool, general, educational, Nepali, docs and 3D source families. New names are discovery keys rather than claims that repository root terms clear all contained data. BFCL is validation-only by default. Engine source/default site scraping exclusions are explicit. Assets, binaries, vendors, generated/minified paths and unresolved third-party components stay excluded. Root and nested vendor patterns are tested, including Windows path escape forms.

Acquisition eligibility requires ACCEPT plus immutable revision, locator, license evidence, canonical-origin verification, path allowlist, explicit source split, file manifest and clearance record bound to that exact manifest. Manifest files have exact paths, byte sizes, SHA256 hashes, owner and normalization content type. Clearance evidence hashes are checked. Missing evidence fails closed.

## Source caps, deficits and fallback

Initial engineering ceilings are 10% of the default category per source; per-owner caps also use 10% of that owner category. Repository aliases share a global cap equal to the largest member source cap. Domain/origin groups share a cap derived from 25% of the largest member default category. These are conservative configuration choices, not user-frozen percentages, available yield or a guarantee that the named inventory can fill the corpus.

The YUG-owned tool-use source has an explicit cap exception up to the full 10M tool-use bucket, so concentration caps do not contradict the owner's required YUG-owned majority. The release contract requires a strict YUG-owned majority, but the current receipt auditor does not certify authorship or enforce that release-quality claim. No owned examples were generated.

Even assuming eventual clearance of every non-rejected candidate, simple cap upper bounds leave at least these planning deficits before stricter group caps or actual source yield:

| Category | Minimum deficit under initial source caps |
|---|---:|
| General high-quality text | 48M |
| Educational / reference | 64M |
| Linux / systems / tools | 12M |
| Security | 2M |
| Instruction | 18M |
| Reasoning | 14M |
| Conversation | 16M |

These are consequences of the initial cap/inventory design, not measured shortages of natural data. Zero in other categories proves neither rights-cleared capacity nor quality. Resolve by explicitly extending qualified sources or reviewing cap design with evidence; no automatic cap raising is implemented.

Fallback is confined to another cleared source inside the same owner category and coverage need, under all source/repository/domain caps, with a recorded substitution. Cross-category transfer, silent cap expansion, duplication, synthetic filling and conditional-source promotion are forbidden. Actual source targets/deficits are unknown until acquisition scope/yield is established: dry-run source_progress reports remaining caps separately rather than mislabeling caps as promised source quotas.

## Implemented checks versus production runtime

| Requirement | Current enforcement / limitation |
|---|---|
| 16 categories, exact quotas, subcategories, 3D margins | Executable specification validation; exact total checked |
| Source statuses, provenance, pins, licenses and path scopes | Registry + fail-closed manifest/clearance validation |
| Source/domain/repository caps | Config validation + receipt accounting; no full acquisition coordinator yet |
| One budget owner | Manifest-file uniqueness and receipt owner checks; descriptive tags give no extra credit |
| Exact duplicate accounting | Receipt auditor rejects repeated normalized hashes globally across categories/splits; existing prepare_stream exact dedup retained |
| Near duplicates | Spec defines review policy; receipt auditor rejects repeated cluster IDs or uncleared review claims. It does NOT compute or validate those clusters |
| Generated/minified/vendor filtering | Manifest path exclusions checked; existing quality helpers preserved; content-marker/manual-quality execution remains a build obligation |
| Source-disjoint split | Explicit group assignments and alias consistency validated; current prepared-corpus builder still needs a production source-group integration |
| Clean token accounting | Frozen tokenizer identity required; integer receipt counts, category/source/group/3D caps and validation separation checked. Counts are not independently re-encoded by this auditor |
| Resume and build checkpoints | Pinned request/cursor/hash contract plus 100M/250M/500M/1B/1.5B/2B milestones specified; existing token-shard resume preserved; acquisition/preparation coordinator still pending |
| Progress/deficits/quality | Dry-run and checkpoint receipt reports; unknown measurements remain null rather than fabricated zeros |
| Evidence identity | Deterministic seals, retained evidence SHA256 verification, frozen files checked |
| Final 2B certification | Never inferred from configuration or receipts alone; actual_corpus_complete remains false |

The receipt auditor is an O(document-count) metadata auditor, not a disk-backed 2B-scale dedup implementation. Its input is already-produced accounting evidence. It rejects detectable accounting contradictions but cannot certify a caller's raw token counts, actual normalized text, cluster detector quality or encoded shard contents. Rejected/review raw-token counts are explicitly distinguished from accepted clean tokens.

## Dry run and reproduction

The dry run validates successfully and is deterministic. It schedules **zero external files / zero download bytes**. All 174 entries have explicit acquisition blockers. Clean accepted production tokens remain **0**; total training deficit remains **2B**. Production validation groups/budget are not yet assigned. No separate 20M pilot was introduced.

Rebuild specification (writes only the two new specification JSON files):

    python -B -m training.experiments.production_corpus_spec

Read-only validation / dry run:

    python -B -m training.production_spec

Require readiness for automation (exit 2 while blocked):

    python -B -m training.production_spec --require-ready

Retain a fresh dry-run output with --output NEW_PATH; existing outputs are refused. Audit future JSONL accounting receipts with --receipts PATH; this never certifies payloads. A large external acquisition is not initiated by either entry point.

## Storage

The existing uint32 token format requires exactly 8,000,000,000 bytes (about 7.45 GiB) for 2B training token IDs. Retained token spools plus final training shards require about 16,000,000,000 bytes (14.90 GiB), excluding headers/manifests, validation, raw and normalized documents, rejected payloads, provenance, dedup indices, temporary files and safety margin. Raw-data download volume cannot be estimated reliably from the current candidate registry; no total disk-sufficiency claim is made.

## Verification

New specification tests: 58 passed, 0 failed (4.12 seconds).

Focused command:

    python -B -m pytest -q tests/test_production_spec.py tests/test_rep_corpus_policy.py tests/test_rep_corpus_sources.py tests/test_rep_corpus_license_verify.py tests/test_training_corpus_prep.py tests/test_representative_mini_corpus.py tests/test_production_pipeline.py

Focused result: **232 passed, 0 failed, 0 skipped**, 32.79 seconds.

Full regression: `python -B -m pytest -q` -> **1129 passed, 0 failed, 2 skipped**, 74.73 seconds. Skips are the existing optional llama.cpp real-model test (tests/test_brain_llama_cpp_adapter.py:136) and CUDA FP16 resume test (tests/test_training_checkpoint_amp.py:99) on this CPU-only host. The 58 new cases explain the increase from 1071 to 1129 passes. The read-only --require-ready CLI exits 2 as designed; its report exactly matches dry_run.json. Evidence hashes and frozen configuration/parameter formula were rechecked after tests. Exact results and final file hashes are in verification.json.

## Acquisition readiness verdict

**NOT READY for controlled production acquisition.** The specification/registry and offline checks are delivered, but no exact production file scope is cleared, candidate yield is unverified, several initial caps leave structural deficits, and the production coordinator/global near-duplicate/source-group integration is not implemented by this specification milestone. Those are explicit remaining implementation/evidence gaps, not hidden behind a successful JSON validation.

Before a large acquisition, resolve the recorded production runtime obligations and publish the first reviewed, immutable per-file download plan with exact URLs/revisions/paths/hashes/byte ceilings and source split assignments. Keep all 16 top-level totals fixed. No 2B completion claim and no full pretraining authorization are issued.
