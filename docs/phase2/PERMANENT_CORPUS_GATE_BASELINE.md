# Phase 2 permanent corpus gate: baseline gap audit

Audit date: 2026-10-23. Starting branch: `main`. Verified HEAD:
`e2839173717d8bc969b91d36ccb48afe51f6aae7`. Starting `git status --short`
was empty. This report is the first Phase 2 deliverable; it implements no gate,
approves no data, and authorizes no acquisition or training.

## Evidence and interpretation

Inspected the live production coordinator, runner, dedup index, source admission,
file approval and license-evidence routes, dataset transport/rights-scope routes,
conditional-license contract, corpus preparation, shard API, production spec and
registry, tracked Phase 1 resources, README, preserved evidence documentation, and
relevant tests. Implementation observations below describe this checkpoint, not
unexecuted policy prose. Historical reports are evidence of their own milestones,
not new production approvals. No external legal conclusion is made by this audit.

SATISFIED means reusable under the existing controlled-input trust boundary, not
proof of universal detection, legal ownership, or 2B-token performance. PARTIAL
means a useful mechanism exists but the permanent *training-eligibility* gate
needs an extension. MISSING means no adequate mandatory content check was found.
No one of the 22 requested requirements is out of scope.

**Totals: 8 SATISFIED, 12 PARTIAL, 2 MISSING, 0 DEFER.**

## Requirement matrix

| # | Requirement / status | Existing implementation and test evidence | Limitation, failure mode and smallest extension |
|---|---|---|---|
| 1 | Provenance — SATISFIED | `source_admission.cached_sources/build`, `production_acquisition.item_key/validate_clearance`; `test_source_admission.py` corruption, changed-revision and changed-content cases; `test_training_corpus_prep.py::test_accepted_documents_retain_full_provenance` | Exact source/revision/path/hash and retained receipts are reusable. Publisher statements still need trusted review; cryptographic binding does not establish copyright ownership. Preserve these fields through added decisions. |
| 2 | Licensing / usage rights — PARTIAL | `source_approval.package_scope/mit_file_scope/mit_file_reason`, `dataset_rights_scope.scope_from_card/assess_provenance`, `conditional_license.validate_approval`; tests in `test_clearance_evidence.py`, `test_dataset_rights_scope.py`, `test_conditional_license.py` | Acquisition checks `production_preparation=APPROVED`, not `commercial_training=APPROVED`. Preparation-only material must not become training-eligible. Add explicit intended-use validation at the future training-corpus seal; preserve narrow supported evidence profiles and REVIEW for unsupported rights chains. |
| 3 | Allowed / conditional / rejected policy — PARTIAL | `source_prepare.LICENSES`, `production_spec.acquisition_blockers`, `validate_clearance`, `conditional_license.CONDITIONAL`; `test_allowed_list_unchanged`, `test_no_promotion_of_conditional_source`, `test_missing_approval_rejected_at_acquisition_boundary` | Source ACCEPT, file CLEARED, dataset CLEAR and production ACCEPTED mean different things. A future adapter could misinterpret evidence coverage as admission. Add an explicit status mapping and mandatory gate receipt, without changing the allowlist or promoting REVIEW/EXCLUDE. |
| 4 | Source identity — SATISFIED | `cached_sources`, `source_admission.build`, `validate_clearance`; `test_changed_revision_not_promoted`, `test_wrong_source_binding` | Revision/locator/registry bindings are checked. Source-group aliases still depend on controlled registry evidence, not automatic proof that two domains are unrelated. Reuse existing identifiers. |
| 5 | Document identity — SATISFIED | `item_key`, `dataset_clearance.decision`, `dataset_transport.build`, `dataset_payload.payload_digest`; source/dataset clearance and corruption tests | Raw item identity and normalized content hash serve different purposes. Preserve both, plus dataset record/payload proofs; never replace a changed record under an old grant. |
| 6 | Exact deduplication — SATISFIED | `production_dedup.DedupIndex.add`, `ProductionBuild.register_prior_production/run`; `test_production_cross_batch.py::test_cross_build_duplicates_and_split_conflicts`, `test_cross_category_duplicate_only_one_accounted` | Exact comparison is of canonical normalized text; raw hashes remain integrity evidence. Prior inventory must be complete and owner-controlled. Hashes cannot discover an omitted independent build. No second exact-dedup system is needed. |
| 7 | Near deduplication — PARTIAL | `production_dedup.features/similarity/DedupIndex.add/assignment`; `test_exact_edited_format_reordered`, `test_code_comments_and_layout`, `test_candidate_overflow_fails_closed` | Lexical heuristic, no recall guarantee or demonstrated 2B throughput. Add a versioned adversarial recall/false-positive corpus and bounded scaling measurements before changing thresholds or algorithm. Reuse the index. |
| 8 | Train/validation split integrity — SATISFIED | `source_admission.holdout_groups`, `ProductionBuild.register/register_references/register_prior_production/run`; cross-batch split tests, `test_validation_duplicate_never_leaks_to_train`, `test_source_group_conflict_rejected_before_build` | Registered groups and detected clusters cannot silently cross splits. Incomplete protected references/undetected near copies remain contamination coverage gaps, not permission to weaken this mechanism. Legacy completed replay is allowed; unbound new admission is not. |
| 9 | PII detection — MISSING | No mandatory payload PII detector found in the live admission path. Provenance/attribution fields are not content screening. | Contact details/personal identifiers can enter text with valid rights. Add bounded deterministic detectors with rule IDs, context and explicit REVIEW/EXCLUDE handling; test emails, international phones, identifiers, multilingual text, benign examples and obfuscations. Do not claim complete PII detection. |
| 10 | Secrets / credentials — MISSING | `.gitignore` protects local Git storage; transport URL credential restrictions do not scan corpus content. `ProductionBuild._prepare` has no credential scanner. | Licensed code may expose keys, tokens/passwords/private keys. Add pre-admission scanning of raw and normalized text with format/prefix/context rules and bounded entropy signals. Test synthetic canaries, multiline keys, placeholders, split strings and malformed inputs. Never log actual detected secrets. |
| 11 | Benchmark contamination — PARTIAL | `register_references` pins a nonempty inventory and indexes validation references; `holdout_groups` excludes retained projects. `test_heldout_inventory_exclusion_and_mandatory_pin`, `test_source_holdout_overrides_grant`; spec benchmark rule is also present. | Existing supplied references are not a comprehensive protected benchmark registry. Missing benchmark material or copied answer fragments can pass. Extend reference schema with benchmark/source/work identity, version, protected splits, hashes and provenance; reuse the same dedup machinery. Add exact, normalized, source-alias and near-copy tests. |
| 12 | Quality filtering — PARTIAL | `_prepare`, `corpus_prep.check_quality/QualityConfig`, `source_prepare.screening`, `source_approval.file_reason`; `test_quota_rejection_and_generated_filter`, `test_check_quality_unit_level` | Strict UTF-8/NUL checks, size bound, generated markers, empty/minimum-length and single-token repetition checks exist. Multi-fragment repetition, boilerplate, truncation and low-information text are not comprehensively checked. Add deterministic bounded metrics with content-type-specific policies and retained reasons; malformed snippets must not be treated like complete programs. |
| 13 | Language tagging — PARTIAL | `validate_clearance` requires a string language; plans/cache records carry labels; `features` uses Python-specific tokenization when labelled Python. | No general language-detection/label-validation confidence gate. Wrong/empty labels can affect dedup routing and composition. Add deterministic bounded script/content checks, explicit unknown/mixed labels and review routing; distinguish human language from programming language. Test Nepali/English/code mixtures without changing normalization. |
| 14 | YUG category tagging — PARTIAL | `validate_clearance` checks owner membership/subcategory and 3D pairing; spec `accounting.one_budget_owner`; `test_production_spec.py`, source-admission allocation checks | Taxonomy validity is enforced, semantic correctness of a supplied label is not. Record label origin and rule/reviewer evidence, retain multiple descriptive tags with one owner. Test ambiguous code/tutorial/multilingual items and prohibit labels overriding rights/safety outcomes. |
| 15 | Category/quota compatibility — PARTIAL | `_limits/_account/verify`, `production_spec.QUOTAS`; `test_quota_rejection_and_generated_filter`, `test_accounting_corruption_detected` | Counters are build-local. `register_prior_production` verifies earlier category totals but does not seed cumulative source/category/group/3D quota counters. Independent distinct-content builds can each consume the same cap. Add a verified cumulative quota view/reservation bound to the complete prior inventory; test two-build cap exhaustion and concurrent reservations. Never rewrite historical totals. |
| 16 | Attribution / NOTICE obligations — PARTIAL | Per-file `license_evidence`/`attribution` mandatory; conditional approval records notices, modifications, reciprocity, release controls; `test_incomplete_never_approves` | Nonempty obligations are plans, not proof notices were assembled/preserved for intended distribution. Add hash-bound obligation inventory and completion evidence for the applicable use. Test missing/mismatched notices and forbidden uses. Final product distribution mechanics belong to a later release gate. |
| 17 | Immutable evidence / manifests — SATISFIED | `production_spec.seal/verify_seal`, `production_corpus.canonical/atomic_json`, `validate_clearance`, pinned plans/cache receipts; `test_changed_clearance_or_plan_refused`, `test_clearance_sidecar_mutation_rejected_on_resume` | Tamper-evident hashes are not signatures, backups or write-once storage. Keep owner-controlled approval provenance and external backups; append new policy decisions rather than rewriting old seals. |
| 18 | Deterministic replay — SATISFIED | `ProductionBuild.run/_cached_tokens/export_shards`, `token_shards.TokenShardStream`; `test_production_resume_no_refetch_or_reencode`, `test_distinct_content_and_replay_without_fetch_encode`, `test_split_build_and_uninterrupted_shards_identical` | Determinism assumes identical pinned inputs/policies. New scanner decisions must become pinned inputs and must invalidate incompatible unfinished resumes. No inference-time model is needed. |
| 19 | Corruption / integrity — SATISFIED | `_acquire/verify/register_prior_production`, shard checksums, cache receipts; committed corruption, partial-download recovery and wrong-payload tests | Hash/size/SQLite validation detects tested corruption; seals do not defend against an authorized actor replacing all evidence. Reuse verification, atomic publication and fail-closed error handling. |
| 20 | Fail-closed behavior — PARTIAL | Missing grants, history, references, changed evidence, candidate overflow and corrupt data raise/refuse admission; `test_missing_corrupt_incompatible_history_fails_closed`, clearance corruption tests | Existing checks fail closed, but absent PII/secrets checks and preparation-vs-training distinction cannot fail closed yet. Make completeness of all required check receipts a mandatory acceptance precondition, with unknown schema/policy/check/error treated as blocked or REVIEW. |
| 21 | Explicit human approval boundaries — PARTIAL | `conditional_license.validate_approval`, exact existing grants in `source_admission.build`; source approval restricted to evidence profiles | Trusted workflow supplies reviewer identity; hashes are not reviewer authentication. Specify who may approve which decision, intended-use scope, revocation and re-review on evidence change. Test review overrides cannot waive hard exclusions or change hashes/splits. No silent whole-repository approval. |
| 22 | Auditability / decision provenance — PARTIAL | Sealed clearance records, reason-coded stages, request/policy identities, progress/admission exports and checkpoints | New content checks need structured rule/version/evidence records rather than one opaque quality ACCEPT flag. Add a compact decision envelope referencing existing evidence; test complete reason trails and deterministic serialization without leaking PII/secrets. |

## Special audits and concrete design boundaries

### Near dedup and cross-batch state

The existing policy is `lexical-bottom128-v1`: five-token shingles (shorter for
short documents), 128-bit hashes, bottom 128 anchors, full stored shingle-set
verification with Jaccard >= 0.5 or containment >= 0.8. Python tokenization removes
comments/layout for matching, with a lexical fallback; other code uses a lexical
scanner. Prose matching casefolds words. Training text is not rewritten by these
features. Exact normalized matches bypass duplicate anchor storage. Connected
components have minimum-member labels; validation wins within a build. Prior
production matches instead block re-admission, including explicit split conflicts.

Limits include 1 MiB documents and 10,000 candidates; overflow raises rather than
silently sampling away candidates. SQLite uses disk-backed candidate tables and
bounded caches. Full shingle blobs and up to 128 anchors per distinct document
still require substantial disk. Boilerplate can create hot anchors and overflow;
short texts, renamed code, partial excerpts and paraphrases need measured recall.
Transitive clusters may suppress distinct related documents. Labels settle within
a pinned universe and are not a promise of invariant labels after adding members.

Each fresh build verifies and indexes prior retained content again. This is safe
for bounded evidence but can become expensive with many builds. No measured 2B
capacity claim is justified. A shared index, if later needed, must be an optimization
of these semantics after profiling, not a new incompatible identity system.

The complete-history inventory is an owner assertion. Ancestor omission is checked,
but an unknown independent build cannot be discovered from hashes alone. Keep a
controlled corpus lineage/catalog and serialize admissions or reserve cumulative
quota capacity against a pinned head. Explicit genesis is not automatic permission
to ignore an existing corpus. Dry-run currently registers references/clearance but
does not call `register_prior_production`; it is not a full Phase 2 readiness proof.

### PII and secrets: layered, conservative proposal only

Scan bounded raw decoded text and normalized text before token accounting. Use
versioned format rules for private keys/provider tokens and credential assignments;
use contextual patterns for emails, phones and personal identifiers; use entropy
only as a supporting signal. Public copyright-author emails are not automatically
equivalent to leaked credentials. Policy must explicitly distinguish legal
attribution, examples/placeholders, public contact information and sensitive data.
Do not auto-delete attribution or redact existing payloads under their old identity.

High-confidence prohibited content should EXCLUDE; uncertain/context-dependent
findings should REVIEW. Scan errors, unsupported required detectors and resource
limits must not produce a pass. Any future approved redaction produces a new
derived document/hash, transformation evidence and a fresh gate run. Keep finding
locations/rule IDs and masked summaries in reports; quarantine access and retention
must be controlled. Tests use fake canaries, never live credentials. Obfuscated,
encoded, unknown-format and culturally variable identifiers remain residual risk;
no detector or combination proves absence of PII.

### Protected evaluation registry

Extend the existing pinned reference inventory, not a competing exclusion index.
Record protected benchmark name/version, source/work aliases, reference IDs, exact
raw and normalized hashes, permitted role and provenance. Where lawful retained
text exists, reuse lexical near matching; where only hashes exist, explicitly report
the weaker detection coverage. Missing required registry state blocks new admission.
Protect questions, answers and derived solution fragments; test copies under new
paths/sources, reordered/edited text and false positives on common code. Never
assume source-disjointness proves absence of benchmark contamination, or that an
unrecognized benchmark name is safe. Registry completeness remains an operational
review responsibility.

### Quality and taxonomy

Keep objective checks distinct from subjective usefulness: decoder errors/NULs,
known generated markers, empty text and excessive repeated fragments can have
explicit policies; semantic correctness, educational value and subtle junk cannot
be inferred from passing those checks. Add line/block repetition, boilerplate
fraction and simple information-density diagnostics, calibrated separately for
code, prose and documentation. Flag suspected truncation, do not silently repair it.
No LLM scoring dependency is justified by this audit.

Reuse the exact 16 `production_spec.QUOTAS`: 440M code; 260M C#/game-dev; 240M
software engineering; 240M general English; 160M educational/reference; 160M Nepali;
120M documentation; 100M graphics; 80M math/science; 60M algorithms; 40M Linux;
20M security; 30M instruction; 20M reasoning; 10M tool-use; 20M conversation.
Total remains 2,000,000,000 unique clean training tokens. Validation is separate.
Preserve the existing 100M internal 3D allocation and its non-additive topic/form
views. Whole-document boundaries leave explicit deficits; no silent truncation,
repetition, owner transfer, source-cap increase or synthetic filler.

Language/domain/format/difficulty are descriptive dimensions, not extra token
budgets. A Nepali Unity debugging instruction has one approved budget owner and
multiple evidenced tags. Category demand never turns uncertain rights or safety
into approval. Phase 2 must enforce cumulative caps, not mistake per-build counters
for a corpus-wide budget. `production_spec.audit_checkpoint` already checks caps
and exclusivity over a supplied receipt stream (`test_global_exclusivity_and_caps`).
Reuse its constraints; it is not currently a mandatory cumulative reservation in
`ProductionBuild.run`, and its own report disclaims payload certification.

## Minimal architecture and state model (proposed, not implemented)

Keep discovery, preparation, rights evidence and exact grant materialization as
they are. Keep `ProductionBuild` as the production boundary, `DedupIndex` as the
dedup engine, and existing shard/replay/accounting abstractions. Add only a small
versioned policy/check-result contract and missing content checks. A seal must bind
raw/normalized identity, source/revision, intended use, all check policy versions,
rights/approval evidence, reference/history identities, split, owner and reasons.
An evidence-only CLEAR/EVIDENCED value must never stand for final ACCEPT.

The suggested long lifecycle conflates evidence, processing progress and decisions.
Do not rename all Phase 1 states. Retain `PENDING -> ACQUIRED -> PREPARED` and
existing rejection/accounting/export states. Add a separate check-result vector:
`PASS / REVIEW / EXCLUDE / ERROR` per required check, with policy and evidence IDs.
Language/classification checks can run alongside quality; prior split groups must
be known before dedup. Rights/safety checks precede accounting. After all required
checks and cluster settlement, verify split and cumulative quota reservation, then
commit ACCEPTED and publish a sealed training-eligible manifest atomically.

Final precedence: EXCLUDE remains excluded; unresolved/error/incomplete checks
cannot ACCEPT; REVIEW requires the authorized bounded review route. ACCEPT is
only possible for the requested use. `SEALED` is a verified publication property,
not permission to skip split/quota checking. Never expose ACCEPT before the split
is validated. Existing completed Phase 1 replay remains historical replay; it does
not silently certify old payloads under new Phase 2 privacy policies. Re-gating
requires a separate decision receipt without altering prior ledgers.

Proposed minimal file boundaries:

- `training/corpus_gate.py`: versioned check receipt, required-check aggregation,
  intended-use decision and final seal verification; no alternate acquisition path.
- `training/content_screening.py`: bounded deterministic PII/secret/quality checks.
- Existing `production_acquisition.py`: invoke checks before accounting/export;
  extend prior-state reuse for cumulative budget checks.
- Existing reference machinery: protected registry validation/schema extension;
  a small `training/protected_references.py` only if shared parsing warrants it.
- Existing specs/policy docs: approved detector policies, controlled review authority,
  intended uses and reference inventory requirements. No license expansion.
- Existing shard/training interfaces: consume verified eligible manifests; explicitly
  separate diagnostic/test data from production authorization, without model or
  training-numerics changes. Training remains unauthorized during implementation.

## Test boundaries, implementation order and exit criteria

1. **First slice: mandatory decision receipt and intended-use enforcement.** Bind
   the minimal receipt to existing identities and enforce it at production admission
   and eligible export. Test missing/unknown/incomplete receipts, REVIEW/EXCLUDE,
   preparation-only conditional approval, changed policy/evidence and replay. Do not
   label placeholder privacy checks PASS. This slice alone does not complete Phase 2.
2. Add bounded secret/PII checks plus explicit policy decisions and masked reporting.
3. Extend protected reference coverage and adversarial near-copy tests using the
   current index. Measure false positives/recall against a fixed diagnostic set.
4. Add quality/language/category evidence and deterministic unknown/review behavior.
5. Close cumulative quota/lineage and concurrent-admission reservations; verify
   obligation inventory and final manifest publication. Include dry-run parity.
6. Bounded integration/replay/corruption tests, clean-checkout full regression,
   measured scaling/storage validation and human review of residual risks.

Unit tests should cover detector boundaries, status precedence, canonical receipts,
intended uses and policy versioning. Integration tests must exercise the real runner,
not only a new helper: data that passes standalone rights checks but fails content
screening must add zero tokens and produce no eligible shard. Replay tests should
replace fetch/encode with failing callbacks. Corruption tests mutate every bound
input and check refusal. Cross-build tests must cover duplicates, split conflicts,
distinct data, prior inventory omissions, cumulative caps and concurrent reservation
conflicts. Use temporary diagnostic payloads and reviewed tracked fixtures; no tests
should need ignored operational artifacts or real secrets.

Measurable Phase 2 completion gates:

- All identified production admission/export routes require a complete, current,
  intended-use-bound gate receipt: zero tested silent admission paths.
- REVIEW, EXCLUDE, ERROR, absent checks and conditional approvals lacking the
  requested use contribute zero accepted training tokens in negative tests.
- Same pinned input/policy/history yields byte-identical decision manifests and
  tokens across fresh runs/resume; completed replay fetches/encodes zero payloads.
- All bound evidence mutations fail closed; immutable prior receipts remain unchanged.
- Exact, normalized, cross-category, cross-build and split conflict tests pass;
  near-duplicate recall/false-positive acceptance criteria are declared before the
  bounded evaluation, with failure/overflow outcomes explicitly tested.
- Known protected source/hash/near-copy canaries cannot enter training. Missing
  required registry state blocks admission; registry coverage limitations recorded.
- Privacy/secrets canary suite and benign-context suite pass predeclared policy
  expectations. Reports do not contain canary secret values. Residual risk accepted
  explicitly, not described as perfect detection.
- Quality/corruption cases, unknown language/category routes and attribution/NOTICE
  evidence failures behave as declared. No label overrides an exclusion.
- Cumulative 16-category/source/group/3D caps cannot be exceeded across builds;
  validation does not fill training quotas; replay cannot double-account.
- Dry-run checks the same required control evidence without fetching/tokenizing;
  it clearly distinguishes untested payload checks from completed acceptance.
- Bounded storage/runtime measurements include dedup features, review cache,
  temporary files and prior-state replay; budgets and overflow tests are explicit.
- Focused tests and full clean-checkout regression have zero failures; frozen model,
  tokenizer, normalization/training semantics and allocations have no changes.

## Non-goals and risks

Defer model/Core v2 work, training execution, 2B acquisition, new license-family
approvals, automatic legal judgments, model-based quality scoring, semantic clone
detection and product release implementation. Their absence does not remove the
need to record obligations/intended-use restrictions in this gate.

The known aggregate review-cache storage ceiling remains a Phase 1 residual risk
and a Phase 2 bounded-resource follow-up. Per-file/chunk bounds and a runner free-disk
estimate do not provide a global cache reservation. Do not infer safe production
scale from small tests. Other risks are source alias omissions, maliciously forged
owner-supplied approvals, incomplete benchmark registries, privacy false negatives,
near-dedup false positives/negatives, language bias, stale policy receipts and
concurrent independent admissions. Controlled inputs and hashes reduce accidental
errors but are not a cryptographic human-authorization system.

## Validation record

Starting commit and clean working tree were verified. This audit changes this report
only. Focused command (Python 3.11 in the existing uv test environment):

```text
python -B -m pytest -q tests/test_production_cross_batch.py tests/test_production_acquisition.py tests/test_conditional_license.py tests/test_dataset_rights_scope.py tests/test_source_admission.py tests/test_training_corpus_prep.py
```

Result: **144 passed, 0 failed, 0 skipped in 67.57 seconds**. These test production
duplicate/split boundaries, replay, corruption, conditional approvals, rights-scope
ambiguity, admission grants and basic quality behavior. They do not prove the newly
proposed Phase 2 controls. `git diff --check` passed; the new report was separately
checked for trailing whitespace. No staged changes. Runtime search found no reference
to `YUG_EXTERNAL_SYSTEMS_RESEARCH` in `training/` or `yug/`.
Full regression is deliberately not rerun for a documentation-only baseline audit;
the prior 1469-pass / 2-skip clean-checkout result remains historical evidence, not
a newly executed result. No corpus was fetched, re-cleared, re-accounted or trained.

## Slice 1 implementation (2026-10-23)

This section supersedes only the baseline's *proposed* authorization-boundary work.
The 22-row matrix remains the historical baseline. `training/corpus_gate.py` now
verifies `yug-corpus-decision-v1` / `explicit-intended-use-v1` receipts using the
existing canonical sealing functions. A sealed `yug-corpus-decisions-v1` bundle
contains per-clearance receipts, with a 16 MiB metadata ceiling. Unknown versions,
duplicate items, absent rows and mismatched hashes cannot authorize a requested use.
No discovery, rights-evidence or approval generator automatically issues these receipts.

Each receipt contains:

- `schema`, `policy`, `identity` (canonical seal), `decision`;
- `binding`: exact clearance identity, full source-record identity, spec identity,
  source ID, revision, document path and payload SHA256;
- `intended_uses`: explicit non-inheriting members of `discovery`, `acquisition`,
  `inspection`, `production_preparation`, `evaluation`, `commercial_training`;
- `authority`: human kind and reviewer ID; `decided_at` is supplied metadata,
  never a nondeterministic verifier timestamp;
- `human_approval_required=true`, `human_approved=true`, explicit `reason`,
  `obligations`, `restrictions`;
- `evidence`: hash-bound local evidence with approval, rights_scope and provenance
  roles. Existing license evidence and conditional approval are reverified as well.

Only ACCEPT plus the exact requested use is authorized. REVIEW, EXCLUDE, REJECT,
missing approval, changed bindings and corrupt evidence fail closed. The controlled
owner workflow supplies receipts and meaningful obligation/restriction text; strings
and seals are not legal conclusions or cryptographic identity authentication. This
slice deliberately requires human authority rather than inventing delegated auto-grants.
It does not parse arbitrary obligation prose into new legal permissions.

`ProductionBuild.run` checks use authorization before acquisition/token accounting
when evaluation/training is requested. The runner exposes `--intended-use` and
`--decision-receipts`, including dry-run checking. Default remains explicitly
preparation-only for Phase 1 compatibility. The request's use and bundle identity
are pinned for resumed authorized builds; dropping/changing them is rejected.
Conditional training additionally requires the existing source-specific approval's
`commercial_training` decision to be APPROVED, not merely production_preparation.
Validation documents in a training-corpus build require explicit evaluation use.

`export_shards(..., intended_use=..., decision_receipts=...)` rechecks payloads,
clearance and evidence and binds authorization into the prepared manifest and thus
the existing shard provenance chain. An existing destination cannot be relabeled
with another use. `replay_for_use` is the authorized production consumer entry point:
it rechecks evidence, export use, payloads and shard identity before returning replay.
Generic `TokenShardStream` remains an integrity decoder, not an authorization API.
The low-level Trainer accepts caller-supplied tensors/sequences; no production
training orchestrator was added or executed. Future production consumers must use
the authorized entry point, not interpret decodable tokens or a `train` split name
as permission. This is not an adversarial sandbox against callers deliberately
constructing tensors outside the production pipeline.

### Phase 1 compatibility and migration

Historical ACCEPTED means admitted preparation/accounting, not expanded use rights.
Existing preparation/replay outputs retain their prior identities and schemas;
absence of a receipt never implies training permission. Migration requires explicit
new receipts referencing the old exact clearance/source/spec and a **new export
destination**. Old seals and exports are not rewritten. No retained corpus was
migrated by this change. A rejected export follows the existing FAILED-state rule;
explicit valid replay recovers the build before a subsequent export attempt.

The existing policy list, grants, source decisions, allocation, dedup and split
requirements remain necessary; a receipt cannot replace them. Diagnostic builds
cannot issue authorized production exports. Preparation may still produce cached
tokens/shards without authorization; these remain preparation artifacts only.

### Gap movement and remaining work

Rows 2, 3, 20, 21 and 22 are reduced, but remain PARTIAL: use authorization is now
explicit; the full privacy/quality/contamination decision vector, operational reviewer
control and release obligations are not complete. Rows 17/18 are strengthened, still
SATISFIED. Totals remain **8 SATISFIED / 12 PARTIAL / 2 MISSING / 0 DEFER** under
the full Phase 2 criteria. No missing safety check is represented as PASS and no
receipt means the whole Phase 2 safety gate is complete.

Recommended Slice 2: bounded deterministic secrets/PII screening, with approved
rule policies, masked findings and mandatory check-result evidence before final
training eligibility. Benchmark registry coverage, cumulative quotas and broader
quality/tagging remain separate later work. No detector, quota redesign, Core v2,
model/tokenizer/training-numerics or allocation change is included here.

### Slice 1 validation commands

Focused: `python -B -m pytest -q tests/test_corpus_gate.py` — **33 passed,
0 failed, 0 skipped**, 11.78 seconds. An earlier 28-case run had one test-sequencing
failure (attempting export after the expected FAILED state); the test was corrected
to explicitly replay/recover, then passed. Five boundary cases were subsequently added.

Affected suite: `python -B -m pytest -q tests/test_corpus_gate.py
tests/test_production_cross_batch.py tests/test_production_acquisition.py
tests/test_production_build.py tests/test_production_spec.py
tests/test_production_pipeline.py tests/test_source_admission.py
tests/test_source_approval.py tests/test_clearance_evidence.py
tests/test_conditional_license.py tests/test_dataset_clearance.py
tests/test_dataset_rights_scope.py` — **341 passed, 0 failed, 0 skipped**,
144.85 seconds. This run collected the earlier 28-case receipt suite; the final
33-case focused run covers the five additional cases. Full regression covers all.

Full `python -B -m pytest -q`: **1502 passed, 0 failed, 2 existing skips** in
219.67 seconds. Skips: real external GGUF adapter (`YUG_TEST_MODEL_PATH`) and CUDA
FP16 GradScaler resume. Tests ran using the existing Python 3.11 uv environment
with pytest/PyYAML/torch/pyarrow; no dependency declaration changed.

Final `git diff --check`: PASS. Reviewed complete tracked diff and new code/tests;
five intended files only (this report, `corpus_gate.py`, `test_corpus_gate.py`,
`production_acquisition.py`, `production_build.py`). Index empty. No frozen
native/tokenizer, training loop/checkpoint/loss/evaluation/config or specification
diff. No corpus acquisition, real approval issuance, production migration, training
job, Core v2, staging, commit or push. Slice 1 is ready for review, not a declaration
that the full permanent corpus safety gate or production training is authorized.

## Slice 1 re-review: authorization manifest integrity repair

The subsequent checkpoint review found a HIGH blocker in `replay_for_use`:
it compared authorization from local `prepared.json` but trusted that file's
recorded identity without recomputing it. `TokenShardStream` separately verified
its embedded manifest. Adding an authorization field to a legacy local manifest,
with valid receipts but unchanged identity/shards, therefore returned 619 tokens.
The earlier Slice 1 passing suite did not cover that attack; its review verdict
was superseded by REVIEW FAILED. Nothing was staged or committed.

The minimal repair recomputes the existing canonical `identity(body)` over every
local manifest field except `prepared_corpus_identity` itself, including the full
authorization field, before trusting authorization. It then constructs the existing
integrity-verified `TokenShardStream` with that verified identity and requires
canonical complete-manifest equality against `stream.manifest['prepared_manifest']`
before returning the stream. No authorization field is excluded from hashing.
The shard decoder's own embedded-manifest and shard checks remain unchanged.

Six adversarial cases cover the exact legacy promotion (with and without local
resealing), authorization-only mutation, wrong local identity, locally resealed
content mismatch and changed embedded authorization. Valid receipts cannot override
these failures. The exact fixture still decodes 619 tokens through generic legacy
integrity replay; authorization-bearing replay rejects it. Valid authorized replay
remains deterministic. No historical manifests are rewritten or grandfathered.

Focused `python -B -m pytest -q tests/test_corpus_gate.py`: **39 passed, 0 failed,
0 skipped**, 17.07 seconds (33 existing plus six new adversarial cases).

The broader Slice 1 command listed above, including production acquisition/build,
cross-batch dedup, accounting and replay suites, passed **352 tests, 0 failed,
0 skipped** in 123.35 seconds after the repair.

Full `python -B -m pytest -q` after the repair: **1508 passed, 0 failed, 2 existing
skips**, 235.57 seconds. Skips remain the external GGUF adapter and CUDA FP16 resume.
`git diff --check` passed. The complete working diff and new test additions were
reviewed; no frozen model/tokenizer/training-numerics/specification diff, staging,
commit, push, production training or Slice 2 work occurred. This repair changes only
`production_acquisition.py`, `test_corpus_gate.py` and this report relative to the
reviewed Slice 1 state. The existing `corpus_gate.py` and `production_build.py`
changes remain pending from Slice 1, untouched by this repair.

The reproduced HIGH finding is verified fixed. No additional unresolved Slice 1
finding was identified in this scoped repair; broader Phase 2 gaps and the known
Phase 1 aggregate review-cache storage residual are unchanged, not claimed closed.
