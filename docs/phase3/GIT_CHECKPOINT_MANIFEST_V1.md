# Phase 2 and Phase 3 software checkpoint manifest v1

Prepared 2026-10-29 from working tree at Git HEAD
`40421b459706d1a952a394197d0dac0e02ff1ea8`. This is a staging
recommendation, not a commit, source approval, corpus receipt, or training
authorization. Paths in `TRACK` are Git-visible and should be reviewed as one
software dependency closure before staging. No files were staged to make this
manifest.

## TRACK: exact software set (49 files including this manifest)

### Modified tracked repository metadata (1)

```text
.gitattributes
```

The two new Phase 3 JSON inputs are exact-byte hash inputs. Their specific
`-text` attributes preserve the current seed CRLF bytes and one-line sealed
review-registry bytes across platforms; no file content or identity was changed.

### Modified tracked implementation (6)

```text
training/clearance_evidence.py
training/experiments/acquisition_readiness_gate.py
training/production_acquisition.py
training/production_build.py
training/source_admission.py
training/source_approval.py
```

### New implementation (10)

```text
training/approval_governance.py
training/category_ownership.py
training/cumulative_accounting.py
training/historical_corpus_import.py
training/language_validation.py
training/notice_attribution.py
training/protected_reference_migration.py
training/protected_references.py
training/quality_filtering.py
training/source_grant_proposals.py
```

### Modified tracked regression tests (7)

```text
tests/test_clearance_evidence.py
tests/test_corpus_gate.py
tests/test_production_acquisition.py
tests/test_production_build.py
tests/test_production_cross_batch.py
tests/test_source_admission.py
tests/test_source_approval.py
```

### New regression tests (13)

```text
tests/test_approval_governance.py
tests/test_category_ownership.py
tests/test_cumulative_accounting.py
tests/test_historical_corpus_import.py
tests/test_language_plan_authority.py
tests/test_language_validation.py
tests/test_notice_attribution.py
tests/test_phase2_integration.py
tests/test_phase3_source_candidate_intake.py
tests/test_protected_reference_migration.py
tests/test_protected_references.py
tests/test_quality_filtering.py
tests/test_source_grant_proposals.py
```

### Phase 2 policy and governance documentation (7)

```text
docs/phase2/APPROVAL_GOVERNANCE_STATUS_V1.md
docs/phase2/CATEGORY_OWNERSHIP_POLICY_V1.md
docs/phase2/CUMULATIVE_ACCOUNTING_V1.md
docs/phase2/LANGUAGE_VALIDATION_POLICY_V1.md
docs/phase2/NOTICE_ATTRIBUTION_POLICY_V1.md
docs/phase2/PROTECTED_REFERENCE_REGISTRY_V1.md
docs/phase2/QUALITY_FILTERING_POLICY_V1.md
```

### Phase 3 documentation and review specifications (5)

```text
docs/phase3/GIT_CHECKPOINT_MANIFEST_V1.md
docs/phase3/SOURCE_CANDIDATE_INTAKE_V1.md
docs/phase3/YUG_OPERATOR_COMMANDS_V1.md
training/specifications/phase3_github_seeds_batch01.json
training/specifications/phase3_source_candidate_registry_v1.json
```

The current production coordinator imports new governance, ownership,
accounting, language, NOTICE, protected-reference, and quality modules. The
protected-reference migration and grant-dossier modules are standalone CLIs
covered by new tests. All current local `training.*`/`yug.*` imports resolve to
tracked or listed files; the test set includes every untracked test module.
The existing tracked `training/specifications/{production_corpus_v1,
source_registry_v1}.json`, frozen tokenizer resource, and Phase 1 fixtures
remain required dependencies; they are already in Git and have no current diff.

## TRACK_RECEIPT: exact candidates, not copied in this preparation

There is a Phase 1 convention at `docs/evidence/phase1/`, with byte-identical
copies and a provenance manifest. There is no reviewed Phase 2/3 destination or
receipt-selection decision yet. Propose `docs/evidence/phase2/` and
`docs/evidence/phase3/`, each with `README.md` and `provenance.json` recording
original path, exact SHA256, copy hash, role, and whether the record is an
external authority pin or only local evidence. Review contents before any copy.
Tracked copies would not replace operational files at their original paths.

Candidate small receipts:

```text
artifacts/acquisition_readiness_gate_v1/heldout_references_manifest.json
artifacts/gsm8k_admission_v1/result.json
artifacts/gsm8k_admission_v1/decisions.jsonl
artifacts/historical_corpus_import_v1/manifest.json
artifacts/phase3_protected_references_v1/manifest.json
artifacts/phase3_github_discovery_batch01_v3/{request,catalog,decisions,complete}.json
artifacts/phase3_github_discovery_batch01_v3/source-{00,01,02,03,04,05,06,07,08,09}/{commit,license}.json
artifacts/phase3_github_prepare_batch01_v1/status.json
artifacts/phase3_github_prepare_batch01_v1/{gh-allalgorithms-python,gh-godotengine-godot-demo-projects,gh-khronosgroup-gltf-blender-io,gh-khronosgroup-vulkan-samples,gh-owasp-wstg,gh-psf-black,gh-python-attrs-attrs}/{request,complete}.json
artifacts/phase3_github_clearance_batch01_v1/{request,complete}.json
```

The brace expressions describe exact existing filenames, not a broad artifact
directory exception. The historical `approval_candidate.json` is in REVIEW:
its stored status still says candidate-only, while later explicit approval and
genesis are separately recorded. Preserve it externally without rewriting it.

## PRESERVE_EXTERNALLY: required operational state

- Exact retained Phase 1 and GSM8K holdout input inventories, plus
  `artifacts/phase3_protected_references_v1/references.jsonl`. The migration
  CLI requires the historical inputs at its fixed relative artifact paths.
- The complete Phase 3 Batch 01 discovery bundle, verified source-preparation
  cache (`artifacts/phase3_github_prepare_batch01_v1/`), and clearance bundle
  (`artifacts/phase3_github_clearance_batch01_v1/`), including inventories,
  blobs, evidence JSONL and SQLite checkpoint. Small tracked receipts alone
  cannot replay their payloads.
- Retained production build state, plans, immutable accepted-item evidence,
  source clearance/license notices, raw/normalized payloads, token cache,
  exports and shards for the historical Batch 1, source approval and MIT
  builds. Restore original relative paths and verify hashes before replay.
- `artifacts/corpus_ledger_v1/ledger.sqlite`, complete prior-production
  identity state, current governance history, ownership approvals, decision
  receipts, and independently trusted ledger/approval/manifest pins. A Git
  receipt is not an independent approval authority.
- The original user-supplied `yug_phase3_source_candidates_v1.json` used to
  derive the review registry, unless a separately reviewed tracked copy is
  later approved. Its input identity must remain independently checkable.

Ignored artifact storage is not a backup. Preserve these bundles with external
manifests, checksums, access controls, and restore instructions. A clean clone
without them reproduces software/tests but not the currently retained corpus.

## SAFE_GENERATED / IGNORE / intentionally excluded

- Re-creatable diagnostic screens, proposal dossiers, and scratch test outputs
  remain under ignored `artifacts/`; they do not authorize a grant.
- `.venv/`, `.uv-cache/`, `__pycache__/`, `.pytest_cache/`, `tmp/`, logs, lock
  files, and `yug.egg-info/` remain ignored.
- Do not stage `SOURCE/`, bulk `artifacts/`, downloaded repository or dataset
  caches, raw corpus, shards, SQLite databases, checkpoints, weights, or
  secrets. Do not force-add artifacts to bypass `.gitignore`.

## Validation and clean-clone limits

- Local import closure: no missing `training.*` or `yug.*` target among the
  current runtime modules; eight untracked modules are directly imported by
  other runtime modules, and the two remaining new modules are CLI/test entry
  points.
- Current CLI documentation includes protected-reference migration,
  review-only clearance, and review-only grant dossiers. The last two create
  no production authorization.
- A clean checkout after this TRACK set is committed can run the current
  software/tests using the tracked Phase 1 resources. It cannot replay Batch
  01, reconstruct the current protected inventory, or verify the historical
  cumulative ledger until the external bundles and authority pins are restored.
- Focused Phase 2/3 migration, proposal, clearance and admission tests:
  111 passed, 0 failed, 0 skipped.
- Full `python -B -m pytest -q`: 1,997 passed, 0 failed, 2 existing skips;
  one environment warning from PyTorch because NumPy is not installed.
- `git diff --check`: PASS after this manifest and operator-guide update.
