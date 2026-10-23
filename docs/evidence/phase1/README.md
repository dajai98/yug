# Phase 1 checkpoint evidence and external state

The seven reports/receipts preserved here are byte-identical historical evidence,
not live runtime inputs or new approvals. Original paths and SHA256 hashes are in
provenance.json. Relative artifact links retain their original meaning and may not
resolve here. Earlier counts and open issues describe those historical milestones.

Preserved evidence covers MIT-heading classification/yield, autonomous-runner
validation, corpus specification and initialization. No corpus payload, token shard,
database, checkpoint, cache or Git object is included.

`tests/fixtures/phase1/legacy_model.py` is the immutable pre-fix oracle; its old
verification disclaimer is historical. `transfer_probe.py` preserves the offline
overlap helpers and payload integrity check; acquisition/encoding drivers are omitted.
Small immutable policy/license evidence, source/acquisition metadata, model-freeze
metadata and the frozen tokenizer live under `training/resources/phase1/`.
The tokenizer is copied exactly, never refitted. Historical paths inside snapshots
are provenance, not executable fallback grants.

Specification regeneration changes evidence locations/hashes and derived identities,
not allocations or source decisions. Existing local production ledgers stay bound to
their original specification and are not rewritten.

## External operational state

Bulk artifacts, historical SOURCE, raw/review downloads, corpus text, token caches,
shards, databases, checkpoints and logs remain local. Ignored storage is not a backup.
Keep controlled separate backups of immutable payloads, license/clearance evidence,
holdout inventories, plans and prior-production ledgers with manifests and hashes.

Resume existing production by restoring its original artifact layout from that backup,
verifying hashes and supplying explicit spec/registry/plan/reference/prior-production
inputs. Never substitute fixtures for operational approval or invent genesis.
Production CLI historical defaults and historical evidence-building experiments still
refer to external artifact inputs and intentionally fail closed when absent.
A Git checkout reproduces tests/specification, not a downloaded production corpus.

The Git attributes pin the checkout line endings of hash-bound source evidence and
preserve immutable snapshots byte-for-byte, so Git line-ending conversion does not
invalidate their hashes. The holdout parser test uses disposable diagnostic records;
the real 59-record holdout verification remains local historical evidence.
