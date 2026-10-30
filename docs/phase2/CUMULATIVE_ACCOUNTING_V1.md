# Phase 2 cumulative corpus accounting v1

This is a production-admission control, not authorization to acquire data or
train. The frozen tokenizer, 16 category allocations, source/group caps and
3D topic/form targets are unchanged.

New production admission requires a shared `--corpus-ledger` SQLite path outside
the individual build directory and an independently trusted
`--approved-corpus-head`. The initial empty head is the `GENESIS` constant in
`training.cumulative_accounting`. Record the new head returned by the ledger
after each committed build in an external approval record; reading a head from
the mutable ledger and calling it approved is not an authority boundary. A
ledger containing a historical genesis additionally requires the externally
approved historical import identity, canonical manifest identity, and exact
manifest file SHA256 on every open, reconstruction, resume and production
continuation. None is inferred from the local manifest or ledger.
A
missing ledger is usable only with externally approved genesis. A ledger that
already has history rejects a second genesis or a stale head. The complete
prior-production inventory must match the ledger's committed build IDs; a
locally resealed inventory cannot omit a committed parent.

The ledger permits one active build at a time. Each item reserves its existing
source, source-owner, repository, domain, training-category or validation-
category, and applicable 3D topic/form usage in a shared SQLite immediate
transaction. Queries include committed and active reservations. A crash leaves
the active reservation in place for exact replay. A successful terminal build
verifies local accepted records and payloads, then seals an append-only commit
entry bound to its parent head, approved plan, path-independent build ID,
accepted-item evidence digest and usage digest. Rerun, export and replay do not
reserve again. `CorpusLedger.reconstruction()` checks the externally pinned
chain and recomputes usage from retained accepted-item and payload evidence.

Explicit `ProductionBuild.abort_corpus_reservation()` releases only an active,
uncommitted build and marks its local state ABORTED. No elapsed-time release or
automatic committed-build subtraction exists. Replacement of a committed build
is unsupported; it needs a separate reviewed lineage decision. A copied build
has the same canonical build ID and cannot consume quota as a new build.
Current-governance rejection during final commit instead rolls back that
uncommitted build's provisional usage and local acceptance, retaining its
verified prepared/token cache for a deterministic retry. This does not alter
committed lineage or turn explicit ABORTED state into a reusable claim.

For an offline check, run `python -B -m training.cumulative_accounting
--ledger <path> --root <repository-root> --approved-head <trusted-head>
--approved-historical-import-identity <trusted-import-id>
--approved-historical-manifest-identity <trusted-manifest-id>
--approved-historical-manifest-sha256 <trusted-file-sha> --reconstruct` for a
historical ledger. The command recomputes usage from committed build databases and
payloads and does not modify the ledger. The same command with
`--abort-build-id <id>` explicitly releases an active reservation; it never
releases a committed build. Both operations require the independently trusted
head rather than accepting the ledger's self-reported head as authority.

Legacy completed builds can retain their existing generic replay path. A
pre-ledger build is never silently imported. The one-time historical import in
`training.historical_corpus_import` checks each retained state database against
its sealed plan, spec and registry snapshots, clearance record, raw and
normalized bytes, token-file hash and size, and all per-item quota dimensions.
It also checks retained source split assignments and prevents one split group
or normalized document from appearing twice in the import.
It requires exactly 174 accepted items, 417,405 training tokens and 20,616
validation tokens. These are retained-set completeness checks, not new caps.
The canonical import identity includes sorted state/plan/spec/registry hashes,
ordered accepted-item evidence and reconstructed usage; it excludes paths.

`python -B -m training.historical_corpus_import --root <root> --manifest
<manifest> --preview` is read-only. To commit, replace `--preview` with
`--commit --ledger <path> --approved-head <externally trusted genesis>
--approved-import-identity <external approval of the preview identity>
--approved-manifest-identity <external approval of the manifest seal>
--approved-manifest-sha256 <external approval of the exact manifest bytes>`.
The approved import is one immutable genesis commit. Any missing old build in
a later prior-production inventory fails before new admission. A copied import
cannot consume quota twice. Historical state databases are not rewritten.
The unpinned Python reconstruction API is likewise an explicit `preview=True`
operation only; authorized reconstruction requires the external manifest pins.
The manifest file hash and seal identity are checked separately from the import
proposal identity. Verification of an existing historical genesis requires all
three pins and rejects a resealed or reformatted manifest even when its accepted
items and usage still produce the same proposal identity. This strengthens the
external verification contract without rewriting the committed genesis entry,
usage or head. Production callers pass the same three pins through
`--approved-historical-import-identity`,
`--approved-historical-manifest-identity` and
`--approved-historical-manifest-sha256`.

The local retained preview manifest is
`artifacts/historical_corpus_import_v1/manifest.json` and remains ignored
operational evidence. Its current read-only preview identity is
`sha256:c1a49d0589fd88618ec57b232ef718570421e31975a9401983472ee2582fc4bf`.
The separately approved canonical manifest identity is
`sha256:28ed4131271eabd794afe92ee269f39923f0795b05122b54f43d7ed13988411c`
and its exact file SHA256 is
`sha256:8d19c4af46aa0597622907da4a9c1ca8df85676ad57280bc8cc5c06a62bb34d2`.
The current workspace's retained local ledger has one historical genesis at
head `sha256:0a6c02abfa37c629b8a7e1622978fe30f84acb0edb270469a632bba1f456c9b6`.
These pins must be supplied from trusted external approval; the ledger cannot
authenticate the operator supplying them. Local ledgers, build databases, payloads and
shards require retained backup; Git is not their backup.

Validation tokens remain outside training-category quotas. They continue to
consume source/source-owner/repository/domain caps and the existing separate
`validation_category` counter (2B per category, not a useful final validation
allocation). No new validation percentage is introduced.
