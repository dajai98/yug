# Phase 2 approval status governance v1

The approval artifact and its SHA256 pin still establish its original scope.
`training.approval_governance` adds a separate, externally pinned status log for
**future** production admission, export and authorized replay. This log does not
modify the approval or any committed corpus item.

Each canonical JSONL event has schema `yug-approval-governance-event-v1`, policy
`yug-approval-governance-v1`, approval type and `sha256:` identity, exact string
scope, previous and new status, replacement identity and scope when applicable,
UTC transition time, reason/decision reference, previous event identity, and a
sealed canonical identity. The complete log's byte SHA256 must be supplied from
outside mutable build storage on every future-use invocation. A missing, stale,
malformed, truncated, reordered or contradictory log fails closed. Appending an
event retains prior lines; duplicate transitions are rejected.

`NONE -> ACTIVE` registers an approval only when its exclusive type/scope has no
current approval. `ACTIVE -> REVOKED` removes current authority. An atomic
`ACTIVE -> SUPERSEDED` transition retires A and makes the named B current for
the identical scope. A remains in the log with its replacement identity.
An already superseded or revoked approval cannot be transitioned again.
All governed types are exclusive per exact scope. A decision receipt's scope is
its item clearance identity; its own intended-use list remains enforced by the
existing receipt verifier. No status event authenticates its human author.

Current production checks cover the externally approved plan digest, ownership
approval, source clearance, every file clearance in the plan, conditional
approval where present, and each use-specific decision receipt. Existing
artifact, evidence, seal and digest checks remain mandatory. `verify()` retains
the historical integrity meaning: a later revocation does not erase what a
completed build recorded. `run`, production export and `replay_for_use` require
the current external status pin. Generic shard decoding is not authorization.
The loader hashes and parses the same bounded log bytes. Admission checks
status after file/artifact validation and immediately before the serialized
corpus reservation; a status failure inside that transaction rolls it back.
Commit rechecks before the corpus head is published. If current governance
rejects an uncommitted build at that boundary, its provisional ledger usage
and local ACCEPTED accounting are rolled back to prepared, cached inputs for
an exact retry. Explicit operator abort remains separate and irreversible.
Export reuse and fresh publication recheck after shard validation. Authorized
replay checks the pinned log after advancing the underlying iterator and
immediately before **each** token is emitted. This per-token check adds log
I/O to the authorized streaming path; generic integrity-only shard decoding
remains separate. These are future-use checks, not a rewrite of completed
historical evidence.

Historical import is a one-time committed genesis. Before an uncommitted import,
the `historical_import` identity must be current for the exact approved manifest
identity and byte SHA256. Its original pins remain immutable historical evidence;
ordinary cumulative ledger head lineage controls subsequent commits. Current
status is not retroactively applied to the already committed genesis and cannot
permit a second genesis. No historical item, approval artifact or ledger entry
is rewritten.

The caller/operator must supply the authentic **current** log byte pin. YUG can
verify its integrity and consistency but cannot authenticate that operator or
discover a later status log if the caller deliberately supplies an old file and
its matching old pin. This is an external authority boundary, not an inferred
approval from local hashes.
