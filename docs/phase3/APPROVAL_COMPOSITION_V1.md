# Phase 3.1B exact-file composition authority

This compiler combines two disjoint first-clearance batches for one registered
source at one pinned revision. It does not grant rights, admit production data,
or authorize training. All examples in tests use synthetic fixture authority.

The original `REVIEW` registry, specification, cache, protected references, and
review bundle define the source policy. Each leaf has its own external human
exact-file decision and per-file rights evidence. The composer replays both
first-clearance compilations against those same immutable inputs. It requires
three independent caller-supplied byte pins: the composition decision and the
two leaf decisions, with the leaf pins ordered by decision identity. A
self-seal or a hash computed by the compiler is not a trust pin.

The composition decision authorizes only the exact union of the two leaf file
sets. The composite clearance names both leaf decisions, manifests, clearances,
plans, and completion receipts. Original leaf bytes and identities remain
unchanged. The active source cap, split, group assignments, and unrelated source
entries remain fixed. The new source snapshot carries an explicit composition
marker so existing specification and NOTICE validators cannot interpret it as a
legacy one-grant source. Incompatible attribution or NOTICE applicability fails.

When the active registry contains another production-cleared source, the
compiler also requires an existing approved plan and a separate caller-supplied
SHA-256 pin for that plan. It validates every prior row against the active
registry, specification, exact source manifest, clearance, ownership evidence,
and any local payload. The prior plan must cover the complete active leaf and
every cleared unrelated manifest; duplicate or missing rows fail. Only the
unrelated rows carry forward unchanged. The new completion receipt records the
prior plan reference and the number of carried files. This pin is an external
authorization input, never inferred from registry membership.

Prospective production uses new plan records bound to the composite clearance.
Carried unrelated records keep their original item and source-clearance
identities. The combined plan has a new byte digest and therefore needs fresh
prospective plan and ownership approvals. An old intended-use receipt cannot be
reused if its source or specification binding changes.
The existing governance log must hold current statuses for the aggregate source,
both independent components, the plan, each file record, and intended-use
receipts. Revocation or supersession of either component blocks future use.
Historical production and accounting remain verifiable. The existing admission,
production, NOTICE, and cumulative ledger components remain the only execution
path. No fixture decision constitutes real production authority.
