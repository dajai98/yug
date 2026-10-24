# Slice 2 export preservation

The coordinator now computes the proposed admission digest and complete prepared
manifest before writing export metadata. Candidate rows are streamed twice rather
than materialized in memory. Screening and use authorization are unchanged.

An existing export must have a matching complete prepared manifest, exact admission
bytes, both valid shard streams, matching embedded provenance, and a compatible
shard request/configuration. A valid completed export is returned without rewriting
its files. Malformed, incompatible or partial export artifacts are refused; choose
a fresh destination rather than repair or migrate historical artifacts in place.
This includes an interrupted first export with metadata but no final shard manifest,
an orphan `admission.part`, or an orphan `prepared.json.tmp` left by atomic
publication. Shard publication temporaries live under `shards/`, which is itself
treated as occupied; the coordinator does not overwrite them on export refusal.
The generic token-shard builder's own resume mechanism is unchanged; it does not
provide production authorization. Normal first export to a fresh destination is
unchanged.

On refusal, historical prepared/admission/shard files remain byte-identical.
The coordinator's operational `state.sqlite`, `progress.json` and writer lock may
still record the failure, including when the export destination is the build root.
These are mutable operational state, not immutable export artifacts. No prior
batch ledger is modified. This is a compatibility preflight, not a general
filesystem transaction or protection against concurrent external file tampering.

The quoted-credential-key repair remains verified and detector revision 2 is
unchanged. The LOW unquoted `null}` value-delimiter behavior and MEDIUM aggregate
review-cache storage ceiling remain OPEN. Placeholder/contact heuristics and
unsupported mixed/escaped/encoded key syntax remain documented limitations.
The subsequent independent re-review verified the Slice 2 repair; Slice 2 is
closed. The residuals above remain open follow-ups.
