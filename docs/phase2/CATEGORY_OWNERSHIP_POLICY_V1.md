# Category ownership evidence v1

Production admission requires one `owner` from the frozen 16-category budget and
one `subcategory` from that owner's existing vocabulary. Neither this policy nor
its implementation changes category percentages or tokenization. `owner` alone
charges the top-level token quota. `tags`, when present, are sorted unique
lowercase descriptors (at most 16) and never select a quota, dedup route, or
authorization. Alternative owner-like keys (`category`, `budget_owner`, etc.)
are forbidden in per-file approval and production records.

Each approved file-manifest entry carries a sealed `ownership_decision`, copied
unchanged into the sealed production-plan record. The decision binds source ID,
revision, path, raw SHA256, one owner, one subcategory, optional normalized
SHA256, tags, decision type, rationale reference, and this policy identity.
The externally supplied production-plan digest pins the record. Current
non-diagnostic production admission also requires a separate sealed ownership
approval artifact outside the mutable build directory. Its independently
supplied SHA256 pins the artifact bytes; the artifact names the exact plan
SHA256, this ownership-policy identity, approver, approval scope
(`production_admission`), and approval time. The build rechecks it on
registration, verification, resume, accounting, and export. A reviewer string
or a newly computed local digest is not approval. The caller must obtain both
expected digests from an actual external approval authority; a caller able to
replace that authority's pins is outside this hash-based trust boundary.
Older builds without this artifact fail closed for current production use.
Diagnostic builds can still exercise integrity paths without acquiring
production authority. The source's
sealed file manifest is an independent approved input; a local production
record cannot invent or replace its decision. Production verification rechecks
the record against the approved file entry before accounting and export.
Existing quality/normalization evidence binds actual normalized bytes and
token counts to the same planned item. If the decision contains a normalized
SHA256, preparation and verification check it explicitly.

Decision types are `structural`, `source_evidence`, and `manual_review`.
Structural routing is deliberately limited to an approved Python code source,
matching source subcategory and `.py` path. The actual bounded raw UTF-8
payload must parse as Python with at least one executable/code construct;
declared type and extension alone are insufficient. Rust and Go need
`source_evidence` or `manual_review` until an adequate deterministic parser
is supported. It requires the source/file approval already in force. Source-evidence
and manual-review routing require a bounded, sealed, separately hashed
file-level decision record containing the exact source/revision/path/raw hash,
owner/subcategory, rationale, decision type, and reviewer. Ambiguous semantic
overlaps, including Unity code/shaders/debugging and multilingual technical
text, need explicit evidence; keyword precedence is not an approval route.
No automatic semantic correctness claim follows from a human or source record.

Subcategory must be present in the approved file entry and match its decision.
A source-level default cannot silently relabel a file. An evidenced override
requires a new file decision and externally approved plan. `tags` are likewise
bound in the decision, but are descriptive only. The admission sidecar carries
the full clearance record, decision identity, owner, subcategory, and tags;
the raw training token stream remains category-free.

Historical plans and builds lacking this policy/decision do not gain current
production admission or authorized export by resealing local state. Prior
retained production may still be inspected as historical integrity evidence
where existing interfaces permit it; this policy does not rewrite that corpus.
Changing category, content, revision, path, or evidence requires a new
externally approved plan. This version does not classify arbitrary prose with
a model, network service, or inferred license permission.
