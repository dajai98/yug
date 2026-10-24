# Slice 2 bounded screening policy v1

Scope: the approved Slice 2 implementation request only. This is not complete
privacy detection, legal clearance, reviewer authentication, or training approval.
The executable policy is `yug-screening-v1` in `training/content_screening.py`.
Its detector identity binds the rule table, implementation revision, limits and
Unicode database version. Changes require a new reviewed policy/implementation
version and fresh evidence; historical artifacts are not silently migrated.

Detector implementation revision 2 repairs quoted credential keys: bare keys and
matching single/double-quoted keys use the same `:`/`=` separators, whitespace,
value limits and outcomes. The result schema and policy identifier are unchanged;
the detector identity changes, so revision-1 evidence must be re-screened rather
than relabeled. Placeholder/contact exceptions are unchanged. This is not a
general JSON parser or support for escaped/encoded keys.

## Rules and outcomes

Both strict UTF-8 decoded raw bytes (optional BOM) and actual normalized text are
scanned. The scan view applies NFKC, converts Unicode decimal digits to ASCII and
removes U+200B/U+200C/U+200D/U+FEFF. It does not rewrite corpus text. Offsets refer
to that view and identify the raw or normalized channel.

| Rule | Outcome |
|---|---|
| PEM private-key header, even without a complete block | EXCLUDE |
| Supported AKIA, ghp_, sk- token shapes | EXCLUDE |
| Credential assignment with non-placeholder value of at least 12 characters | EXCLUDE |
| Shorter non-placeholder credential assignment | REVIEW |
| Email without a recognized public/placeholder context | REVIEW |
| International phone shape with 8–15 digits | REVIEW |
| SSN shape or English/Nepali passport/citizenship/ID label | REVIEW |
| No actionable findings | ACCEPT for screening only |

Credential placeholder prefixes are FAKE_, EXAMPLE_, YOUR_, DUMMY_, TEST_, `<`
and `${`; exact values changeme, redacted, none, null are also placeholders.
These exceptions apply only to credential assignments, never private-key/provider
rules. Reserved example email domains are benign. Emails on a line explicitly
marked copyright, ©, contact, support or documentation are treated as public
contact/attribution. These narrow lexical contexts are heuristics, not proof that
a value is public. A sensitive value disguised as an example/public contact,
unsupported provider format, encoded or split credential, or unsupported language
can evade this baseline. Such limits remain explicit residual risks.

EXCLUDE dominates REVIEW; REVIEW dominates ACCEPT. No automatic redaction occurs.
REVIEW/EXCLUDE items remain FILTERED in the existing coordinator with the explicit
`screening_REVIEW`/`screening_EXCLUDE` reason and a retained result.

## Bounds, evidence and enforcement

Each raw and normalized input is at most 1 MiB; each channel permits at most 512
candidate matches, including benign matches. At most 256 findings in total are
allowed. Exceeding any bound, decoding errors or scanner exceptions withholds
eligibility. Exceptions at the coordinator boundary use fixed sanitized messages.
This is bounded deterministic work, not a general-purpose detector timeout or
aggregate storage reservation.

`yug-screening-result-v1` binds document identity, exact raw SHA256, normalized
SHA256, policy, detector, outcome and masked findings under the existing canonical
seal. Findings contain rule/outcome/channel/view offsets and `[MASKED]`, never
matched text. Evidence is local and immutable for a build/item.

The coordinator scans before preparation succeeds; verifies results before
accounting and when checking prepared/accepted state; verifies again on export
and authorized replay. Verification recomputes the exact expected result from
bounded retained bytes and checks normalization, so a forged/resealed result does
not supply evidence. A streaming digest of ordered accepted item/result identities
plus policy/detector is embedded in the prepared manifest and thus in shards.
Authorized replay requires the current digest and the Slice 1 complete local/
embedded manifest agreement. Prior receipts alone are insufficient.

Old builds lacking evidence cannot resume eligible processing or export through
this coordinator. No historical ledger/result is rewritten to grandfather them.
The existing generic TokenShardStream integrity decoder remains available and
does not confer authorization. Prior batches may remain read-only dedup references;
that use does not authorize their contents or account their tokens again.

The known MEDIUM aggregate review-cache storage-budget issue remains an accepted
bounded Phase 1 residual and OPEN Phase 2 follow-up. This slice does not resolve it.
No benchmark expansion, quota redesign, broader quality/tagging, Core v2, model,
tokenizer, training-numerics, allocation or training execution is included.
