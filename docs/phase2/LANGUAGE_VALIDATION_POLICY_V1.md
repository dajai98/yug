# Phase 2 production language validation v1

This is a bounded, deterministic **script-consistency** check, not a language
detector. It does not establish that Devanagari text is Nepali rather than Hindi,
or identify Nepali written in Latin transliteration. It never supplies source
clearance, changes category ownership, or grants an intended use.

## Labels and routing

Prose and Markdown may use `en`, `ne`, `mixed`, or `unknown`. Code may use
`python`, `rust`, `go`, `csharp`, `javascript`, `typescript`,
`javascript_typescript`, `c`, `cpp`, `c_cpp`, `mixed`, or `unknown`. The
programming labels are limited to retained production languages and existing
approved code-scope/subcategory names; REVIEW-only Java and Dart are not
promoted. A code label describes the programming-language route, not the
language of identifiers or comments. `mixed` and `unknown` permit code with
multilingual comments or insufficient natural-language evidence. Empty,
missing, null, and unsupported labels fail clearance deterministically.

For prose/Markdown, `en` and `ne` are checked only against clear Latin or
Devanagari script contradictions. A `mixed` label requires meaningful evidence
of both scripts. `unknown` remains permitted when script cannot establish a
natural language, including other Latin-script languages and short structured
text. Code bypasses natural-language script checks. No script result is used as
a language-specific token quota.

## Fixed calibration and limits

The expected outcomes for 23 synthetic cases were declared in
`tests/test_language_validation.py` before these thresholds were set. Controls
cover English prose/docs, Devanagari Nepali prose and Nepali technical terms,
balanced and asymmetric mixed text, Python/Rust/Go/C#/Unity, code comments,
config, URLs, identifiers, punctuation, and false/unsupported labels. The
expected-limit controls explicitly include Hindi in Devanagari and Latin
transliteration. The earlier audit's small diagnostic set was not used alone.

As a second read-only calibration, all 59 `en`/`ne` documents in the retained
frozen tokenizer sample (31 English + 8 Nepali train; 16 English + 4 Nepali
validation) passed these script checks. Their labels come from that earlier
sample and are not independent language adjudications. All 174 retained
production documents (121 `rust`, 49 `python`, 4 `go`) passed the label route;
these are code, not natural-language calibration. Neither set establishes a
production mixed-language error rate.

For one document, count Unicode letters once, separating Latin, Devanagari and
other letters. Script decisions apply only when Latin + Devanagari is at least
75% of letters. Clear single-script contradictions require at least 80 letters
of that script, at most 2 of the other, and at least 85% of all letters.
Clearly mixed contradiction requires at least 30 letters in each script and a
minority share of at least 30% of Latin + Devanagari. A `mixed` label requires
at least 20 letters in each script and a minority share of at least 20%.
These deliberately conservative engineering thresholds are versioned in
`training/language_validation.py`; they are not linguistic quality guarantees.
Short or ambiguous text retains `unknown` rather than receiving an invented
English/Nepali decision. Latin-transliterated Nepali can be flagged under a
`ne` claim; it needs separate evidence or `unknown`, not a claim that the text
is English. Hindi labeled `ne` is not distinguishable by this policy.

The scan is local, deterministic and O(n) in document characters. Production's
existing 1 MiB item limit bounds its work. No network, model, service, new
dependency or corpus-wide language detector is involved.

## Evidence and replay boundary

The request pins the full `script-consistency-v1` policy identity. Real
production admission and every resume require an **externally approved plan
SHA256** supplied by the caller. The runner's `--approved-plan-sha256` must be
recorded outside the writable build directory and checked against its compiled
plan before admission. Computing a new digest from a modified local plan is not
approval. The SQLite plan hash and sealed records are integrity evidence, not
independent authority; both are checked against the caller's original digest.
Diagnostic builds may bind a plan on first registration, but resuming them
requires the original digest explicitly. Old mutable-plan state lacking this
binding cannot be upgraded in place.

Verification compares every stored record to the approved plan. For indexed
items, it checks dedup's exact normalized hash, split and record provenance,
and recomputes non-exact shingle features from normalized content using the
current content-type/language route. Exact duplicates have no shingle blob and
must have an earlier indexed item with the same normalized hash. Accepted
content is rechecked under the current script rule during verification. A
resealed local record/clearance/hash change cannot redefine the plan or reuse
stale route-specific dedup features.
The prepared manifest includes the language policy and its complete identity
continues to bind shard provenance. Existing intended-use receipts still bind
clearance identity; this policy does not change authorization.

Historical build files are never rewritten. An old production state without
this request policy and pinned plan path cannot resume or produce a newly
authorized export under v1. Generic integrity-only shard decoding remains
separate from authorization. A fresh explicitly approved build is needed for
new v1 language evidence; preserved historical tokens are not silently
reclassified.
