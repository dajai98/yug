# Phase 2 bounded quality filtering v1

This production-only extension calls the unchanged, historically hash-pinned
`training.corpus_prep.check_quality` first. The deterministic synthetic cases and
predeclared ACCEPT/REJECT expectations are retained in
`tests/test_quality_filtering.py`. They are diagnostic fixtures, not corpus data
or a validated estimator of downstream model quality.

## Calibration and limits

The fixed set has 8 pathological cases and 13 legitimate controls across prose,
Markdown and code. Before setting thresholds, repeated-line fractions were
1.0 for the 20-fold paragraph, alternating blocks and 15-fold function; 0.918
for boilerplate documentation. The highest legitimate-control fraction was
0.333 (common code keywords). Repeated three-line block fractions were at
least 0.918 in those four bad cases and 0 in the controls. The two single-line
filler/placeholder cases had 150/105 words and unique-word fractions 0.013/0.029;
the shortest legitimate prose control had only 8 words. Punctuation had zero
alphanumeric characters; the lowest legitimate-control alphanumeric fraction
was 0.409 for a Markdown table. Zero-width padding had an invisible-character
fraction above 0.99; controls had zero.

The conservative v1 checks are:

- Repeated lines or fixed three-line blocks: at least 10 nonempty lines and
  200 characters, with at least three occurrences of eligible lines/blocks and
  at least 80% repeated/dominant character coverage. Lines shorter than 12
  characters and blocks shorter than 48 characters do not establish repetition.
- Prose/Markdown only: at least 80 words and unique-word fraction at most 0.05.
  Code is exempt from this lexical-diversity rule.
- At least 80 characters and alphanumeric fraction below 0.20 for prose/Markdown
  or below 0.10 for code; or invisible/control fraction at least 0.20.
- The existing empty, 20-character minimum and six-identical-token rules remain.
  Production searches every character of the existing maximum 1,048,576-byte
  decoded item for the configured generated markers.

These are intentionally conservative screens for the demonstrated extremes,
not a claim to catch every low-quality document. Short legitimate Python such
as `print("Hi")` still fails the existing minimum-length rule. General
truncation and content-type mismatch remain outside this slice. Legitimate
repetitive reference material may require future measured review; thresholds
must not be loosened or tightened silently.

## Evidence and compatibility

`bounded-quality-v1`, the thresholds, baseline quality configuration,
normalization routes and configured generated markers form the build's
`quality_policy` identity. Every processed item retains a compact decision:
policy identity, normalized-content SHA256 where decoding/normalization ran,
PASS/FILTERED/NOT_EVALUATED, and a reason. No full payload is duplicated in the
decision record. The build request and prepared/shard-bound manifest carry the
policy. Incompatible historical or in-progress builds require a fresh build;
the coordinator does not migrate or rewrite them. Generic shard decoding does
not confer authorization.

Each new decision also retains a canonical identity over its item, policy,
normalized-content hash, outcome and reason. Verification rejects an altered
decision identity and recomputes accepted quality results from the verified raw
payload under the current policy before accounting, export or authorized replay.
Quality-filtered reasons are likewise rechecked. Historical decision tables
without the identity field are incompatible and require a fresh build; their
records are not upgraded in place.

Metrics use fixed-width windows and counters, O(n) time and memory for a
document of at most 1 MiB. No substring enumeration, parser, model, network
service or new corpus route is used.
